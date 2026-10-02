#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <opencv2/opencv.hpp>

#define CHECK_HIP(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__constant__ float d_k5_1d[5];
__constant__ float d_k7_1d[7];

__global__ void preprocess_bgr_kernel(const unsigned char* in_bgr, float* out_gray, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out_gray[idx] = (0.299f * in_bgr[idx * 3 + 2] + 0.587f * in_bgr[idx * 3 + 1] + 0.114f * in_bgr[idx * 3 + 0]) / 255.0f;
    }
}

__global__ void blur_1d_h_kernel(const float* in, float* out, int width, int height, const float* __restrict__ kernel, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float sum = 0.0f;
        int row_offset = y * width;
        for (int k = -radius; k <= radius; ++k) {
            int px = min(max(x + k, 0), width - 1);
            sum += in[row_offset + px] * kernel[k + radius];
        }
        out[row_offset + x] = sum;
    }
}

__global__ void unsharp_v_kernel(const float* in_orig, const float* in_blurred_h, float* out, int width, int height, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float blur_val = 0.0f;
        for (int k = -2; k <= 2; ++k) {
            int py = min(max(y + k, 0), height - 1);
            blur_val += in_blurred_h[py * width + x] * d_k5_1d[k + 2];
        }
        float current = in_orig[y * width + x];
        out[y * width + x] = current + (current - blur_val) * strength;
    }
}

__global__ void diffusion_v_kernel(const float* in_blurred_h, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float blur_val = 0.0f;
        for (int k = -3; k <= 3; ++k) {
            int py = min(max(y + k, 0), height - 1);
            blur_val += in_blurred_h[py * width + x] * d_k7_1d[k + 3];
        }
        out[y * width + x] = fminf(fmaxf(blur_val, 0.0f), 1.0f);
    }
}

__global__ void postprocess_downscale_half_kernel(const float* in, unsigned char* out, int src_width, int src_height) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_width = src_width / 2;
    int dst_height = src_height / 2;

    if (out_x < dst_width && out_y < dst_height) {
        int src_x = out_x * 2;
        int src_y = out_y * 2;

        float p00 = in[src_y * src_width + src_x];
        float p10 = in[src_y * src_width + src_x + 1];
        float p01 = in[(src_y + 1) * src_width + src_x];
        float p11 = in[(src_y + 1) * src_width + src_x + 1];

        float avg = (p00 + p10 + p01 + p11) * 0.25f;
        out[out_y * dst_width + out_x] = (unsigned char)(fminf(fmaxf(avg, 0.0f), 1.0f) * 255.0f);
    }
}

// --- Threading Variables ---
std::mutex raw_mutex;
std::condition_variable raw_cv;
cv::Mat shared_raw_frame;
bool new_raw_ready = false;

std::mutex display_mutex;
std::condition_variable display_cv;
cv::Mat shared_display_frame;
bool new_display_ready = false;

std::atomic<bool> system_running(true);

// --- Stage 1: The Harvester (Camera Thread) ---
void camera_thread_func(cv::VideoCapture* cap) {
    cv::Mat acquired_frame;
    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (system_running) {
        if (!cap->read(acquired_frame) || acquired_frame.empty()) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(raw_mutex);
            std::swap(shared_raw_frame, acquired_frame);
            new_raw_ready = true;
        }
        raw_cv.notify_one();

        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - last_time).count();
        if (elapsed >= 2.0) {
            std::cout << "[Camera Ingest]: " << std::fixed << std::setprecision(1) 
                      << (frame_count / elapsed) << " FPS" << std::endl;
            frame_count = 0;
            last_time = current_time;
        }
    }
}

// --- Stage 2: The Number Cruncher (GPU Compute Thread) ---
void gpu_thread_func(int width, int height) {
    size_t pixels = (size_t)width * height;
    int display_width = width / 2;
    int display_height = height / 2;
    size_t display_pixels = (size_t)display_width * display_height;

    int iterations = 35;
    float sharpen_strength = 5.0f;

    unsigned char *h_in_bgr, *h_out_display;
    CHECK_HIP(hipHostMalloc(&h_in_bgr, pixels * 3));
    CHECK_HIP(hipHostMalloc(&h_out_display, display_pixels));

    unsigned char *d_in_bgr, *d_out_display;
    float *d_bufA, *d_bufB, *d_temp;
    CHECK_HIP(hipMalloc(&d_in_bgr, pixels * 3));
    CHECK_HIP(hipMalloc(&d_out_display, display_pixels));
    CHECK_HIP(hipMalloc(&d_bufA, pixels * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_bufB, pixels * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_temp, pixels * sizeof(float)));

    dim3 block2D(16, 16);
    dim3 grid2D((width + block2D.x - 1) / block2D.x, (height + block2D.y - 1) / block2D.y);
    dim3 grid_display((display_width + block2D.x - 1) / block2D.x, (display_height + block2D.y - 1) / block2D.y);

    int threads1D = 256;
    int blocks1D = (pixels + threads1D - 1) / threads1D;

    cv::Mat local_raw;
    cv::Mat local_out_display(display_height, display_width, CV_8UC1);

    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    hipStream_t compute_stream;
    CHECK_HIP(hipStreamCreate(&compute_stream));

    while (system_running) {
        {
            std::unique_lock<std::mutex> lock(raw_mutex);
            raw_cv.wait_for(lock, std::chrono::milliseconds(50), [] {
                return new_raw_ready || !system_running;
            });

            if (!system_running) break;
            if (!new_raw_ready || shared_raw_frame.empty()) continue;

            std::swap(local_raw, shared_raw_frame);
            new_raw_ready = false;
        }

        memcpy(h_in_bgr, local_raw.data, pixels * 3);
        CHECK_HIP(hipMemcpyAsync(d_in_bgr, h_in_bgr, pixels * 3, hipMemcpyHostToDevice, compute_stream));

        hipLaunchKernelGGL(preprocess_bgr_kernel, dim3(blocks1D), dim3(threads1D), 0, compute_stream, d_in_bgr, d_bufA, pixels);

        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(blur_1d_h_kernel, grid2D, block2D, 0, compute_stream, d_bufA, d_temp, width, height, d_k5_1d, 2);
            hipLaunchKernelGGL(unsharp_v_kernel, grid2D, block2D, 0, compute_stream, d_bufA, d_temp, d_bufB, width, height, sharpen_strength);

            hipLaunchKernelGGL(blur_1d_h_kernel, grid2D, block2D, 0, compute_stream, d_bufB, d_temp, width, height, d_k7_1d, 3);
            hipLaunchKernelGGL(diffusion_v_kernel, grid2D, block2D, 0, compute_stream, d_temp, d_bufA, width, height);
        }

        hipLaunchKernelGGL(postprocess_downscale_half_kernel, grid_display, block2D, 0, compute_stream, d_bufA, d_out_display, width, height);

        CHECK_HIP(hipMemcpyAsync(h_out_display, d_out_display, display_pixels, hipMemcpyDeviceToHost, compute_stream));
        CHECK_HIP(hipStreamSynchronize(compute_stream));

        memcpy(local_out_display.data, h_out_display, display_pixels);

        {
            std::lock_guard<std::mutex> lock(display_mutex);
            std::swap(shared_display_frame, local_out_display);
            new_display_ready = true;
        }
        display_cv.notify_one();

        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - last_time).count();
        if (elapsed >= 2.0) {
            std::cout << "[GPU Pipeline]:  " << std::fixed << std::setprecision(1) 
                      << (frame_count / elapsed) << " FPS" << std::endl;
            frame_count = 0;
            last_time = current_time;
        }
    }

    hipStreamDestroy(compute_stream);
    hipFree(d_in_bgr); hipFree(d_out_display); hipFree(d_bufA); hipFree(d_bufB); hipFree(d_temp);
    hipHostFree(h_in_bgr); hipHostFree(h_out_display);
}

void generate_gaussian_kernel_1d(float* k, int size, float sigma) {
    float sum = 0.0f;
    int center = size / 2;
    for (int i = -center; i <= center; ++i) {
        float val = expf(-(i * i) / (2.0f * sigma * sigma));
        k[i + center] = val;
        sum += val;
    }
    for (int i = 0; i < size; ++i) k[i] /= sum;
}

// --- Stage 3: The Painter (Main UI Thread) ---
int main() {
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double cam_fps = cap.get(cv::CAP_PROP_FPS);
    int fourcc_code = (int)cap.get(cv::CAP_PROP_FOURCC);
    char fourcc_str[] = {
        (char)(fourcc_code & 0XFF),
        (char)((fourcc_code >> 8) & 0XFF),
        (char)((fourcc_code >> 16) & 0XFF),
        (char)((fourcc_code >> 24) & 0XFF),
        0
    };
    std::cout << "Camera initialized: " << width << "x" << height 
              << " @ " << cam_fps << " FPS, FOURCC: " << fourcc_str << std::endl;

    float h_k5_1d[5], h_k7_1d[7];
    generate_gaussian_kernel_1d(h_k5_1d, 5, 1.0f);
    generate_gaussian_kernel_1d(h_k7_1d, 7, 1.8f);
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_k5_1d), h_k5_1d, sizeof(float) * 5));
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_k7_1d), h_k7_1d, sizeof(float) * 7));

    std::cout << "Starting GUI Pipeline. Press ESC in the window to exit." << std::endl;

    std::thread cam_thread(camera_thread_func, &cap);
    std::thread compute_thread(gpu_thread_func, width, height);

    cv::namedWindow("Turing Camera", cv::WINDOW_NORMAL);
    cv::Mat local_display;
    auto last_gui_time = std::chrono::high_resolution_clock::now();
    int gui_frame_count = 0;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(display_mutex);
            display_cv.wait_for(lock, std::chrono::milliseconds(30), [] {
                return new_display_ready || !system_running;
            });

            if (!system_running) break;

            if (new_display_ready && !shared_display_frame.empty()) {
                std::swap(local_display, shared_display_frame);
                new_display_ready = false;
            }
        }

        if (!local_display.empty()) {
            cv::imshow("Turing Camera", local_display);
            gui_frame_count++;
        }

        auto current_gui_time = std::chrono::high_resolution_clock::now();
        double gui_elapsed = std::chrono::duration<double>(current_gui_time - last_gui_time).count();
        if (gui_elapsed >= 2.0) {
            std::cout << "[GUI Display]:   " << std::fixed << std::setprecision(1) 
                      << (gui_frame_count / gui_elapsed) << " FPS" << std::endl;
            gui_frame_count = 0;
            last_gui_time = current_gui_time;
        }

        if (cv::waitKey(1) == 27) break;
    }
    
    system_running = false;
    cam_thread.join();
    compute_thread.join();
    
    cap.release();
    cv::destroyAllWindows();
    std::cout << "\nSuccessfully shut down." << std::endl;

    return 0;
}
