#include <iostream>
#include <thread>
#include <mutex>
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

__constant__ float d_k5[25];
__constant__ float d_k7[49];

// --- GPU Kernels ---
__global__ void preprocess_bgr_kernel(const unsigned char* in_bgr, float* out_gray, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out_gray[idx] = (0.299f * in_bgr[idx * 3 + 2] + 0.587f * in_bgr[idx * 3 + 1] + 0.114f * in_bgr[idx * 3 + 0]) / 255.0f;
    }
}

__global__ void unsharp_kernel(const float* in, float* out, int width, int height, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float blur_val = 0.0f;
        for (int ky = -2; ky <= 2; ++ky) {
            for (int kx = -2; kx <= 2; ++kx) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                blur_val += in[py * width + px] * d_k5[(ky + 2) * 5 + (kx + 2)];
            }
        }
        float current = in[y * width + x];
        out[y * width + x] = current + (current - blur_val) * strength;
    }
}

__global__ void diffusion_kernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float blur_val = 0.0f;
        for (int ky = -3; ky <= 3; ++ky) {
            for (int kx = -3; kx <= 3; ++kx) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                blur_val += in[py * width + px] * d_k7[(ky + 3) * 7 + (kx + 3)];
            }
        }
        out[y * width + x] = fminf(fmaxf(blur_val, 0.0f), 1.0f);
    }
}

__global__ void postprocess_kernel(const float* in, unsigned char* out, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) out[idx] = (unsigned char)(in[idx] * 255.0f);
}

// --- Threading Variables ---
std::mutex raw_mutex;
cv::Mat shared_raw_frame;
std::atomic<bool> new_raw_ready(false);

std::mutex display_mutex;
cv::Mat shared_display_frame;
std::atomic<bool> new_display_ready(false);

std::atomic<bool> system_running(true);

// --- Stage 1: The Harvester (Camera Thread) ---
void camera_thread_func(cv::VideoCapture* cap) {
    cv::Mat temp_frame;
    while (system_running) {
        (*cap) >> temp_frame;
        if (temp_frame.empty()) continue;
        
        std::lock_guard<std::mutex> lock(raw_mutex);
        temp_frame.copyTo(shared_raw_frame);
        new_raw_ready = true;
    }
}

// --- Stage 2: The Number Cruncher (GPU Compute Thread) ---
void gpu_thread_func(int width, int height) {
    size_t pixels = width * height;
    int iterations = 100;
    float sharpen_strength = 5.0f;

    unsigned char *h_in_bgr, *h_out_gray;
    CHECK_HIP(hipHostMalloc(&h_in_bgr, pixels * 3));
    CHECK_HIP(hipHostMalloc(&h_out_gray, pixels));

    unsigned char *d_in_bgr, *d_out_gray;
    float *d_bufA, *d_bufB;
    CHECK_HIP(hipMalloc(&d_in_bgr, pixels * 3));
    CHECK_HIP(hipMalloc(&d_out_gray, pixels));
    CHECK_HIP(hipMalloc(&d_bufA, pixels * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_bufB, pixels * sizeof(float)));

    dim3 block2D(16, 16);
    dim3 grid2D((width + block2D.x - 1) / block2D.x, (height + block2D.y - 1) / block2D.y);
    int threads1D = 256;
    int blocks1D = (pixels + threads1D - 1) / threads1D;

    cv::Mat local_raw;
    cv::Mat out_frame(height, width, CV_8UC1, h_out_gray);

    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (system_running) {
        bool got_new = false;
        {
            std::lock_guard<std::mutex> lock(raw_mutex);
            if (new_raw_ready) {
                shared_raw_frame.copyTo(local_raw);
                new_raw_ready = false;
                got_new = true;
            }
        }

        if (!got_new || local_raw.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        memcpy(h_in_bgr, local_raw.data, pixels * 3);
        CHECK_HIP(hipMemcpy(d_in_bgr, h_in_bgr, pixels * 3, hipMemcpyHostToDevice));
        
        hipLaunchKernelGGL(preprocess_bgr_kernel, dim3(blocks1D), dim3(threads1D), 0, 0, d_in_bgr, d_bufA, pixels);
        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(unsharp_kernel, grid2D, block2D, 0, 0, d_bufA, d_bufB, width, height, sharpen_strength);
            hipLaunchKernelGGL(diffusion_kernel, grid2D, block2D, 0, 0, d_bufB, d_bufA, width, height);
        }
        hipLaunchKernelGGL(postprocess_kernel, dim3(blocks1D), dim3(threads1D), 0, 0, d_bufA, d_out_gray, pixels);

        CHECK_HIP(hipMemcpy(h_out_gray, d_out_gray, pixels, hipMemcpyDeviceToHost));
        CHECK_HIP(hipDeviceSynchronize());

        {
            std::lock_guard<std::mutex> lock(display_mutex);
            out_frame.copyTo(shared_display_frame);
            new_display_ready = true;
        }

        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - last_time).count();
        if (elapsed >= 1.0) {
            std::cout << "\rTrue GPU Compute: " << std::fixed << std::setprecision(1) << (frame_count / elapsed) << " FPS   " << std::flush;
            frame_count = 0;
            last_time = current_time;
        }
    }

    hipFree(d_in_bgr); hipFree(d_out_gray); hipFree(d_bufA); hipFree(d_bufB);
    hipHostFree(h_in_bgr); hipHostFree(h_out_gray);
}

void generate_gaussian_kernel(float* k, int size, float sigma) {
    float sum = 0.0f;
    int center = size / 2;
    for (int y = -center; y <= center; ++y) {
        for (int x = -center; x <= center; ++x) {
            float val = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            k[(y + center) * size + (x + center)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < size * size; ++i) k[i] /= sum;
}

// --- Stage 3: The Painter (Main UI Thread) ---
int main() {
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    // Force Format Order to bypass V4L2 5-FPS limit
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FPS, 30);

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    float h_k5[25], h_k7[49];
    generate_gaussian_kernel(h_k5, 5, 1.0f);
    generate_gaussian_kernel(h_k7, 7, 1.8f);
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_k5), h_k5, sizeof(float) * 25));
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_k7), h_k7, sizeof(float) * 49));

    std::cout << "Starting GUI Pipeline. Press ESC in the window to exit." << std::endl;

    std::thread cam_thread(camera_thread_func, &cap);
    std::thread compute_thread(gpu_thread_func, width, height);

    cv::namedWindow("Turing Camera", cv::WINDOW_NORMAL);
    cv::Mat local_display, display_downscaled;

    while (true) {
        bool got_display = false;
        {
            std::lock_guard<std::mutex> lock(display_mutex);
            if (new_display_ready) {
                shared_display_frame.copyTo(local_display);
                new_display_ready = false;
                got_display = true;
            }
        }

        if (got_display && !local_display.empty()) {
            // Shrink the output right before display to un-choke the WSLg RDP connection
            cv::resize(local_display, display_downscaled, cv::Size(width / 2, height / 2));
            cv::imshow("Turing Camera", display_downscaled);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
