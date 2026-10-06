#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <opencv2/opencv.hpp>

#include "turing_core.h"

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
void camera_thread_func(cv::VideoCapture* cap, bool use_stdin_pipe, int width, int height) {
    cv::Mat acquired_frame;
    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    if (use_stdin_pipe) {
        size_t frame_bytes = (size_t)width * height * 3;
        acquired_frame.create(height, width, CV_8UC3);

        while (system_running) {
            size_t bytes_read = 0;
            while (bytes_read < frame_bytes && system_running) {
                size_t n = fread(acquired_frame.data + bytes_read, 1, frame_bytes - bytes_read, stdin);
                if (n == 0) {
                    if (feof(stdin) || ferror(stdin)) {
                        system_running = false;
                        break;
                    }
                }
                bytes_read += n;
            }

            if (!system_running || bytes_read < frame_bytes) break;

            {
                std::lock_guard<std::mutex> lock(raw_mutex);
                acquired_frame.copyTo(shared_raw_frame);
                new_raw_ready = true;
            }
            raw_cv.notify_one();

            frame_count++;
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - last_time).count();
            if (elapsed >= 2.0) {
                std::cout << "[Pipe Ingest]:   " << std::fixed << std::setprecision(1) 
                          << (frame_count / elapsed) << " FPS" << std::endl;
                frame_count = 0;
                last_time = current_time;
            }
        }
        return;
    }

    while (system_running) {
        if (!cap->read(acquired_frame) || acquired_frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(raw_mutex);
            acquired_frame.copyTo(shared_raw_frame);
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

    int iterations = TURING_ITERS_LIVE;
    float sharpen_strength = TURING_SHARPEN;

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

    dim3 block2D = turing_block_2d();
    dim3 grid2D = turing_grid_2d(width, height);
    dim3 grid_display = turing_grid_2d(display_width, display_height);

    int threads1D = TURING_BLOCK_THREADS;
    int blocks1D = turing_blocks_1d(pixels);

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

            shared_raw_frame.copyTo(local_raw);
            new_raw_ready = false;
        }

        if (local_raw.empty() || local_raw.cols != width || local_raw.rows != height) {
            continue;
        }

        memcpy(h_in_bgr, local_raw.data, pixels * 3);
        CHECK_HIP(hipMemcpyAsync(d_in_bgr, h_in_bgr, pixels * 3, hipMemcpyHostToDevice, compute_stream));

        hipLaunchKernelGGL(preprocess_bgr_kernel, dim3(blocks1D), dim3(threads1D), 0, compute_stream, d_in_bgr, d_bufA, pixels);

        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(blur_1d_h_k5_kernel, grid2D, block2D, 0, compute_stream, d_bufA, d_temp, width, height);
            hipLaunchKernelGGL(unsharp_v_kernel, grid2D, block2D, 0, compute_stream, d_bufA, d_temp, d_bufB, width, height, sharpen_strength);

            hipLaunchKernelGGL(blur_1d_h_k7_kernel, grid2D, block2D, 0, compute_stream, d_bufB, d_temp, width, height);
            hipLaunchKernelGGL(diffusion_v_kernel, grid2D, block2D, 0, compute_stream, d_temp, d_bufA, width, height);
        }

        hipLaunchKernelGGL(postprocess_downscale_half_kernel, grid_display, block2D, 0, compute_stream, d_bufA, d_out_display, width, height);

        CHECK_HIP(hipMemcpyAsync(h_out_display, d_out_display, display_pixels, hipMemcpyDeviceToHost, compute_stream));
        CHECK_HIP(hipStreamSynchronize(compute_stream));

        memcpy(local_out_display.data, h_out_display, display_pixels);

        {
            std::lock_guard<std::mutex> lock(display_mutex);
            local_out_display.copyTo(shared_display_frame);
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

// --- Stage 3: The Painter (Main UI Thread) ---
int main(int argc, char** argv) {
    std::string video_source = (argc > 1) ? argv[1] : "0";
    cv::VideoCapture cap;
    bool use_stdin_pipe = (video_source == "-" || video_source == "pipe" || video_source == "pipe:0");

    int width = 1920;
    int height = 1080;

    if (use_stdin_pipe) {
        if (argc > 3) {
            width = std::stoi(argv[2]);
            height = std::stoi(argv[3]);
        }
        std::cout << "Streaming directly from STDIN pipe: " << width << "x" << height << " BGR24" << std::endl;
    } else {
        bool is_device_index = !video_source.empty() && 
            std::all_of(video_source.begin(), video_source.end(), ::isdigit);

        if (is_device_index) {
            int dev_id = std::stoi(video_source);
#if defined(_WIN32) || defined(_WIN64)
            cap.open(dev_id, cv::CAP_DSHOW);
#else
            cap.open(dev_id, cv::CAP_V4L2);
#endif
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open camera device " << dev_id << std::endl;
                return 1;
            }

            // ORDER MATTERS for OpenCV's DirectShow backend:
            //  - set(FOURCC) resets its internal m_fourcc to -1 after a successful
            //    setup, so any later WIDTH/HEIGHT/FPS set() re-creates the stream
            //    with the *default* subtype (RGB24). The camera then falls back to
            //    uncompressed YUY2, which saturates USB bandwidth and throttles to
            //    ~5 FPS.
            //  - set(FOURCC) must therefore run LAST, after size and FPS are
            //    negotiated. It re-applies MJPG at the current size while carrying
            //    over the frame rate negotiated by the previous step.
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
            cap.set(cv::CAP_PROP_FPS, 30);
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        } else {
            std::cout << "Connecting to stream: " << video_source << std::endl;
            cap.open(video_source, cv::CAP_FFMPEG);
            if (!cap.isOpened()) {
                cap.open(video_source);
            }
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not connect to stream " << video_source << std::endl;
                return 1;
            }
        }

        width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double cam_fps = cap.get(cv::CAP_PROP_FPS);
        int fourcc_code = (int)cap.get(cv::CAP_PROP_FOURCC);
        char fourcc_str[] = {
            (char)(fourcc_code & 0XFF),
            (char)((fourcc_code >> 8) & 0XFF),
            (char)((fourcc_code >> 16) & 0XFF),
            (char)((fourcc_code >> 24) & 0XFF),
            0
        };
        std::cout << "Stream connected: " << width << "x" << height 
                  << " @ " << cam_fps << " FPS, FOURCC: " << fourcc_str << std::endl;
    }

    upload_gaussian_constants();

    std::cout << "Starting GUI Pipeline. Press ESC in the window to exit." << std::endl;

    std::thread cam_thread(camera_thread_func, &cap, use_stdin_pipe, width, height);
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
                shared_display_frame.copyTo(local_display);
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
    
    if (cap.isOpened()) cap.release();
    cv::destroyAllWindows();
    std::cout << "\nSuccessfully shut down." << std::endl;

    return 0;
}
