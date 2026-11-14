#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <hip/hip_runtime.h>
#include <opencv2/opencv.hpp>

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include <fcntl.h>
#endif

#include "turing_core.h"

using Clock = std::chrono::steady_clock;

constexpr long kMaxDim = 16384;

// --- Threading Variables ---
// Each cv::Mat buffer is owned by exactly one party at a time: its producer,
// the shared slot, or its consumer. Ownership moves only via std::swap under
// the matching mutex, so no two threads ever touch the same buffer.
std::mutex raw_mutex;
std::condition_variable raw_cv;
cv::Mat shared_raw_frame;          // guarded by raw_mutex
bool new_raw_ready = false;        // guarded by raw_mutex

std::mutex display_mutex;
std::condition_variable display_cv;
cv::Mat shared_display_frame;      // guarded by display_mutex
bool new_display_ready = false;    // guarded by display_mutex

std::atomic<bool> system_running(true);
std::atomic<bool> cam_done(false);  // set when the camera thread exits

// Sets cam_done on every exit path of the camera thread.
struct DoneFlag {
    ~DoneFlag() { cam_done = true; }
};

// Flips system_running and wakes every waiter. The flag is written while each
// mutex is held, so a waiter cannot test its predicate, miss the change, and
// then sleep past the notify.
void request_shutdown() {
    {
        std::lock_guard<std::mutex> lock(raw_mutex);
        system_running = false;
    }
    raw_cv.notify_all();
    {
        std::lock_guard<std::mutex> lock(display_mutex);
        system_running = false;
    }
    display_cv.notify_all();
}

// Publishes a producer-owned frame in O(1). On return `frame` holds the buffer
// the consumer handed back last time (or an empty Mat), which nobody else uses.
void publish_raw_frame(cv::Mat& frame) {
    {
        std::lock_guard<std::mutex> lock(raw_mutex);
        std::swap(frame, shared_raw_frame);
        new_raw_ready = true;
    }
    raw_cv.notify_one();
}

// Parses the whole string as a base-10 integer in [lo, hi]. Never throws.
static bool parse_bounded_long(const std::string& s, long lo, long hi, long& out) {
    if (s.empty()) return false;
    try {
        size_t pos = 0;
        long v = std::stol(s, &pos, 10);
        if (pos != s.size()) return false;
        if (v < lo || v > hi) return false;
        out = v;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// Periodic "[tag] X FPS" reporter. tick() is called once per frame that was
// actually produced or displayed.
struct FpsMeter {
    const char* tag;
    Clock::time_point last;
    int count = 0;

    explicit FpsMeter(const char* t) : tag(t), last(Clock::now()) {}

    void tick() {
        ++count;
        Clock::time_point now = Clock::now();
        double elapsed = std::chrono::duration<double>(now - last).count();
        if (elapsed >= 2.0) {
            std::cout << tag << std::fixed << std::setprecision(1)
                      << (count / elapsed) << " FPS" << std::endl;
            count = 0;
            last = now;
        }
    }
};

// --- Stage 1a: Pipe ingest (stdin, raw BGR24 frames) ---
static void pipe_ingest(int width, int height, cv::Mat& acquired_frame) {
    const size_t frame_bytes = (size_t)width * height * 3;
    FpsMeter fps("[Pipe Ingest]:   ");

    while (system_running) {
        // Allocates only if the buffer is empty (it is after a hand-off the
        // consumer has not returned yet); otherwise a no-op.
        acquired_frame.create(height, width, CV_8UC3);

        size_t bytes_read = 0;
        while (bytes_read < frame_bytes && system_running) {
            size_t n = fread(acquired_frame.data + bytes_read, 1, frame_bytes - bytes_read, stdin);
            if (n == 0) {
                if (feof(stdin) || ferror(stdin)) break;
                // No data and no EOF: back off instead of spinning.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            bytes_read += n;
        }

        if (bytes_read < frame_bytes) {
            // EOF, read error, or shutdown in the middle of a frame.
            if (system_running) std::cerr << "Pipe input ended (EOF or read error)." << std::endl;
            request_shutdown();
            return;
        }

        publish_raw_frame(acquired_frame);
        fps.tick();
    }
}

// --- Stage 1b: Capture ingest (camera / stream via cv::VideoCapture) ---
static void capture_ingest(cv::VideoCapture* cap, cv::Mat& acquired_frame, bool pending_probe) {
    cv::Mat convert_buf;                 // producer-owned scratch for format conversion
    FpsMeter fps("[Camera Ingest]: ");
    bool warned_format = false;
    int consecutive_failures = 0;
    Clock::time_point failure_start;

    while (system_running) {
        bool got;
        if (pending_probe) {
            // The probe frame read in run_pipeline is published first, not dropped.
            pending_probe = false;
            got = true;
        } else {
            got = cap->read(acquired_frame) && !acquired_frame.empty();
        }

        if (!got) {
            if (consecutive_failures == 0) failure_start = Clock::now();
            ++consecutive_failures;
            double failing_for = std::chrono::duration<double>(Clock::now() - failure_start).count();
            if (failing_for >= 2.0) {
                std::cerr << "Camera stream lost (" << consecutive_failures
                          << " consecutive read failures over " << std::fixed << std::setprecision(1)
                          << failing_for << " s)." << std::endl;
                request_shutdown();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        consecutive_failures = 0;

        const int ch = acquired_frame.channels();
        if (acquired_frame.depth() != CV_8U || (ch != 1 && ch != 3 && ch != 4)) {
            if (!warned_format) {
                std::cerr << "[Camera Ingest] WARNING: unsupported frame format (depth "
                          << acquired_frame.depth() << ", " << ch
                          << " channels); dropping such frames. Further drops are silent." << std::endl;
                warned_format = true;
            }
            continue;
        }

        // Normalise to continuous CV_8UC3 BGR before handing off.
        cv::Mat* out = &acquired_frame;
        if (ch == 1) {
            cv::cvtColor(acquired_frame, convert_buf, cv::COLOR_GRAY2BGR);
            out = &convert_buf;
        } else if (ch == 4) {
            cv::cvtColor(acquired_frame, convert_buf, cv::COLOR_BGRA2BGR);
            out = &convert_buf;
        } else if (!acquired_frame.isContinuous()) {
            convert_buf = acquired_frame.clone();
            out = &convert_buf;
        }

        publish_raw_frame(*out);
        fps.tick();
    }
}

// --- Stage 1: The Harvester (Camera Thread) ---
void camera_thread_func(cv::VideoCapture* cap, bool use_stdin_pipe, int width, int height, cv::Mat first_frame) {
    DoneFlag done_flag;  // sets cam_done however this function exits
    try {
        cv::Mat acquired_frame = std::move(first_frame);
        if (use_stdin_pipe) {
            pipe_ingest(width, height, acquired_frame);
        } else {
            capture_ingest(cap, acquired_frame, !acquired_frame.empty());
        }
    } catch (const std::exception& e) {
        std::cerr << "[Camera Ingest] Fatal: " << e.what() << std::endl;
        request_shutdown();
    } catch (...) {
        std::cerr << "[Camera Ingest] Fatal: unknown exception" << std::endl;
        request_shutdown();
    }
}

// --- Stage 2: The Number Cruncher (GPU Compute Thread) ---
void gpu_thread_func(int width, int height) {
    size_t pixels = (size_t)width * height;
    int display_width = width / 2;
    int display_height = height / 2;
    size_t display_pixels = (size_t)display_width * display_height;

    int iterations = TURING_ITERS_LIVE;

    unsigned char *h_in_bgr, *h_out_display;
    CHECK_HIP(hipHostMalloc(&h_in_bgr, pixels * 3));
    CHECK_HIP(hipHostMalloc(&h_out_display, display_pixels));

    unsigned char *d_in_bgr, *d_out_display;
    float *d_bufA, *d_bufB;
    CHECK_HIP(hipMalloc(&d_in_bgr, pixels * 3));
    CHECK_HIP(hipMalloc(&d_out_display, display_pixels));
    CHECK_HIP(hipMalloc(&d_bufA, pixels * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_bufB, pixels * sizeof(float)));

    dim3 block2D = turing_block_2d();
    dim3 grid_display = turing_grid_2d(display_width, display_height);

    int threads1D = TURING_BLOCK_THREADS;
    int blocks1D = turing_blocks_1d(pixels);

    cv::Mat local_raw;                 // consumer-owned input frame
    cv::Mat local_out_display;         // producer-owned display frame
    bool warned_mismatch = false;

    FpsMeter gpu_fps("[GPU Pipeline]:  ");

    hipStream_t compute_stream;
    CHECK_HIP(hipStreamCreate(&compute_stream));

    // The loop is inside a try so that an exception still reaches the cleanup below.
    try {
        while (system_running) {
            {
                std::unique_lock<std::mutex> lock(raw_mutex);
                raw_cv.wait_for(lock, std::chrono::milliseconds(50), [] {
                    return new_raw_ready || !system_running;
                });

                if (!system_running) break;
                if (!new_raw_ready || shared_raw_frame.empty()) continue;

                std::swap(local_raw, shared_raw_frame);   // O(1); shared gets our old buffer back
                new_raw_ready = false;
            }

            if (local_raw.empty() || local_raw.cols != width || local_raw.rows != height ||
                local_raw.type() != CV_8UC3 || !local_raw.isContinuous()) {
                if (!warned_mismatch) {
                    std::cerr << "[GPU Pipeline] WARNING: frame " << local_raw.cols << "x" << local_raw.rows
                              << " type " << local_raw.type() << " does not match expected "
                              << width << "x" << height << " CV_8UC3; skipping it. Further mismatches are silent."
                              << std::endl;
                    warned_mismatch = true;
                }
                continue;
            }

            memcpy(h_in_bgr, local_raw.data, pixels * 3);
            CHECK_HIP(hipMemcpyAsync(d_in_bgr, h_in_bgr, pixels * 3, hipMemcpyHostToDevice, compute_stream));

            hipLaunchKernelGGL(preprocess_bgr_kernel, dim3(blocks1D), dim3(threads1D), 0, compute_stream, d_in_bgr, d_bufA, pixels);

            turing_iterate(d_bufA, d_bufB, width, height, 1, iterations, compute_stream);

            hipLaunchKernelGGL(postprocess_downscale_half_kernel, grid_display, block2D, 0, compute_stream, d_bufA, d_out_display, width, height);
            CHECK_HIP(hipGetLastError());   // launch errors for this frame's kernels

            CHECK_HIP(hipMemcpyAsync(h_out_display, d_out_display, display_pixels, hipMemcpyDeviceToHost, compute_stream));
            CHECK_HIP(hipStreamSynchronize(compute_stream));

            // Re-create in case the display consumer swapped our buffer away last time (no-op otherwise).
            local_out_display.create(display_height, display_width, CV_8UC1);
            memcpy(local_out_display.data, h_out_display, display_pixels);

            {
                std::lock_guard<std::mutex> lock(display_mutex);
                std::swap(local_out_display, shared_display_frame);
                new_display_ready = true;
            }
            display_cv.notify_one();

            gpu_fps.tick();
        }
    } catch (const std::exception& e) {
        std::cerr << "[GPU Pipeline] Fatal: " << e.what() << std::endl;
        request_shutdown();
    } catch (...) {
        std::cerr << "[GPU Pipeline] Fatal: unknown exception" << std::endl;
        request_shutdown();
    }

    // Always runs: the GPU thread releases its own resources. Errors here are
    // ignored on purpose (we are shutting down; a CHECK_HIP exit would race main).
    (void)hipStreamDestroy(compute_stream);
    (void)hipFree(d_in_bgr); (void)hipFree(d_out_display); (void)hipFree(d_bufA); (void)hipFree(d_bufB);
    (void)hipHostFree(h_in_bgr); (void)hipHostFree(h_out_display);
}

// Stops the workers and joins them. Returns false when the camera thread is still
// blocked after ~2 s (cap.read / fread cannot be interrupted); the caller must then
// exit without touching the capture object.
static bool shutdown_threads(std::thread& cam_thread, std::thread& compute_thread) {
    request_shutdown();

    // The GPU thread wakes on the notify, or within 50 ms, and exits after its current frame.
    if (compute_thread.joinable()) compute_thread.join();

    if (!cam_thread.joinable()) return true;

    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (!cam_done.load() && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (cam_done.load()) {
        cam_thread.join();
        return true;
    }
    std::cerr << "Camera thread is still blocked in capture (cap.read or stdin). Forcing exit." << std::endl;
    return false;
}

// --- Stage 3: The Painter (Main UI Thread) ---
// Owns the setup and the display loop. Threads are declared by the caller so
// that main() can always join them, even when this function throws.
static int run_pipeline(int argc, char** argv, cv::VideoCapture& cap,
                        std::thread& cam_thread, std::thread& compute_thread) {
    const std::string video_source = (argc > 1) ? argv[1] : "0";
    const bool use_stdin_pipe = (video_source == "-" || video_source == "pipe" || video_source == "pipe:0");

    int width = 1920;
    int height = 1080;
    cv::Mat probe;

    if (use_stdin_pipe) {
        if (argc == 3) {
            std::cerr << "Usage: turing_live - [width height]\n"
                      << "       give both width and height, or neither (default 1920x1080)." << std::endl;
            return 1;
        }
        if (argc > 3) {
            long w = 0, h = 0;
            if (!parse_bounded_long(argv[2], 1, kMaxDim, w) || !parse_bounded_long(argv[3], 1, kMaxDim, h)) {
                std::cerr << "Error: pipe width and height must be integers in 1.." << kMaxDim
                          << " (got '" << argv[2] << "' and '" << argv[3] << "')." << std::endl;
                return 1;
            }
            width = (int)w;
            height = (int)h;
        }
#if defined(_WIN32) || defined(_WIN64)
        // Windows stdin is text mode by default: CRLF translation and 0x1A-as-EOF corrupt raw bytes.
        if (_setmode(_fileno(stdin), _O_BINARY) == -1) {
            std::cerr << "Error: could not switch stdin to binary mode." << std::endl;
            return 1;
        }
#endif
        std::cout << "Streaming directly from STDIN pipe: " << width << "x" << height << " BGR24" << std::endl;
    } else {
        // Digits only; the numeric value is parsed safely below.
        bool is_device_index = !video_source.empty() &&
            std::all_of(video_source.begin(), video_source.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            });

        if (is_device_index) {
            long dev_id_l = 0;
            if (!parse_bounded_long(video_source, 0, std::numeric_limits<int>::max(), dev_id_l)) {
                std::cerr << "Error: camera index '" << video_source << "' is out of range." << std::endl;
                return 1;
            }
            int dev_id = (int)dev_id_l;
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

        // The probe frame is the source of truth for size. It is handed to the
        // camera thread as its first frame, so it is not lost.
        if (!cap.read(probe) || probe.empty()) probe.release();

        double cam_fps = cap.get(cv::CAP_PROP_FPS);
        int fourcc_code = (int)cap.get(cv::CAP_PROP_FOURCC);
        if (!probe.empty()) {
            width = probe.cols;
            height = probe.rows;
        } else {
            width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        }
        if (width <= 0 || height <= 0) {
            std::cerr << "Error: could not determine the frame size of the stream (got "
                      << width << "x" << height << ")." << std::endl;
            return 1;
        }

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

    // Create the window before any thread starts, so a missing display fails cleanly.
    cv::namedWindow("Turing Camera", cv::WINDOW_NORMAL);

    std::cout << "Starting GUI Pipeline. Press ESC in the window to exit." << std::endl;

    cam_thread = std::thread(camera_thread_func, &cap, use_stdin_pipe, width, height, std::move(probe));
    compute_thread = std::thread(gpu_thread_func, width, height);

    cv::Mat local_display;             // main-thread-owned
    FpsMeter gui_fps("[GUI Display]:   ");

    while (true) {
        bool got_new = false;
        {
            std::unique_lock<std::mutex> lock(display_mutex);
            display_cv.wait_for(lock, std::chrono::milliseconds(30), [] {
                return new_display_ready || !system_running;
            });

            if (!system_running) break;

            if (new_display_ready && !shared_display_frame.empty()) {
                std::swap(local_display, shared_display_frame);   // O(1)
                new_display_ready = false;
                got_new = true;
            }
        }

        // Only newly produced frames are shown and counted.
        if (got_new && !local_display.empty()) {
            cv::imshow("Turing Camera", local_display);
            gui_fps.tick();
        }

        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}

int main(int argc, char** argv) {
    std::thread cam_thread;
    std::thread compute_thread;
    cv::VideoCapture cap;   // declared before the threads that use it; it is released only after they join
    int rc = 1;

    try {
        rc = run_pipeline(argc, argv, cap, cam_thread, compute_thread);
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        rc = 1;
    } catch (...) {
        std::cerr << "Fatal error: unknown exception" << std::endl;
        rc = 1;
    }

    // Whatever happened above, stop and join the workers before any joinable std::thread is destroyed.
    const bool cam_joined = shutdown_threads(cam_thread, compute_thread);

    try {
        cv::destroyAllWindows();
    } catch (...) {
        // No display or already gone: nothing to clean up.
    }

    if (!cam_joined) {
        // Only the camera thread is stuck inside the capture call. Its memory is
        // not freed here, because the thread may still be using it.
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(0);
    }

    if (cap.isOpened()) cap.release();
    if (rc == 0) std::cout << "\nSuccessfully shut down." << std::endl;
    return rc;
}
