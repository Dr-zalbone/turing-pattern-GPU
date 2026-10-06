#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <iomanip>
#include <hip/hip_runtime.h>

#include "turing_core.h"

// UCRT only declares the underscored POSIX names, and bare `popen`/`pclose`
// are absent from _CRT_INTERNAL_NONSTDC_NAMES, so map them ourselves.
#ifdef _WIN32
#define popen  _popen
#define pclose _pclose
// _popen opens the stream in TEXT mode by default, which treats 0x1A as EOF and
// rewrites 0x0A -> 0x0D 0x0A. Both pipes carry raw grayscale bytes, so that
// silently truncates reads and corrupts writes. Ask for binary instead.
#define POPEN_READ  "rb"
#define POPEN_WRITE "wb"
#else
#define POPEN_READ  "r"
#define POPEN_WRITE "w"
#endif

// --- CPU Helpers ---

// Quote a path so it survives both POSIX sh and Windows cmd.exe.
static std::string shell_quote(const std::string& path) {
    return "\"" + path + "\"";
}

// Photo mode is selected purely by the *input* extension, so the documented
// video invocation stays exactly as it always was.
static bool has_image_extension(const std::string& path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= path.size()) return false;
    std::string ext;
    for (size_t i = dot + 1; i < path.size(); ++i) {
        char c = path[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        ext += c;
    }
    return ext == "jpg"  || ext == "jpeg" || ext == "png"  || ext == "bmp" ||
           ext == "webp" || ext == "tif"  || ext == "tiff" || ext == "pgm"  ||
           ext == "ppm";
}

// Ask ffprobe for the pixel size of a still image, e.g. "width=1600" /
// "height=1200". Parsing by key keeps us independent of ffprobe's field order.
static bool detect_image_size(const std::string& path, int& width, int& height) {
    std::string cmd = "ffprobe -v error -select_streams v:0 "
                      "-show_entries stream=width,height "
                      "-of default=noprint_wrappers=1 " + shell_quote(path);
    FILE* pipe = popen(cmd.c_str(), POPEN_READ);
    if (!pipe) return false;

    width = height = 0;
    char line[128];
    while (fgets(line, sizeof(line), pipe) != NULL) {
        if      (strncmp(line, "width=",  6) == 0) width  = atoi(line + 6);
        else if (strncmp(line, "height=", 7) == 0) height = atoi(line + 7);
    }
    pclose(pipe);
    return width > 0 && height > 0;
}

long get_total_frames(const std::string& input_path) {
    std::string cmd = "ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 " + shell_quote(input_path);
    FILE* pipe = popen(cmd.c_str(), POPEN_READ);
    if (!pipe) return 0;
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    pclose(pipe);
    try { return std::stol(result); } catch(...) { return 0; }
}

void draw_progress_bar(long current, long total, double elapsed_seconds) {
    if (total <= 0) return;
    float progress = (float)current / total;
    int barWidth = 40;
    
    double fps = current / elapsed_seconds;
    double eta_seconds = (total - current) / fps;
    
    int eta_m = (int)eta_seconds / 60;
    int eta_s = (int)eta_seconds % 60;

    std::cout << "\r[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total << ") "
              << std::fixed << std::setprecision(1) << fps << " fps | ETA: "
              << std::setfill('0') << std::setw(2) << eta_m << "m"
              << std::setfill('0') << std::setw(2) << eta_s << "s " << std::flush;
}

static void print_usage_video() {
    std::cerr << "Usage: ./turing <input> <output> <width> <height> [batch_size]\n";
}

static void print_usage_photo() {
    std::cerr << "Usage: ./turing <input> <output> [width height]\n"
              << "       photo mode (input is an image: jpg/png/bmp/webp/tif)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        if (argc >= 2 && has_image_extension(argv[1])) print_usage_photo();
        else                                             print_usage_video();
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    bool image_mode = has_image_extension(input_path);

    int width = 0, height = 0;
    int batch_size = 1;

    if (image_mode) {
        if (!has_image_extension(output_path)) {
            std::cerr << "Error: photo mode needs an image output extension "
                         "(png/jpg/bmp/webp/tif), got: " << output_path << "\n";
            return 1;
        }
        if (argc == 5) {
            width = std::stoi(argv[3]);
            height = std::stoi(argv[4]);
        } else if (argc == 3) {
            if (!detect_image_size(input_path, width, height)) {
                std::cerr << "Error: could not read image dimensions from " << input_path
                          << ".\nPass them explicitly: ./turing <input> <output> <width> <height>\n";
                return 1;
            }
            std::cout << "Detected " << width << "x" << height << " from " << input_path << std::endl;
        } else {
            print_usage_photo();
            return 1;
        }
        batch_size = 1;
    } else {
        if (argc < 5 || argc > 6) {
            print_usage_video();
            return 1;
        }
        width = std::stoi(argv[3]);
        height = std::stoi(argv[4]);
        batch_size = (argc == 6) ? std::stoi(argv[5]) : 32;
    }

    size_t pixels_per_frame = width * height;
    size_t batch_pixels = pixels_per_frame * batch_size;
    int iterations = TURING_ITERS_OFFLINE;
    float sharpen_strength = TURING_SHARPEN;

    long total_frames = image_mode ? 0 : get_total_frames(input_path);

    upload_gaussian_constants();

    unsigned char *h_in, *h_out;
    CHECK_HIP(hipHostMalloc(&h_in, batch_pixels));
    CHECK_HIP(hipHostMalloc(&h_out, batch_pixels));

    unsigned char *d_in, *d_out;
    float *d_bufA, *d_bufB, *d_temp;
    CHECK_HIP(hipMalloc(&d_in, batch_pixels));
    CHECK_HIP(hipMalloc(&d_out, batch_pixels));
    CHECK_HIP(hipMalloc(&d_bufA, batch_pixels * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_bufB, batch_pixels * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_temp, batch_pixels * sizeof(float)));

    std::string read_cmd, write_cmd;
    if (image_mode) {
        // One still frame in, one still frame out; the encoder is picked from
        // the output extension, and there is no -r / -c:v because it is not a
        // video.
        read_cmd  = "ffmpeg -v error -i " + shell_quote(input_path) +
                    " -f rawvideo -pix_fmt gray -vcodec rawvideo -";
        write_cmd = "ffmpeg -v error -y -f rawvideo -vcodec rawvideo -s " +
                    std::to_string(width) + "x" + std::to_string(height) +
                    " -pix_fmt gray -i - -frames:v 1 " + shell_quote(output_path);
    } else {
        read_cmd  = "ffmpeg -v error -i " + shell_quote(input_path) +
                    " -f image2pipe -pix_fmt gray -vcodec rawvideo -";
        write_cmd = "ffmpeg -v error -y -f rawvideo -vcodec rawvideo -s " +
                    std::to_string(width) + "x" + std::to_string(height) +
                    " -pix_fmt gray -r 30 -i - -c:v libx264 -preset ultrafast -pix_fmt yuv420p " +
                    shell_quote(output_path);
    }

    FILE* pipe_in = popen(read_cmd.c_str(), POPEN_READ);
    FILE* pipe_out = popen(write_cmd.c_str(), POPEN_WRITE);
    if (!pipe_in || !pipe_out) {
        std::cerr << "\nError: could not launch ffmpeg. Is it on PATH?\n";
        if (pipe_in) pclose(pipe_in);
        if (pipe_out) pclose(pipe_out);
        return 1;
    }

    dim3 block2D = turing_block_2d();
    int threads1D = TURING_BLOCK_THREADS;

    long processed_frames = 0;
    size_t frames_read = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while ((frames_read = fread(h_in, pixels_per_frame, batch_size, pipe_in)) > 0) {
        
        size_t active_pixels = frames_read * pixels_per_frame;
        int blocks1D = turing_blocks_1d(active_pixels);
        dim3 grid3D = turing_grid_2d(width, height, (int)frames_read);

        CHECK_HIP(hipMemcpy(d_in, h_in, active_pixels, hipMemcpyHostToDevice));
        
        hipLaunchKernelGGL(preprocess_gray_kernel, dim3(blocks1D), dim3(threads1D), 0, 0,
                           d_in, d_bufA, active_pixels);

        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(blur_1d_h_k5_kernel, grid3D, block2D, 0, 0, d_bufA, d_temp, width, height);
            hipLaunchKernelGGL(unsharp_v_kernel,    grid3D, block2D, 0, 0, d_bufA, d_temp, d_bufB, width, height, sharpen_strength);
            hipLaunchKernelGGL(blur_1d_h_k7_kernel, grid3D, block2D, 0, 0, d_bufB, d_temp, width, height);
            hipLaunchKernelGGL(diffusion_v_kernel,  grid3D, block2D, 0, 0, d_temp, d_bufA, width, height);
        }

        hipLaunchKernelGGL(postprocess_kernel, dim3(blocks1D), dim3(threads1D), 0, 0,
                           d_bufA, d_out, active_pixels);

        CHECK_HIP(hipMemcpy(h_out, d_out, active_pixels, hipMemcpyDeviceToHost));
        CHECK_HIP(hipDeviceSynchronize());
        
        fwrite(h_out, pixels_per_frame, frames_read, pipe_out);
        
        processed_frames += frames_read;

        if (!image_mode) {
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(current_time - start_time).count();
            draw_progress_bar(processed_frames, total_frames, elapsed_seconds);
        }
    }

    if (image_mode) {
        if (processed_frames == 0) {
            std::cerr << "Error: could not decode " << input_path << std::endl;
        } else {
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time).count();
            std::cout << "Processed " << width << "x" << height << " in "
                      << std::fixed << std::setprecision(1) << ms << " ms -> "
                      << output_path << std::endl;
        }
    } else {
        std::cout << "\nFinished successfully!" << std::endl;
    }

    pclose(pipe_in);
    pclose(pipe_out);
    hipFree(d_in); hipFree(d_out); hipFree(d_bufA); hipFree(d_bufB); hipFree(d_temp);
    hipHostFree(h_in); hipHostFree(h_out);

    return (image_mode && processed_frames == 0) ? 1 : 0;
}
