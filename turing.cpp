#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <filesystem>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <climits>
#include <sstream>
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

// Upper bounds for command-line numbers.
static const long kMaxDim   = 16384;
static const long kMaxBatch = 1024;

// Automatic batch size (video). Kept conservative: the GPU may be shared with
// the desktop, and the speed-up flattens out well before this many frames.
static const int    kAutoBatchMax        = 64;
static const double kAutoBatchGpuShare   = 0.25;               // of free VRAM
static const size_t kAutoBatchHostCap    = (size_t)256 << 20;  // pinned host bytes
static const size_t kDeviceBytesPerPixel = 1 + 1 + 4 + 4;      // d_in, d_out, d_bufA, d_bufB
static const size_t kHostBytesPerPixel   = 1 + 1;              // h_in, h_out
static const int    kAutoBatchFallback   = 8;                  // if VRAM can't be queried

// --- CPU Helpers ---

// Quote a path for the shell that popen() hands the command to.
// POSIX sh: single quotes, with each embedded ' written as '\''.
// Windows cmd.exe: double quotes (paths containing " are rejected in main).
static std::string shell_quote(const std::string& path) {
#ifdef _WIN32
    return "\"" + path + "\"";
#else
    std::string out = "'";
    for (char c : path) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
#endif
}

// Windows double quoting cannot carry a '"' character. NTFS never allows one in
// a file name, but a bad argument should still get a clear error, not a broken command.
static bool path_is_shell_safe(const std::string& path) {
#ifdef _WIN32
    return path.find('"') == std::string::npos;
#else
    (void)path;
    return true;
#endif
}

// True only when both paths exist and refer to the same file.
static bool same_file(const std::string& a, const std::string& b) {
    std::error_code ec;
    if (!std::filesystem::exists(a, ec) || ec) return false;
    if (!std::filesystem::exists(b, ec) || ec) return false;
    bool same = std::filesystem::equivalent(a, b, ec);
    return same && !ec;
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

// Whole-string integer in [1, max_value]. Rejects "", "abc", "12x", "0", "-3",
// and values that overflow long.
static bool parse_int_in_range(const std::string& text, long max_value, int& out) {
    try {
        size_t used = 0;
        long v = std::stol(text, &used, 10);
        if (used != text.size() || v <= 0 || v > max_value) return false;
        out = (int)v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_int_arg(const char* name, const std::string& text, long max_value, int& out) {
    if (parse_int_in_range(text, max_value, out)) return true;
    std::cerr << "Error: " << name << " must be a whole number from 1 to " << max_value
              << ", got: '" << text << "'\n";
    return false;
}

// Ask ffprobe for the pixel size of the first video stream (a still image is
// a one-frame video stream too), e.g. "width=1600" / "height=1200". Parsing by
// key keeps us independent of ffprobe's field order.
// ffmpeg auto-rotates while decoding, so a stream carrying a 90/270 degree
// rotation (portrait phone video) is reported with width and height swapped:
// that is the size of the frames the decoder actually hands us. Newer files
// store it as display-matrix side data ("rotation=-90"), older ones as a
// "rotate" tag ("TAG:rotate=90").
static bool detect_source_size(const std::string& path, int& width, int& height) {
    std::string cmd = "ffprobe -v error -select_streams v:0 "
                      "-show_entries stream=width,height:stream_tags=rotate:stream_side_data=rotation "
                      "-of default=noprint_wrappers=1 " + shell_quote(path);
    FILE* pipe = popen(cmd.c_str(), POPEN_READ);
    if (!pipe) return false;

    width = height = 0;
    int rotation = 0;
    char line[256];
    while (fgets(line, sizeof(line), pipe) != NULL) {
        if      (strncmp(line, "width=",  6) == 0)      width    = atoi(line + 6);
        else if (strncmp(line, "height=", 7) == 0)      height   = atoi(line + 7);
        else if (strncmp(line, "rotation=", 9) == 0)    rotation = atoi(line + 9);
        else if (strncmp(line, "TAG:rotate=", 11) == 0) rotation = atoi(line + 11);
    }
    pclose(pipe);
    if (width <= 0 || height <= 0) return false;
    int angle = ((rotation % 360) + 360) % 360;
    if (angle == 90 || angle == 270) std::swap(width, height);
    return true;
}

// Largest batch that comfortably fits: at most kAutoBatchGpuShare of the free
// VRAM, at most kAutoBatchHostCap of pinned host memory, at most kAutoBatchMax
// frames, never more frames than the video has, and small enough that 32-bit
// kernel indexing (blockIdx.x * blockDim.x) can't overflow.
static int auto_batch_size(size_t pixels_per_frame, long total_frames, std::string& why) {
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess || free_bytes == 0) {
        (void)hipGetLastError();
        why = "auto, GPU memory unknown";
        return kAutoBatchFallback;
    }

    size_t by_gpu  = (size_t)(free_bytes * kAutoBatchGpuShare) / (pixels_per_frame * kDeviceBytesPerPixel);
    size_t by_host = kAutoBatchHostCap / (pixels_per_frame * kHostBytesPerPixel);
    size_t by_idx  = (size_t)INT_MAX / pixels_per_frame;

    size_t batch = std::min({ (size_t)kAutoBatchMax, by_gpu, by_host, by_idx });
    if (total_frames > 0) batch = std::min(batch, (size_t)total_frames);
    if (batch < 1) batch = 1;

    std::ostringstream os;
    os << "auto, " << (free_bytes >> 20) << " MiB GPU memory free";
    why = os.str();
    return (int)batch;
}

// Accept "N" or "N/D" with both parts greater than zero. ffprobe reports
// r_frame_rate as "0/0" when it genuinely doesn't know, so reject that.
static bool is_valid_rate(const std::string& s) {
    size_t slash = s.find('/');
    std::string num = (slash == std::string::npos) ? s : s.substr(0, slash);
    std::string den = (slash == std::string::npos) ? "1" : s.substr(slash + 1);
    if (num.empty() || den.empty()) return false;
    if (den.find('/') != std::string::npos) return false;   // more than one slash
    for (char c : num) if (c < '0' || c > '9') return false;
    for (char c : den) if (c < '0' || c > '9') return false;
    try { return std::stol(num) > 0 && std::stol(den) > 0; } catch (...) { return false; }
}

// The source frame rate as a rational, e.g. "30/1", "24000/1001". Passing the
// rational straight through keeps NTSC rates exact instead of rounding 29.97.
// Falls back to 30/1 (the old hardcoded value) with a warning if ffprobe can't tell us.
static std::string detect_frame_rate(const std::string& path) {
    std::string cmd = "ffprobe -v error -select_streams v:0 "
                      "-show_entries stream=r_frame_rate "
                      "-of default=noprint_wrappers=1 " + shell_quote(path);
    std::string rate;
    FILE* pipe = popen(cmd.c_str(), POPEN_READ);
    if (pipe) {
        char line[128];
        while (fgets(line, sizeof(line), pipe) != NULL) {
            if (strncmp(line, "r_frame_rate=", 13) != 0) continue;
            std::string value = line + 13;
            while (!value.empty() && (value.back() == '\n' || value.back() == '\r' ||
                                      value.back() == ' '  || value.back() == '\t'))
                value.pop_back();
            if (is_valid_rate(value)) rate = value;
        }
        pclose(pipe);
    }
    if (rate.empty()) {
        std::cerr << "Warning: could not read the frame rate from " << path
                  << "; assuming 30/1\n";
        return "30/1";
    }
    return rate;
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
    if (total <= 0 || elapsed_seconds <= 0.0) return;
    // The packet count can be inaccurate, so keep the bar and ETA in range.
    float progress = (float)current / total;
    if (progress > 1.0f) progress = 1.0f;
    int barWidth = 40;

    double fps = current / elapsed_seconds;
    double eta_seconds = (fps > 0.0) ? (total - current) / fps : 0.0;
    if (eta_seconds < 0.0) eta_seconds = 0.0;

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
    std::cerr << "Usage: ./turing <input> <output> [width height] [--batch N] [--no-audio]\n"
              << "       width height  output size (default: the source size)\n"
              << "       --batch N     frames per GPU batch, 1-" << kMaxBatch
              << " (default: picked from free GPU memory)\n"
              << "       --no-audio    don't copy the source audio into the output\n";
}

static void print_usage_photo() {
    std::cerr << "Usage: ./turing <input> <output> [width height]\n"
              << "       photo mode (input is an image: jpg/png/bmp/webp/tif)\n";
}

int main(int argc, char** argv) {
#ifndef _WIN32
    // A dead encoder should make fwrite() come back short, not kill us with SIGPIPE.
    std::signal(SIGPIPE, SIG_IGN);
#endif

    // Flags may appear anywhere on the command line and are stripped first, so
    // every positional argument keeps the position it has always had.
    bool copy_audio = true;
    bool batch_given = false;          // --batch / -b / --batch=N seen
    std::string batch_flag;
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-audio" || arg == "-an") { copy_audio = false; continue; }
        if (arg == "--batch" || arg == "-b") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " needs a value, e.g. " << arg << " 16\n";
                return 1;
            }
            batch_given = true;
            batch_flag = argv[++i];
            continue;
        }
        if (arg.compare(0, 8, "--batch=") == 0) {
            batch_given = true;
            batch_flag = arg.substr(8);
            continue;
        }
        pos.push_back(argv[i]);
    }

    if (pos.size() < 2) {
        if (pos.size() == 1 && has_image_extension(pos[0])) print_usage_photo();
        else                                                 print_usage_video();
        return 1;
    }

    std::string input_path = pos[0];
    std::string output_path = pos[1];
    bool image_mode = has_image_extension(input_path);

    if (!path_is_shell_safe(input_path) || !path_is_shell_safe(output_path)) {
        std::cerr << "Error: paths containing a double quote are not supported on Windows\n";
        return 1;
    }
    // Writing the output would truncate the file ffmpeg is still reading.
    if (same_file(input_path, output_path)) {
        std::cerr << "Error: input and output are the same file: " << input_path << "\n";
        return 1;
    }

    int width = 0, height = 0;
    int batch_size = 1;
    std::string batch_source;          // shown in the "Batch size" line; empty = auto
    std::string frame_rate = "30/1";

    if (image_mode) {
        if (!has_image_extension(output_path)) {
            std::cerr << "Error: photo mode needs an image output extension "
                         "(png/jpg/bmp/webp/tif), got: " << output_path << "\n";
            return 1;
        }
        if (pos.size() == 4) {
            if (!parse_int_arg("width", pos[2], kMaxDim, width) ||
                !parse_int_arg("height", pos[3], kMaxDim, height)) {
                print_usage_photo();
                return 1;
            }
        } else if (pos.size() == 2) {
            if (!detect_source_size(input_path, width, height)) {
                std::cerr << "Error: could not read image dimensions from " << input_path
                          << ".\nPass them explicitly: ./turing <input> <output> <width> <height>\n";
                return 1;
            }
            std::cout << "Detected " << width << "x" << height << " from " << input_path << std::endl;
        } else {
            print_usage_photo();
            return 1;
        }
        if (batch_given) std::cerr << "Note: --batch is ignored in photo mode\n";
        batch_size = 1;
    } else {
        // in out | in out W H | in out W H batch (old form, still accepted)
        if (pos.size() != 2 && pos.size() != 4 && pos.size() != 5) {
            print_usage_video();
            return 1;
        }
        if (pos.size() == 5 && batch_given) {
            std::cerr << "Error: batch size given both as a 5th argument and with --batch; use only --batch\n";
            return 1;
        }
        if (batch_given) {
            if (!parse_int_arg("--batch", batch_flag, kMaxBatch, batch_size)) {
                print_usage_video();
                return 1;
            }
            batch_source = "--batch";
        }
        if (pos.size() == 5) {
            if (!parse_int_arg("batch_size", pos[4], kMaxBatch, batch_size)) {
                print_usage_video();
                return 1;
            }
            batch_source = "5th argument";
            std::cerr << "Note: the positional batch_size is deprecated; use --batch "
                      << batch_size << " instead\n";
        }

        if (pos.size() >= 4) {
            if (!parse_int_arg("width", pos[2], kMaxDim, width) ||
                !parse_int_arg("height", pos[3], kMaxDim, height)) {
                print_usage_video();
                return 1;
            }
            if (width % 2 != 0 || height % 2 != 0) {
                std::cerr << "Error: video output needs even width and height for yuv420p "
                             "(got " << width << "x" << height << ")\n";
                return 1;
            }
        } else {
            if (!detect_source_size(input_path, width, height)) {
                std::cerr << "Error: could not read the video size from " << input_path
                          << ".\nPass it explicitly: ./turing <input> <output> <width> <height>\n";
                return 1;
            }
            if (width > kMaxDim || height > kMaxDim) {
                std::cerr << "Error: source is " << width << "x" << height << ", larger than "
                          << kMaxDim << "; pass a smaller <width> <height> to downscale\n";
                return 1;
            }
            // yuv420p needs even sizes; drop the odd row/column rather than fail.
            int even_w = width & ~1, even_h = height & ~1;
            if (even_w < 2 || even_h < 2) {
                std::cerr << "Error: source is too small (" << width << "x" << height << ")\n";
                return 1;
            }
            std::cout << "Detected " << width << "x" << height << " from " << input_path;
            if (even_w != width || even_h != height)
                std::cout << " (using " << even_w << "x" << even_h << ": yuv420p needs even sizes)";
            std::cout << std::endl;
            width = even_w;
            height = even_h;
        }
        frame_rate = detect_frame_rate(input_path);
    }

    size_t pixels_per_frame = (size_t)width * (size_t)height;
    int iterations = TURING_ITERS_OFFLINE;

    long total_frames = image_mode ? 0 : get_total_frames(input_path);

    if (!image_mode) {
        if (batch_source.empty())
            batch_size = auto_batch_size(pixels_per_frame, total_frames, batch_source);
        std::cout << "Batch size: " << batch_size << " (" << batch_source << ")" << std::endl;
    }
    size_t batch_pixels = pixels_per_frame * (size_t)batch_size;

    upload_gaussian_constants();

    unsigned char *h_in, *h_out;
    CHECK_HIP(hipHostMalloc(&h_in, batch_pixels));
    CHECK_HIP(hipHostMalloc(&h_out, batch_pixels));

    unsigned char *d_in, *d_out;
    float *d_bufA, *d_bufB;
    CHECK_HIP(hipMalloc(&d_in, batch_pixels));
    CHECK_HIP(hipMalloc(&d_out, batch_pixels));
    CHECK_HIP(hipMalloc(&d_bufA, batch_pixels * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_bufB, batch_pixels * sizeof(float)));

    // The decoder always scales to the requested size, so the frames match what
    // turing processes even when the source has different dimensions.
    std::string scale_filter = " -vf scale=" + std::to_string(width) + ":" + std::to_string(height);

    std::string read_cmd, write_cmd;
    if (image_mode) {
        // One still frame in, one still frame out; the encoder is picked from
        // the output extension, and there is no -r / -c:v because it is not a
        // video.
        read_cmd  = "ffmpeg -v error -nostdin -i " + shell_quote(input_path) +
                    " -frames:v 1" + scale_filter +
                    " -f rawvideo -pix_fmt gray -vcodec rawvideo -";
        write_cmd = "ffmpeg -v error -y -f rawvideo -vcodec rawvideo -s " +
                    std::to_string(width) + "x" + std::to_string(height) +
                    " -pix_fmt gray -i - -frames:v 1 " + shell_quote(output_path);
    } else {
        read_cmd  = "ffmpeg -v error -nostdin -i " + shell_quote(input_path) +
                    scale_filter +
                    " -f image2pipe -pix_fmt gray -vcodec rawvideo -";
        // -r sits before -i because it times the raw pipe; the output inherits
        // that rate. For audio the original file is opened as a second input
        // and mapped explicitly, so the processed frames remain the only video
        // stream while its audio is copied across untouched.
        write_cmd = "ffmpeg -v error -y -f rawvideo -vcodec rawvideo -s " +
                    std::to_string(width) + "x" + std::to_string(height) +
                    " -pix_fmt gray -r " + frame_rate + " -i -";
        if (copy_audio) {
            write_cmd += " -i " + shell_quote(input_path) + " -map 0:v -map \"1:a?\"";
        }
        write_cmd += " -c:v libx264 -preset ultrafast -pix_fmt yuv420p";
        if (copy_audio) write_cmd += " -c:a copy";
        write_cmd += " " + shell_quote(output_path);
    }

    FILE* pipe_in = popen(read_cmd.c_str(), POPEN_READ);
    FILE* pipe_out = popen(write_cmd.c_str(), POPEN_WRITE);
    if (!pipe_in || !pipe_out) {
        std::cerr << "\nError: could not launch ffmpeg. Is it on PATH?\n";
        if (pipe_in) pclose(pipe_in);
        if (pipe_out) pclose(pipe_out);
        return 1;
    }

    int threads1D = TURING_BLOCK_THREADS;

    long processed_frames = 0;
    size_t frames_read = 0;
    bool write_failed = false;

    auto start_time = std::chrono::high_resolution_clock::now();

    while ((frames_read = fread(h_in, pixels_per_frame, batch_size, pipe_in)) > 0) {

        size_t active_pixels = frames_read * pixels_per_frame;
        int blocks1D = turing_blocks_1d(active_pixels);

        CHECK_HIP(hipMemcpy(d_in, h_in, active_pixels, hipMemcpyHostToDevice));

        hipLaunchKernelGGL(preprocess_gray_kernel, dim3(blocks1D), dim3(threads1D), 0, 0,
                           d_in, d_bufA, active_pixels);

        turing_iterate(d_bufA, d_bufB, width, height, (int)frames_read, iterations, 0);

        hipLaunchKernelGGL(postprocess_kernel, dim3(blocks1D), dim3(threads1D), 0, 0,
                           d_bufA, d_out, active_pixels);

        CHECK_HIP(hipMemcpy(h_out, d_out, active_pixels, hipMemcpyDeviceToHost));
        CHECK_HIP(hipDeviceSynchronize());

        size_t written = fwrite(h_out, pixels_per_frame, frames_read, pipe_out);
        if (written != frames_read) {
            std::cerr << "\nError: encoder stopped (ffmpeg failed)" << std::endl;
            write_failed = true;
            break;
        }

        processed_frames += frames_read;

        if (!image_mode) {
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(current_time - start_time).count();
            draw_progress_bar(processed_frames, total_frames, elapsed_seconds);
        }
    }

    // Processing time excludes the encoder's final flush in pclose below.
    auto end_time = std::chrono::high_resolution_clock::now();
    bool read_failed = ferror(pipe_in) != 0;

    // Close the encoder first: it only finalizes the output file once its stdin
    // closes, and that wait is where a failed encode shows up as a non-zero status.
    int enc_status = pclose(pipe_out);
    int dec_status = pclose(pipe_in);

    bool ok = !write_failed && !read_failed && enc_status == 0 && dec_status == 0 &&
              processed_frames > 0;
    if (!ok) {
        if (!write_failed) {
            if (read_failed)           std::cerr << "\nError: read error on the decoder pipe" << std::endl;
            else if (dec_status != 0)  std::cerr << "\nError: ffmpeg could not decode " << input_path << std::endl;
            else if (enc_status != 0)  std::cerr << "\nError: ffmpeg failed writing " << output_path << std::endl;
            else                       std::cerr << "\nError: no frames were decoded from " << input_path << std::endl;
        }
        (void)hipFree(d_in); (void)hipFree(d_out); (void)hipFree(d_bufA); (void)hipFree(d_bufB);
        (void)hipHostFree(h_in); (void)hipHostFree(h_out);
        return 1;
    }

    if (image_mode) {
        double ms = std::chrono::duration<double, std::milli>(
                        end_time - start_time).count();
        std::cout << "Processed " << width << "x" << height << " in "
                  << std::fixed << std::setprecision(1) << ms << " ms -> "
                  << output_path << std::endl;
    } else {
        std::cout << "\n";
    }
    std::cout << "Finished successfully!" << std::endl;

    (void)hipFree(d_in); (void)hipFree(d_out); (void)hipFree(d_bufA); (void)hipFree(d_bufB);
    (void)hipHostFree(h_in); (void)hipHostFree(h_out);

    return 0;
}
