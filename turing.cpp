#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <iomanip>
#include <hip/hip_runtime.h>

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

__global__ void preprocess_kernel(const unsigned char* in, float* out, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = in[idx] / 255.0f;
    }
}

__global__ void unsharp_kernel(const float* in, float* out, int width, int height, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height) {
        size_t frame_offset = (size_t)z * width * height;
        float blur_val = 0.0f;
        int k_center = 2;
        
        for (int ky = -2; ky <= 2; ++ky) {
            for (int kx = -2; kx <= 2; ++kx) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                blur_val += in[frame_offset + py * width + px] * d_k5[(ky + k_center) * 5 + (kx + k_center)];
            }
        }
        
        float current = in[frame_offset + y * width + x];
        out[frame_offset + y * width + x] = current + (current - blur_val) * strength;
    }
}

__global__ void diffusion_kernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height) {
        size_t frame_offset = (size_t)z * width * height;
        float blur_val = 0.0f;
        int k_center = 3;
        
        for (int ky = -3; ky <= 3; ++ky) {
            for (int kx = -3; kx <= 3; ++kx) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                blur_val += in[frame_offset + py * width + px] * d_k7[(ky + k_center) * 7 + (kx + k_center)];
            }
        }
        out[frame_offset + y * width + x] = fminf(fmaxf(blur_val, 0.0f), 1.0f);
    }
}

__global__ void postprocess_kernel(const float* in, unsigned char* out, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = (unsigned char)(in[idx] * 255.0f);
    }
}

// --- CPU Helpers ---

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

long get_total_frames(const std::string& input_path) {
    std::string cmd = "ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 " + input_path;
    FILE* pipe = popen(cmd.c_str(), "r");
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

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: ./turing <input> <output> <width> <height> [batch_size]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int width = std::stoi(argv[3]);
    int height = std::stoi(argv[4]);
    int batch_size = (argc == 6) ? std::stoi(argv[5]) : 32;
    
    size_t pixels_per_frame = width * height;
    size_t batch_pixels = pixels_per_frame * batch_size;
    int iterations = 100;
    float sharpen_strength = 5.0f;

    long total_frames = get_total_frames(input_path);

    float h_k5[25], h_k7[49];
    generate_gaussian_kernel(h_k5, 5, 1.0f);
    generate_gaussian_kernel(h_k7, 7, 1.8f);
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_k5), h_k5, sizeof(float) * 25));
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_k7), h_k7, sizeof(float) * 49));

    unsigned char *h_in, *h_out;
    CHECK_HIP(hipHostMalloc(&h_in, batch_pixels));
    CHECK_HIP(hipHostMalloc(&h_out, batch_pixels));

    unsigned char *d_in, *d_out;
    float *d_bufA, *d_bufB;
    CHECK_HIP(hipMalloc(&d_in, batch_pixels));
    CHECK_HIP(hipMalloc(&d_out, batch_pixels));
    CHECK_HIP(hipMalloc(&d_bufA, batch_pixels * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_bufB, batch_pixels * sizeof(float)));

    std::string read_cmd = "ffmpeg -v error -i " + input_path + " -f image2pipe -pix_fmt gray -vcodec rawvideo -";
    std::string write_cmd = "ffmpeg -v error -y -f rawvideo -vcodec rawvideo -s " + std::to_string(width) + "x" + std::to_string(height) + 
                            " -pix_fmt gray -r 30 -i - -c:v libx264 -preset ultrafast -pix_fmt yuv420p " + output_path;

    FILE* pipe_in = popen(read_cmd.c_str(), "r");
    FILE* pipe_out = popen(write_cmd.c_str(), "w");

    dim3 block2D(16, 16, 1);
    int threads1D = 256;

    long processed_frames = 0;
    size_t frames_read = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while ((frames_read = fread(h_in, pixels_per_frame, batch_size, pipe_in)) > 0) {
        
        size_t active_pixels = frames_read * pixels_per_frame;
        int blocks1D = (active_pixels + threads1D - 1) / threads1D;
        dim3 grid3D((width + block2D.x - 1) / block2D.x, (height + block2D.y - 1) / block2D.y, frames_read);

        CHECK_HIP(hipMemcpy(d_in, h_in, active_pixels, hipMemcpyHostToDevice));
        
        hipLaunchKernelGGL(preprocess_kernel, dim3(blocks1D), dim3(threads1D), 0, 0, d_in, d_bufA, active_pixels);

        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(unsharp_kernel, grid3D, block2D, 0, 0, d_bufA, d_bufB, width, height, sharpen_strength);
            hipLaunchKernelGGL(diffusion_kernel, grid3D, block2D, 0, 0, d_bufB, d_bufA, width, height);
        }

        hipLaunchKernelGGL(postprocess_kernel, dim3(blocks1D), dim3(threads1D), 0, 0, d_bufA, d_out, active_pixels);

        CHECK_HIP(hipMemcpy(h_out, d_out, active_pixels, hipMemcpyDeviceToHost));
        CHECK_HIP(hipDeviceSynchronize());
        
        fwrite(h_out, pixels_per_frame, frames_read, pipe_out);
        
        processed_frames += frames_read;
        
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(current_time - start_time).count();
        draw_progress_bar(processed_frames, total_frames, elapsed_seconds);
    }

    std::cout << "\nFinished successfully!" << std::endl;

    pclose(pipe_in);
    pclose(pipe_out);
    hipFree(d_in); hipFree(d_out); hipFree(d_bufA); hipFree(d_bufB);
    hipHostFree(h_in); hipHostFree(h_out);

    return 0;
}
