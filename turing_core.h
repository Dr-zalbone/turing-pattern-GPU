// ===========================================================================
//  turing_core.h - shared GPU core for the video, photo and live-camera apps
//
//  Header-only on purpose: every binary is a single translation unit, so the
//  __constant__ definitions below are safe here and the documented one-line
//  build commands keep working (  hipcc -O3 -ffast-math turing.cpp -o turing ).
//
//  The effect is a separable filter, so the 5x5 unsharp and 7x7 diffusion
//  convolutions are evaluated as a horizontal pass followed by a vertical
//  pass: 24 multiply-adds per pixel per iteration instead of 74, with
//  clamp-to-edge boundaries that make the result mathematically identical to
//  the full 2D convolution (they differ only in float summation order).
//
//  Kernels that run over a batch read the frame index from blockIdx.z. Callers
//  that process a single frame launch a 2-D grid, so blockIdx.z is 0 and the
//  offset collapses to a no-op - their launch code is unchanged.
// ===========================================================================
#ifndef TURING_CORE_H
#define TURING_CORE_H

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// --- Error checking -------------------------------------------------------
#define CHECK_HIP(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// --- Effect parameters: the single source of truth ------------------------
constexpr float TURING_SIGMA_SMALL  = 1.0f;  // sigma of the 5-tap unsharp kernel
constexpr float TURING_SIGMA_LARGE  = 1.8f;  // sigma of the 7-tap diffusion kernel
constexpr float TURING_SHARPEN      = 5.0f;  // unsharp strength
constexpr int   TURING_ITERS_LIVE   = 90;    // realtime budget, camera path
constexpr int   TURING_ITERS_OFFLINE = 100;  // video / photo path

// --- Launch configuration -------------------------------------------------
constexpr int TURING_BLOCK_SIDE   = 16;   // 16x16 for the 2-D passes
constexpr int TURING_BLOCK_THREADS = 256; // for the 1-D flat passes

inline dim3 turing_block_2d() {
    return dim3(TURING_BLOCK_SIDE, TURING_BLOCK_SIDE);
}

inline dim3 turing_grid_2d(int width, int height, int frames = 1) {
    return dim3((width  + TURING_BLOCK_SIDE - 1) / TURING_BLOCK_SIDE,
                (height + TURING_BLOCK_SIDE - 1) / TURING_BLOCK_SIDE,
                frames);
}

inline int turing_blocks_1d(size_t elements) {
    return (int)((elements + TURING_BLOCK_THREADS - 1) / TURING_BLOCK_THREADS);
}

// --- Separable Gaussian taps ----------------------------------------------
__constant__ float d_k5_1d[5];
__constant__ float d_k7_1d[7];

inline void generate_gaussian_kernel_1d(float* k, int size, float sigma) {
    float sum = 0.0f;
    int center = size / 2;
    for (int i = -center; i <= center; ++i) {
        float val = expf(-(i * i) / (2.0f * sigma * sigma));
        k[i + center] = val;
        sum += val;
    }
    for (int i = 0; i < size; ++i) k[i] /= sum;
}

inline void upload_gaussian_constants() {
    float h_k5_1d[5], h_k7_1d[7];
    generate_gaussian_kernel_1d(h_k5_1d, 5, TURING_SIGMA_SMALL);
    generate_gaussian_kernel_1d(h_k7_1d, 7, TURING_SIGMA_LARGE);
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_k5_1d), h_k5_1d, sizeof(float) * 5));
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_k7_1d), h_k7_1d, sizeof(float) * 7));
}

// --- Input preprocessing --------------------------------------------------
// Both are 1-D flattened over the whole (possibly multi-frame) buffer, so they
// need no frame offset.

__global__ void preprocess_gray_kernel(const unsigned char* in, float* out, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = in[idx] / 255.0f;
    }
}

__global__ void preprocess_bgr_kernel(const unsigned char* in_bgr, float* out_gray, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out_gray[idx] = (0.299f * in_bgr[idx * 3 + 2] + 0.587f * in_bgr[idx * 3 + 1] + 0.114f * in_bgr[idx * 3 + 0]) / 255.0f;
    }
}

// --- Separable passes -----------------------------------------------------
// blockIdx.z selects the frame inside a batch.

__global__ void blur_1d_h_k5_kernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        size_t base = (size_t)blockIdx.z * width * height;
        float sum = 0.0f;
        for (int k = -2; k <= 2; ++k) {
            int px = min(max(x + k, 0), width - 1);
            sum += in[base + y * width + px] * d_k5_1d[k + 2];
        }
        out[base + y * width + x] = sum;
    }
}

__global__ void blur_1d_h_k7_kernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        size_t base = (size_t)blockIdx.z * width * height;
        float sum = 0.0f;
        for (int k = -3; k <= 3; ++k) {
            int px = min(max(x + k, 0), width - 1);
            sum += in[base + y * width + px] * d_k7_1d[k + 3];
        }
        out[base + y * width + x] = sum;
    }
}

__global__ void unsharp_v_kernel(const float* in_orig, const float* in_blurred_h, float* out,
                                 int width, int height, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        size_t base = (size_t)blockIdx.z * width * height;
        float blur_val = 0.0f;
        for (int k = -2; k <= 2; ++k) {
            int py = min(max(y + k, 0), height - 1);
            blur_val += in_blurred_h[base + py * width + x] * d_k5_1d[k + 2];
        }
        float current = in_orig[base + y * width + x];
        out[base + y * width + x] = current + (current - blur_val) * strength;
    }
}

__global__ void diffusion_v_kernel(const float* in_blurred_h, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        size_t base = (size_t)blockIdx.z * width * height;
        float blur_val = 0.0f;
        for (int k = -3; k <= 3; ++k) {
            int py = min(max(y + k, 0), height - 1);
            blur_val += in_blurred_h[base + py * width + x] * d_k7_1d[k + 3];
        }
        out[base + y * width + x] = fminf(fmaxf(blur_val, 0.0f), 1.0f);
    }
}

// --- Output ---------------------------------------------------------------
// Flat 1-D over the active buffer: works for any batch size, no frame offset.

__global__ void postprocess_kernel(const float* in, unsigned char* out, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = (unsigned char)(in[idx] * 255.0f);
    }
}

// Camera path only: 2x2 box average into a half-resolution display frame.
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

#endif // TURING_CORE_H
