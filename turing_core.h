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

// --- Fused separable passes -----------------------------------------------
// One block computes a TILE_W x TILE_H tile of output. The input tile plus an
// R-pixel halo is staged into shared memory with clamp-to-edge coordinates,
// which is exactly what the clamped global reads of the original per-pass
// kernels did. The horizontal pass is then computed for every staged row and
// the vertical pass reads those rows from shared memory, so the intermediate
// buffers never touch global memory.
//
// Every thread takes part in the loads and in every __syncthreads(), including
// threads whose output pixel lies outside the frame. There are no early returns
// before the barriers.
//
// blockIdx.z selects the frame inside a batch.

constexpr int TURING_TILE_W = 32;    // output tile width  (= blockDim.x)
constexpr int TURING_TILE_H = 8;     // output tile height (= blockDim.y)
constexpr int TURING_UNSHARP_R = 2;  // radius of the 5-tap kernel
constexpr int TURING_DIFF_R    = 3;  // radius of the 7-tap kernel

// Unsharp: 5-tap horizontal + 5-tap vertical blur of A, then the sharpen step.
// B = A + (A - blur) * strength, with A taken from the staged tile.
__global__ void unsharp_fused_kernel(const float* __restrict__ A, float* __restrict__ B,
                                     int width, int height, float strength) {
    constexpr int R  = TURING_UNSHARP_R;
    constexpr int K  = 2 * R + 1;
    constexpr int AW = TURING_TILE_W + 2 * R;
    constexpr int AH = TURING_TILE_H + 2 * R;
    __shared__ float sA[AH][AW];            // input tile, clamped at load time
    __shared__ float sT[AH][TURING_TILE_W]; // horizontal 5-tap of sA

    const int tid  = threadIdx.y * TURING_TILE_W + threadIdx.x;
    const int nthr = TURING_TILE_W * TURING_TILE_H;
    const int x0 = blockIdx.x * TURING_TILE_W;
    const int y0 = blockIdx.y * TURING_TILE_H;
    const size_t base = (size_t)blockIdx.z * width * height;
    const float* Ab = A + base;
    float* Bb = B + base;

    // Stage A with clamp-to-edge coordinates.
    for (int i = tid; i < AH * AW; i += nthr) {
        int r = i / AW, c = i % AW;
        int gy = min(max(y0 - R + r, 0), height - 1);
        int gx = min(max(x0 - R + c, 0), width - 1);
        sA[r][c] = Ab[gy * width + gx];
    }
    __syncthreads();

    // Horizontal blur for every staged row. Row r therefore equals the original
    // horizontal pass at clamped row clamp(y0 - R + r).
    for (int i = tid; i < AH * TURING_TILE_W; i += nthr) {
        int r = i / TURING_TILE_W, c = i % TURING_TILE_W;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sA[r][c + k] * d_k5_1d[k];
        sT[r][c] = sum;
    }
    __syncthreads();

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = x0 + tx, y = y0 + ty;
    if (x < width && y < height) {
        float blur_val = 0.0f;
        for (int k = 0; k < K; ++k) blur_val += sT[ty + k][tx] * d_k5_1d[k];
        float current = sA[ty + R][tx + R];
        Bb[y * width + x] = current + (current - blur_val) * strength;
    }
}

// Diffusion: 7-tap horizontal + 7-tap vertical blur of B, clamped to [0, 1].
__global__ void diffusion_fused_kernel(const float* __restrict__ B, float* __restrict__ Aout,
                                       int width, int height) {
    constexpr int R  = TURING_DIFF_R;
    constexpr int K  = 2 * R + 1;
    constexpr int BW = TURING_TILE_W + 2 * R;
    constexpr int BH = TURING_TILE_H + 2 * R;
    __shared__ float sB[BH][BW];
    __shared__ float sT[BH][TURING_TILE_W];

    const int tid  = threadIdx.y * TURING_TILE_W + threadIdx.x;
    const int nthr = TURING_TILE_W * TURING_TILE_H;
    const int x0 = blockIdx.x * TURING_TILE_W;
    const int y0 = blockIdx.y * TURING_TILE_H;
    const size_t base = (size_t)blockIdx.z * width * height;
    const float* Bb = B + base;
    float* Ob = Aout + base;

    for (int i = tid; i < BH * BW; i += nthr) {
        int r = i / BW, c = i % BW;
        int gy = min(max(y0 - R + r, 0), height - 1);
        int gx = min(max(x0 - R + c, 0), width - 1);
        sB[r][c] = Bb[gy * width + gx];
    }
    __syncthreads();

    for (int i = tid; i < BH * TURING_TILE_W; i += nthr) {
        int r = i / TURING_TILE_W, c = i % TURING_TILE_W;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sB[r][c + k] * d_k7_1d[k];
        sT[r][c] = sum;
    }
    __syncthreads();

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = x0 + tx, y = y0 + ty;
    if (x < width && y < height) {
        float blur_val = 0.0f;
        for (int k = 0; k < K; ++k) blur_val += sT[ty + k][tx] * d_k7_1d[k];
        Ob[y * width + x] = fminf(fmaxf(blur_val, 0.0f), 1.0f);
    }
}

// One Turing iteration pair per loop turn: unsharp (state -> scratch), then
// diffusion (scratch -> state). Runs `iters` iterations in place on d_state.
// d_state and d_scratch each hold frames*width*height floats.
static inline void turing_iterate(float* d_state, float* d_scratch, int width, int height,
                                  int frames, int iters, hipStream_t stream) {
    dim3 block(TURING_TILE_W, TURING_TILE_H);
    dim3 grid((width  + TURING_TILE_W - 1) / TURING_TILE_W,
              (height + TURING_TILE_H - 1) / TURING_TILE_H,
              frames);
    const float strength = TURING_SHARPEN;
    for (int i = 0; i < iters; ++i) {
        hipLaunchKernelGGL(unsharp_fused_kernel,   grid, block, 0, stream,
                           d_state, d_scratch, width, height, strength);
        hipLaunchKernelGGL(diffusion_fused_kernel, grid, block, 0, stream,
                           d_scratch, d_state, width, height);
    }
    CHECK_HIP(hipGetLastError());
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
