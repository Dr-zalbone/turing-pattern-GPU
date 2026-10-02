// Shadow shim for HIP SDK's __clang_hip_runtime_wrapper.h
//
// Bug: the stock wrapper force-includes <cmath> BEFORE the HIP device-math
// forward declarations. Microsoft STL 14.51 (VS 2026) added constexpr
// isgreater/isless/isunordered/... overloads; clang implicitly marks constexpr
// functions __host__ __device__ in HIP mode, which then collide with HIP's
// __device__ overloads ("cannot overload __host__ __device__ function").
//
// Fix: include __clang_cuda_math_forward_declares.h FIRST (as upstream LLVM
// documents: "We need to do this, and do it before cmath is included"), then
// chain to the real SDK wrapper via include_next.
#if defined(__HIP__) || defined(__CUDA__)
#include <__clang_cuda_math_forward_declares.h>
#endif

#include_next <__clang_hip_runtime_wrapper.h>
