// Newton-Schulz-style iterative matrix inverse kernel.
// Written by Zhe Feng
#pragma once
#include <torch/types.h>

namespace laplace {

#if defined(WITH_CUDA) || defined(WITH_HIP)
void NewtonInverse_forward_cuda(
    const at::Tensor mat,
    at::Tensor output,
    int iter);

void NewtonInverse_backward_cuda(
    const at::Tensor grad_output,
    const at::Tensor inverse,
    at::Tensor grad_mat);

bool NewtonInverse_supports(int N);
#endif

inline void NewtonInverse_forward(
    const at::Tensor mat,
    at::Tensor output,
    int64_t iter) {
  TORCH_CHECK(mat.is_cuda(), "mat must be on GPU");
  TORCH_CHECK(mat.dim() == 4, "mat must be 4D (B,H,N,N)");
  TORCH_CHECK(mat.size(-1) == mat.size(-2), "last two dims of mat must be equal");
#if defined(WITH_CUDA) || defined(WITH_HIP)
  NewtonInverse_forward_cuda(mat, output, static_cast<int>(iter));
#else
  AT_ERROR("Not compiled with GPU support");
#endif
}

inline void NewtonInverse_backward(
    const at::Tensor grad_output,
    const at::Tensor inverse,
    at::Tensor grad_mat) {
  TORCH_CHECK(grad_output.is_cuda() && inverse.is_cuda(), "tensors must be on GPU");
  TORCH_CHECK(grad_output.dim() == 4 && inverse.dim() == 4, "must be 4D");
#if defined(WITH_CUDA) || defined(WITH_HIP)
  NewtonInverse_backward_cuda(grad_output, inverse, grad_mat);
#else
  AT_ERROR("Not compiled with GPU support");
#endif
}

inline bool NewtonInverse_is_supported(int64_t N) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  return NewtonInverse_supports(static_cast<int>(N));
#else
  return false;
#endif
}

} // namespace laplace
