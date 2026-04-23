// Copyright (c) Facebook, Inc. and its affiliates.
// Modified by Zhe Feng
#pragma once
#include <torch/types.h>

namespace laplace {

#if defined(WITH_CUDA) || defined(WITH_HIP)
    void LaplaceSubtraction_forward_cuda(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output
    );

    void LaplaceSubtraction_backward_query_cuda(
        const at::Tensor gradOutput,
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor gradQuery
    );

    void LaplaceSubtraction_backward_key_cuda(
        const at::Tensor gradOutput,
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor gradKey
    );

    void LaplaceSubtraction_reduce_forward_cuda(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output
    );
#endif


inline void LaplaceSubtraction_forward(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output) {
    if (query.is_cuda() && key.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
        TORCH_CHECK(query.is_cuda(), "query tensor is not on GPU!");
        TORCH_CHECK(key.is_cuda(),   "key tensor is not on GPU!");
        return LaplaceSubtraction_forward_cuda(query, key, output);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

inline void LaplaceSubtraction_backward_query(
        const at::Tensor gradOutput,
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor gradQuery) {
    if (gradOutput.is_cuda() && query.is_cuda() && key.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
        return LaplaceSubtraction_backward_query_cuda(
            gradOutput,
            query,
            key,
            gradQuery
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

inline void LaplaceSubtraction_backward_key(
        const at::Tensor gradOutput,
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor gradKey) {
    if (gradOutput.is_cuda() && query.is_cuda() && key.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
        return LaplaceSubtraction_backward_key_cuda(
            gradOutput,
            query,
            key,
            gradKey
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

// 如果需要“reduce”版本，也做相应接口
inline void LaplaceSubtraction_reduce_forward(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output) {
    if (query.is_cuda() && key.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
        return LaplaceSubtraction_reduce_forward_cuda(query, key, output);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

} // namespace laplace
