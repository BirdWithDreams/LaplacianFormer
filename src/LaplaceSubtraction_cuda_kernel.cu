// Laplace L1-distance attention kernel.
// Written by Zhe Feng

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <type_traits>

using namespace at;

namespace {

template <typename scalar_t>
struct AccT { using type = float; };
template <> struct AccT<double> { using type = double; };

template <typename scalar_t>
__device__ __forceinline__ typename AccT<scalar_t>::type to_acc(scalar_t v) {
    return static_cast<typename AccT<scalar_t>::type>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_acc(typename AccT<scalar_t>::type v) {
    return static_cast<scalar_t>(v);
}

template <typename T>
__device__ __forceinline__ T my_abs(T v) { return v < T(0) ? -v : v; }

// Branchless sign emitting {-1, 0, 1}.
template <typename T>
__device__ __forceinline__ T sign_func(T v) {
    return static_cast<T>((v > T(0)) - (v < T(0)));
}

} // namespace

// ---------------------------------------------------------------------------
// Tiled forward kernel.
// Output tile: [BM rows of HW1] x [BN cols of HW2], reduction over C.
// Thread tile: each thread computes TM x TN outputs.
// ---------------------------------------------------------------------------
template <typename scalar_t,
          int BM, int BN, int BK,
          int TM, int TN>
__global__ void laplace_forward_tiled_kernel(
        const scalar_t* __restrict__ query,  
        const scalar_t* __restrict__ key,    
        scalar_t* __restrict__ output,        
        int HW1, int HW2, int C)
{
    using acc_t = typename AccT<scalar_t>::type;

    constexpr int THREADS_X = BN / TN;   
    constexpr int THREADS_Y = BM / TM;   
    constexpr int BLOCK_THREADS = THREADS_X * THREADS_Y;

    const int bh = blockIdx.z;
    const int m_base = blockIdx.y * BM;
    const int n_base = blockIdx.x * BN;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const scalar_t* Q = query  + bh * HW1 * C;
    const scalar_t* K = key    + bh * C   * HW2;
    scalar_t*       O = output + bh * HW1 * HW2;

    __shared__ acc_t sA[BM][BK + 1];
    __shared__ acc_t sB[BK][BN];

    acc_t acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = acc_t(0);

    for (int kbase = 0; kbase < C; kbase += BK) {
        #pragma unroll
        for (int idx = tid; idx < BM * BK; idx += BLOCK_THREADS) {
            int r = idx / BK;
            int c = idx - r * BK;
            int gm = m_base + r;
            int gk = kbase + c;
            acc_t v = (gm < HW1 && gk < C)
                        ? to_acc<scalar_t>(Q[gm * C + gk])
                        : acc_t(0);
            sA[r][c] = v;
        }
        #pragma unroll
        for (int idx = tid; idx < BK * BN; idx += BLOCK_THREADS) {
            int r = idx / BN;
            int c = idx - r * BN;
            int gk = kbase + r;
            int gn = n_base + c;
            acc_t v = (gk < C && gn < HW2)
                        ? to_acc<scalar_t>(K[gk * HW2 + gn])
                        : acc_t(0);
            sB[r][c] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc_t a_reg[TM];
            acc_t b_reg[TN];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) a_reg[tm] = sA[ty * TM + tm][kk];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) b_reg[tn] = sB[kk][tx * TN + tn];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc_t d = a_reg[tm] - b_reg[tn];
                    acc[tm][tn] += my_abs<acc_t>(d);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int gm = m_base + ty * TM + tm;
        if (gm >= HW1) continue;
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            int gn = n_base + tx * TN + tn;
            if (gn < HW2) {
                O[gm * HW2 + gn] = from_acc<scalar_t>(acc[tm][tn]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tiled backward-query kernel.
// Output[p, c] = sum_k out_diff[p, k] * sign(query[p, c] - key[c, k])
// Output tile: [BM rows of HW1] x [BC cols of C], reduction over HW2.
// ---------------------------------------------------------------------------
template <typename scalar_t,
          int BM, int BC, int BK,
          int TM, int TC,
          bool SPLIT_K>
__global__ void laplace_backward_query_tiled_kernel(
        const scalar_t* __restrict__ out_diff, 
        const scalar_t* __restrict__ query,    
        const scalar_t* __restrict__ key,      
        scalar_t* __restrict__ grad_query,     
        int HW1, int HW2, int C,
        int split_k, int k_chunk)
{
    using acc_t = typename AccT<scalar_t>::type;

    constexpr int THREADS_X = BC / TC;   
    constexpr int THREADS_Y = BM / TM;   
    constexpr int BLOCK_THREADS = THREADS_X * THREADS_Y;

    const int bh_split = blockIdx.z;
    const int bh = SPLIT_K ? (bh_split / split_k) : bh_split;
    const int split_idx = SPLIT_K ? (bh_split % split_k) : 0;
    const int k_start = SPLIT_K ? (split_idx * k_chunk) : 0;
    const int k_end   = SPLIT_K ? min(HW2, k_start + k_chunk) : HW2;
    const int m_base = blockIdx.y * BM;
    const int c_base = blockIdx.x * BC;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const scalar_t* OD = out_diff + bh * HW1 * HW2;
    const scalar_t* Q  = query    + bh * HW1 * C;
    const scalar_t* K  = key      + bh * C   * HW2;
    scalar_t*       GQ = grad_query + bh * HW1 * C;

    acc_t q_reg[TM][TC];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int gm = m_base + ty * TM + tm;
        #pragma unroll
        for (int tc = 0; tc < TC; ++tc) {
            int gc = c_base + tx * TC + tc;
            q_reg[tm][tc] = (gm < HW1 && gc < C)
                             ? to_acc<scalar_t>(Q[gm * C + gc])
                             : acc_t(0);
        }
    }

    __shared__ acc_t sOD[BM][BK + 1]; 
    __shared__ acc_t sK [BC][BK + 1]; 

    acc_t acc[TM][TC];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TC; ++j)
            acc[i][j] = acc_t(0);

    for (int kbase = k_start; kbase < k_end; kbase += BK) {
        #pragma unroll
        for (int idx = tid; idx < BM * BK; idx += BLOCK_THREADS) {
            int r = idx / BK;
            int c = idx - r * BK;
            int gm = m_base + r;
            int gk = kbase + c;
            acc_t v = (gm < HW1 && gk < k_end)
                        ? to_acc<scalar_t>(OD[gm * HW2 + gk])
                        : acc_t(0);
            sOD[r][c] = v;
        }
        #pragma unroll
        for (int idx = tid; idx < BC * BK; idx += BLOCK_THREADS) {
            int r = idx / BK;
            int c = idx - r * BK;
            int gc = c_base + r;
            int gk = kbase + c;
            acc_t v = (gc < C && gk < k_end)
                        ? to_acc<scalar_t>(K[gc * HW2 + gk])
                        : acc_t(0);
            sK[r][c] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc_t od_reg[TM];
            acc_t k_reg[TC];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) od_reg[tm] = sOD[ty * TM + tm][kk];
            #pragma unroll
            for (int tc = 0; tc < TC; ++tc) k_reg[tc]  = sK[tx * TC + tc][kk];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tc = 0; tc < TC; ++tc) {
                    acc_t d = q_reg[tm][tc] - k_reg[tc];
                    acc[tm][tc] += od_reg[tm] * sign_func<acc_t>(d);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int gm = m_base + ty * TM + tm;
        if (gm >= HW1) continue;
        #pragma unroll
        for (int tc = 0; tc < TC; ++tc) {
            int gc = c_base + tx * TC + tc;
            if (gc < C) {
                if (SPLIT_K) {
                    atomicAdd(&GQ[gm * C + gc], from_acc<scalar_t>(acc[tm][tc]));
                } else {
                    GQ[gm * C + gc] = from_acc<scalar_t>(acc[tm][tc]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tiled backward-key kernel.
// Output[c, k] = -sum_p out_diff[p, k] * sign(query[p, c] - key[c, k])
// Output tile: [BC rows of C] x [BN cols of HW2], reduction over HW1.
// ---------------------------------------------------------------------------
template <typename scalar_t,
          int BC, int BN, int BP,
          int TC, int TN,
          bool SPLIT_P>
__global__ void laplace_backward_key_tiled_kernel(
        const scalar_t* __restrict__ out_diff, 
        const scalar_t* __restrict__ query,    
        const scalar_t* __restrict__ key,      
        scalar_t* __restrict__ grad_key,     
        int HW1, int HW2, int C,
        int split_p, int p_chunk)
{
    using acc_t = typename AccT<scalar_t>::type;

    constexpr int THREADS_X = BN / TN;   
    constexpr int THREADS_Y = BC / TC;   
    constexpr int BLOCK_THREADS = THREADS_X * THREADS_Y;

    const int bh_split = blockIdx.z;
    const int bh = SPLIT_P ? (bh_split / split_p) : bh_split;
    const int split_idx = SPLIT_P ? (bh_split % split_p) : 0;
    const int p_start = SPLIT_P ? (split_idx * p_chunk) : 0;
    const int p_end   = SPLIT_P ? min(HW1, p_start + p_chunk) : HW1;
    const int c_base = blockIdx.y * BC;
    const int n_base = blockIdx.x * BN;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const scalar_t* OD = out_diff + bh * HW1 * HW2;
    const scalar_t* Q  = query    + bh * HW1 * C;
    const scalar_t* K  = key      + bh * C   * HW2;
    scalar_t*       GK = grad_key + bh * C   * HW2;

    acc_t k_reg[TC][TN];
    #pragma unroll
    for (int tc = 0; tc < TC; ++tc) {
        int gc = c_base + ty * TC + tc;
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            int gn = n_base + tx * TN + tn;
            k_reg[tc][tn] = (gc < C && gn < HW2)
                             ? to_acc<scalar_t>(K[gc * HW2 + gn])
                             : acc_t(0);
        }
    }

    __shared__ acc_t sOD[BP][BN + 1]; 
    __shared__ acc_t sQ [BP][BC + 1]; 

    acc_t acc[TC][TN];
    #pragma unroll
    for (int i = 0; i < TC; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = acc_t(0);

    for (int pbase = p_start; pbase < p_end; pbase += BP) {
        #pragma unroll
        for (int idx = tid; idx < BP * BN; idx += BLOCK_THREADS) {
            int r = idx / BN;
            int c = idx - r * BN;
            int gp = pbase + r;
            int gn = n_base + c;
            acc_t v = (gp < p_end && gn < HW2)
                        ? to_acc<scalar_t>(OD[gp * HW2 + gn])
                        : acc_t(0);
            sOD[r][c] = v;
        }
        #pragma unroll
        for (int idx = tid; idx < BP * BC; idx += BLOCK_THREADS) {
            int r = idx / BC;
            int c = idx - r * BC;
            int gp = pbase + r;
            int gc = c_base + c;
            acc_t v = (gp < p_end && gc < C)
                        ? to_acc<scalar_t>(Q[gp * C + gc])
                        : acc_t(0);
            sQ[r][c] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int pp = 0; pp < BP; ++pp) {
            acc_t q_reg[TC];
            acc_t od_reg[TN];
            #pragma unroll
            for (int tc = 0; tc < TC; ++tc) q_reg[tc]  = sQ[pp][ty * TC + tc];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) od_reg[tn] = sOD[pp][tx * TN + tn];
            #pragma unroll
            for (int tc = 0; tc < TC; ++tc) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc_t d = q_reg[tc] - k_reg[tc][tn];
                    acc[tc][tn] -= od_reg[tn] * sign_func<acc_t>(d);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tc = 0; tc < TC; ++tc) {
        int gc = c_base + ty * TC + tc;
        if (gc >= C) continue;
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            int gn = n_base + tx * TN + tn;
            if (gn < HW2) {
                if (SPLIT_P) {
                    atomicAdd(&GK[gc * HW2 + gn], from_acc<scalar_t>(acc[tc][tn]));
                } else {
                    GK[gc * HW2 + gn] = from_acc<scalar_t>(acc[tc][tn]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy reduce forward (one block per output element with block reduction)
// Kept so the reduce test case and public API stay the same.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void subtraction_reduce_laplace_fwd_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    scalar_t* __restrict__ output,
    int total_elems,
    int input_channels,
    int batch_size,
    int num_head,
    int q_len,
    int k_len)
{
    extern __shared__ __align__(8) unsigned char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    int idx = blockIdx.x;
    if (idx >= total_elems) return;

    int kq = k_len * q_len;
    int nh = num_head * kq;
    int b  = idx / nh;
    int tmp = idx % nh;
    int h  = tmp / kq;
    int tmp2 = tmp % kq;
    int p  = tmp2 / k_len;
    int q  = tmp2 % k_len;

    int query_offset = ((b * num_head + h) * q_len + p) * input_channels;
    int key_base     = (b * num_head + h) * input_channels * k_len;

    scalar_t thread_sum = scalar_t(0);
    for (int c = threadIdx.x; c < input_channels; c += blockDim.x) {
        scalar_t diff = query[query_offset + c] - key[key_base + c * k_len + q];
        thread_sum += (diff >= 0 ? diff : -diff);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[idx] = sdata[0];
    }
}

namespace {
    inline int get_reduce_threads(int n) { return n < 256 ? n : 256; }
}

namespace laplace {

    void subtraction_laplace_forward_cuda(
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor output,
            int batch_size, int num_head,
            int q_len, int k_len, int input_channels)
    {
        at::cuda::CUDAGuard device_guard(query.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        constexpr int BM = 64, BN = 64, BK = 16;
        constexpr int TM = 4,  TN = 4;

        dim3 block(BN / TN, BM / TM);
        dim3 grid((k_len + BN - 1) / BN, (q_len + BM - 1) / BM, batch_size * num_head);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            query.scalar_type(), "laplace_forward_tiled", ([&] {
                laplace_forward_tiled_kernel<scalar_t, BM, BN, BK, TM, TN>
                    <<<grid, block, 0, stream>>>(
                        query.data_ptr<scalar_t>(),
                        key.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        q_len, k_len, input_channels);
            }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error in laplace_forward_tiled: %s\n", cudaGetErrorString(err));
        }
    }

    void subtraction_laplace_backward_query_cuda(
            const at::Tensor output_diff,
            const at::Tensor query_data,
            const at::Tensor key_data,
            at::Tensor query_diff,
            int batch_size, int num_head,
            int q_len, int k_len, int input_channels)
    {
        at::cuda::CUDAGuard device_guard(query_data.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        constexpr int BM = 64, BC = 32, BK = 32;
        constexpr int TM = 4,  TC = 2;

        dim3 block(BC / TC, BM / TM);   
        int gx = (input_channels + BC - 1) / BC;
        int gy = (q_len + BM - 1) / BM;
        int base_blocks = gx * gy * batch_size * num_head;

        int target_blocks = 192;
        int max_split = (k_len + BK - 1) / BK;
        int split_k = 1;
        if (base_blocks < target_blocks && max_split > 1) {
            split_k = std::min(max_split,
                               (target_blocks + base_blocks - 1) / base_blocks);
            if (split_k < 1) split_k = 1;
        }

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            query_data.scalar_type(), "laplace_backward_query_tiled", ([&] {
                if (split_k > 1) {
                    query_diff.zero_();
                    int k_chunk = ((k_len + split_k - 1) / split_k + BK - 1) / BK * BK;
                    if (k_chunk < BK) k_chunk = BK;
                    dim3 grid(gx, gy, batch_size * num_head * split_k);
                    laplace_backward_query_tiled_kernel<scalar_t, BM, BC, BK, TM, TC, true>
                        <<<grid, block, 0, stream>>>(
                            output_diff.data_ptr<scalar_t>(),
                            query_data.data_ptr<scalar_t>(),
                            key_data.data_ptr<scalar_t>(),
                            query_diff.data_ptr<scalar_t>(),
                            q_len, k_len, input_channels,
                            split_k, k_chunk);
                } else {
                    dim3 grid(gx, gy, batch_size * num_head);
                    laplace_backward_query_tiled_kernel<scalar_t, BM, BC, BK, TM, TC, false>
                        <<<grid, block, 0, stream>>>(
                            output_diff.data_ptr<scalar_t>(),
                            query_data.data_ptr<scalar_t>(),
                            key_data.data_ptr<scalar_t>(),
                            query_diff.data_ptr<scalar_t>(),
                            q_len, k_len, input_channels,
                            1, k_len);
                }
            }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error in laplace_backward_query_tiled: %s\n", cudaGetErrorString(err));
        }
    }

    void subtraction_laplace_backward_key_cuda(
            const at::Tensor output_diff,
            const at::Tensor query_data,
            const at::Tensor key_data,
            at::Tensor key_diff,
            int batch_size, int num_head,
            int q_len, int k_len, int input_channels)
    {
        at::cuda::CUDAGuard device_guard(query_data.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        constexpr int BC = 32, BN = 64, BP = 32;
        constexpr int TC = 2,  TN = 4;

        dim3 block(BN / TN, BC / TC);   
        int gx = (k_len + BN - 1) / BN;
        int gy = (input_channels + BC - 1) / BC;
        int base_blocks = gx * gy * batch_size * num_head;

        int target_blocks = 192;
        int max_split = (q_len + BP - 1) / BP;
        int split_p = 1;
        if (base_blocks < target_blocks && max_split > 1) {
            split_p = std::min(max_split,
                               (target_blocks + base_blocks - 1) / base_blocks);
            if (split_p < 1) split_p = 1;
        }

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            query_data.scalar_type(), "laplace_backward_key_tiled", ([&] {
                if (split_p > 1) {
                    key_diff.zero_();
                    int p_chunk = ((q_len + split_p - 1) / split_p + BP - 1) / BP * BP;
                    if (p_chunk < BP) p_chunk = BP;
                    dim3 grid(gx, gy, batch_size * num_head * split_p);
                    laplace_backward_key_tiled_kernel<scalar_t, BC, BN, BP, TC, TN, true>
                        <<<grid, block, 0, stream>>>(
                            output_diff.data_ptr<scalar_t>(),
                            query_data.data_ptr<scalar_t>(),
                            key_data.data_ptr<scalar_t>(),
                            key_diff.data_ptr<scalar_t>(),
                            q_len, k_len, input_channels,
                            split_p, p_chunk);
                } else {
                    dim3 grid(gx, gy, batch_size * num_head);
                    laplace_backward_key_tiled_kernel<scalar_t, BC, BN, BP, TC, TN, false>
                        <<<grid, block, 0, stream>>>(
                            output_diff.data_ptr<scalar_t>(),
                            query_data.data_ptr<scalar_t>(),
                            key_data.data_ptr<scalar_t>(),
                            key_diff.data_ptr<scalar_t>(),
                            q_len, k_len, input_channels,
                            1, q_len);
                }
            }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error in laplace_backward_key_tiled: %s\n", cudaGetErrorString(err));
        }
    }

    void subtraction_reduce_laplace_forward_cuda(
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor output,
            int batch_size, int num_head,
            int q_len, int k_len, int input_channels)
    {
        int total_elems = batch_size * num_head * q_len * k_len;

        at::cuda::CUDAGuard device_guard(query.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        int blockSize = get_reduce_threads(input_channels);
        int gridSize  = total_elems;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            query.scalar_type(), "subtraction_reduce_laplace_forward_cuda", ([&] {
                size_t sharedMemSize = blockSize * sizeof(scalar_t);

                subtraction_reduce_laplace_fwd_kernel<<<gridSize, blockSize,
                                                        sharedMemSize, stream>>>(
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    total_elems,
                    input_channels,
                    batch_size,
                    num_head,
                    q_len,
                    k_len);
            }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error in subtraction_reduce_laplace_forward_cuda: %s\n", cudaGetErrorString(err));
        }
    }

} // namespace laplace
