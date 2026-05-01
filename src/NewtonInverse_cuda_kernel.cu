// Fused Newton-Schulz-style matrix inverse kernels.
// Written by Zhe Feng


#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdio.h>

// cp.async helpers (sm_80+).
#define IS_SM80 (__CUDA_ARCH__ >= 800)

namespace {

template <typename scalar_t>
struct AccT { using type = float; };
template <> struct AccT<double> { using type = double; };

template <typename scalar_t>
__device__ __forceinline__ typename AccT<scalar_t>::type to_acc(scalar_t v) {
    return static_cast<typename AccT<scalar_t>::type>(v);
}

template <typename scalar_t, typename acc_t>
__device__ __forceinline__ scalar_t from_acc(acc_t v) {
    return static_cast<scalar_t>(v);
}

// 16x16 = 256 thread block, each thread owns a TMxTN output tile.
template <int N_MAX> struct Blk {
    static constexpr int TH = 16;
    static constexpr int TW = 16;
    static constexpr int TM = N_MAX / TH;
    static constexpr int TN = N_MAX / TW;
    static_assert(TM * TH == N_MAX, "N_MAX must be multiple of 16");
    static_assert(TN * TW == N_MAX, "N_MAX must be multiple of 16");
};

template <typename acc_t, int N_MAX>
struct SmemLayout {
    static constexpr int STRIDE     = N_MAX + 1;   // +1 to avoid bank conflicts on column broadcasts
    static constexpr int NELEM      = N_MAX * STRIDE;
    static constexpr int BYTES_3BUF = 3 * NELEM * sizeof(acc_t);
    static constexpr int BYTES_2BUF = 2 * NELEM * sizeof(acc_t);
    static constexpr int BYTES_BWD  = 2 * NELEM * sizeof(acc_t);
};

// Partial-unroll factor for the inner kk loop. Full unrolling at N_MAX=128
// blows up code size and register pressure (212 regs at fp32 in baseline).
// Unrolling in groups of 8 keeps the kk body small enough to fit tidy ILP
// pipelines while still feeding FMAs back-to-back. Small-N kernels still do
// a full unroll (UF >= N_MAX means all iterations unrolled).
template <int N_MAX>
struct MatmulUnroll {
    // Keep short loops fully unrolled, but cap long ones. At N_MAX=96,112,128
    // a full unroll gives huge straight-line kk bodies (many kB of SASS) that
    // spill registers and bust the I-cache. UF=16 halves the inner-loop
    // iteration count vs UF=8 but keeps each body small enough to schedule
    // well.
    static constexpr int UF = (N_MAX <= 64) ? N_MAX : 16;
};

// ----------------------------------------------------------------------------
// cp.async helpers (sm_80+). At lower arch we fall back to a normal load.
// ----------------------------------------------------------------------------
#if IS_SM80
__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit()   { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
#endif

// ----------------------------------------------------------------------------
// alpha reduction helper: max over row-sums of the original `mat`
// ----------------------------------------------------------------------------
template <typename acc_t, int N_MAX, int BLOCK_THREADS>
__device__ __forceinline__ acc_t compute_alpha(
        acc_t* __restrict__ sPbuf, int STRIDE, int N, int tid)
{
    // Each thread tid<N owns one row sum of sPbuf (P = mat + 0.01*I).
    // The Newton-Schulz scaling must use the regularized matrix P that is
    // actually inverted. Using mat alone makes alpha explode when the
    // Laplacian kernel values are tiny and the diagonal perturbation
    // dominates, which causes the iteration to diverge.
    acc_t my_row = acc_t(-1e30);
    if (tid < N) {
        acc_t s = acc_t(0);
        #pragma unroll
        for (int j = 0; j < N_MAX; ++j) {
            if (j < N) s += sPbuf[tid * STRIDE + j];
        }
        my_row = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_t other = __shfl_xor_sync(0xffffffff, my_row, off);
        if (other > my_row) my_row = other;
    }
    __shared__ acc_t sWarpMax[BLOCK_THREADS / 32];
    if ((tid & 31) == 0) sWarpMax[tid >> 5] = my_row;
    __syncthreads();
    __shared__ acc_t sAlpha;
    if (tid == 0) {
        constexpr int NW = BLOCK_THREADS / 32;
        acc_t m = sWarpMax[0];
        #pragma unroll
        for (int w = 1; w < NW; ++w) {
            if (sWarpMax[w] > m) m = sWarpMax[w];
        }
        sAlpha = acc_t(1.8) / (m * m);
    }
    __syncthreads();
    return sAlpha;
}

// ----------------------------------------------------------------------------
// Forward kernel: 3-buffer variant (P, V, T all in smem)
// ----------------------------------------------------------------------------
template <typename scalar_t, int N_MAX>
__launch_bounds__(256, 1)
__global__ void newton_inverse_forward_kernel_3buf(
        const scalar_t* __restrict__ mat_in,
        scalar_t* __restrict__ V_out,
        int N, int iters)
{
    using acc_t = typename AccT<scalar_t>::type;
    using B = Blk<N_MAX>;
    constexpr int STRIDE = SmemLayout<acc_t, N_MAX>::STRIDE;
    constexpr int NELEM  = SmemLayout<acc_t, N_MAX>::NELEM;
    constexpr int UF     = MatmulUnroll<N_MAX>::UF;

    const int bh  = blockIdx.x;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * B::TW + tx;
    constexpr int BLOCK_THREADS = B::TH * B::TW;

    const scalar_t* M = mat_in + bh * N * N;
    scalar_t*       O = V_out  + bh * N * N;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    acc_t* sPbuf = reinterpret_cast<acc_t*>(smem_raw);
    acc_t* sVbuf = sPbuf + NELEM;
    acc_t* sTbuf = sVbuf + NELEM;

    auto P = [&](int i, int j) -> acc_t& { return sPbuf[i * STRIDE + j]; };
    auto V = [&](int i, int j) -> acc_t& { return sVbuf[i * STRIDE + j]; };
    auto T = [&](int i, int j) -> acc_t& { return sTbuf[i * STRIDE + j]; };

    // Zero the padding region (columns/rows >= N up to N_MAX) so matmul can
    // iterate the full N_MAX without guarding k.
    if (N < N_MAX) {
        for (int idx = tid; idx < NELEM; idx += BLOCK_THREADS) {
            sPbuf[idx] = acc_t(0);
        }
        __syncthreads();
    }

    for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
        int i = idx / N;
        int j = idx - i * N;
        acc_t v = to_acc(M[i * N + j]);
        P(i, j) = v + ((i == j) ? acc_t(0.01) : acc_t(0));
    }
    __syncthreads();

    const acc_t alpha = compute_alpha<acc_t, N_MAX, BLOCK_THREADS>(
            sPbuf, STRIDE, N, tid);

    for (int idx = tid; idx < NELEM; idx += BLOCK_THREADS) {
        sVbuf[idx] = sPbuf[idx] * alpha;
    }
    __syncthreads();

    for (int it = 0; it < iters; ++it) {
        // matmul1: acc1 = P @ V
        acc_t acc1[B::TM][B::TN];
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn) acc1[tm][tn] = acc_t(0);

        #pragma unroll UF
        for (int kk = 0; kk < N_MAX; ++kk) {
            acc_t a_reg[B::TM];
            acc_t b_reg[B::TN];
            #pragma unroll
            for (int tm = 0; tm < B::TM; ++tm) a_reg[tm] = P(ty * B::TM + tm, kk);
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn) b_reg[tn] = V(kk, tx * B::TN + tn);
            #pragma unroll
            for (int tm = 0; tm < B::TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < B::TN; ++tn)
                    acc1[tm][tn] += a_reg[tm] * b_reg[tn];
        }

        // Cache old V for the "2*V_old" contribution before we overwrite V.
        acc_t v_cache[B::TM][B::TN];
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn)
                v_cache[tm][tn] = V(ty * B::TM + tm, tx * B::TN + tn);

        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn)
                T(ty * B::TM + tm, tx * B::TN + tn) = acc1[tm][tn];
        __syncthreads();

        // matmul2: acc2 = V @ T
        acc_t acc2[B::TM][B::TN];
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn) acc2[tm][tn] = acc_t(0);

        #pragma unroll UF
        for (int kk = 0; kk < N_MAX; ++kk) {
            acc_t a_reg[B::TM];
            acc_t b_reg[B::TN];
            #pragma unroll
            for (int tm = 0; tm < B::TM; ++tm) a_reg[tm] = V(ty * B::TM + tm, kk);
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn) b_reg[tn] = T(kk, tx * B::TN + tn);
            #pragma unroll
            for (int tm = 0; tm < B::TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < B::TN; ++tn)
                    acc2[tm][tn] += a_reg[tm] * b_reg[tn];
        }
        __syncthreads();

        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn)
                V(ty * B::TM + tm, tx * B::TN + tn) = acc_t(2) * v_cache[tm][tn] - acc2[tm][tn];
        __syncthreads();
    }

    for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
        int i = idx / N;
        int j = idx - i * N;
        O[i * N + j] = from_acc<scalar_t>(V(i, j));
    }
}

// ----------------------------------------------------------------------------
// Forward kernel: 2-buffer variant (V in smem, slot A holds P then T)
//
// Per iteration:
//   1. Reload P into slot A from gmem (first iter: loaded as part of init).
//   2. matmul1 acc1 = V @ P, read from both slots.
//   3. Cache thread's tile of V_old into registers.
//   4. Overwrite slot A with acc1 (= W = V @ P).
//   5. matmul2 acc2 = W @ V_old, read slot A (now T) and slot B (still V_old).
//   6. Write V_new = 2*v_cache - acc2 into slot B.
//
// This associativity (V @ P then @ V) matches the python reference exactly
// (V @ P @ V is left-associative in python), so there is no numerical drift
// relative to the reference due to ordering.
//
// sm_80+ optimization: the P reload in step (1) is issued as cp.async at the
// top of the next iteration and its completion is waited on just before the
// first smem read that touches slot A. The intent is for the reload to
// overlap the v_cache write-back and the V-final-write sync.
// ----------------------------------------------------------------------------
template <typename scalar_t, int N_MAX>
__launch_bounds__(256, 1)
__global__ void newton_inverse_forward_kernel_2buf(
        const scalar_t* __restrict__ mat_in,
        scalar_t* __restrict__ V_out,
        int N, int iters)
{
    using acc_t = typename AccT<scalar_t>::type;
    using B = Blk<N_MAX>;
    constexpr int STRIDE = SmemLayout<acc_t, N_MAX>::STRIDE;
    constexpr int NELEM  = SmemLayout<acc_t, N_MAX>::NELEM;
    constexpr int UF     = MatmulUnroll<N_MAX>::UF;

    const int bh  = blockIdx.x;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * B::TW + tx;
    constexpr int BLOCK_THREADS = B::TH * B::TW;

    const scalar_t* M = mat_in + bh * N * N;
    scalar_t*       O = V_out  + bh * N * N;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    acc_t* sAbuf = reinterpret_cast<acc_t*>(smem_raw);   // holds P then T
    acc_t* sVbuf = sAbuf + NELEM;                        // V across iterations

    auto A = [&](int i, int j) -> acc_t& { return sAbuf[i * STRIDE + j]; };
    auto V = [&](int i, int j) -> acc_t& { return sVbuf[i * STRIDE + j]; };

    // Initial zero of the padding band; only needed once since we always
    // reload the same (N<N_MAX) block of P into the same slot.
    if (N < N_MAX) {
        for (int idx = tid; idx < NELEM; idx += BLOCK_THREADS) {
            sAbuf[idx] = acc_t(0);
            sVbuf[idx] = acc_t(0);
        }
        __syncthreads();
    }

    for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
        int i = idx / N;
        int j = idx - i * N;
        acc_t v = to_acc(M[i * N + j]);
        A(i, j) = v + ((i == j) ? acc_t(0.01) : acc_t(0));
    }
    __syncthreads();

    const acc_t alpha = compute_alpha<acc_t, N_MAX, BLOCK_THREADS>(
            sAbuf, STRIDE, N, tid);

    for (int idx = tid; idx < NELEM; idx += BLOCK_THREADS) {
        sVbuf[idx] = sAbuf[idx] * alpha;
    }
    __syncthreads();

    // Helper: kick off (possibly async) reload of P's live NxN region into
    // slot A. sm_80+ uses cp.async; older arches use ordinary LD/ST.
    auto reload_P_async = [&]() {
#if IS_SM80
        if constexpr (std::is_same<acc_t, float>::value && std::is_same<scalar_t, float>::value) {
            // Fast path: async 4B-at-a-time copies, no conversion.
            for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
                int i = idx / N;
                int j = idx - i * N;
                acc_t* dst = &sAbuf[i * STRIDE + j];
                const scalar_t* src = &M[i * N + j];
                cp_async_4B(dst, src);
            }
            cp_async_commit();
            // Diagonal +0.01 perturbation is applied post-copy at wait time.
        } else {
            for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
                int i = idx / N;
                int j = idx - i * N;
                acc_t v = to_acc(M[i * N + j]);
                A(i, j) = v + ((i == j) ? acc_t(0.01) : acc_t(0));
            }
        }
#else
        for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
            int i = idx / N;
            int j = idx - i * N;
            acc_t v = to_acc(M[i * N + j]);
            A(i, j) = v + ((i == j) ? acc_t(0.01) : acc_t(0));
        }
#endif
    };

    auto wait_P_and_fix_diag = [&]() {
#if IS_SM80
        if constexpr (std::is_same<acc_t, float>::value && std::is_same<scalar_t, float>::value) {
            cp_async_wait_all();
            __syncthreads();
            // Apply +0.01 diagonal. Only row=col threads touch this.
            for (int idx = tid; idx < N; idx += BLOCK_THREADS) {
                A(idx, idx) += acc_t(0.01);
            }
            __syncthreads();
            return;
        }
#endif
        __syncthreads();
    };

    for (int it = 0; it < iters; ++it) {
        if (it > 0) {
            // The previous iter's last __syncthreads() closed the V-write
            // window. Slot A still holds T; we're free to overwrite with P.
            reload_P_async();
            wait_P_and_fix_diag();
        }

        // matmul1: acc1 = V @ P
        acc_t acc1[B::TM][B::TN];
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn) acc1[tm][tn] = acc_t(0);

        #pragma unroll UF
        for (int kk = 0; kk < N_MAX; ++kk) {
            acc_t a_reg[B::TM];
            acc_t b_reg[B::TN];
            #pragma unroll
            for (int tm = 0; tm < B::TM; ++tm) a_reg[tm] = V(ty * B::TM + tm, kk);
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn) b_reg[tn] = A(kk, tx * B::TN + tn);
            #pragma unroll
            for (int tm = 0; tm < B::TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < B::TN; ++tn)
                    acc1[tm][tn] += a_reg[tm] * b_reg[tn];
        }

        // Cache thread's V tile (used later for 2*V_old).
        acc_t v_cache[B::TM][B::TN];
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn)
                v_cache[tm][tn] = V(ty * B::TM + tm, tx * B::TN + tn);

        __syncthreads();  // all matmul1 reads of P done before we clobber slot A

        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn)
                A(ty * B::TM + tm, tx * B::TN + tn) = acc1[tm][tn];
        __syncthreads();

        // matmul2: acc2 = T @ V   (T now lives in slot A)
        acc_t acc2[B::TM][B::TN];
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn) acc2[tm][tn] = acc_t(0);

        #pragma unroll UF
        for (int kk = 0; kk < N_MAX; ++kk) {
            acc_t a_reg[B::TM];
            acc_t b_reg[B::TN];
            #pragma unroll
            for (int tm = 0; tm < B::TM; ++tm) a_reg[tm] = A(ty * B::TM + tm, kk);
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn) b_reg[tn] = V(kk, tx * B::TN + tn);
            #pragma unroll
            for (int tm = 0; tm < B::TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < B::TN; ++tn)
                    acc2[tm][tn] += a_reg[tm] * b_reg[tn];
        }
        __syncthreads();  // all matmul2 reads of V done before we overwrite V

        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn)
                V(ty * B::TM + tm, tx * B::TN + tn) = acc_t(2) * v_cache[tm][tn] - acc2[tm][tn];
        __syncthreads();
    }

    for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
        int i = idx / N;
        int j = idx - i * N;
        O[i * N + j] = from_acc<scalar_t>(V(i, j));
    }
}

// ----------------------------------------------------------------------------
// Backward kernel: grad_mat = -(V^T @ G @ V^T)   (unchanged, just wider N_MAX)
// ----------------------------------------------------------------------------
template <typename scalar_t, int N_MAX>
__launch_bounds__(256, 1)
__global__ void newton_inverse_backward_kernel(
        const scalar_t* __restrict__ grad_out,
        const scalar_t* __restrict__ inv_in,
        scalar_t* __restrict__ grad_mat_out,
        int N)
{
    using acc_t = typename AccT<scalar_t>::type;
    using B = Blk<N_MAX>;
    constexpr int STRIDE = SmemLayout<acc_t, N_MAX>::STRIDE;
    constexpr int NELEM  = SmemLayout<acc_t, N_MAX>::NELEM;
    constexpr int UF     = MatmulUnroll<N_MAX>::UF;

    const int bh  = blockIdx.x;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * B::TW + tx;
    constexpr int BLOCK_THREADS = B::TH * B::TW;

    const scalar_t* G  = grad_out     + bh * N * N;
    const scalar_t* Vg = inv_in       + bh * N * N;
    scalar_t*       O  = grad_mat_out + bh * N * N;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    acc_t* sVbuf = reinterpret_cast<acc_t*>(smem_raw);
    acc_t* sGbuf = sVbuf + NELEM;

    auto V  = [&](int i, int j) -> acc_t& { return sVbuf[i * STRIDE + j]; };
    auto SG = [&](int i, int j) -> acc_t& { return sGbuf[i * STRIDE + j]; };

    for (int idx = tid; idx < NELEM; idx += BLOCK_THREADS) {
        sVbuf[idx] = acc_t(0);
        sGbuf[idx] = acc_t(0);
    }
    __syncthreads();

    for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
        int i = idx / N;
        int j = idx - i * N;
        V(i, j)  = to_acc(Vg[i * N + j]);
        SG(i, j) = to_acc(G[i * N + j]);
    }
    __syncthreads();

    acc_t acc1[B::TM][B::TN];
    #pragma unroll
    for (int tm = 0; tm < B::TM; ++tm)
        #pragma unroll
        for (int tn = 0; tn < B::TN; ++tn) acc1[tm][tn] = acc_t(0);

    #pragma unroll UF
    for (int pp = 0; pp < N_MAX; ++pp) {
        acc_t a_reg[B::TM];
        acc_t b_reg[B::TN];
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm) a_reg[tm] = V(pp, ty * B::TM + tm);
        #pragma unroll
        for (int tn = 0; tn < B::TN; ++tn) b_reg[tn] = SG(pp, tx * B::TN + tn);
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn)
                acc1[tm][tn] += a_reg[tm] * b_reg[tn];
    }
    __syncthreads();
    #pragma unroll
    for (int tm = 0; tm < B::TM; ++tm)
        #pragma unroll
        for (int tn = 0; tn < B::TN; ++tn)
            SG(ty * B::TM + tm, tx * B::TN + tn) = acc1[tm][tn];
    __syncthreads();

    acc_t acc2[B::TM][B::TN];
    #pragma unroll
    for (int tm = 0; tm < B::TM; ++tm)
        #pragma unroll
        for (int tn = 0; tn < B::TN; ++tn) acc2[tm][tn] = acc_t(0);

    #pragma unroll UF
    for (int kk = 0; kk < N_MAX; ++kk) {
        acc_t a_reg[B::TM];
        acc_t b_reg[B::TN];
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm) a_reg[tm] = SG(ty * B::TM + tm, kk);
        #pragma unroll
        for (int tn = 0; tn < B::TN; ++tn) b_reg[tn] = V(tx * B::TN + tn, kk);
        #pragma unroll
        for (int tm = 0; tm < B::TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < B::TN; ++tn)
                acc2[tm][tn] += a_reg[tm] * b_reg[tn];
    }

    #pragma unroll
    for (int tm = 0; tm < B::TM; ++tm) {
        int i = ty * B::TM + tm;
        if (i >= N) continue;
        #pragma unroll
        for (int tn = 0; tn < B::TN; ++tn) {
            int j = tx * B::TN + tn;
            if (j < N) O[i * N + j] = from_acc<scalar_t>(-acc2[tm][tn]);
        }
    }
}

// ----------------------------------------------------------------------------
// Host-side dispatch
// ----------------------------------------------------------------------------

// Returns the device's per-block opt-in shared memory limit (cached).
static int query_max_smem() {
    static int cached = -1;
    if (cached >= 0) return cached;
    int dev = 0;
    cudaGetDevice(&dev);
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cached = v;
    return v;
}

// Set the SM carveout so the full 164KB shmem is available (A100) — otherwise
// an opt-in of e.g. 50KB is rounded up to the nearest carveout tier (64KB on
// sm_80), causing multi-block-per-SM occupancy to collapse to 1 block/SM.
// Opt in to the maximum shared-memory carveout (164KB on sm_80, ~228KB on
// sm_90) so that blocks with per-block smem between the default 48KB and the
// device max can still achieve >1 block/SM occupancy.
//
// Two knobs must align:
//   1. PreferredSharedMemoryCarveout = MaxShared    (selects the carveout tier)
//   2. MaxDynamicSharedMemorySize    = device-max   (keeps the kernel eligible
//                                                    for that tier; if it is
//                                                    set to just `smem_bytes`
//                                                    then the driver may
//                                                    collapse the carveout
//                                                    down to the smallest
//                                                    tier that fits a single
//                                                    block of smem_bytes).
template <typename FnPtr>
static void smem_carveout_max(FnPtr fn, size_t smem_bytes) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    int device_max_smem = query_max_smem();
    if (smem_bytes > 48 * 1024) {
        TORCH_CHECK(
            smem_bytes <= static_cast<size_t>(device_max_smem),
            "NewtonInverse kernel requires ",
            smem_bytes,
            " bytes of dynamic shared memory, but this device only allows ",
            device_max_smem,
            " bytes per block"
        );
        C10_CUDA_CHECK(cudaFuncSetAttribute(fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }
}

template <typename scalar_t, int N_MAX>
void launch_forward_3buf(const scalar_t* mat_ptr, scalar_t* V_ptr,
                         int BH, int N, int iters, cudaStream_t stream)
{
    using acc_t = typename AccT<scalar_t>::type;
    using B = Blk<N_MAX>;

    dim3 block(B::TW, B::TH);
    dim3 grid(BH);
    size_t smem_bytes = SmemLayout<acc_t, N_MAX>::BYTES_3BUF;

    auto kernel = newton_inverse_forward_kernel_3buf<scalar_t, N_MAX>;
    smem_carveout_max(kernel, smem_bytes);
    kernel<<<grid, block, smem_bytes, stream>>>(mat_ptr, V_ptr, N, iters);
}

template <typename scalar_t, int N_MAX>
void launch_forward_2buf(const scalar_t* mat_ptr, scalar_t* V_ptr,
                         int BH, int N, int iters, cudaStream_t stream)
{
    using acc_t = typename AccT<scalar_t>::type;
    using B = Blk<N_MAX>;

    dim3 block(B::TW, B::TH);
    dim3 grid(BH);
    size_t smem_bytes = SmemLayout<acc_t, N_MAX>::BYTES_2BUF;

    auto kernel = newton_inverse_forward_kernel_2buf<scalar_t, N_MAX>;
    smem_carveout_max(kernel, smem_bytes);
    kernel<<<grid, block, smem_bytes, stream>>>(mat_ptr, V_ptr, N, iters);
}

template <typename scalar_t, int N_MAX>
void launch_backward(const scalar_t* G_ptr, const scalar_t* V_ptr,
                     scalar_t* out_ptr, int BH, int N, cudaStream_t stream)
{
    using acc_t = typename AccT<scalar_t>::type;
    using B = Blk<N_MAX>;

    dim3 block(B::TW, B::TH);
    dim3 grid(BH);
    size_t smem_bytes = SmemLayout<acc_t, N_MAX>::BYTES_BWD;

    auto kernel = newton_inverse_backward_kernel<scalar_t, N_MAX>;
    smem_carveout_max(kernel, smem_bytes);
    kernel<<<grid, block, smem_bytes, stream>>>(G_ptr, V_ptr, out_ptr, N);
}

inline int pick_n_max(int N) {
    if (N <= 0)   return 0;
    if (N <= 16)  return 16;
    if (N <= 32)  return 32;
    if (N <= 48)  return 48;
    if (N <= 64)  return 64;
    if (N <= 80)  return 80;
    if (N <= 96)  return 96;
    if (N <= 112) return 112;
    if (N <= 128) return 128;
    return 0;
}

// Chooses 3-buffer (fused P,V,T) when it fits in the device's opt-in smem,
// otherwise falls back to 2-buffer (reloads P each iter). 2-buffer is the
// only option at N_MAX=128.
template <typename scalar_t>
void dispatch_forward(const scalar_t* mat_ptr, scalar_t* V_ptr,
                      int BH, int N, int iters, cudaStream_t stream)
{
    int nmax = pick_n_max(N);
    int max_smem = query_max_smem();
    using acc_t = typename AccT<scalar_t>::type;

    auto try3 = [&](int fit_bytes) { return fit_bytes <= max_smem; };

    switch (nmax) {
        case 16:  launch_forward_3buf<scalar_t, 16> (mat_ptr, V_ptr, BH, N, iters, stream); break;
        case 32:  launch_forward_3buf<scalar_t, 32> (mat_ptr, V_ptr, BH, N, iters, stream); break;
        case 48:  launch_forward_3buf<scalar_t, 48> (mat_ptr, V_ptr, BH, N, iters, stream); break;
        case 64:  launch_forward_3buf<scalar_t, 64> (mat_ptr, V_ptr, BH, N, iters, stream); break;
        case 80:  launch_forward_3buf<scalar_t, 80> (mat_ptr, V_ptr, BH, N, iters, stream); break;
        case 96:
            if (try3(SmemLayout<acc_t, 96>::BYTES_3BUF))
                launch_forward_3buf<scalar_t, 96>(mat_ptr, V_ptr, BH, N, iters, stream);
            else
                launch_forward_2buf<scalar_t, 96>(mat_ptr, V_ptr, BH, N, iters, stream);
            break;
        case 112:
            if (try3(SmemLayout<acc_t, 112>::BYTES_3BUF))
                launch_forward_3buf<scalar_t, 112>(mat_ptr, V_ptr, BH, N, iters, stream);
            else
                launch_forward_2buf<scalar_t, 112>(mat_ptr, V_ptr, BH, N, iters, stream);
            break;
        case 128:
            // 3-buffer is 192KB for fp32, exceeds all current devices; always 2-buffer.
            launch_forward_2buf<scalar_t, 128>(mat_ptr, V_ptr, BH, N, iters, stream);
            break;
        default:
            TORCH_CHECK(false, "NewtonInverse CUDA path only supports N <= 128 (got ", N, ")");
    }
}

template <typename scalar_t>
void dispatch_backward(const scalar_t* G_ptr, const scalar_t* V_ptr,
                       scalar_t* out_ptr, int BH, int N, cudaStream_t stream)
{
    int nmax = pick_n_max(N);
    switch (nmax) {
        case 16:  launch_backward<scalar_t, 16> (G_ptr, V_ptr, out_ptr, BH, N, stream); break;
        case 32:  launch_backward<scalar_t, 32> (G_ptr, V_ptr, out_ptr, BH, N, stream); break;
        case 48:  launch_backward<scalar_t, 48> (G_ptr, V_ptr, out_ptr, BH, N, stream); break;
        case 64:  launch_backward<scalar_t, 64> (G_ptr, V_ptr, out_ptr, BH, N, stream); break;
        case 80:  launch_backward<scalar_t, 80> (G_ptr, V_ptr, out_ptr, BH, N, stream); break;
        case 96:  launch_backward<scalar_t, 96> (G_ptr, V_ptr, out_ptr, BH, N, stream); break;
        case 112: launch_backward<scalar_t, 112>(G_ptr, V_ptr, out_ptr, BH, N, stream); break;
        case 128: launch_backward<scalar_t, 128>(G_ptr, V_ptr, out_ptr, BH, N, stream); break;
        default:
            TORCH_CHECK(false, "NewtonInverse CUDA path only supports N <= 128 (got ", N, ")");
    }
}

} // namespace

namespace laplace {

bool NewtonInverse_supports(int N) {
    if (pick_n_max(N) == 0) return false;
    // At N <= 80 the 3-buffer kernel always fits in 99KB opt-in.
    // At N >= 96 we rely on either 3-buf (on A100-class devices) or 2-buf
    // (smem budget 2*N*(N+1)*4, the tightest case is N=128 at ~129KB which
    // needs >=164KB opt-in, i.e., sm_80/sm_90).
    int nmax = pick_n_max(N);
    int smem_2buf = 2 * nmax * (nmax + 1) * sizeof(float);
    return smem_2buf <= query_max_smem();
}

void NewtonInverse_forward_cuda(const at::Tensor mat, at::Tensor output, int iter) {
    at::cuda::CUDAGuard device_guard(mat.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(mat.is_contiguous(), "mat must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(mat.sizes() == output.sizes(), "mat and output must have the same shape");

    int B  = mat.size(0);
    int H  = mat.size(1);
    int N  = mat.size(2);
    int BH = B * H;

    TORCH_CHECK(NewtonInverse_supports(N),
                "NewtonInverse: N=", N,
                " is not supported on this device (insufficient smem or out of range)");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        mat.scalar_type(), "newton_inverse_forward", ([&] {
            dispatch_forward<scalar_t>(
                mat.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                BH, N, iter, stream);
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in NewtonInverse_forward_cuda: %s\n", cudaGetErrorString(err));
    }
}

void NewtonInverse_backward_cuda(const at::Tensor grad_output,
                                 const at::Tensor inverse,
                                 at::Tensor grad_mat)
{
    at::cuda::CUDAGuard device_guard(inverse.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(inverse.is_contiguous(), "inverse must be contiguous");
    TORCH_CHECK(grad_mat.is_contiguous(), "grad_mat must be contiguous");
    TORCH_CHECK(grad_output.sizes() == inverse.sizes() &&
                inverse.sizes() == grad_mat.sizes(),
                "shape mismatch");

    int B  = inverse.size(0);
    int H  = inverse.size(1);
    int N  = inverse.size(2);
    int BH = B * H;

    TORCH_CHECK(NewtonInverse_supports(N),
                "NewtonInverse: N=", N,
                " is not supported on this device (insufficient smem or out of range)");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        inverse.scalar_type(), "newton_inverse_backward", ([&] {
            dispatch_backward<scalar_t>(
                grad_output.data_ptr<scalar_t>(),
                inverse.data_ptr<scalar_t>(),
                grad_mat.data_ptr<scalar_t>(),
                BH, N, stream);
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in NewtonInverse_backward_cuda: %s\n", cudaGetErrorString(err));
    }
}

} // namespace laplace
