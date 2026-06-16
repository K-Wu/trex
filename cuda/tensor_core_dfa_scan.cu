/*
 * tensor_core_dfa_scan.cu — v3: Optimized
 *
 * Int8 Tensor-Core Accelerated DFA Simulation via Parallel Prefix Scan
 *
 * Two execution paths:
 *   Basic:     Self-contained per-call (alloc/compute/free). Simple, correct.
 *   Optimized: GPUContext with persistent memory + Blelloch work-efficient scan.
 *              Eliminates malloc overhead and reduces O(L log L) → O(L) work.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_90 -std=c++17 cuda/tensor_core_dfa_scan.cu -o build/dfa_scan
 *   nvcc -O3 -arch=sm_90 -std=c++17 -DBUILD_LIB -shared -Xcompiler -fPIC \
 *        cuda/tensor_core_dfa_scan.cu -o build/libdfa_scan.so
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

using namespace nvcuda;

// ─── Configuration ─────────────────────────────────────────────────────────

constexpr int TILE = 16;
constexpr int TILE_ELEMS = TILE * TILE;  // 256
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;  // 256

// Per-warp shared memory: B transpose (256 bytes) + int32 accumulator (1024 bytes)
constexpr int SMEM_PER_WARP = TILE_ELEMS + TILE_ELEMS * (int)sizeof(int32_t);
// Downsweep needs extra 256 bytes for saved matrix
constexpr int SMEM_PER_WARP_DS = SMEM_PER_WARP + TILE_ELEMS;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// ─── Host-side DFA ─────────────────────────────────────────────────────────

struct DFAHost {
    int n_states;
    int n_states_padded;   // always 16
    int alphabet_size;
    int start_state;
    int8_t *accept_mask;   // [16]
    int8_t *trans_matrices; // [alphabet_size * 256] row-major: T[char][dst*16+src]
};

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static int ilog2(int n) {
    int k = 0;
    while ((1 << k) < n) k++;
    return k;
}

// ─── Device: WMMA 16×16 matmul helper ──────────────────────────────────────

/*
 * Compute C = A × B where both A, B are 16×16 int8 row-major in global memory.
 * B is transposed into shared memory before WMMA load (col_major requirement).
 * Result clamped to {0,1} for DFA/NFA correctness.
 *
 * smem_b: [256] int8_t  — shared memory for B transpose
 * smem_c: [256] int32_t — shared memory for accumulator store
 */
__device__ __forceinline__ void warp_matmul_16x16(
    const int8_t *__restrict__ a_global,
    const int8_t *__restrict__ b_global,
    int8_t *__restrict__ c_global,
    int8_t *smem_b,
    int32_t *smem_c,
    int lane
) {
    // Transpose B from row-major to col-major in shared memory
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int r = e / TILE, c = e % TILE;
        smem_b[c * TILE + r] = b_global[r * TILE + c];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c_frag;

    wmma::fill_fragment(c_frag, 0);
    wmma::load_matrix_sync(a_frag, a_global, TILE);
    wmma::load_matrix_sync(b_frag, smem_b, TILE);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(smem_c, c_frag, TILE, wmma::mem_row_major);
    __syncwarp();

    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
        c_global[e] = (int8_t)(smem_c[e] > 0 ? 1 : 0);
}

// ─── Device Kernels ────────────────────────────────────────────────────────

__global__ void gather_matrices_kernel(
    const int *input_chars,
    const int8_t *trans_mats,
    int8_t *gathered,
    int L
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)L * TILE_ELEMS) return;
    int pos = (int)(idx / TILE_ELEMS);
    int elem = (int)(idx % TILE_ELEMS);
    gathered[(long long)pos * TILE_ELEMS + elem] =
        trans_mats[input_chars[pos] * TILE_ELEMS + elem];
}

// Hillis-Steele inclusive prefix scan step (work-INefficient, O(L log L))
__global__ void hillis_steele_step_kernel(
    const int8_t *__restrict__ src,
    int8_t *__restrict__ dst,
    int stride,
    int padded_L
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int i = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int lane = threadIdx.x % WARP_SIZE;
    if (i >= padded_L) return;

    long long off_i = (long long)i * TILE_ELEMS;

    if (i < stride) {
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            dst[off_i + e] = src[off_i + e];
        return;
    }

    extern __shared__ char smem[];
    int8_t  *sb = (int8_t *)(smem + warp_in_block * SMEM_PER_WARP);
    int32_t *sc = (int32_t *)(sb + TILE_ELEMS);

    warp_matmul_16x16(
        src + off_i,
        src + (long long)(i - stride) * TILE_ELEMS,
        dst + off_i,
        sb, sc, lane
    );
}

// Blelloch upsweep step: buf[idx] = buf[idx] @ buf[idx - half]
__global__ void blelloch_upsweep_step(
    int8_t *buf,
    int stride,
    int half,
    int padded_L
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int pair = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int lane = threadIdx.x % WARP_SIZE;
    int idx = (pair + 1) * stride - 1;
    if (idx >= padded_L) return;

    long long off_idx = (long long)idx * TILE_ELEMS;
    long long off_src = (long long)(idx - half) * TILE_ELEMS;

    extern __shared__ char smem[];
    int8_t  *sb = (int8_t *)(smem + warp_in_block * SMEM_PER_WARP);
    int32_t *sc = (int32_t *)(sb + TILE_ELEMS);

    warp_matmul_16x16(buf + off_idx, buf + off_src, buf + off_idx, sb, sc, lane);
}

// Blelloch downsweep step: temp=buf[left]; buf[left]=buf[idx]; buf[idx]=buf[idx]@temp
__global__ void blelloch_downsweep_step(
    int8_t *buf,
    int stride,
    int half,
    int padded_L
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int pair = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int lane = threadIdx.x % WARP_SIZE;
    int idx = (pair + 1) * stride - 1;
    if (idx >= padded_L) return;
    int left = idx - half;

    long long off_idx = (long long)idx * TILE_ELEMS;
    long long off_left = (long long)left * TILE_ELEMS;

    extern __shared__ char smem[];
    int8_t  *saved   = (int8_t *)(smem + warp_in_block * SMEM_PER_WARP_DS);
    int8_t  *sb      = saved + TILE_ELEMS;
    int32_t *sc      = (int32_t *)(sb + TILE_ELEMS);

    // Save buf[left]
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
        saved[e] = buf[off_left + e];
    __syncwarp();

    // buf[left] = buf[idx]
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
        buf[off_left + e] = buf[off_idx + e];
    __syncwarp();

    // buf[idx] = saved @ buf[idx]  (local partial LEFT, inherited prefix RIGHT)
    warp_matmul_16x16(saved, buf + off_idx, buf + off_idx, sb, sc, lane);
}

// Batched inclusive conversion: result[i] = orig[i] @ exclusive[i]
__global__ void inclusive_convert_kernel(
    const int8_t *__restrict__ orig,
    const int8_t *__restrict__ excl,
    int8_t *__restrict__ result,
    int L
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int i = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int lane = threadIdx.x % WARP_SIZE;
    if (i >= L) return;

    long long off = (long long)i * TILE_ELEMS;

    extern __shared__ char smem[];
    int8_t  *sb = (int8_t *)(smem + warp_in_block * SMEM_PER_WARP);
    int32_t *sc = (int32_t *)(sb + TILE_ELEMS);

    warp_matmul_16x16(orig + off, excl + off, result + off, sb, sc, lane);
}

// Set a 16×16 identity matrix on device
__global__ void set_identity_kernel(int8_t *buf, long long offset) {
    int idx = threadIdx.x;
    if (idx < TILE_ELEMS) {
        int r = idx / TILE, c = idx % TILE;
        buf[offset + idx] = (r == c) ? 1 : 0;
    }
}

// Fill range with identity matrices
__global__ void fill_identities_kernel(int8_t *buf, int start_pos, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * TILE_ELEMS) return;
    int pos = idx / TILE_ELEMS;
    int elem = idx % TILE_ELEMS;
    int r = elem / TILE, c = elem % TILE;
    buf[(long long)(start_pos + pos) * TILE_ELEMS + elem] = (r == c) ? 1 : 0;
}

// ─── GPUContext: persistent device memory for amortized queries ────────────

struct GPUContext {
    DFAHost dfa;
    int max_L;
    int max_padded_L;

    // Device memory (persistent)
    int8_t *d_trans;     // [alphabet_size * 256]
    int    *d_input;     // [max_L]
    int8_t *d_orig;      // [max_padded_L * 256] — original gathered matrices
    int8_t *d_work;      // [max_padded_L * 256] — working buffer (scan)
    int8_t *d_buf_b;     // [max_padded_L * 256] — second buffer (Hillis-Steele)

    bool use_blelloch;

    void init(const DFAHost &h_dfa, int max_l, bool blelloch = true) {
        dfa = h_dfa;
        max_L = max_l;
        max_padded_L = next_pow2(max_l);
        use_blelloch = blelloch;

        CHECK_CUDA(cudaMalloc(&d_trans, dfa.alphabet_size * TILE_ELEMS));
        CHECK_CUDA(cudaMalloc(&d_input, max_L * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_orig,  (size_t)max_padded_L * TILE_ELEMS));
        CHECK_CUDA(cudaMalloc(&d_work,  (size_t)max_padded_L * TILE_ELEMS));
        CHECK_CUDA(cudaMalloc(&d_buf_b, (size_t)max_padded_L * TILE_ELEMS));

        CHECK_CUDA(cudaMemcpy(d_trans, dfa.trans_matrices,
                              dfa.alphabet_size * TILE_ELEMS, cudaMemcpyHostToDevice));
    }

    void destroy() {
        cudaFree(d_trans);
        cudaFree(d_input);
        cudaFree(d_orig);
        cudaFree(d_work);
        cudaFree(d_buf_b);
    }

    bool simulate(const int *h_input, int L) {
        if (L == 0)
            return dfa.accept_mask[dfa.start_state] != 0;
        if (L > max_L) {
            fprintf(stderr, "GPUContext: L=%d exceeds max_L=%d\n", L, max_L);
            return false;
        }

        int padded_L = next_pow2(L);
        int log2L = ilog2(padded_L);

        // H2D: input characters
        CHECK_CUDA(cudaMemcpy(d_input, h_input, L * sizeof(int), cudaMemcpyHostToDevice));

        // Gather transition matrices → d_orig
        {
            long long total = (long long)L * TILE_ELEMS;
            int grid = (int)((total + 255) / 256);
            gather_matrices_kernel<<<grid, 256>>>(d_input, d_trans, d_orig, L);
        }

        // Fill padded positions with identity
        if (padded_L > L) {
            int count = padded_L - L;
            int grid = (count * TILE_ELEMS + 255) / 256;
            fill_identities_kernel<<<grid, 256>>>(d_orig, L, count);
        }

        int smem_hs = WARPS_PER_BLOCK * SMEM_PER_WARP;
        int smem_ds = WARPS_PER_BLOCK * SMEM_PER_WARP_DS;

        int8_t *result_buf;

        if (use_blelloch) {
            // Copy orig → work for in-place Blelloch
            CHECK_CUDA(cudaMemcpy(d_work, d_orig,
                                  (size_t)padded_L * TILE_ELEMS, cudaMemcpyDeviceToDevice));

            // Upsweep
            for (int d = 0; d < log2L; d++) {
                int stride = 1 << (d + 1);
                int half = 1 << d;
                int n_pairs = padded_L / stride;
                if (n_pairs == 0) continue;
                int grid = (n_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                blelloch_upsweep_step<<<grid, BLOCK_SIZE, smem_hs>>>(
                    d_work, stride, half, padded_L);
                CHECK_CUDA(cudaGetLastError());
            }

            // Set root to identity
            set_identity_kernel<<<1, TILE_ELEMS>>>(d_work, (long long)(padded_L - 1) * TILE_ELEMS);

            // Downsweep
            for (int d = log2L - 1; d >= 0; d--) {
                int stride = 1 << (d + 1);
                int half = 1 << d;
                int n_pairs = padded_L / stride;
                if (n_pairs == 0) continue;
                int grid = (n_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                blelloch_downsweep_step<<<grid, BLOCK_SIZE, smem_ds>>>(
                    d_work, stride, half, padded_L);
                CHECK_CUDA(cudaGetLastError());
            }

            // Inclusive conversion: d_buf_b[i] = d_orig[i] @ d_work[i]
            {
                int grid = (padded_L + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                inclusive_convert_kernel<<<grid, BLOCK_SIZE, smem_hs>>>(
                    d_orig, d_work, d_buf_b, padded_L);
                CHECK_CUDA(cudaGetLastError());
            }
            result_buf = d_buf_b;

        } else {
            // Hillis-Steele
            int8_t *src = d_orig;
            int8_t *dst = d_work;
            for (int d = 0; d < log2L; d++) {
                int s = 1 << d;
                int grid = (padded_L + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                hillis_steele_step_kernel<<<grid, BLOCK_SIZE, smem_hs>>>(
                    src, dst, s, padded_L);
                CHECK_CUDA(cudaGetLastError());
                int8_t *tmp = src; src = dst; dst = tmp;
            }
            result_buf = src;
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        // Read back prefix product at position L-1
        int8_t h_final[TILE_ELEMS];
        CHECK_CUDA(cudaMemcpy(h_final, result_buf + (size_t)(L - 1) * TILE_ELEMS,
                              TILE_ELEMS, cudaMemcpyDeviceToHost));

        // Matrix-vector: final_state = prefix[L-1] @ start_vec
        int8_t start_vec[TILE] = {0};
        start_vec[dfa.start_state] = 1;
        int32_t final_state[TILE] = {0};
        for (int r = 0; r < TILE; r++)
            for (int c = 0; c < TILE; c++)
                final_state[r] += (int32_t)h_final[r * TILE + c] * (int32_t)start_vec[c];

        for (int i = 0; i < dfa.n_states; i++)
            if (final_state[i] > 0 && dfa.accept_mask[i] != 0)
                return true;
        return false;
    }
};

// ─── Host: Sequential reference ────────────────────────────────────────────

static bool simulate_sequential(const DFAHost &dfa, const int *input, int L) {
    int state = dfa.start_state;
    for (int i = 0; i < L; i++) {
        int c = input[i];
        for (int d = 0; d < dfa.n_states_padded; d++)
            if (dfa.trans_matrices[c * TILE_ELEMS + d * TILE + state] == 1) {
                state = d; break;
            }
    }
    return dfa.accept_mask[state] != 0;
}

// ─── Basic self-contained simulation (for Python bridge backward compat) ───

static bool simulate_dfa_tensor_core(
    const DFAHost &dfa, const int *h_input, int L
) {
    if (L == 0) return dfa.accept_mask[dfa.start_state] != 0;

    int padded_L = next_pow2(L);
    int log2L = ilog2(padded_L);

    int *d_input; int8_t *d_trans, *d_buf_a, *d_buf_b;
    CHECK_CUDA(cudaMalloc(&d_input, L * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_trans, dfa.alphabet_size * TILE_ELEMS));
    CHECK_CUDA(cudaMalloc(&d_buf_a, (size_t)padded_L * TILE_ELEMS));
    CHECK_CUDA(cudaMalloc(&d_buf_b, (size_t)padded_L * TILE_ELEMS));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, L * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_trans, dfa.trans_matrices,
                          dfa.alphabet_size * TILE_ELEMS, cudaMemcpyHostToDevice));

    if (padded_L > L) {
        int count = padded_L - L;
        int grid = (count * TILE_ELEMS + 255) / 256;
        fill_identities_kernel<<<grid, 256>>>(d_buf_a, L, count);
    }

    {
        long long total = (long long)L * TILE_ELEMS;
        int grid = (int)((total + 255) / 256);
        gather_matrices_kernel<<<grid, 256>>>(d_input, d_trans, d_buf_a, L);
    }

    int8_t *src = d_buf_a, *dst = d_buf_b;
    int smem = WARPS_PER_BLOCK * SMEM_PER_WARP;
    for (int d = 0; d < log2L; d++) {
        int s = 1 << d;
        int grid = (padded_L + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        hillis_steele_step_kernel<<<grid, BLOCK_SIZE, smem>>>(src, dst, s, padded_L);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        int8_t *t = src; src = dst; dst = t;
    }

    int8_t h_final[TILE_ELEMS];
    CHECK_CUDA(cudaMemcpy(h_final, src + (size_t)(L - 1) * TILE_ELEMS,
                          TILE_ELEMS, cudaMemcpyDeviceToHost));

    int8_t sv[TILE] = {0}; sv[dfa.start_state] = 1;
    int32_t fs[TILE] = {0};
    for (int r = 0; r < TILE; r++)
        for (int c = 0; c < TILE; c++)
            fs[r] += (int32_t)h_final[r * TILE + c] * (int32_t)sv[c];

    bool accepted = false;
    for (int i = 0; i < dfa.n_states; i++)
        if (fs[i] > 0 && dfa.accept_mask[i] != 0) { accepted = true; break; }

    cudaFree(d_input); cudaFree(d_trans); cudaFree(d_buf_a); cudaFree(d_buf_b);
    return accepted;
}

// ─── DFA construction helpers ──────────────────────────────────────────────

struct DFABuilder {
    DFAHost dfa;
    DFABuilder(int n_states, int alphabet_size, int start_state) {
        dfa.n_states = n_states;
        dfa.n_states_padded = TILE;
        dfa.alphabet_size = alphabet_size;
        dfa.start_state = start_state;
        dfa.accept_mask = new int8_t[TILE]();
        dfa.trans_matrices = new int8_t[alphabet_size * TILE_ELEMS]();
        for (int c = 0; c < alphabet_size; c++)
            for (int s = n_states; s < TILE; s++)
                dfa.trans_matrices[c * TILE_ELEMS + s * TILE + s] = 1;
    }
    void set_accept(int state) { dfa.accept_mask[state] = 1; }
    void set_trans(int char_idx, int src, int dst) {
        dfa.trans_matrices[char_idx * TILE_ELEMS + dst * TILE + src] = 1;
    }
    ~DFABuilder() { delete[] dfa.accept_mask; delete[] dfa.trans_matrices; }
};

// ─── Python bridge (ctypes) ────────────────────────────────────────────────

extern "C" {

int gpu_simulate_dfa(
    int n_states, int alphabet_size, int start_state,
    const int8_t *accept_mask, const int8_t *trans_matrices,
    const int *input_chars, int L
) {
    DFAHost dfa;
    dfa.n_states = n_states; dfa.n_states_padded = TILE;
    dfa.alphabet_size = alphabet_size; dfa.start_state = start_state;
    dfa.accept_mask = const_cast<int8_t*>(accept_mask);
    dfa.trans_matrices = const_cast<int8_t*>(trans_matrices);
    return simulate_dfa_tensor_core(dfa, input_chars, L) ? 1 : 0;
}

void gpu_simulate_dfa_batch(
    int n_states, int alphabet_size, int start_state,
    const int8_t *accept_mask, const int8_t *trans_matrices,
    const int *all_input_chars, int *results,
    int batch_size, int L
) {
    DFAHost dfa;
    dfa.n_states = n_states; dfa.n_states_padded = TILE;
    dfa.alphabet_size = alphabet_size; dfa.start_state = start_state;
    dfa.accept_mask = const_cast<int8_t*>(accept_mask);
    dfa.trans_matrices = const_cast<int8_t*>(trans_matrices);

    if (batch_size <= 0 || L <= 0) {
        bool empty_accept = dfa.accept_mask[dfa.start_state] != 0;
        for (int b = 0; b < batch_size; b++) results[b] = empty_accept ? 1 : 0;
        return;
    }

    GPUContext ctx;
    ctx.init(dfa, L, true);
    for (int b = 0; b < batch_size; b++)
        results[b] = ctx.simulate(all_input_chars + (long long)b * L, L) ? 1 : 0;
    ctx.destroy();
}

int gpu_device_check() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 2)) return -2;
    return 0;
}

}  // extern "C"

// ─── Benchmark ─────────────────────────────────────────────────────────────

struct BenchResult {
    double total_ms;
    double throughput_gbps;
    bool correct;
};

static BenchResult bench_context(
    GPUContext &ctx, const int *h_input, int L, bool expected,
    int n_warmup = 5, int n_iters = 100
) {
    for (int i = 0; i < n_warmup; i++) ctx.simulate(h_input, L);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    bool last = false;
    for (int i = 0; i < n_iters; i++) last = ctx.simulate(h_input, L);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cudaEventDestroy(start); cudaEventDestroy(stop);

    BenchResult r;
    r.total_ms = elapsed_ms / n_iters;
    r.throughput_gbps = (double)L / (r.total_ms * 1e6);
    r.correct = (last == expected);
    return r;
}

// ─── Main: tests + benchmarks ──────────────────────────────────────────────

#ifndef BUILD_LIB

static int g_tests = 0, g_pass = 0;
static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) { g_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

static void test_abb_pattern() {
    printf("\n--- (a|b)*abb ---\n");
    DFABuilder b(5, 2, 0);
    b.set_accept(3);
    b.set_trans(0, 0, 1); b.set_trans(0, 1, 1); b.set_trans(0, 2, 1);
    b.set_trans(0, 3, 1); b.set_trans(0, 4, 4);
    b.set_trans(1, 0, 0); b.set_trans(1, 1, 2); b.set_trans(1, 2, 3);
    b.set_trans(1, 3, 0); b.set_trans(1, 4, 4);

    auto &dfa = b.dfa;

    // Basic tests with self-contained function
    int t_abb[] = {0, 1, 1};
    check("abb → accept (basic)", simulate_dfa_tensor_core(dfa, t_abb, 3) == true);
    int t_ab[] = {0, 1};
    check("ab → reject (basic)", simulate_dfa_tensor_core(dfa, t_ab, 2) == false);
    check("empty → reject (basic)", simulate_dfa_tensor_core(dfa, nullptr, 0) == false);

    // Test GPUContext with both scan algorithms
    for (int use_bk = 0; use_bk <= 1; use_bk++) {
        const char *algo = use_bk ? "blelloch" : "hillis-steele";
        GPUContext ctx;
        ctx.init(dfa, 8192, use_bk);

        {
            int t[] = {0, 1, 1};
            char msg[128]; snprintf(msg, sizeof(msg), "abb → accept (%s)", algo);
            check(msg, ctx.simulate(t, 3) == true);
        }
        {
            int t[] = {0, 1};
            char msg[128]; snprintf(msg, sizeof(msg), "ab → reject (%s)", algo);
            check(msg, ctx.simulate(t, 2) == false);
        }
        {
            char msg[128]; snprintf(msg, sizeof(msg), "empty → reject (%s)", algo);
            check(msg, ctx.simulate(nullptr, 0) == false);
        }

        // Cross-validate random strings
        srand(12345 + use_bk);
        int lengths[] = {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64,
                         127, 128, 255, 256, 511, 512, 1023, 1024, 4096, 8192};
        for (int li = 0; li < 22; li++) {
            int L = lengths[li];
            int *input = new int[L];
            for (int j = 0; j < L; j++) input[j] = rand() % 2;
            bool seq = simulate_sequential(dfa, input, L);
            bool gpu = ctx.simulate(input, L);
            char msg[128];
            snprintf(msg, sizeof(msg), "%s random L=%d: seq=%d gpu=%d", algo, L, seq, gpu);
            check(msg, seq == gpu);
            delete[] input;
        }

        // Known-accepting strings
        for (int li = 0; li < 22; li++) {
            int L = lengths[li];
            if (L < 3) continue;
            int *input = new int[L];
            for (int j = 0; j < L; j++) input[j] = rand() % 2;
            input[L-3] = 0; input[L-2] = 1; input[L-1] = 1;
            bool gpu = ctx.simulate(input, L);
            bool seq = simulate_sequential(dfa, input, L);
            char msg[128];
            snprintf(msg, sizeof(msg), "%s abb-suffix L=%d", algo, L);
            check(msg, gpu == seq && gpu == true);
            delete[] input;
        }

        ctx.destroy();
    }
}

static void test_even_a() {
    printf("\n--- even number of a's ---\n");
    DFABuilder b(2, 2, 0);
    b.set_accept(0);
    b.set_trans(0, 0, 1); b.set_trans(0, 1, 0);
    b.set_trans(1, 0, 0); b.set_trans(1, 1, 1);

    GPUContext ctx;
    ctx.init(b.dfa, 16384, true);

    check("empty → accept", ctx.simulate(nullptr, 0) == true);
    int t1[] = {0}; check("a → reject", ctx.simulate(t1, 1) == false);
    int t2[] = {0, 0}; check("aa → accept", ctx.simulate(t2, 2) == true);
    int t3[] = {1}; check("b → accept", ctx.simulate(t3, 1) == true);

    srand(54321);
    for (int L : {10, 100, 1000, 4096, 8192, 16384}) {
        int *input = new int[L];
        for (int j = 0; j < L; j++) input[j] = rand() % 2;
        bool seq = simulate_sequential(b.dfa, input, L);
        bool gpu = ctx.simulate(input, L);
        char msg[64]; snprintf(msg, sizeof(msg), "random L=%d agree", L);
        check(msg, seq == gpu);
        delete[] input;
    }
    ctx.destroy();
}

static void test_binary_div3() {
    printf("\n--- binary divisible by 3 ---\n");
    DFABuilder b(3, 2, 0);
    b.set_accept(0);
    b.set_trans(0, 0, 0); b.set_trans(0, 1, 2); b.set_trans(0, 2, 1);
    b.set_trans(1, 0, 1); b.set_trans(1, 1, 0); b.set_trans(1, 2, 2);

    GPUContext ctx;
    ctx.init(b.dfa, 8192, true);

    check("empty → accept", ctx.simulate(nullptr, 0) == true);
    int t1[] = {1,1}; check("'11'(=3) → accept", ctx.simulate(t1, 2) == true);
    int t2[] = {1,0}; check("'10'(=2) → reject", ctx.simulate(t2, 2) == false);

    srand(99999);
    for (int trial = 0; trial < 20; trial++) {
        int L = 1 + rand() % 5000;
        int *input = new int[L];
        for (int j = 0; j < L; j++) input[j] = rand() % 2;
        bool seq = simulate_sequential(b.dfa, input, L);
        bool gpu = ctx.simulate(input, L);
        char msg[64]; snprintf(msg, sizeof(msg), "div3 random L=%d agree", L);
        check(msg, seq == gpu);
        delete[] input;
    }
    ctx.destroy();
}

int main() {
    printf("=== Int8 Tensor Core DFA Scan v3 ===\n");

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 2)) {
        printf("ERROR: int8 WMMA requires SM >= 7.2\n"); return 1;
    }

    test_abb_pattern();
    test_even_a();
    test_binary_div3();

    printf("\n=== Results: %d / %d passed ===\n", g_pass, g_tests);
    if (g_pass != g_tests) { printf("SOME TESTS FAILED\n"); return 1; }

    // Throughput benchmark: Blelloch vs Hillis-Steele
    printf("\n=== Throughput Benchmark: (a|b)*abb ===\n");
    {
        DFABuilder b(5, 2, 0);
        b.set_accept(3);
        b.set_trans(0, 0, 1); b.set_trans(0, 1, 1); b.set_trans(0, 2, 1);
        b.set_trans(0, 3, 1); b.set_trans(0, 4, 4);
        b.set_trans(1, 0, 0); b.set_trans(1, 1, 2); b.set_trans(1, 2, 3);
        b.set_trans(1, 3, 0); b.set_trans(1, 4, 4);

        int max_L = 1 << 24;
        GPUContext ctx_bk, ctx_hs;
        ctx_bk.init(b.dfa, max_L, true);
        ctx_hs.init(b.dfa, max_L, false);

        printf("  %10s  %12s  %12s  %12s  %12s\n",
               "Length", "BK time(ms)", "BK GB/s", "HS time(ms)", "HS GB/s");

        for (int log_len = 6; log_len <= 24; log_len += 2) {
            int L = 1 << log_len;
            int *input = new int[L];
            srand(42);
            for (int j = 0; j < L; j++) input[j] = rand() % 2;
            if (L >= 3) { input[L-3] = 0; input[L-2] = 1; input[L-1] = 1; }

            bool expected = simulate_sequential(b.dfa, input, L);
            int iters = L > (1 << 20) ? 20 : 100;
            auto bk = bench_context(ctx_bk, input, L, expected, 3, iters);
            auto hs = bench_context(ctx_hs, input, L, expected, 3, iters);
            printf("  %10d  %12.4f  %12.3f  %12.4f  %12.3f  %s\n",
                   L, bk.total_ms, bk.throughput_gbps,
                   hs.total_ms, hs.throughput_gbps,
                   (bk.correct && hs.correct) ? "" : "WRONG");
            delete[] input;
        }

        ctx_bk.destroy();
        ctx_hs.destroy();
    }

    return 0;
}

#endif  // BUILD_LIB
