/*
 * kgram_evolution.cu -- K-gram TC State Evolution via WMMA
 *
 * Precomputes product matrices for all σ^k possible k-grams, then
 * processes k characters per WMMA MMA call (single-string-per-warp).
 * This increases effective arithmetic intensity by k× compared to
 * per-character evolution.
 *
 * N=16 only (single WMMA tile).
 *
 * Threading model:
 *   Each warp processes ONE string (1 column of the 16×16 tile)
 *   4 warps per block → 4 strings per block
 *   Grid: ceil(B / STRINGS_PER_BLOCK)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace nvcuda;

constexpr int TILE = 16;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int STRINGS_PER_BLOCK = WARPS_PER_BLOCK;

constexpr int STRINGS_PER_WARP_V2 = 2;
constexpr int STRINGS_PER_BLOCK_V2 = WARPS_PER_BLOCK * STRINGS_PER_WARP_V2;

// cp.async helpers (SM 8.0+) — async global→shared memory copy
__device__ __forceinline__ void cp_async_8(void *dst_shared, const void *src_global) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                 :: "r"(dst_addr), "l"(src_global));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)


// ---- K-gram Evolution Kernel (N=16, single WMMA tile) --------------------
//
// Shared memory layout per block:
//   S_sh[4][16*16]       1024 bytes  (state matrices, col-major, only col 0 used)
//   acc_sh[4][16*16]     4096 bytes  (int32 accumulators, row-major)
//   Total:               5120 bytes

__global__ void kgram_evolution_kernel(
    const int8_t  *__restrict__ T_kgram,     // [n_entries, TILE, TILE]
    const int8_t  *__restrict__ T_base,      // [sigma, TILE, TILE]
    const uint8_t *__restrict__ input,       // [L, B_padded]
    const int8_t  *__restrict__ accept_mask, // [TILE]
    const int8_t  *__restrict__ start_vec,   // [TILE]
    int *__restrict__ results,               // [B]
    int B, int B_padded, int L,
    int sigma, int k
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int string_id = blockIdx.x * STRINGS_PER_BLOCK + warp_in_block;

    extern __shared__ char smem_raw[];
    int8_t  *S_base  = (int8_t *)smem_raw;
    int32_t *acc_base = (int32_t *)(S_base + WARPS_PER_BLOCK * TILE_ELEMS);

    int8_t  *S_sh   = S_base + warp_in_block * TILE_ELEMS;
    int32_t *acc_sh = acc_base + warp_in_block * TILE_ELEMS;

    // Initialize S from start vector (col-major, only column 0)
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        int col = e / TILE;
        S_sh[e] = (col == 0) ? start_vec[row] : (int8_t)0;
    }
    __syncwarp();

    if (string_id >= B) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_T;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_S;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_acc;

    wmma::load_matrix_sync(frag_S, S_sh, TILE);

    // Main loop: process k characters per iteration
    int pos = 0;
    for (; pos + k <= L; pos += k) {
        // Compute k-gram index (all lanes compute same value — same string)
        uint32_t idx = 0;
        bool valid = true;
        for (int i = 0; i < k; i++) {
            uint8_t ch = input[(pos + i) * B_padded + string_id];
            if (ch >= (uint8_t)sigma) { valid = false; break; }
            idx = idx * (uint32_t)sigma + (uint32_t)ch;
        }

        if (valid) {
            wmma::load_matrix_sync(frag_T, &T_kgram[idx * TILE_ELEMS], TILE);
            wmma::fill_fragment(frag_acc, 0);
            wmma::mma_sync(frag_acc, frag_T, frag_S, frag_acc);
            wmma::store_matrix_sync(acc_sh, frag_acc, TILE, wmma::mem_row_major);
            __syncwarp();

            if (lane < TILE) {
                S_sh[lane] = (int8_t)(acc_sh[lane * TILE] > 0 ? 1 : 0);
            }
            __syncwarp();
            wmma::load_matrix_sync(frag_S, S_sh, TILE);
        } else {
            // Hit identity/padding — process remaining valid chars then stop
            for (int i = 0; i < k && (pos + i) < L; i++) {
                uint8_t ch = input[(pos + i) * B_padded + string_id];
                if (ch >= (uint8_t)sigma) break;
                wmma::load_matrix_sync(frag_T, &T_base[ch * TILE_ELEMS], TILE);
                wmma::fill_fragment(frag_acc, 0);
                wmma::mma_sync(frag_acc, frag_T, frag_S, frag_acc);
                wmma::store_matrix_sync(acc_sh, frag_acc, TILE, wmma::mem_row_major);
                __syncwarp();
                if (lane < TILE) {
                    S_sh[lane] = (int8_t)(acc_sh[lane * TILE] > 0 ? 1 : 0);
                }
                __syncwarp();
                wmma::load_matrix_sync(frag_S, S_sh, TILE);
            }
            pos = L;
            break;
        }
    }

    // Tail: remaining L % k characters (per-character MMA)
    for (; pos < L; pos++) {
        uint8_t ch = input[pos * B_padded + string_id];
        if (ch >= (uint8_t)sigma) break;
        wmma::load_matrix_sync(frag_T, &T_base[ch * TILE_ELEMS], TILE);
        wmma::fill_fragment(frag_acc, 0);
        wmma::mma_sync(frag_acc, frag_T, frag_S, frag_acc);
        wmma::store_matrix_sync(acc_sh, frag_acc, TILE, wmma::mem_row_major);
        __syncwarp();
        if (lane < TILE) {
            S_sh[lane] = (int8_t)(acc_sh[lane * TILE] > 0 ? 1 : 0);
        }
        __syncwarp();
        wmma::load_matrix_sync(frag_S, S_sh, TILE);
    }

    // Accept check: column 0 of S_sh vs accept_mask
    if (lane == 0) {
        int accepted = 0;
        for (int r = 0; r < TILE; r++) {
            if (S_sh[r] > 0 && accept_mask[r] != 0) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}


// ---- Pipelined V2 Kernel (dual-string per warp + async prefetch) ---------
//
// Each warp processes TWO strings. While string B is processed synchronously,
// string A's k-gram table entry is prefetched asynchronously into shared memory
// via cp.async. This hides one global memory load per iteration behind compute.
//
// Shared memory per block:
//   S_sh[8][TILE_ELEMS]      2048 bytes (int8, state per string)
//   acc_sh[8][TILE_ELEMS]    8192 bytes (int32, accumulators)
//   T_stage[4][TILE_ELEMS]   1024 bytes (int8, prefetch staging per warp)
//   Total:                  11264 bytes

__global__ void __launch_bounds__(BLOCK_SIZE, 16) kgram_evolution_v2_kernel(
    const int8_t  *__restrict__ T_kgram,
    const int8_t  *__restrict__ T_base,
    const uint8_t *__restrict__ input,
    const int8_t  *__restrict__ accept_mask,
    const int8_t  *__restrict__ start_vec,
    int *__restrict__ results,
    int B, int B_padded, int L,
    int sigma, int k
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int sid_A = blockIdx.x * STRINGS_PER_BLOCK_V2 + warp_id * 2;
    int sid_B = sid_A + 1;
    bool valid_A = (sid_A < B);
    bool valid_B = (sid_B < B);

    extern __shared__ char smem_raw[];

    constexpr int N_SLOTS = WARPS_PER_BLOCK * 2;
    int8_t  *S_all    = (int8_t *)smem_raw;
    int32_t *acc_all  = (int32_t *)(S_all + N_SLOTS * TILE_ELEMS);
    int8_t  *T_st_all = (int8_t *)(acc_all + N_SLOTS * TILE_ELEMS);

    int slot_A = warp_id * 2;
    int slot_B = warp_id * 2 + 1;

    int8_t  *S_sh_A   = S_all + slot_A * TILE_ELEMS;
    int8_t  *S_sh_B   = S_all + slot_B * TILE_ELEMS;
    int32_t *acc_sh_A = acc_all + slot_A * TILE_ELEMS;
    int32_t *acc_sh_B = acc_all + slot_B * TILE_ELEMS;
    int8_t  *T_stage  = T_st_all + warp_id * TILE_ELEMS;

    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        int col = e / TILE;
        int8_t val = (col == 0) ? start_vec[row] : (int8_t)0;
        S_sh_A[e] = val;
        S_sh_B[e] = val;
    }
    __syncwarp();

    if (!valid_A && !valid_B) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_T;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_S_A, frag_S_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_acc;

    wmma::load_matrix_sync(frag_S_A, S_sh_A, TILE);
    wmma::load_matrix_sync(frag_S_B, S_sh_B, TILE);

    bool active_A = valid_A, active_B = valid_B;

    // ---- Macros for repeated MMA + threshold blocks ----
    #define DO_MMA_THRESHOLD(fragS, S_sh_ptr, acc_sh_ptr)       \
        wmma::fill_fragment(frag_acc, 0);                       \
        wmma::mma_sync(frag_acc, frag_T, fragS, frag_acc);     \
        wmma::store_matrix_sync(acc_sh_ptr, frag_acc, TILE,     \
                                wmma::mem_row_major);           \
        __syncwarp();                                           \
        if (lane < TILE) {                                      \
            S_sh_ptr[lane] = (int8_t)(acc_sh_ptr[lane*TILE] > 0 ? 1 : 0); \
        }                                                       \
        __syncwarp();                                           \
        wmma::load_matrix_sync(fragS, S_sh_ptr, TILE);

    // Main k-gram loop
    int pos = 0;
    for (; pos + k <= L; pos += k) {
        if (!active_A && !active_B) break;

        uint32_t idx_A = 0, idx_B = 0;
        bool kv_A = active_A, kv_B = active_B;

        if (kv_A) {
            for (int i = 0; i < k; i++) {
                uint8_t ch = input[(pos + i) * B_padded + sid_A];
                if (ch >= (uint8_t)sigma) { kv_A = false; break; }
                idx_A = idx_A * (uint32_t)sigma + (uint32_t)ch;
            }
        }
        if (kv_B) {
            for (int i = 0; i < k; i++) {
                uint8_t ch = input[(pos + i) * B_padded + sid_B];
                if (ch >= (uint8_t)sigma) { kv_B = false; break; }
                idx_B = idx_B * (uint32_t)sigma + (uint32_t)ch;
            }
        }

        if (kv_A && kv_B) {
            // PIPELINED: async prefetch T_A → T_stage, sync process B
            cp_async_8(T_stage + lane * 8,
                       &T_kgram[idx_A * TILE_ELEMS + lane * 8]);
            cp_async_commit();

            wmma::load_matrix_sync(frag_T, &T_kgram[idx_B * TILE_ELEMS], TILE);
            DO_MMA_THRESHOLD(frag_S_B, S_sh_B, acc_sh_B)

            cp_async_wait_all();
            __syncwarp();

            wmma::load_matrix_sync(frag_T, T_stage, TILE);
            DO_MMA_THRESHOLD(frag_S_A, S_sh_A, acc_sh_A)

        } else if (kv_A) {
            wmma::load_matrix_sync(frag_T, &T_kgram[idx_A * TILE_ELEMS], TILE);
            DO_MMA_THRESHOLD(frag_S_A, S_sh_A, acc_sh_A)
        } else if (kv_B) {
            wmma::load_matrix_sync(frag_T, &T_kgram[idx_B * TILE_ELEMS], TILE);
            DO_MMA_THRESHOLD(frag_S_B, S_sh_B, acc_sh_B)
        }

        // Per-char fallback for strings that just hit padding
        if (!kv_A && active_A) {
            for (int i = 0; i < k && (pos + i) < L; i++) {
                uint8_t ch = input[(pos + i) * B_padded + sid_A];
                if (ch >= (uint8_t)sigma) break;
                wmma::load_matrix_sync(frag_T, &T_base[ch * TILE_ELEMS], TILE);
                DO_MMA_THRESHOLD(frag_S_A, S_sh_A, acc_sh_A)
            }
            active_A = false;
        }
        if (!kv_B && active_B) {
            for (int i = 0; i < k && (pos + i) < L; i++) {
                uint8_t ch = input[(pos + i) * B_padded + sid_B];
                if (ch >= (uint8_t)sigma) break;
                wmma::load_matrix_sync(frag_T, &T_base[ch * TILE_ELEMS], TILE);
                DO_MMA_THRESHOLD(frag_S_B, S_sh_B, acc_sh_B)
            }
            active_B = false;
        }
    }

    // Tail: remaining L%k characters
    for (; pos < L; pos++) {
        if (active_A) {
            uint8_t ch = input[pos * B_padded + sid_A];
            if (ch < (uint8_t)sigma) {
                wmma::load_matrix_sync(frag_T, &T_base[ch * TILE_ELEMS], TILE);
                DO_MMA_THRESHOLD(frag_S_A, S_sh_A, acc_sh_A)
            } else {
                active_A = false;
            }
        }
        if (active_B) {
            uint8_t ch = input[pos * B_padded + sid_B];
            if (ch < (uint8_t)sigma) {
                wmma::load_matrix_sync(frag_T, &T_base[ch * TILE_ELEMS], TILE);
                DO_MMA_THRESHOLD(frag_S_B, S_sh_B, acc_sh_B)
            } else {
                active_B = false;
            }
        }
    }

    #undef DO_MMA_THRESHOLD

    // Accept check for both strings
    if (lane == 0) {
        if (valid_A) {
            int accepted = 0;
            for (int r = 0; r < TILE; r++) {
                if (S_sh_A[r] > 0 && accept_mask[r] != 0) { accepted = 1; break; }
            }
            results[sid_A] = accepted;
        }
        if (valid_B) {
            int accepted = 0;
            for (int r = 0; r < TILE; r++) {
                if (S_sh_B[r] > 0 && accept_mask[r] != 0) { accepted = 1; break; }
            }
            results[sid_B] = accepted;
        }
    }
}


// ---- Engine Struct --------------------------------------------------------

struct KGramEngine {
    int N;
    int sigma;
    int k;
    int n_entries;
    int max_B;
    int max_L;

    int8_t  *d_T_kgram;
    int8_t  *d_T_base;
    int8_t  *d_accept;
    int8_t  *d_start_vec;
    uint8_t *d_input;
    int     *d_results;

    int B_padded_max;

    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;

    bool initialized;

    void init(const int8_t *T_kgram, const int8_t *T_base,
              const int8_t *accept_mask, const int8_t *start_vec,
              int n, int sig, int _k, int _n_entries,
              int maxB, int maxL) {
        N = n;
        sigma = sig;
        k = _k;
        n_entries = _n_entries;
        max_B = maxB;
        max_L = maxL;
        B_padded_max = ((maxB + STRINGS_PER_BLOCK_V2 - 1) / STRINGS_PER_BLOCK_V2) * STRINGS_PER_BLOCK_V2;

        size_t table_bytes = (size_t)n_entries * TILE_ELEMS;
        CHECK_CUDA(cudaMalloc(&d_T_kgram, table_bytes));
        CHECK_CUDA(cudaMemcpy(d_T_kgram, T_kgram, table_bytes, cudaMemcpyHostToDevice));

        size_t base_bytes = (size_t)sig * TILE_ELEMS;
        CHECK_CUDA(cudaMalloc(&d_T_base, base_bytes));
        CHECK_CUDA(cudaMemcpy(d_T_base, T_base, base_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&d_accept, TILE));
        CHECK_CUDA(cudaMemcpy(d_accept, accept_mask, TILE, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&d_start_vec, TILE));
        CHECK_CUDA(cudaMemcpy(d_start_vec, start_vec, TILE, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&d_input, (size_t)maxL * B_padded_max));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)maxB * sizeof(int)));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

        initialized = true;
    }

    void destroy() {
        if (!initialized) return;
        cudaFree(d_T_kgram);
        cudaFree(d_T_base);
        cudaFree(d_accept);
        cudaFree(d_start_vec);
        cudaFree(d_input);
        cudaFree(d_results);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start);
        cudaEventDestroy(ev_kern_stop);
        initialized = false;
    }

    int dispatch(const uint8_t *h_input, int B, int L, int B_padded,
                 int *h_results, float *kernel_ms, float *total_ms) {
        if (!initialized) return -1;

        CHECK_CUDA(cudaEventRecord(ev_start));

        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        int n_blocks = (B + STRINGS_PER_BLOCK - 1) / STRINGS_PER_BLOCK;
        int smem = WARPS_PER_BLOCK * TILE_ELEMS
                 + WARPS_PER_BLOCK * TILE_ELEMS * (int)sizeof(int32_t);

        kgram_evolution_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
            d_T_kgram, d_T_base, d_input, d_accept, d_start_vec,
            d_results, B, B_padded, L, sigma, k);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results, (size_t)B * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);

        return 0;
    }

    int dispatch_v2(const uint8_t *h_input, int B, int L, int B_padded,
                    int *h_results, float *kernel_ms, float *total_ms) {
        if (!initialized) return -1;

        CHECK_CUDA(cudaEventRecord(ev_start));

        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        int n_blocks = (B + STRINGS_PER_BLOCK_V2 - 1) / STRINGS_PER_BLOCK_V2;
        constexpr int N_SLOTS = WARPS_PER_BLOCK * 2;
        int smem = N_SLOTS * TILE_ELEMS
                 + N_SLOTS * TILE_ELEMS * (int)sizeof(int32_t)
                 + WARPS_PER_BLOCK * TILE_ELEMS;

        kgram_evolution_v2_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
            d_T_kgram, d_T_base, d_input, d_accept, d_start_vec,
            d_results, B, B_padded, L, sigma, k);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results, (size_t)B * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);

        return 0;
    }
};


// ---- Global Engine --------------------------------------------------------

static KGramEngine g_kgram_engine = {};


// ---- C API ----------------------------------------------------------------

extern "C" {

int kgram_engine_device_check() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 7) return -2;
    return 0;
}

int kgram_engine_init(
    const int8_t *T_kgram,
    const int8_t *T_base,
    const int8_t *accept_mask,
    const int8_t *start_vec,
    int N, int sigma, int k, int n_entries,
    int max_B, int max_L
) {
    if (g_kgram_engine.initialized) g_kgram_engine.destroy();
    g_kgram_engine.init(T_kgram, T_base, accept_mask, start_vec,
                        N, sigma, k, n_entries, max_B, max_L);
    return 0;
}

void kgram_engine_destroy() {
    g_kgram_engine.destroy();
}

int kgram_engine_dispatch(
    const uint8_t *input,
    int B, int L,
    int *results,
    float *kernel_ms, float *total_ms
) {
    int B_padded = ((B + STRINGS_PER_BLOCK - 1) / STRINGS_PER_BLOCK) * STRINGS_PER_BLOCK;
    return g_kgram_engine.dispatch(input, B, L, B_padded, results, kernel_ms, total_ms);
}

int kgram_engine_dispatch_v2(
    const uint8_t *input,
    int B, int L,
    int *results,
    float *kernel_ms, float *total_ms
) {
    int B_padded = ((B + STRINGS_PER_BLOCK_V2 - 1) / STRINGS_PER_BLOCK_V2) * STRINGS_PER_BLOCK_V2;
    return g_kgram_engine.dispatch_v2(input, B, L, B_padded, results, kernel_ms, total_ms);
}

void kgram_prepare_input(
    const char *strings_concat,
    const int *offsets,
    uint8_t *output,
    int B, int B_padded, int L,
    const int *char_to_idx,
    int identity_idx)
{
    memset(output, (uint8_t)identity_idx, (size_t)L * B_padded);
    for (int b = 0; b < B; b++) {
        int str_start = offsets[b];
        int str_len = offsets[b + 1] - str_start;
        int len = (str_len < L) ? str_len : L;
        for (int t = 0; t < len; t++) {
            unsigned char ch = (unsigned char)strings_concat[str_start + t];
            int idx = char_to_idx[ch];
            if (idx < 0) idx = identity_idx;
            output[t * B_padded + b] = (uint8_t)idx;
        }
    }
}

}  // extern "C"


// ---- Built-in Tests -------------------------------------------------------

#ifndef BUILD_LIB

static int g_tests = 0, g_pass = 0;
static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) { g_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

static bool simulate_sequential_ref(
    int N, const int8_t *start_vec,
    const int8_t *accept_mask, const int8_t *trans_matrices,
    const uint8_t *chars, int L, int sigma
) {
    std::vector<int8_t> state(N), new_state(N);
    memcpy(state.data(), start_vec, N);

    for (int t = 0; t < L; t++) {
        int c = chars[t];
        if (c >= sigma) continue;
        const int8_t *T = trans_matrices + c * N * N;
        memset(new_state.data(), 0, N);
        for (int row = 0; row < N; row++) {
            int32_t sum = 0;
            for (int kk = 0; kk < N; kk++)
                sum += (int32_t)T[row * N + kk] * (int32_t)state[kk];
            new_state[row] = (int8_t)(sum > 0 ? 1 : 0);
        }
        memcpy(state.data(), new_state.data(), N);
    }

    for (int r = 0; r < N; r++)
        if (state[r] > 0 && accept_mask[r] != 0) return true;
    return false;
}

static void precompute_kgram_table(
    const int8_t *T_base, int N, int sigma, int k,
    std::vector<int8_t> &table
) {
    int n_entries = 1;
    for (int i = 0; i < k; i++) n_entries *= sigma;
    table.resize((size_t)n_entries * N * N);

    for (int idx = 0; idx < n_entries; idx++) {
        std::vector<int> chars(k);
        int tmp = idx;
        for (int i = k - 1; i >= 0; i--) {
            chars[i] = tmp % sigma;
            tmp /= sigma;
        }

        std::vector<int8_t> acc(N * N, 0);
        for (int i = 0; i < N; i++) acc[i * N + i] = 1;

        for (int ci = 0; ci < k; ci++) {
            int c = chars[ci];
            const int8_t *T = T_base + c * N * N;
            std::vector<int8_t> result(N * N, 0);
            for (int r = 0; r < N; r++) {
                for (int col = 0; col < N; col++) {
                    int32_t sum = 0;
                    for (int m = 0; m < N; m++)
                        sum += (int32_t)T[r * N + m] * (int32_t)acc[m * N + col];
                    result[r * N + col] = (int8_t)(sum > 0 ? 1 : 0);
                }
            }
            acc = result;
        }

        memcpy(&table[idx * N * N], acc.data(), N * N);
    }
}

int main() {
    printf("=== kgram_evolution built-in tests ===\n");

    // Even-A DFA: 2 real states + 14 padded = N=16
    int8_t accept[TILE] = {};
    accept[0] = 1;

    int8_t T_base[2 * TILE_ELEMS] = {};
    for (int s = 2; s < TILE; s++) {
        T_base[0 * TILE_ELEMS + s * TILE + s] = 1;
        T_base[1 * TILE_ELEMS + s * TILE + s] = 1;
    }
    T_base[0 * TILE_ELEMS + 1 * TILE + 0] = 1;  // T[a]: 0->1
    T_base[0 * TILE_ELEMS + 0 * TILE + 1] = 1;  // T[a]: 1->0
    T_base[1 * TILE_ELEMS + 0 * TILE + 0] = 1;  // T[b]: 0->0
    T_base[1 * TILE_ELEMS + 1 * TILE + 1] = 1;  // T[b]: 1->1

    int8_t start_vec[TILE] = {};
    start_vec[0] = 1;

    int sigma = 2;

    struct TestCase { const char *name; std::vector<uint8_t> chars; bool expected; };
    std::vector<TestCase> tests = {
        {"empty",     {},                           true},
        {"a",         {0},                          false},
        {"aa",        {0,0},                        true},
        {"b",         {1},                          true},
        {"ab",        {0,1},                        false},
        {"aabb",      {0,0,1,1},                    true},
        {"aaab",      {0,0,0,1},                    false},
        {"aaaabb",    {0,0,0,0,1,1},                true},
        {"aabba",     {0,0,1,1,0},                  false},
        {"aabbaa",    {0,0,1,1,0,0},                true},
    };

    // Verify reference simulation
    for (auto &tc : tests) {
        bool ref = simulate_sequential_ref(TILE, start_vec, accept, T_base,
                                           tc.chars.data(), (int)tc.chars.size(), sigma);
        char buf[256];
        snprintf(buf, sizeof(buf), "ref_%s", tc.name);
        check(buf, ref == tc.expected);
    }

    // Batch GPU test with k=4
    int k = 4;
    int n_entries = 1;
    for (int i = 0; i < k; i++) n_entries *= sigma;

    std::vector<int8_t> T_kgram;
    precompute_kgram_table(T_base, TILE, sigma, k, T_kgram);
    printf("Precomputed %d k-gram matrices (k=%d, sigma=%d)\n", n_entries, k, sigma);

    int B = (int)tests.size();
    int L_max = 0;
    for (auto &tc : tests) L_max = std::max(L_max, (int)tc.chars.size());
    if (L_max == 0) L_max = 1;

    int B_padded = ((B + STRINGS_PER_BLOCK - 1) / STRINGS_PER_BLOCK) * STRINGS_PER_BLOCK;

    std::vector<uint8_t> input(L_max * B_padded, (uint8_t)sigma);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < (int)tests[b].chars.size(); t++) {
            input[t * B_padded + b] = tests[b].chars[t];
        }
    }

    kgram_engine_init(T_kgram.data(), T_base, accept, start_vec,
                      TILE, sigma, k, n_entries, B_padded, L_max);

    std::vector<int> results(B, -1);
    float kern_ms = 0, total_ms = 0;
    kgram_engine_dispatch(input.data(), B, L_max, results.data(), &kern_ms, &total_ms);

    for (int i = 0; i < B; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "gpu_k%d_%s", k, tests[i].name);
        check(buf, (results[i] != 0) == tests[i].expected);
    }
    printf("k=%d kernel: %.3f ms, total: %.3f ms\n", k, kern_ms, total_ms);

    // Test with k=1
    std::vector<int8_t> T_kgram_k1;
    precompute_kgram_table(T_base, TILE, sigma, 1, T_kgram_k1);

    kgram_engine_init(T_kgram_k1.data(), T_base, accept, start_vec,
                      TILE, sigma, 1, sigma, B_padded, L_max);

    std::vector<int> results_k1(B, -1);
    kgram_engine_dispatch(input.data(), B, L_max, results_k1.data(), &kern_ms, &total_ms);

    for (int i = 0; i < B; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "gpu_k1_%s", tests[i].name);
        check(buf, (results_k1[i] != 0) == tests[i].expected);
    }
    printf("k=1 kernel: %.3f ms, total: %.3f ms\n", kern_ms, total_ms);

    kgram_engine_destroy();

    // ---- V2 (pipelined) tests ----
    printf("\n--- V2 pipelined kernel tests ---\n");

    int B_padded_v2 = ((B + STRINGS_PER_BLOCK_V2 - 1) / STRINGS_PER_BLOCK_V2) * STRINGS_PER_BLOCK_V2;
    std::vector<uint8_t> input_v2(L_max * B_padded_v2, (uint8_t)sigma);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < (int)tests[b].chars.size(); t++) {
            input_v2[t * B_padded_v2 + b] = tests[b].chars[t];
        }
    }

    // V2 with k=4
    kgram_engine_init(T_kgram.data(), T_base, accept, start_vec,
                      TILE, sigma, k, n_entries, B_padded_v2, L_max);

    std::vector<int> results_v2(B, -1);
    kgram_engine_dispatch_v2(input_v2.data(), B, L_max, results_v2.data(), &kern_ms, &total_ms);

    for (int i = 0; i < B; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "v2_k%d_%s", k, tests[i].name);
        check(buf, (results_v2[i] != 0) == tests[i].expected);
    }
    printf("v2 k=%d kernel: %.3f ms, total: %.3f ms\n", k, kern_ms, total_ms);

    // V2 with k=1
    kgram_engine_init(T_kgram_k1.data(), T_base, accept, start_vec,
                      TILE, sigma, 1, sigma, B_padded_v2, L_max);

    std::vector<int> results_v2_k1(B, -1);
    kgram_engine_dispatch_v2(input_v2.data(), B, L_max, results_v2_k1.data(), &kern_ms, &total_ms);

    for (int i = 0; i < B; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "v2_k1_%s", tests[i].name);
        check(buf, (results_v2_k1[i] != 0) == tests[i].expected);
    }
    printf("v2 k=1 kernel: %.3f ms, total: %.3f ms\n", kern_ms, total_ms);

    // V2 with odd batch size (tests boundary handling)
    int B_odd = 7;
    int B_padded_v2_odd = ((B_odd + STRINGS_PER_BLOCK_V2 - 1) / STRINGS_PER_BLOCK_V2) * STRINGS_PER_BLOCK_V2;
    std::vector<uint8_t> input_v2_odd(L_max * B_padded_v2_odd, (uint8_t)sigma);
    for (int b = 0; b < B_odd; b++) {
        for (int t = 0; t < (int)tests[b].chars.size(); t++) {
            input_v2_odd[t * B_padded_v2_odd + b] = tests[b].chars[t];
        }
    }

    kgram_engine_init(T_kgram.data(), T_base, accept, start_vec,
                      TILE, sigma, k, n_entries, B_padded_v2_odd, L_max);

    std::vector<int> results_v2_odd(B_odd, -1);
    kgram_engine_dispatch_v2(input_v2_odd.data(), B_odd, L_max, results_v2_odd.data(), &kern_ms, &total_ms);

    for (int i = 0; i < B_odd; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "v2_odd_%s", tests[i].name);
        check(buf, (results_v2_odd[i] != 0) == tests[i].expected);
    }
    printf("v2 odd B=%d kernel: %.3f ms, total: %.3f ms\n", B_odd, kern_ms, total_ms);

    kgram_engine_destroy();

    printf("\n%d/%d tests passed\n", g_pass, g_tests);
    return (g_pass == g_tests) ? 0 : 1;
}

#endif  // BUILD_LIB
