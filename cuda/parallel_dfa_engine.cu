/*
 * parallel_dfa_engine.cu — v4: Multi-String Parallel Engine
 *
 * Three execution regimes:
 *   R1: Warp-per-string sequential chain (many short strings, L <= 1024)
 *   R3: Decoupled look-back persistent scan (few long strings, L > 1024)
 *   Adaptive dispatch selects regime based on (B, L_max)
 *
 * Features:
 *   - CSR variable-length string support (offsets array)
 *   - Multi-DFA single-pass fusion
 *   - Kernel-only and end-to-end timing
 *   - Stream compaction for cascading pipelines
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

// R1 config: 4 warps per block leaves more blocks per SM
constexpr int R1_WARPS_PER_BLOCK = 4;
constexpr int R1_BLOCK_SIZE = R1_WARPS_PER_BLOCK * WARP_SIZE;  // 128

// R3 config: tile size for decoupled look-back
constexpr int R3_TILE_SIZE = 64;   // positions per tile (sequential chain per tile)
constexpr int R3_WARPS_PER_BLOCK = 4;
constexpr int R3_BLOCK_SIZE = R3_WARPS_PER_BLOCK * WARP_SIZE;

// Decoupled look-back status flags
constexpr int STATUS_INVALID = 0;
constexpr int STATUS_AGGREGATE = 1;
constexpr int STATUS_PREFIX = 2;

// Shared memory per warp: B transpose (256 bytes) + int32 accum (1024 bytes)
constexpr int SMEM_PER_WARP = TILE_ELEMS + TILE_ELEMS * (int)sizeof(int32_t);

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// ─── Device: WMMA 16x16 matmul (same as v3) ──────────────────────────────

__device__ __forceinline__ void warp_matmul_16x16(
    const int8_t *__restrict__ a_global,
    const int8_t *__restrict__ b_global,
    int8_t *__restrict__ c_global,
    int8_t *smem_b,
    int32_t *smem_c,
    int lane
) {
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

// In-place: C = A x C (accumulate into c_global, using temp buffer)
__device__ __forceinline__ void warp_matmul_inplace_right(
    const int8_t *__restrict__ a_global,
    int8_t *__restrict__ c_global,
    int8_t *smem_b,     // 256 bytes for B transpose
    int32_t *smem_c,    // 1024 bytes for accumulator
    int8_t *smem_tmp,   // 256 bytes temp for old C
    int lane
) {
    // Save C to temp
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
        smem_tmp[e] = c_global[e];
    __syncwarp();

    // Transpose saved-C into smem_b (it becomes B operand)
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int r = e / TILE, c = e % TILE;
        smem_b[c * TILE + r] = smem_tmp[r * TILE + c];
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

// ─── R1: Warp-per-string mega-kernel ──────────────────────────────────────
//
// Each warp processes one string by sequential matmul chain.
// Strings are variable-length, packed in CSR format (offsets array).
// Each warp: gather transition matrix for each char, chain-multiply.
// Result: prefix product stored in shared memory, then accept check.

// Shared memory layout per warp for R1:
//   accum[256]:    current running product (int8)
//   smem_b[256]:   B transpose workspace (int8)
//   smem_c[256]:   int32 accumulator for WMMA output
//   tmp[256]:      temp for in-place matmul
constexpr int R1_SMEM_PER_WARP = TILE_ELEMS * 4 + TILE_ELEMS * (int)(sizeof(int32_t) - 1);
// More precisely: accum(256) + smem_b(256) + smem_c(1024) + tmp(256) = 1792 bytes

__global__ void r1_batch_short_strings_kernel(
    const int8_t *__restrict__ trans_matrices,  // [alphabet_size * 256]
    const int    *__restrict__ all_chars,        // [total_chars] concatenated
    const int    *__restrict__ offsets,           // [B+1] string start positions
    int          *__restrict__ results,           // [B] output: 1=accept, 0=reject
    const int8_t *__restrict__ accept_mask,       // [16]
    int start_state,
    int B
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int string_id = blockIdx.x * R1_WARPS_PER_BLOCK + warp_in_block;
    if (string_id >= B) return;

    int str_start = offsets[string_id];
    int str_end   = offsets[string_id + 1];
    int L = str_end - str_start;

    // Shared memory for this warp
    extern __shared__ char smem_raw[];
    char *warp_smem = smem_raw + warp_in_block * (TILE_ELEMS * 3 + TILE_ELEMS * (int)sizeof(int32_t));
    int8_t  *accum   = (int8_t *)warp_smem;
    int8_t  *smem_b  = accum + TILE_ELEMS;
    int32_t *smem_c  = (int32_t *)(smem_b + TILE_ELEMS);
    int8_t  *tmp     = (int8_t *)(smem_c + TILE_ELEMS);

    // Initialize accum to identity matrix
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int r = e / TILE, c = e % TILE;
        accum[e] = (r == c) ? (int8_t)1 : (int8_t)0;
    }
    __syncwarp();

    if (L == 0) {
        // Empty string: check if start state is accepting
        if (lane == 0)
            results[string_id] = (accept_mask[start_state] != 0) ? 1 : 0;
        return;
    }

    // Sequential chain: accum = T[c_{L-1}] x T[c_{L-2}] x ... x T[c_0]
    // We compute left-to-right: accum = T[c_0], then accum = T[c_1] x accum, ...
    // Convention: T[c][dst][src], so prefix = T[c_{i}] @ prefix_{i-1}
    // After full chain, final_state = accum @ start_vec

    // First matrix: just copy T[c_0] into accum
    {
        int ch = all_chars[str_start];
        const int8_t *mat = trans_matrices + ch * TILE_ELEMS;
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            accum[e] = mat[e];
        __syncwarp();
    }

    // Chain remaining: accum = T[c_i] @ accum
    for (int i = 1; i < L; i++) {
        int ch = all_chars[str_start + i];
        const int8_t *mat = trans_matrices + ch * TILE_ELEMS;
        // accum = mat @ accum (mat is A, accum is B)
        warp_matmul_inplace_right(mat, accum, smem_b, smem_c, tmp, lane);
        __syncwarp();
    }

    // Extract result: final_state_vec = accum @ start_vec
    // start_vec has 1 at start_state, 0 elsewhere
    // final_state[r] = accum[r * TILE + start_state]
    // Accept if any accepting state has nonzero entry
    if (lane == 0) {
        int accepted = 0;
        for (int r = 0; r < TILE; r++) {
            if (accum[r * TILE + start_state] > 0 && accept_mask[r] != 0) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}


// ─── Multi-DFA: Run K DFAs on same batch in one kernel ───────────────────
//
// Each warp handles one (string, dfa) pair. Grid covers B*K warps total.
// This amortizes kernel launch overhead and allows the GPU to read input
// characters once across multiple DFA evaluations.

__global__ void multi_dfa_batch_kernel(
    const int8_t *const *__restrict__ trans_ptrs,  // [K] pointers to trans matrices
    const int8_t *const *__restrict__ accept_ptrs, // [K] pointers to accept masks
    const int    *__restrict__ start_states,         // [K]
    const int    *__restrict__ all_chars,             // [total_chars] concatenated
    const int    *__restrict__ offsets,               // [B+1]
    int          *__restrict__ results,               // [B * K] row-major: results[b*K+k]
    int B, int K
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int global_warp = blockIdx.x * R1_WARPS_PER_BLOCK + warp_in_block;
    if (global_warp >= B * K) return;

    int string_id = global_warp / K;
    int dfa_id    = global_warp % K;

    int str_start = offsets[string_id];
    int str_end   = offsets[string_id + 1];
    int L = str_end - str_start;

    const int8_t *trans  = trans_ptrs[dfa_id];
    const int8_t *accept = accept_ptrs[dfa_id];
    int start_state = start_states[dfa_id];

    extern __shared__ char smem_raw[];
    char *warp_smem = smem_raw + warp_in_block * (TILE_ELEMS * 3 + TILE_ELEMS * (int)sizeof(int32_t));
    int8_t  *accum   = (int8_t *)warp_smem;
    int8_t  *smem_b  = accum + TILE_ELEMS;
    int32_t *smem_c  = (int32_t *)(smem_b + TILE_ELEMS);
    int8_t  *tmp     = (int8_t *)(smem_c + TILE_ELEMS);

    if (L == 0) {
        if (lane == 0)
            results[string_id * K + dfa_id] = (accept[start_state] != 0) ? 1 : 0;
        return;
    }

    // First matrix
    {
        int ch = all_chars[str_start];
        const int8_t *mat = trans + ch * TILE_ELEMS;
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            accum[e] = mat[e];
        __syncwarp();
    }

    for (int i = 1; i < L; i++) {
        int ch = all_chars[str_start + i];
        const int8_t *mat = trans + ch * TILE_ELEMS;
        warp_matmul_inplace_right(mat, accum, smem_b, smem_c, tmp, lane);
        __syncwarp();
    }

    if (lane == 0) {
        int accepted = 0;
        for (int r = 0; r < TILE; r++) {
            if (accum[r * TILE + start_state] > 0 && accept[r] != 0) {
                accepted = 1;
                break;
            }
        }
        results[string_id * K + dfa_id] = accepted;
    }
}


// ─── Stream Compaction: select surviving indices ─────────────────────────

__global__ void compact_indices_kernel(
    const int *__restrict__ mask,       // [B] 0 or 1
    int       *__restrict__ out_indices, // [B] compacted surviving indices
    int       *__restrict__ out_count,   // [1] number of survivors
    int B
) {
    // Simple single-block compaction for moderate B.
    // For very large B, use CUB DeviceSelect.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;
    if (mask[idx]) {
        int pos = atomicAdd(out_count, 1);
        out_indices[pos] = idx;
    }
}


// ─── R3: Decoupled Look-back Persistent Kernel ───────────────────────────
//
// For long strings: partition into tiles, each block scans a tile sequentially,
// then coordinates with preceding blocks via decoupled look-back protocol.
//
// Per-tile data in global memory:
//   tile_aggregates[tile_id * 256]:  aggregate product for this tile
//   tile_prefixes[tile_id * 256]:    inclusive prefix (all tiles up to this one)
//   tile_status[tile_id]:            STATUS_INVALID / AGGREGATE / PREFIX

// Shared memory per block for R3:
// We use 1 warp for the scan chain and look-back.
// warp 0: does all the work. Remaining warps idle (could optimize later).
// smem: accum(256) + smem_b(256) + smem_c(1024) + tmp(256) + lookback_accum(256) = 2048 bytes
constexpr int R3_SMEM_SIZE = TILE_ELEMS * 4 + TILE_ELEMS * (int)sizeof(int32_t);

__device__ void set_identity_smem(int8_t *buf, int lane) {
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int r = e / TILE, c = e % TILE;
        buf[e] = (r == c) ? (int8_t)1 : (int8_t)0;
    }
    __syncwarp();
}

__global__ void r3_decoupled_lookback_kernel(
    const int8_t *__restrict__ trans_matrices,
    const int    *__restrict__ chars,
    int           L,
    int8_t       *__restrict__ tile_aggregates,  // [n_tiles * 256]
    int8_t       *__restrict__ tile_prefixes,     // [n_tiles * 256]
    volatile int *__restrict__ tile_status,       // [n_tiles]
    int8_t       *__restrict__ final_result,      // [256] output: prefix product at position L-1
    int           n_tiles,
    int           tile_size
) {
    int tile_id = blockIdx.x;
    if (tile_id >= n_tiles) return;

    int lane = threadIdx.x % WARP_SIZE;

    // Only warp 0 does work
    if (threadIdx.x >= WARP_SIZE) return;

    extern __shared__ char smem_raw[];
    int8_t  *accum         = (int8_t *)smem_raw;
    int8_t  *smem_b        = accum + TILE_ELEMS;
    int32_t *smem_c        = (int32_t *)(smem_b + TILE_ELEMS);
    int8_t  *tmp           = (int8_t *)(smem_c + TILE_ELEMS);
    int8_t  *lookback_buf  = tmp + TILE_ELEMS;

    // ── Phase 1: Compute tile aggregate via sequential chain ──
    int tile_start = tile_id * tile_size;
    int tile_end = min(tile_start + tile_size, L);
    int tile_len = tile_end - tile_start;

    if (tile_len == 0) {
        // Empty tile: aggregate = identity
        set_identity_smem(accum, lane);
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            tile_aggregates[(long long)tile_id * TILE_ELEMS + e] = accum[e];
        __syncwarp();
        __threadfence();
        if (lane == 0) tile_status[tile_id] = STATUS_AGGREGATE;
        return;
    }

    // accum = T[c_0]
    {
        int ch = chars[tile_start];
        const int8_t *mat = trans_matrices + ch * TILE_ELEMS;
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            accum[e] = mat[e];
        __syncwarp();
    }

    // accum = T[c_i] @ accum for i = 1..tile_len-1
    for (int i = 1; i < tile_len; i++) {
        int ch = chars[tile_start + i];
        const int8_t *mat = trans_matrices + ch * TILE_ELEMS;
        warp_matmul_inplace_right(mat, accum, smem_b, smem_c, tmp, lane);
        __syncwarp();
    }

    // Store aggregate
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
        tile_aggregates[(long long)tile_id * TILE_ELEMS + e] = accum[e];
    __syncwarp();
    __threadfence();
    if (lane == 0) tile_status[tile_id] = STATUS_AGGREGATE;

    // ── Phase 2: Decoupled look-back ──
    if (tile_id == 0) {
        // First tile: prefix = aggregate
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            tile_prefixes[e] = accum[e];
        __syncwarp();
        __threadfence();
        if (lane == 0) tile_status[tile_id] = STATUS_PREFIX;
    } else {
        // Look back to compute inclusive prefix
        // lookback_buf = identity (will accumulate predecessors' aggregates)
        set_identity_smem(lookback_buf, lane);

        // Look-back: accumulate predecessors' contributions.
        // We scan from tile_id-1 backwards. Each predecessor is EARLIER
        // in the string, so its contribution goes on the RIGHT:
        //   lookback_buf = lookback_buf @ pred
        // After the loop, prefix[tile_id] = agg[tile_id] @ lookback_buf.

        int look = tile_id - 1;
        while (look >= 0) {
            int status;
            if (lane == 0) {
                do {
                    status = tile_status[look];
                } while (status == STATUS_INVALID);
            }
            status = __shfl_sync(0xFFFFFFFF, status, 0);

            if (status == STATUS_PREFIX) {
                const int8_t *pred_prefix = tile_prefixes + (long long)look * TILE_ELEMS;

                for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
                    tmp[e] = pred_prefix[e];
                __syncwarp();

                // lookback_buf = lookback_buf @ pred_prefix
                // lookback_buf is A (later tiles), pred_prefix is B (earlier)
                warp_matmul_16x16(lookback_buf, tmp, lookback_buf, smem_b, smem_c, lane);
                __syncwarp();
                break;
            } else {
                const int8_t *pred_agg = tile_aggregates + (long long)look * TILE_ELEMS;

                for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
                    tmp[e] = pred_agg[e];
                __syncwarp();

                // lookback_buf = lookback_buf @ pred_agg
                warp_matmul_16x16(lookback_buf, tmp, lookback_buf, smem_b, smem_c, lane);
                __syncwarp();
                look--;
            }
        }

        // prefix[tile_id] = agg[tile_id] @ lookback_buf
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            accum[e] = tile_aggregates[(long long)tile_id * TILE_ELEMS + e];
        __syncwarp();

        warp_matmul_16x16(accum, lookback_buf,
                          tile_prefixes + (long long)tile_id * TILE_ELEMS,
                          smem_b, smem_c, lane);
        __syncwarp();
        __threadfence();
        if (lane == 0) tile_status[tile_id] = STATUS_PREFIX;
    }

    // ── Phase 3: If this is the last tile covering position L-1, output result ──
    if (tile_end >= L && tile_start < L) {
        // This tile contains the final position
        // The inclusive prefix for this tile = prefix product of all positions 0..tile_end-1
        // But we need prefix at position L-1 specifically.
        // Since we computed the full tile aggregate, and the look-back gives us prefix up to
        // the end of this tile, the tile prefix IS the prefix product at position tile_end-1.
        // If tile_end-1 == L-1, we're done. If tile has padding (tile_end > L), the
        // positions >= L were not processed (tile_len = L - tile_start), so the aggregate
        // only covers up to L-1. The tile prefix is correct.
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            final_result[e] = tile_prefixes[(long long)tile_id * TILE_ELEMS + e];
    }
}


// ─── Batch Dispatch for R3 (multiple long strings) ───────────────────────
// For a batch of long strings, host dispatches R3 per string using the
// same kernel with offset pointer arithmetic on the host side.


// ─── Host-side Engine ─────────────────────────────────────────────────────

struct ParallelEngine {
    // DFA parameters
    int n_states;
    int alphabet_size;
    int start_state;
    int8_t *h_accept_mask;    // [16] host
    int8_t *h_trans_matrices; // [alphabet_size * 256] host

    // Persistent device memory
    int8_t *d_trans;          // [alphabet_size * 256]
    int8_t *d_accept;         // [16]

    // Buffers (sized for max workload)
    int    *d_chars;          // [max_total_chars]
    int    *d_offsets;        // [max_B + 1]
    int    *d_results;        // [max_B]
    int     max_total_chars;
    int     max_B;

    // R3 buffers
    int8_t *d_tile_agg;      // [max_tiles * 256]
    int8_t *d_tile_prefix;   // [max_tiles * 256]
    int    *d_tile_status;   // [max_tiles]
    int8_t *d_final_result;  // [256]
    int     max_tiles;

    // Timing
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;

    void init(int n_st, int alpha_sz, int start_st,
              const int8_t *accept, const int8_t *trans,
              int max_chars, int max_batch) {
        n_states = n_st;
        alphabet_size = alpha_sz;
        start_state = start_st;
        h_accept_mask = new int8_t[TILE];
        h_trans_matrices = new int8_t[alpha_sz * TILE_ELEMS];
        memcpy(h_accept_mask, accept, TILE);
        memcpy(h_trans_matrices, trans, alpha_sz * TILE_ELEMS);

        max_total_chars = max_chars;
        max_B = max_batch;

        CHECK_CUDA(cudaMalloc(&d_trans, alpha_sz * TILE_ELEMS));
        CHECK_CUDA(cudaMalloc(&d_accept, TILE));
        CHECK_CUDA(cudaMalloc(&d_chars, (size_t)max_chars * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_offsets, (size_t)(max_batch + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)max_batch * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_trans, trans, alpha_sz * TILE_ELEMS, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, accept, TILE, cudaMemcpyHostToDevice));

        // R3 buffers: max tiles for a single string of max_chars
        max_tiles = (max_chars + R3_TILE_SIZE - 1) / R3_TILE_SIZE + 1;
        CHECK_CUDA(cudaMalloc(&d_tile_agg,    (size_t)max_tiles * TILE_ELEMS));
        CHECK_CUDA(cudaMalloc(&d_tile_prefix,  (size_t)max_tiles * TILE_ELEMS));
        CHECK_CUDA(cudaMalloc(&d_tile_status,  (size_t)max_tiles * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_final_result, TILE_ELEMS));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));
    }

    void destroy() {
        delete[] h_accept_mask;
        delete[] h_trans_matrices;
        cudaFree(d_trans); cudaFree(d_accept);
        cudaFree(d_chars); cudaFree(d_offsets); cudaFree(d_results);
        cudaFree(d_tile_agg); cudaFree(d_tile_prefix);
        cudaFree(d_tile_status); cudaFree(d_final_result);
        cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start); cudaEventDestroy(ev_kern_stop);
    }

    // ── R1 dispatch: many short strings ──
    // h_chars: concatenated character indices
    // h_offsets: [B+1] CSR offsets into h_chars
    // h_results: [B] output
    // Returns: kernel time in ms (via kernel_ms), total time in ms (via total_ms)
    void dispatch_r1(const int *h_chars, const int *h_offsets,
                     int *h_results, int B, int total_chars,
                     float *kernel_ms, float *total_ms) {
        CHECK_CUDA(cudaEventRecord(ev_start));

        // H2D
        CHECK_CUDA(cudaMemcpy(d_chars, h_chars, (size_t)total_chars * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (size_t)(B + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Kernel
        int grid = (B + R1_WARPS_PER_BLOCK - 1) / R1_WARPS_PER_BLOCK;
        int smem = R1_WARPS_PER_BLOCK * (TILE_ELEMS * 3 + TILE_ELEMS * (int)sizeof(int32_t));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        r1_batch_short_strings_kernel<<<grid, R1_BLOCK_SIZE, smem>>>(
            d_trans, d_chars, d_offsets, d_results,
            d_accept, start_state, B);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        // D2H
        CHECK_CUDA(cudaMemcpy(h_results, d_results, (size_t)B * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);
    }

    // ── R3 dispatch: single long string ──
    // Returns accept/reject. Timing via kernel_ms / total_ms.
    bool dispatch_r3(const int *h_chars, int L,
                     float *kernel_ms, float *total_ms) {
        if (L == 0) return h_accept_mask[start_state] != 0;

        int n_tiles = (L + R3_TILE_SIZE - 1) / R3_TILE_SIZE;

        CHECK_CUDA(cudaEventRecord(ev_start));

        // H2D
        CHECK_CUDA(cudaMemcpy(d_chars, h_chars, (size_t)L * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Clear tile status
        CHECK_CUDA(cudaMemset(d_tile_status, 0, (size_t)n_tiles * sizeof(int)));

        // Kernel
        // Launch n_tiles blocks, each with R3_BLOCK_SIZE threads
        int smem = TILE_ELEMS * 5 + TILE_ELEMS * (int)sizeof(int32_t);
        // accum(256) + smem_b(256) + smem_c(1024) + tmp(256) + lookback_buf(256) = 2048

        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        r3_decoupled_lookback_kernel<<<n_tiles, R3_BLOCK_SIZE, smem>>>(
            d_trans, d_chars, L,
            d_tile_agg, d_tile_prefix, d_tile_status,
            d_final_result, n_tiles, R3_TILE_SIZE);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        // D2H: read final result matrix
        int8_t h_final[TILE_ELEMS];
        CHECK_CUDA(cudaMemcpy(h_final, d_final_result, TILE_ELEMS,
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);

        // Accept check
        for (int r = 0; r < n_states; r++)
            if (h_final[r * TILE + start_state] > 0 && h_accept_mask[r] != 0)
                return true;
        return false;
    }

    // ── Adaptive dispatch: batch of variable-length strings ──
    // Uses R1 for short strings, R3 for long strings.
    // h_chars[total_chars], h_offsets[B+1], h_results[B]
    void dispatch_adaptive(const int *h_chars, const int *h_offsets,
                           int *h_results, int B, int total_chars,
                           float *kernel_ms, float *total_ms) {
        // Find max length
        int L_max = 0;
        for (int i = 0; i < B; i++) {
            int len = h_offsets[i + 1] - h_offsets[i];
            if (len > L_max) L_max = len;
        }

        // Simple regime selection
        if (B >= 4 && L_max <= 1024) {
            // R1: all strings short enough for warp-per-string
            dispatch_r1(h_chars, h_offsets, h_results, B, total_chars,
                        kernel_ms, total_ms);
        } else if (B == 1) {
            // Single long string: R3
            bool result = dispatch_r3(h_chars, h_offsets[1] - h_offsets[0],
                                      kernel_ms, total_ms);
            h_results[0] = result ? 1 : 0;
        } else {
            // Mixed: bin into short (R1) and long (R3)
            // For now, process all via R1 if average is short, else R3 per string
            float avg_len = (float)total_chars / B;
            if (avg_len <= 1024) {
                dispatch_r1(h_chars, h_offsets, h_results, B, total_chars,
                            kernel_ms, total_ms);
            } else {
                // Process each string individually via R3
                // (future: concurrent streams)
                CHECK_CUDA(cudaEventRecord(ev_start));
                CHECK_CUDA(cudaEventRecord(ev_kern_start));
                for (int i = 0; i < B; i++) {
                    int off = h_offsets[i];
                    int len = h_offsets[i + 1] - off;
                    float km, tm;
                    bool r = dispatch_r3(h_chars + off, len, &km, &tm);
                    h_results[i] = r ? 1 : 0;
                }
                CHECK_CUDA(cudaEventRecord(ev_kern_stop));
                CHECK_CUDA(cudaEventRecord(ev_stop));
                CHECK_CUDA(cudaEventSynchronize(ev_stop));
                if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
                if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);
            }
        }
    }
};

// ─── Multi-DFA Engine ─────────────────────────────────────────────────────

struct MultiDFAEngine {
    static constexpr int MAX_DFAS = 8;

    int K;  // number of DFAs
    int8_t *d_trans_all[MAX_DFAS];
    int8_t *d_accept_all[MAX_DFAS];
    int     h_start_states[MAX_DFAS];
    int     h_n_states[MAX_DFAS];

    // Device pointer arrays
    int8_t **d_trans_ptrs;
    int8_t **d_accept_ptrs;
    int    *d_start_states;

    // Shared buffers
    int    *d_chars;
    int    *d_offsets;
    int    *d_results;  // [max_B * MAX_DFAS]
    int     max_total_chars;
    int     max_B;

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;

    void init(int max_chars, int max_batch) {
        K = 0;
        max_total_chars = max_chars;
        max_B = max_batch;
        CHECK_CUDA(cudaMalloc(&d_chars, (size_t)max_chars * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_offsets, (size_t)(max_batch + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)max_batch * MAX_DFAS * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_trans_ptrs, MAX_DFAS * sizeof(int8_t *)));
        CHECK_CUDA(cudaMalloc(&d_accept_ptrs, MAX_DFAS * sizeof(int8_t *)));
        CHECK_CUDA(cudaMalloc(&d_start_states, MAX_DFAS * sizeof(int)));
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));
    }

    void add_dfa(int n_states, int alphabet_size, int start_state,
                 const int8_t *accept, const int8_t *trans) {
        if (K >= MAX_DFAS) return;
        int k = K++;
        h_n_states[k] = n_states;
        h_start_states[k] = start_state;
        CHECK_CUDA(cudaMalloc(&d_trans_all[k], alphabet_size * TILE_ELEMS));
        CHECK_CUDA(cudaMalloc(&d_accept_all[k], TILE));
        CHECK_CUDA(cudaMemcpy(d_trans_all[k], trans, alphabet_size * TILE_ELEMS,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept_all[k], accept, TILE,
                              cudaMemcpyHostToDevice));
        // Update device pointer arrays
        CHECK_CUDA(cudaMemcpy(d_trans_ptrs + k, &d_trans_all[k], sizeof(int8_t *),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept_ptrs + k, &d_accept_all[k], sizeof(int8_t *),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_start_states + k, &start_state, sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    // Run all K DFAs on B strings. results[b*K + k] = accept/reject.
    void dispatch(const int *h_chars, const int *h_offsets,
                  int *h_results, int B, int total_chars,
                  float *kernel_ms, float *total_ms) {
        CHECK_CUDA(cudaEventRecord(ev_start));
        CHECK_CUDA(cudaMemcpy(d_chars, h_chars, (size_t)total_chars * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (size_t)(B + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

        int total_warps = B * K;
        int grid = (total_warps + R1_WARPS_PER_BLOCK - 1) / R1_WARPS_PER_BLOCK;
        int smem = R1_WARPS_PER_BLOCK * (TILE_ELEMS * 3 + TILE_ELEMS * (int)sizeof(int32_t));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        multi_dfa_batch_kernel<<<grid, R1_BLOCK_SIZE, smem>>>(
            (const int8_t *const *)d_trans_ptrs,
            (const int8_t *const *)d_accept_ptrs,
            d_start_states,
            d_chars, d_offsets, d_results,
            B, K);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results, (size_t)B * K * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);
    }

    void destroy() {
        for (int k = 0; k < K; k++) {
            cudaFree(d_trans_all[k]);
            cudaFree(d_accept_all[k]);
        }
        cudaFree(d_chars); cudaFree(d_offsets); cudaFree(d_results);
        cudaFree(d_trans_ptrs); cudaFree(d_accept_ptrs); cudaFree(d_start_states);
        cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start); cudaEventDestroy(ev_kern_stop);
    }
};

// ─── Host: Sequential reference ───────────────────────────────────────────

static bool simulate_sequential_host(
    int n_states, int start_state,
    const int8_t *accept_mask, const int8_t *trans_matrices,
    const int *input, int L
) {
    if (L == 0) return accept_mask[start_state] != 0;
    int state = start_state;
    for (int i = 0; i < L; i++) {
        int c = input[i];
        for (int d = 0; d < TILE; d++)
            if (trans_matrices[c * TILE_ELEMS + d * TILE + state] == 1) {
                state = d; break;
            }
    }
    return accept_mask[state] != 0;
}

// ─── Python bridge (ctypes) ──────────────────────────────────────────────

#ifdef BUILD_LIB

extern "C" {

// Reuse v3 bridge functions for backward compat
// New parallel engine bridge:

struct EngineHandle {
    ParallelEngine engine;
    bool initialized;
};

static EngineHandle g_engine = {.initialized = false};

int engine_init(int n_states, int alphabet_size, int start_state,
                const int8_t *accept_mask, const int8_t *trans_matrices,
                int max_total_chars, int max_batch) {
    if (g_engine.initialized) g_engine.engine.destroy();
    g_engine.engine.init(n_states, alphabet_size, start_state,
                         accept_mask, trans_matrices,
                         max_total_chars, max_batch);
    g_engine.initialized = true;
    return 0;
}

void engine_destroy() {
    if (g_engine.initialized) {
        g_engine.engine.destroy();
        g_engine.initialized = false;
    }
}

// Dispatch batch of variable-length strings
// Returns kernel_ms and total_ms via pointers
int engine_dispatch_batch(
    const int *chars, const int *offsets, int *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms
) {
    if (!g_engine.initialized) return -1;
    g_engine.engine.dispatch_adaptive(chars, offsets, results, B, total_chars,
                                      kernel_ms, total_ms);
    return 0;
}

// Dispatch single string (convenience)
int engine_dispatch_single(const int *chars, int L, float *kernel_ms, float *total_ms) {
    if (!g_engine.initialized) return -2;
    int off[2] = {0, L};
    int result;
    g_engine.engine.dispatch_adaptive(chars, off, &result, 1, L, kernel_ms, total_ms);
    return result;
}

int engine_device_check() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 2)) return -2;
    return 0;
}

}  // extern "C"

#endif  // BUILD_LIB


// ─── Test + Benchmark (standalone build) ──────────────────────────────────

#ifndef BUILD_LIB

static int g_tests = 0, g_pass = 0;
static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) { g_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

// Helper: build (a|b)*abb DFA
struct AbbDFA {
    int8_t accept[TILE];
    int8_t trans[2 * TILE_ELEMS];  // alphabet=2
    AbbDFA() {
        memset(accept, 0, sizeof(accept));
        memset(trans, 0, sizeof(trans));
        accept[3] = 1;
        // Self-loops for padding states
        for (int c = 0; c < 2; c++)
            for (int s = 5; s < TILE; s++)
                trans[c * TILE_ELEMS + s * TILE + s] = 1;
        // char 'a' (0): all states -> 1 except dead state 4
        trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1;  // 0->1
        trans[0 * TILE_ELEMS + 1 * TILE + 1] = 1;  // 1->1
        trans[0 * TILE_ELEMS + 1 * TILE + 2] = 1;  // 2->1
        trans[0 * TILE_ELEMS + 1 * TILE + 3] = 1;  // 3->1
        trans[0 * TILE_ELEMS + 4 * TILE + 4] = 1;  // 4->4 (dead)
        // char 'b' (1): 0->0, 1->2, 2->3, 3->0
        trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1;  // 0->0
        trans[1 * TILE_ELEMS + 2 * TILE + 1] = 1;  // 1->2
        trans[1 * TILE_ELEMS + 3 * TILE + 2] = 1;  // 2->3
        trans[1 * TILE_ELEMS + 0 * TILE + 3] = 1;  // 3->0
        trans[1 * TILE_ELEMS + 4 * TILE + 4] = 1;  // 4->4 (dead)
    }
};

// Helper: "even number of a's" DFA (2 states, alphabet=2)
struct EvenADFA {
    int8_t accept[TILE];
    int8_t trans[2 * TILE_ELEMS];
    EvenADFA() {
        memset(accept, 0, sizeof(accept));
        memset(trans, 0, sizeof(trans));
        accept[0] = 1;  // state 0 = even count (accept)
        for (int c = 0; c < 2; c++)
            for (int s = 2; s < TILE; s++)
                trans[c * TILE_ELEMS + s * TILE + s] = 1;
        // 'a' (0): 0->1, 1->0
        trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1;
        trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1;
        // 'b' (1): 0->0, 1->1
        trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1;
        trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1;
    }
};

static void test_multi_dfa() {
    printf("\n--- Multi-DFA correctness ---\n");
    AbbDFA dfa1;     // (a|b)*abb: accepts strings ending in "abb"
    EvenADFA dfa2;   // even_a: accepts strings with even number of 'a's

    MultiDFAEngine eng;
    eng.init(1 << 20, 1 << 16);
    eng.add_dfa(5, 2, 0, dfa1.accept, dfa1.trans);
    eng.add_dfa(2, 2, 0, dfa2.accept, dfa2.trans);

    // Known tests
    {
        // "abb": dfa1=accept, dfa2=odd a's (1 'a') -> reject
        int chars[] = {0, 1, 1};
        int offsets[] = {0, 3};
        int results[2];
        eng.dispatch(chars, offsets, results, 1, 3, nullptr, nullptr);
        check("MultiDFA 'abb' dfa1=accept", results[0] == 1);
        check("MultiDFA 'abb' dfa2=reject", results[1] == 0);
    }
    {
        // "aabb": dfa1=accept (ends in abb), dfa2=even a's (2) -> accept
        int chars[] = {0, 0, 1, 1};
        int offsets[] = {0, 4};
        int results[2];
        eng.dispatch(chars, offsets, results, 1, 4, nullptr, nullptr);
        check("MultiDFA 'aabb' dfa1=accept", results[0] == 1);
        check("MultiDFA 'aabb' dfa2=accept", results[1] == 1);
    }
    {
        // "ab": dfa1=reject, dfa2=odd a's -> reject
        int chars[] = {0, 1};
        int offsets[] = {0, 2};
        int results[2];
        eng.dispatch(chars, offsets, results, 1, 2, nullptr, nullptr);
        check("MultiDFA 'ab' dfa1=reject", results[0] == 0);
        check("MultiDFA 'ab' dfa2=reject", results[1] == 0);
    }

    // Large batch cross-validation
    {
        int B = 5000;
        int L = 50;
        int total = B * L;
        int *chars = new int[total];
        int *offsets = new int[B + 1];
        int *results = new int[B * 2];
        srand(42);
        for (int i = 0; i < total; i++) chars[i] = rand() % 2;
        for (int i = 0; i <= B; i++) offsets[i] = i * L;

        eng.dispatch(chars, offsets, results, B, total, nullptr, nullptr);

        int m1 = 0, m2 = 0;
        for (int i = 0; i < B; i++) {
            bool exp1 = simulate_sequential_host(5, 0, dfa1.accept, dfa1.trans,
                                                  chars + offsets[i], L);
            bool exp2 = simulate_sequential_host(2, 0, dfa2.accept, dfa2.trans,
                                                  chars + offsets[i], L);
            if (results[i * 2 + 0] != (exp1 ? 1 : 0)) m1++;
            if (results[i * 2 + 1] != (exp2 ? 1 : 0)) m2++;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "MultiDFA batch B=%d dfa1 (%d mismatches)", B, m1);
        check(msg, m1 == 0);
        snprintf(msg, sizeof(msg), "MultiDFA batch B=%d dfa2 (%d mismatches)", B, m2);
        check(msg, m2 == 0);
        delete[] chars; delete[] offsets; delete[] results;
    }

    eng.destroy();
}

static void test_r1_basic() {
    printf("\n--- R1: Basic correctness ---\n");
    AbbDFA dfa;
    int max_chars = 1 << 23;  // 8M: enough for variable-length tests
    int max_batch = 1 << 16;
    ParallelEngine eng;
    eng.init(5, 2, 0, dfa.accept, dfa.trans, max_chars, max_batch);

    // Single known strings
    {
        int chars[] = {0, 1, 1};  // "abb" -> accept
        int offsets[] = {0, 3};
        int results[1];
        eng.dispatch_r1(chars, offsets, results, 1, 3, nullptr, nullptr);
        check("R1 single 'abb' -> accept", results[0] == 1);
    }
    {
        int chars[] = {0, 1};  // "ab" -> reject
        int offsets[] = {0, 2};
        int results[1];
        eng.dispatch_r1(chars, offsets, results, 1, 2, nullptr, nullptr);
        check("R1 single 'ab' -> reject", results[0] == 0);
    }

    // Batch of 4
    {
        // "abb"(accept), "ab"(reject), "aabb"(accept), "b"(reject)
        int chars[] = {0,1,1, 0,1, 0,0,1,1, 1};
        int offsets[] = {0, 3, 5, 9, 10};
        int results[4];
        eng.dispatch_r1(chars, offsets, results, 4, 10, nullptr, nullptr);
        check("R1 batch[0] 'abb' -> accept", results[0] == 1);
        check("R1 batch[1] 'ab' -> reject", results[1] == 0);
        check("R1 batch[2] 'aabb' -> accept", results[2] == 1);
        check("R1 batch[3] 'b' -> reject", results[3] == 0);
    }

    // Empty string in batch
    {
        int chars[] = {0, 1, 1};
        int offsets[] = {0, 0, 3};  // empty, then "abb"
        int results[2];
        eng.dispatch_r1(chars, offsets, results, 2, 3, nullptr, nullptr);
        check("R1 batch empty -> reject", results[0] == 0);
        check("R1 batch 'abb' -> accept", results[1] == 1);
    }

    // Large batch cross-validation against sequential
    {
        int B = 10000;
        int L_each = 50;
        int total = B * L_each;
        int *chars = new int[total];
        int *offsets = new int[B + 1];
        int *results = new int[B];
        srand(42);
        for (int i = 0; i < total; i++) chars[i] = rand() % 2;
        for (int i = 0; i <= B; i++) offsets[i] = i * L_each;

        eng.dispatch_r1(chars, offsets, results, B, total, nullptr, nullptr);

        int mismatches = 0;
        for (int i = 0; i < B; i++) {
            bool expected = simulate_sequential_host(
                5, 0, dfa.accept, dfa.trans, chars + offsets[i], L_each);
            if (results[i] != (expected ? 1 : 0)) mismatches++;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "R1 batch B=%d L=%d cross-validate (%d mismatches)", B, L_each, mismatches);
        check(msg, mismatches == 0);
        delete[] chars; delete[] offsets; delete[] results;
    }

    // Variable-length batch cross-validation
    {
        int B = 5000;
        int lengths[] = {1, 5, 10, 50, 100, 200, 500, 1000};
        int n_lens = 8;
        srand(123);

        // Compute offsets
        int *offsets = new int[B + 1];
        offsets[0] = 0;
        for (int i = 0; i < B; i++) {
            int len = lengths[rand() % n_lens];
            offsets[i + 1] = offsets[i] + len;
        }
        int total = offsets[B];
        int *chars = new int[total];
        for (int i = 0; i < total; i++) chars[i] = rand() % 2;
        int *results = new int[B];

        eng.dispatch_r1(chars, offsets, results, B, total, nullptr, nullptr);

        int mismatches = 0;
        for (int i = 0; i < B; i++) {
            int len = offsets[i + 1] - offsets[i];
            bool expected = simulate_sequential_host(
                5, 0, dfa.accept, dfa.trans, chars + offsets[i], len);
            if (results[i] != (expected ? 1 : 0)) mismatches++;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "R1 variable-length B=%d cross-validate (%d mismatches)", B, mismatches);
        check(msg, mismatches == 0);
        delete[] chars; delete[] offsets; delete[] results;
    }

    // Early/late accept patterns
    {
        int B = 1000;
        int L = 100;
        int total = B * L;
        int *chars = new int[total];
        int *offsets = new int[B + 1];
        int *results = new int[B];
        srand(999);

        for (int i = 0; i < B; i++) {
            offsets[i] = i * L;
            for (int j = 0; j < L; j++) chars[i * L + j] = rand() % 2;
            if (i % 3 == 0) {
                // Late accept: "abb" at end
                chars[i * L + L - 3] = 0;
                chars[i * L + L - 2] = 1;
                chars[i * L + L - 1] = 1;
            } else if (i % 3 == 1) {
                // Early accept: "abb" at start (but may not accept at end)
                chars[i * L + 0] = 0;
                chars[i * L + 1] = 1;
                chars[i * L + 2] = 1;
            }
            // i % 3 == 2: fully random
        }
        offsets[B] = total;

        eng.dispatch_r1(chars, offsets, results, B, total, nullptr, nullptr);

        int mismatches = 0;
        for (int i = 0; i < B; i++) {
            bool expected = simulate_sequential_host(
                5, 0, dfa.accept, dfa.trans, chars + offsets[i], L);
            if (results[i] != (expected ? 1 : 0)) mismatches++;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "R1 early/late accept B=%d (%d mismatches)", B, mismatches);
        check(msg, mismatches == 0);
        delete[] chars; delete[] offsets; delete[] results;
    }

    eng.destroy();
}

static void test_r3_basic() {
    printf("\n--- R3: Decoupled look-back correctness ---\n");
    AbbDFA dfa;
    int max_chars = 1 << 24;
    ParallelEngine eng;
    eng.init(5, 2, 0, dfa.accept, dfa.trans, max_chars, 1);

    // Short strings (tiles=1)
    {
        int chars[] = {0, 1, 1};
        check("R3 'abb' -> accept", eng.dispatch_r3(chars, 3, nullptr, nullptr) == true);
    }
    {
        int chars[] = {0, 1};
        check("R3 'ab' -> reject", eng.dispatch_r3(chars, 2, nullptr, nullptr) == false);
    }
    {
        check("R3 empty -> reject", eng.dispatch_r3(nullptr, 0, nullptr, nullptr) == false);
    }

    // Cross-validate at various lengths including multi-tile
    srand(77777);
    int test_lengths[] = {1, 2, 3, 10, 32, 63, 64, 65, 100, 128, 200, 256,
                          500, 512, 1000, 1024, 2000, 4096, 8192, 16384,
                          65536, 100000, 262144, 1048576};
    for (int li = 0; li < 24; li++) {
        int L = test_lengths[li];
        int *chars = new int[L];
        for (int j = 0; j < L; j++) chars[j] = rand() % 2;
        bool seq = simulate_sequential_host(5, 0, dfa.accept, dfa.trans, chars, L);
        bool gpu = eng.dispatch_r3(chars, L, nullptr, nullptr);
        char msg[128];
        snprintf(msg, sizeof(msg), "R3 random L=%d seq=%d gpu=%d", L, seq, gpu);
        check(msg, seq == gpu);
        delete[] chars;
    }

    // Known accepting: "abb" suffix
    for (int L : {3, 64, 65, 128, 200, 1000, 10000, 100000}) {
        int *chars = new int[L];
        srand(L);
        for (int j = 0; j < L; j++) chars[j] = rand() % 2;
        chars[L - 3] = 0; chars[L - 2] = 1; chars[L - 1] = 1;
        bool seq = simulate_sequential_host(5, 0, dfa.accept, dfa.trans, chars, L);
        bool gpu = eng.dispatch_r3(chars, L, nullptr, nullptr);
        char msg[128];
        snprintf(msg, sizeof(msg), "R3 abb-suffix L=%d", L);
        check(msg, seq == gpu && gpu == true);
        delete[] chars;
    }

    eng.destroy();
}

static void test_adaptive() {
    printf("\n--- Adaptive dispatch ---\n");
    AbbDFA dfa;
    ParallelEngine eng;
    eng.init(5, 2, 0, dfa.accept, dfa.trans, 1 << 22, 1 << 16);

    // Batch of short strings -> R1
    {
        int B = 1000;
        int L = 20;
        int total = B * L;
        int *chars = new int[total];
        int *offsets = new int[B + 1];
        int *results = new int[B];
        srand(55);
        for (int i = 0; i < total; i++) chars[i] = rand() % 2;
        for (int i = 0; i <= B; i++) offsets[i] = i * L;

        eng.dispatch_adaptive(chars, offsets, results, B, total, nullptr, nullptr);

        int mismatches = 0;
        for (int i = 0; i < B; i++) {
            bool expected = simulate_sequential_host(
                5, 0, dfa.accept, dfa.trans, chars + offsets[i], L);
            if (results[i] != (expected ? 1 : 0)) mismatches++;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "Adaptive short-batch B=%d L=%d (%d mismatches)", B, L, mismatches);
        check(msg, mismatches == 0);
        delete[] chars; delete[] offsets; delete[] results;
    }

    // Single long string -> R3
    {
        int L = 100000;
        int *chars = new int[L];
        srand(66);
        for (int j = 0; j < L; j++) chars[j] = rand() % 2;
        int offsets[] = {0, L};
        int results[1];
        eng.dispatch_adaptive(chars, offsets, results, 1, L, nullptr, nullptr);
        bool expected = simulate_sequential_host(5, 0, dfa.accept, dfa.trans, chars, L);
        check("Adaptive single long L=100K", results[0] == (expected ? 1 : 0));
        delete[] chars;
    }

    eng.destroy();
}

static void bench_r1() {
    printf("\n=== R1 Benchmark: Warp-per-string ===\n");
    AbbDFA dfa;
    ParallelEngine eng;
    eng.init(5, 2, 0, dfa.accept, dfa.trans, 1 << 24, 1 << 20);

    printf("  %10s  %8s  %12s  %12s  %12s  %12s\n",
           "B", "L", "kern(ms)", "total(ms)", "kern GB/s", "total GB/s");

    int batch_sizes[] = {1000, 10000, 100000, 1000000};
    int str_lengths[] = {10, 50, 100, 500, 1000};

    for (int bi = 0; bi < 4; bi++) {
        for (int li = 0; li < 5; li++) {
            int B = batch_sizes[bi];
            int L = str_lengths[li];
            long long total_chars = (long long)B * L;
            if (total_chars > (1 << 24)) continue;

            int *chars = new int[total_chars];
            int *offsets = new int[B + 1];
            int *results = new int[B];
            srand(42);
            for (long long i = 0; i < total_chars; i++) chars[i] = rand() % 2;
            for (int i = 0; i <= B; i++) offsets[i] = i * L;

            // Warmup
            for (int w = 0; w < 3; w++)
                eng.dispatch_r1(chars, offsets, results, B, (int)total_chars, nullptr, nullptr);

            int iters = 20;
            float kern_ms_total = 0, total_ms_total = 0;
            for (int it = 0; it < iters; it++) {
                float km, tm;
                eng.dispatch_r1(chars, offsets, results, B, (int)total_chars, &km, &tm);
                kern_ms_total += km;
                total_ms_total += tm;
            }
            float kern_ms = kern_ms_total / iters;
            float tot_ms = total_ms_total / iters;
            double kern_gbs = (double)total_chars / (kern_ms * 1e6);
            double tot_gbs = (double)total_chars / (tot_ms * 1e6);

            printf("  %10d  %8d  %12.4f  %12.4f  %12.3f  %12.3f\n",
                   B, L, kern_ms, tot_ms, kern_gbs, tot_gbs);

            delete[] chars; delete[] offsets; delete[] results;
        }
    }
    eng.destroy();
}

static void bench_r3() {
    printf("\n=== R3 Benchmark: Decoupled look-back ===\n");
    AbbDFA dfa;
    int max_L = 1 << 24;
    ParallelEngine eng;
    eng.init(5, 2, 0, dfa.accept, dfa.trans, max_L, 1);

    printf("  %12s  %12s  %12s  %12s  %12s\n",
           "Length", "kern(ms)", "total(ms)", "kern GB/s", "total GB/s");

    for (int log_len = 6; log_len <= 24; log_len += 2) {
        int L = 1 << log_len;
        int *chars = new int[L];
        srand(42);
        for (int j = 0; j < L; j++) chars[j] = rand() % 2;

        // Warmup
        for (int w = 0; w < 3; w++)
            eng.dispatch_r3(chars, L, nullptr, nullptr);

        int iters = L > (1 << 20) ? 10 : 50;
        float kern_ms_total = 0, total_ms_total = 0;
        for (int it = 0; it < iters; it++) {
            float km, tm;
            eng.dispatch_r3(chars, L, &km, &tm);
            kern_ms_total += km;
            total_ms_total += tm;
        }
        float kern_ms = kern_ms_total / iters;
        float tot_ms = total_ms_total / iters;
        double kern_gbs = (double)L / (kern_ms * 1e6);
        double tot_gbs = (double)L / (tot_ms * 1e6);

        printf("  %12d  %12.4f  %12.4f  %12.3f  %12.3f\n",
               L, kern_ms, tot_ms, kern_gbs, tot_gbs);

        delete[] chars;
    }
    eng.destroy();
}

static void bench_variable_length() {
    printf("\n=== Variable-Length Batch Benchmark ===\n");
    AbbDFA dfa;
    ParallelEngine eng;
    eng.init(5, 2, 0, dfa.accept, dfa.trans, 1 << 24, 1 << 20);

    printf("  %10s  %10s  %10s  %12s  %12s  %12s  %12s\n",
           "B", "L_min", "L_max", "kern(ms)", "total(ms)", "kern GB/s", "total GB/s");

    // Uniform distribution of lengths
    for (int B : {10000, 100000}) {
        for (auto [Lmin, Lmax] : std::initializer_list<std::pair<int,int>>{
                {1, 100}, {10, 1000}, {50, 500}, {1, 1000}}) {
            srand(42);
            int *offsets = new int[B + 1];
            offsets[0] = 0;
            for (int i = 0; i < B; i++) {
                int len = Lmin + rand() % (Lmax - Lmin + 1);
                offsets[i + 1] = offsets[i] + len;
            }
            int total = offsets[B];
            if (total > (1 << 24)) { delete[] offsets; continue; }

            int *chars = new int[total];
            for (int i = 0; i < total; i++) chars[i] = rand() % 2;
            int *results = new int[B];

            // Warmup
            for (int w = 0; w < 3; w++)
                eng.dispatch_r1(chars, offsets, results, B, total, nullptr, nullptr);

            int iters = 10;
            float kern_ms_total = 0, total_ms_total = 0;
            for (int it = 0; it < iters; it++) {
                float km, tm;
                eng.dispatch_r1(chars, offsets, results, B, total, &km, &tm);
                kern_ms_total += km;
                total_ms_total += tm;
            }
            float kern_ms = kern_ms_total / iters;
            float tot_ms = total_ms_total / iters;
            double kern_gbs = (double)total / (kern_ms * 1e6);
            double tot_gbs = (double)total / (tot_ms * 1e6);

            printf("  %10d  %10d  %10d  %12.4f  %12.4f  %12.3f  %12.3f\n",
                   B, Lmin, Lmax, kern_ms, tot_ms, kern_gbs, tot_gbs);

            delete[] chars; delete[] offsets; delete[] results;
        }
    }
    eng.destroy();
}

int main() {
    printf("=== TERX Parallel DFA Engine v4 ===\n");
    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    test_r1_basic();
    test_r3_basic();
    test_adaptive();
    test_multi_dfa();

    printf("\n=== Results: %d / %d passed ===\n", g_pass, g_tests);
    if (g_pass != g_tests) { printf("SOME TESTS FAILED\n"); return 1; }

    bench_r1();
    bench_r3();
    bench_variable_length();

    return 0;
}

#endif  // BUILD_LIB
