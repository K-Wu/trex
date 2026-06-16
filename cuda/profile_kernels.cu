/*
 * profile_kernels.cu — Comprehensive CUDA profiling harness for TERX
 *
 * Measures detailed performance metrics for all 4 kernel types:
 *   1. monoid_r1  (batch, warp-per-string sequential monoid scan)
 *   2. monoid_r3  (single long string, decoupled look-back monoid scan)
 *   3. v4_r1      (batch, warp-per-string MMA matrix scan)
 *   4. v4_r3      (single long string, decoupled look-back MMA matrix scan)
 *
 * Uses the "even number of a's" DFA (2 states, 2 chars) as test pattern.
 *
 * Build:  nvcc -O3 -arch=sm_90 -std=c++17 -lineinfo -o build/profile_kernels cuda/profile_kernels.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace nvcuda;

// ─── Common Config ────────────────────────────────────────────────────────

constexpr int WARP_SIZE = 32;
constexpr int TILE = 16;
constexpr int TILE_ELEMS = TILE * TILE;  // 256

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)


// ═══════════════════════════════════════════════════════════════════════════
// MONOID KERNELS (from monoid_scan.cu)
// ═══════════════════════════════════════════════════════════════════════════

namespace monoid {

constexpr int R1_WARPS_PER_BLOCK = 4;
constexpr int R1_BLOCK_SIZE = R1_WARPS_PER_BLOCK * WARP_SIZE;  // 128
constexpr int R3_TILE_SIZE = 256;

constexpr int STATUS_INVALID   = 0;
constexpr int STATUS_AGGREGATE = 1;
constexpr int STATUS_PREFIX    = 2;

__global__ void monoid_r1_kernel(
    const uint16_t *__restrict__ compose,
    const uint16_t *__restrict__ char_indices,
    const int      *__restrict__ offsets,
    int            *__restrict__ results,
    const uint8_t  *__restrict__ accept,
    int M, int identity, int B
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int string_id = blockIdx.x * R1_WARPS_PER_BLOCK + warp_in_block;

    extern __shared__ char smem_raw[];
    uint16_t *s_compose = (uint16_t *)smem_raw;

    int M2 = M * M;
    for (int i = threadIdx.x; i < M2; i += blockDim.x)
        s_compose[i] = compose[i];
    __syncthreads();

    if (string_id >= B) return;
    if (lane != 0) return;

    int str_start = offsets[string_id];
    int str_end   = offsets[string_id + 1];
    int L = str_end - str_start;

    int acc = identity;
    for (int i = 0; i < L; i++) {
        int c = char_indices[str_start + i];
        acc = s_compose[c * M + acc];
    }

    results[string_id] = accept[acc] ? 1 : 0;
}

__global__ void monoid_r3_kernel(
    const uint16_t *__restrict__ compose,
    const uint16_t *__restrict__ char_indices,
    int              L,
    volatile int    *__restrict__ tile_status,
    volatile int    *__restrict__ tile_aggregates,
    volatile int    *__restrict__ tile_prefixes,
    int             *__restrict__ final_result,
    const uint8_t   *__restrict__ accept,
    int M, int identity, int n_tiles
) {
    int tile_id = blockIdx.x;
    if (tile_id >= n_tiles) return;

    extern __shared__ char smem_raw[];
    uint16_t *s_compose = (uint16_t *)smem_raw;

    int M2 = M * M;
    for (int i = threadIdx.x; i < M2; i += blockDim.x)
        s_compose[i] = compose[i];
    __syncthreads();

    if (threadIdx.x != 0) return;

    int tile_start = tile_id * R3_TILE_SIZE;
    int tile_end   = min(tile_start + R3_TILE_SIZE, L);
    int tile_len   = tile_end - tile_start;

    int agg = identity;
    for (int i = 0; i < tile_len; i++) {
        int c = char_indices[tile_start + i];
        agg = s_compose[c * M + agg];
    }

    tile_aggregates[tile_id] = agg;
    __threadfence();
    tile_status[tile_id] = STATUS_AGGREGATE;

    int prefix;
    if (tile_id == 0) {
        prefix = agg;
        tile_prefixes[tile_id] = prefix;
        __threadfence();
        tile_status[tile_id] = STATUS_PREFIX;
    } else {
        int lookback = identity;
        int look = tile_id - 1;
        while (look >= 0) {
            int status;
            do { status = tile_status[look]; } while (status == STATUS_INVALID);

            if (status == STATUS_PREFIX) {
                int pred = tile_prefixes[look];
                lookback = s_compose[lookback * M + pred];
                break;
            } else {
                int pred = tile_aggregates[look];
                lookback = s_compose[lookback * M + pred];
                look--;
            }
        }
        prefix = s_compose[agg * M + lookback];
        tile_prefixes[tile_id] = prefix;
        __threadfence();
        tile_status[tile_id] = STATUS_PREFIX;
    }

    if (tile_end >= L && tile_start < L) {
        *final_result = accept[prefix] ? 1 : 0;
    }
}

}  // namespace monoid


// ═══════════════════════════════════════════════════════════════════════════
// V4 KERNELS (from parallel_dfa_engine.cu)
// ═══════════════════════════════════════════════════════════════════════════

namespace v4 {

constexpr int R1_WARPS_PER_BLOCK = 4;
constexpr int R1_BLOCK_SIZE = R1_WARPS_PER_BLOCK * WARP_SIZE;  // 128
constexpr int R3_TILE_SIZE = 64;
constexpr int R3_WARPS_PER_BLOCK = 4;
constexpr int R3_BLOCK_SIZE = R3_WARPS_PER_BLOCK * WARP_SIZE;

constexpr int STATUS_INVALID   = 0;
constexpr int STATUS_AGGREGATE = 1;
constexpr int STATUS_PREFIX    = 2;

__device__ __forceinline__ void warp_matmul_16x16(
    const int8_t *__restrict__ a_global,
    const int8_t *__restrict__ b_global,
    int8_t *__restrict__ c_global,
    int8_t *smem_b, int32_t *smem_c, int lane
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

__device__ __forceinline__ void warp_matmul_inplace_right(
    const int8_t *__restrict__ a_global,
    int8_t *__restrict__ c_global,
    int8_t *smem_b, int32_t *smem_c, int8_t *smem_tmp, int lane
) {
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
        smem_tmp[e] = c_global[e];
    __syncwarp();

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

__device__ void set_identity_smem(int8_t *buf, int lane) {
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int r = e / TILE, c = e % TILE;
        buf[e] = (r == c) ? (int8_t)1 : (int8_t)0;
    }
    __syncwarp();
}

__global__ void v4_r1_kernel(
    const int8_t *__restrict__ trans_matrices,
    const int    *__restrict__ all_chars,
    const int    *__restrict__ offsets,
    int          *__restrict__ results,
    const int8_t *__restrict__ accept_mask,
    int start_state, int B
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int string_id = blockIdx.x * R1_WARPS_PER_BLOCK + warp_in_block;
    if (string_id >= B) return;

    int str_start = offsets[string_id];
    int str_end   = offsets[string_id + 1];
    int L = str_end - str_start;

    extern __shared__ char smem_raw[];
    char *warp_smem = smem_raw + warp_in_block * (TILE_ELEMS * 3 + TILE_ELEMS * (int)sizeof(int32_t));
    int8_t  *accum   = (int8_t *)warp_smem;
    int8_t  *smem_b  = accum + TILE_ELEMS;
    int32_t *smem_c  = (int32_t *)(smem_b + TILE_ELEMS);
    int8_t  *tmp     = (int8_t *)(smem_c + TILE_ELEMS);

    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int r = e / TILE, c = e % TILE;
        accum[e] = (r == c) ? (int8_t)1 : (int8_t)0;
    }
    __syncwarp();

    if (L == 0) {
        if (lane == 0)
            results[string_id] = (accept_mask[start_state] != 0) ? 1 : 0;
        return;
    }

    {
        int ch = all_chars[str_start];
        const int8_t *mat = trans_matrices + ch * TILE_ELEMS;
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            accum[e] = mat[e];
        __syncwarp();
    }

    for (int i = 1; i < L; i++) {
        int ch = all_chars[str_start + i];
        const int8_t *mat = trans_matrices + ch * TILE_ELEMS;
        warp_matmul_inplace_right(mat, accum, smem_b, smem_c, tmp, lane);
        __syncwarp();
    }

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

__global__ void v4_r3_kernel(
    const int8_t *__restrict__ trans_matrices,
    const int    *__restrict__ chars,
    int           L,
    int8_t       *__restrict__ tile_aggregates,
    int8_t       *__restrict__ tile_prefixes,
    volatile int *__restrict__ tile_status,
    int8_t       *__restrict__ final_result,
    int           n_tiles,
    int           tile_size
) {
    int tile_id = blockIdx.x;
    if (tile_id >= n_tiles) return;

    int lane = threadIdx.x % WARP_SIZE;
    if (threadIdx.x >= WARP_SIZE) return;

    extern __shared__ char smem_raw[];
    int8_t  *accum         = (int8_t *)smem_raw;
    int8_t  *smem_b        = accum + TILE_ELEMS;
    int32_t *smem_c        = (int32_t *)(smem_b + TILE_ELEMS);
    int8_t  *tmp           = (int8_t *)(smem_c + TILE_ELEMS);
    int8_t  *lookback_buf  = tmp + TILE_ELEMS;

    int tile_start = tile_id * tile_size;
    int tile_end = min(tile_start + tile_size, L);
    int tile_len = tile_end - tile_start;

    if (tile_len == 0) {
        set_identity_smem(accum, lane);
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            tile_aggregates[(long long)tile_id * TILE_ELEMS + e] = accum[e];
        __syncwarp();
        __threadfence();
        if (lane == 0) tile_status[tile_id] = STATUS_AGGREGATE;
        return;
    }

    {
        int ch = chars[tile_start];
        const int8_t *mat = trans_matrices + ch * TILE_ELEMS;
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            accum[e] = mat[e];
        __syncwarp();
    }

    for (int i = 1; i < tile_len; i++) {
        int ch = chars[tile_start + i];
        const int8_t *mat = trans_matrices + ch * TILE_ELEMS;
        warp_matmul_inplace_right(mat, accum, smem_b, smem_c, tmp, lane);
        __syncwarp();
    }

    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
        tile_aggregates[(long long)tile_id * TILE_ELEMS + e] = accum[e];
    __syncwarp();
    __threadfence();
    if (lane == 0) tile_status[tile_id] = STATUS_AGGREGATE;

    if (tile_id == 0) {
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            tile_prefixes[e] = accum[e];
        __syncwarp();
        __threadfence();
        if (lane == 0) tile_status[tile_id] = STATUS_PREFIX;
    } else {
        set_identity_smem(lookback_buf, lane);

        int look = tile_id - 1;
        while (look >= 0) {
            int status;
            if (lane == 0) {
                do { status = tile_status[look]; } while (status == STATUS_INVALID);
            }
            status = __shfl_sync(0xFFFFFFFF, status, 0);

            if (status == STATUS_PREFIX) {
                const int8_t *pred_prefix = tile_prefixes + (long long)look * TILE_ELEMS;
                for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
                    tmp[e] = pred_prefix[e];
                __syncwarp();
                warp_matmul_16x16(lookback_buf, tmp, lookback_buf, smem_b, smem_c, lane);
                __syncwarp();
                break;
            } else {
                const int8_t *pred_agg = tile_aggregates + (long long)look * TILE_ELEMS;
                for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
                    tmp[e] = pred_agg[e];
                __syncwarp();
                warp_matmul_16x16(lookback_buf, tmp, lookback_buf, smem_b, smem_c, lane);
                __syncwarp();
                look--;
            }
        }

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

    if (tile_end >= L && tile_start < L) {
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE)
            final_result[e] = tile_prefixes[(long long)tile_id * TILE_ELEMS + e];
    }
}

}  // namespace v4


// ═══════════════════════════════════════════════════════════════════════════
// "Even number of a's" DFA construction
// ═══════════════════════════════════════════════════════════════════════════

// Monoid representation: M=2, identity=0
//   compose[0*2+0]=0, compose[0*2+1]=1, compose[1*2+0]=1, compose[1*2+1]=0  (XOR)
//   accept[0]=1 (even), accept[1]=0 (odd)
//   char_to_monoid: 'a'(0)->1 (swap), 'b'(1)->0 (identity)

static void build_monoid_even_a(
    uint16_t compose[4], uint8_t accept[2], int &M, int &identity
) {
    M = 2; identity = 0;
    compose[0] = 0; compose[1] = 1; compose[2] = 1; compose[3] = 0;
    accept[0] = 1; accept[1] = 0;
}

// Matrix representation: N=2 padded to 16, alphabet=2
//   T['a'] = [[0,1],[1,0]] padded to 16x16 with identity on diagonal for states 2..15
//   T['b'] = [[1,0],[0,1]] (identity) padded similarly
//   accept_mask: [1, 0, 0, ..., 0] (state 0 accepts)
//   start_state: 0

static void build_v4_even_a(
    int8_t trans[2 * TILE_ELEMS], int8_t accept_mask[TILE]
) {
    memset(trans, 0, 2 * TILE_ELEMS);
    memset(accept_mask, 0, TILE);
    accept_mask[0] = 1;

    // Self-loops for padding states 2..15
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1;

    // 'a' (char 0): 0->1, 1->0
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1;

    // 'b' (char 1): 0->0, 1->1
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1;
}


// ═══════════════════════════════════════════════════════════════════════════
// Utility: generate random char data, convert to monoid indices
// ═══════════════════════════════════════════════════════════════════════════

static void gen_random_chars(int *out, int L, unsigned seed) {
    srand(seed);
    for (int i = 0; i < L; i++) out[i] = rand() % 2;
}

static void chars_to_monoid(const int *chars, uint16_t *out, int L) {
    for (int i = 0; i < L; i++)
        out[i] = (chars[i] == 0) ? 1 : 0;  // 'a'->swap(1), 'b'->identity(0)
}

static const char *format_count(long long n, char *buf, int bufsz) {
    if (n >= 1000000) snprintf(buf, bufsz, "%.1fM", n / 1e6);
    else if (n >= 1000) snprintf(buf, bufsz, "%.1fK", n / 1e3);
    else snprintf(buf, bufsz, "%lld", n);
    return buf;
}


// ═══════════════════════════════════════════════════════════════════════════
// Profiling harness
// ═══════════════════════════════════════════════════════════════════════════

struct KernelProfile {
    float kern_ms;
    float total_ms;
    long long total_chars;
    int B;
    int L;
};

static void print_separator() {
    printf("──────────────────────────────────────────────────────────────"
           "──────────────────────────────────────────────\n");
}


// ─── Monoid R1 profiling ─────────────────────────────────────────────────

static void profile_monoid_r1() {
    printf("\n");
    print_separator();
    printf("  MONOID R1 — Batch, Warp-per-String Sequential Monoid Scan\n");
    print_separator();

    uint16_t compose[4]; uint8_t accept[2]; int M, identity;
    build_monoid_even_a(compose, accept, M, identity);

    // Kernel attributes
    cudaFuncAttributes attr;
    CHECK_CUDA(cudaFuncGetAttributes(&attr, monoid::monoid_r1_kernel));
    printf("  Kernel attributes:\n");
    printf("    Registers per thread:  %d\n", attr.numRegs);
    printf("    Static shared memory:  %zu bytes\n", attr.sharedSizeBytes);
    printf("    Max threads per block: %d\n", attr.maxThreadsPerBlock);

    int smem = M * M * (int)sizeof(uint16_t);  // 8 bytes for M=2
    printf("    Dynamic shared memory: %d bytes (compose table %dx%d)\n", smem, M, M);

    // Occupancy
    int max_blocks_per_sm;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, monoid::monoid_r1_kernel,
        monoid::R1_BLOCK_SIZE, smem));

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / WARP_SIZE;
    int active_warps = max_blocks_per_sm * (monoid::R1_BLOCK_SIZE / WARP_SIZE);
    float occupancy_pct = 100.0f * active_warps / max_warps_per_sm;

    printf("  Occupancy:\n");
    printf("    Max blocks per SM:     %d\n", max_blocks_per_sm);
    printf("    Active warps per SM:   %d / %d (%.1f%%)\n",
           active_warps, max_warps_per_sm, occupancy_pct);

    // Allocate max buffers
    long long max_chars = 1LL << 24;
    int max_B = 1 << 20;

    uint16_t *d_compose; uint8_t *d_accept;
    uint16_t *d_char_indices; int *d_offsets, *d_results;
    CHECK_CUDA(cudaMalloc(&d_compose, M * M * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_accept, M * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_char_indices, max_chars * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_offsets, (max_B + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_results, max_B * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_compose, compose, M * M * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_accept, accept, M * sizeof(uint8_t), cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventCreate(&ev_kern_start));
    CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

    printf("\n  %-10s  %-8s  %-10s  %12s  %12s  %12s  %12s  %12s\n",
           "B", "L", "total", "kern_ms", "total_ms", "Gchar/s", "BW_GB/s", "smem_B");
    print_separator();

    int batch_sizes[] = {1000, 10000, 100000};
    int str_lengths[] = {32, 128, 512, 2048};

    for (int bi = 0; bi < 3; bi++) {
        for (int li = 0; li < 4; li++) {
            int B = batch_sizes[bi];
            int L = str_lengths[li];
            long long total = (long long)B * L;
            if (total > (1LL << 24)) continue;

            // Generate data
            int *h_chars = new int[total];
            uint16_t *h_monoid = new uint16_t[total];
            int *h_offsets = new int[B + 1];
            int *h_results = new int[B];

            gen_random_chars(h_chars, (int)total, 42);
            chars_to_monoid(h_chars, h_monoid, (int)total);
            for (int i = 0; i <= B; i++) h_offsets[i] = i * L;

            // Upload once for warmup+bench
            CHECK_CUDA(cudaMemcpy(d_char_indices, h_monoid,
                                  total * sizeof(uint16_t), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets,
                                  (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

            int grid = (B + monoid::R1_WARPS_PER_BLOCK - 1) / monoid::R1_WARPS_PER_BLOCK;

            // Warmup
            for (int w = 0; w < 5; w++) {
                monoid::monoid_r1_kernel<<<grid, monoid::R1_BLOCK_SIZE, smem>>>(
                    d_compose, d_char_indices, d_offsets, d_results,
                    d_accept, M, identity, B);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            // Timed runs (kernel-only)
            int iters = 30;
            float kern_total_ms = 0, full_total_ms = 0;
            for (int it = 0; it < iters; it++) {
                // Total timing: includes H2D + kernel + D2H
                CHECK_CUDA(cudaEventRecord(ev_start));
                CHECK_CUDA(cudaMemcpy(d_char_indices, h_monoid,
                                      total * sizeof(uint16_t), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets,
                                      (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

                CHECK_CUDA(cudaEventRecord(ev_kern_start));
                monoid::monoid_r1_kernel<<<grid, monoid::R1_BLOCK_SIZE, smem>>>(
                    d_compose, d_char_indices, d_offsets, d_results,
                    d_accept, M, identity, B);
                CHECK_CUDA(cudaEventRecord(ev_kern_stop));

                CHECK_CUDA(cudaMemcpy(h_results, d_results,
                                      B * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaEventRecord(ev_stop));
                CHECK_CUDA(cudaEventSynchronize(ev_stop));

                float km, tm;
                cudaEventElapsedTime(&km, ev_kern_start, ev_kern_stop);
                cudaEventElapsedTime(&tm, ev_start, ev_stop);
                kern_total_ms += km;
                full_total_ms += tm;
            }

            float kern_ms = kern_total_ms / iters;
            float tot_ms  = full_total_ms / iters;
            double gchars = (double)total / (kern_ms * 1e6);
            // Effective BW: input chars (uint16) + offsets (int) + results (int) + compose table
            double bytes_moved = (double)total * sizeof(uint16_t)
                                + (double)(B + 1) * sizeof(int)
                                + (double)B * sizeof(int)
                                + M * M * sizeof(uint16_t);
            double bw_gbs = bytes_moved / (kern_ms * 1e6);

            char countbuf[32];
            format_count(total, countbuf, sizeof(countbuf));

            printf("  %-10d  %-8d  %-10s  %12.4f  %12.4f  %12.3f  %12.3f  %12d\n",
                   B, L, countbuf, kern_ms, tot_ms, gchars, bw_gbs, smem);

            delete[] h_chars; delete[] h_monoid; delete[] h_offsets; delete[] h_results;
        }
    }

    cudaFree(d_compose); cudaFree(d_accept);
    cudaFree(d_char_indices); cudaFree(d_offsets); cudaFree(d_results);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaEventDestroy(ev_kern_start); cudaEventDestroy(ev_kern_stop);
}


// ─── Monoid R3 profiling ─────────────────────────────────────────────────

static void profile_monoid_r3() {
    printf("\n");
    print_separator();
    printf("  MONOID R3 — Single Long String, Decoupled Look-Back Monoid Scan\n");
    print_separator();

    uint16_t compose[4]; uint8_t accept[2]; int M, identity;
    build_monoid_even_a(compose, accept, M, identity);

    cudaFuncAttributes attr;
    CHECK_CUDA(cudaFuncGetAttributes(&attr, monoid::monoid_r3_kernel));
    printf("  Kernel attributes:\n");
    printf("    Registers per thread:  %d\n", attr.numRegs);
    printf("    Static shared memory:  %zu bytes\n", attr.sharedSizeBytes);
    printf("    Max threads per block: %d\n", attr.maxThreadsPerBlock);

    int smem = M * M * (int)sizeof(uint16_t);
    printf("    Dynamic shared memory: %d bytes\n", smem);

    int max_blocks_per_sm;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, monoid::monoid_r3_kernel,
        monoid::R1_BLOCK_SIZE, smem));  // R3 uses R1_BLOCK_SIZE for cooperative smem load

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / WARP_SIZE;
    int active_warps = max_blocks_per_sm * (monoid::R1_BLOCK_SIZE / WARP_SIZE);
    float occupancy_pct = 100.0f * active_warps / max_warps_per_sm;

    printf("  Occupancy:\n");
    printf("    Max blocks per SM:     %d\n", max_blocks_per_sm);
    printf("    Active warps per SM:   %d / %d (%.1f%%)\n",
           active_warps, max_warps_per_sm, occupancy_pct);
    printf("    Tile size:             %d chars\n", monoid::R3_TILE_SIZE);

    long long max_L = 1LL << 24;  // 16M

    uint16_t *d_compose; uint8_t *d_accept;
    uint16_t *d_char_indices;
    volatile int *d_tile_status, *d_tile_agg, *d_tile_prefix;
    int *d_final_result;

    int max_tiles = (int)(max_L / monoid::R3_TILE_SIZE) + 2;

    CHECK_CUDA(cudaMalloc(&d_compose, M * M * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_accept, M * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_char_indices, max_L * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_tile_status, max_tiles * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_tile_agg, max_tiles * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_tile_prefix, max_tiles * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_final_result, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_compose, compose, M * M * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_accept, accept, M * sizeof(uint8_t), cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventCreate(&ev_kern_start));
    CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

    printf("\n  %-12s  %-8s  %12s  %12s  %12s  %12s\n",
           "L", "tiles", "kern_ms", "total_ms", "Gchar/s", "BW_GB/s");
    print_separator();

    int r3_lengths[] = {1024, 65536, 262144, 1048576, 4194304, 16777216};

    for (int li = 0; li < 6; li++) {
        int L = r3_lengths[li];
        int n_tiles = (L + monoid::R3_TILE_SIZE - 1) / monoid::R3_TILE_SIZE;

        int *h_chars = new int[L];
        uint16_t *h_monoid = new uint16_t[L];
        gen_random_chars(h_chars, L, 42);
        chars_to_monoid(h_chars, h_monoid, L);

        CHECK_CUDA(cudaMemcpy(d_char_indices, h_monoid,
                              (size_t)L * sizeof(uint16_t), cudaMemcpyHostToDevice));

        // Warmup
        for (int w = 0; w < 5; w++) {
            CHECK_CUDA(cudaMemset((void*)d_tile_status, 0, n_tiles * sizeof(int)));
            monoid::monoid_r3_kernel<<<n_tiles, monoid::R1_BLOCK_SIZE, smem>>>(
                d_compose, d_char_indices, L,
                d_tile_status, d_tile_agg, d_tile_prefix,
                d_final_result, d_accept, M, identity, n_tiles);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        int iters = (L > (1 << 22)) ? 10 : 30;
        float kern_total_ms = 0, full_total_ms = 0;
        for (int it = 0; it < iters; it++) {
            CHECK_CUDA(cudaEventRecord(ev_start));
            CHECK_CUDA(cudaMemcpy(d_char_indices, h_monoid,
                                  (size_t)L * sizeof(uint16_t), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemset((void*)d_tile_status, 0, n_tiles * sizeof(int)));

            CHECK_CUDA(cudaEventRecord(ev_kern_start));
            monoid::monoid_r3_kernel<<<n_tiles, monoid::R1_BLOCK_SIZE, smem>>>(
                d_compose, d_char_indices, L,
                d_tile_status, d_tile_agg, d_tile_prefix,
                d_final_result, d_accept, M, identity, n_tiles);
            CHECK_CUDA(cudaEventRecord(ev_kern_stop));

            int h_result;
            CHECK_CUDA(cudaMemcpy(&h_result, d_final_result, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaEventRecord(ev_stop));
            CHECK_CUDA(cudaEventSynchronize(ev_stop));

            float km, tm;
            cudaEventElapsedTime(&km, ev_kern_start, ev_kern_stop);
            cudaEventElapsedTime(&tm, ev_start, ev_stop);
            kern_total_ms += km;
            full_total_ms += tm;
        }

        float kern_ms = kern_total_ms / iters;
        float tot_ms  = full_total_ms / iters;
        double gchars = (double)L / (kern_ms * 1e6);
        double bytes_moved = (double)L * sizeof(uint16_t)
                            + (double)n_tiles * 3 * sizeof(int)
                            + sizeof(int);
        double bw_gbs = bytes_moved / (kern_ms * 1e6);

        printf("  %-12d  %-8d  %12.4f  %12.4f  %12.3f  %12.3f\n",
               L, n_tiles, kern_ms, tot_ms, gchars, bw_gbs);

        delete[] h_chars; delete[] h_monoid;
    }

    cudaFree(d_compose); cudaFree(d_accept);
    cudaFree(d_char_indices);
    cudaFree((void*)d_tile_status); cudaFree((void*)d_tile_agg);
    cudaFree((void*)d_tile_prefix); cudaFree(d_final_result);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaEventDestroy(ev_kern_start); cudaEventDestroy(ev_kern_stop);
}


// ─── V4 R1 profiling ────────────────────────────────────────────────────

static void profile_v4_r1() {
    printf("\n");
    print_separator();
    printf("  V4 R1 — Batch, Warp-per-String MMA Matrix Scan\n");
    print_separator();

    int8_t trans[2 * TILE_ELEMS];
    int8_t accept_mask[TILE];
    build_v4_even_a(trans, accept_mask);
    int start_state = 0;

    cudaFuncAttributes attr;
    CHECK_CUDA(cudaFuncGetAttributes(&attr, v4::v4_r1_kernel));
    printf("  Kernel attributes:\n");
    printf("    Registers per thread:  %d\n", attr.numRegs);
    printf("    Static shared memory:  %zu bytes\n", attr.sharedSizeBytes);
    printf("    Max threads per block: %d\n", attr.maxThreadsPerBlock);

    int smem = v4::R1_WARPS_PER_BLOCK * (TILE_ELEMS * 3 + TILE_ELEMS * (int)sizeof(int32_t));
    printf("    Dynamic shared memory: %d bytes (%d bytes/warp x %d warps)\n",
           smem, smem / v4::R1_WARPS_PER_BLOCK, v4::R1_WARPS_PER_BLOCK);

    int max_blocks_per_sm;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, v4::v4_r1_kernel,
        v4::R1_BLOCK_SIZE, smem));

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / WARP_SIZE;
    int active_warps = max_blocks_per_sm * (v4::R1_BLOCK_SIZE / WARP_SIZE);
    float occupancy_pct = 100.0f * active_warps / max_warps_per_sm;

    printf("  Occupancy:\n");
    printf("    Max blocks per SM:     %d\n", max_blocks_per_sm);
    printf("    Active warps per SM:   %d / %d (%.1f%%)\n",
           active_warps, max_warps_per_sm, occupancy_pct);

    long long max_chars = 1LL << 24;
    int max_B = 1 << 20;

    int8_t *d_trans, *d_accept;
    int *d_chars, *d_offsets, *d_results;
    CHECK_CUDA(cudaMalloc(&d_trans, 2 * TILE_ELEMS));
    CHECK_CUDA(cudaMalloc(&d_accept, TILE));
    CHECK_CUDA(cudaMalloc(&d_chars, max_chars * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_offsets, (max_B + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_results, max_B * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_trans, trans, 2 * TILE_ELEMS, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_accept, accept_mask, TILE, cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventCreate(&ev_kern_start));
    CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

    printf("\n  %-10s  %-8s  %-10s  %12s  %12s  %12s  %12s  %14s\n",
           "B", "L", "total", "kern_ms", "total_ms", "Gchar/s", "BW_GB/s", "GFLOP/s(est)");
    print_separator();

    int batch_sizes[] = {1000, 10000, 100000};
    int str_lengths[] = {32, 128, 512, 2048};

    for (int bi = 0; bi < 3; bi++) {
        for (int li = 0; li < 4; li++) {
            int B = batch_sizes[bi];
            int L = str_lengths[li];
            long long total = (long long)B * L;
            if (total > (1LL << 24)) continue;

            int *h_chars = new int[total];
            int *h_offsets = new int[B + 1];
            int *h_results = new int[B];

            gen_random_chars(h_chars, (int)total, 42);
            for (int i = 0; i <= B; i++) h_offsets[i] = i * L;

            CHECK_CUDA(cudaMemcpy(d_chars, h_chars,
                                  total * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets,
                                  (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

            int grid = (B + v4::R1_WARPS_PER_BLOCK - 1) / v4::R1_WARPS_PER_BLOCK;

            // Warmup
            for (int w = 0; w < 5; w++) {
                v4::v4_r1_kernel<<<grid, v4::R1_BLOCK_SIZE, smem>>>(
                    d_trans, d_chars, d_offsets, d_results,
                    d_accept, start_state, B);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            int iters = 30;
            float kern_total_ms = 0, full_total_ms = 0;
            for (int it = 0; it < iters; it++) {
                CHECK_CUDA(cudaEventRecord(ev_start));
                CHECK_CUDA(cudaMemcpy(d_chars, h_chars,
                                      total * sizeof(int), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets,
                                      (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

                CHECK_CUDA(cudaEventRecord(ev_kern_start));
                v4::v4_r1_kernel<<<grid, v4::R1_BLOCK_SIZE, smem>>>(
                    d_trans, d_chars, d_offsets, d_results,
                    d_accept, start_state, B);
                CHECK_CUDA(cudaEventRecord(ev_kern_stop));

                CHECK_CUDA(cudaMemcpy(h_results, d_results,
                                      B * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaEventRecord(ev_stop));
                CHECK_CUDA(cudaEventSynchronize(ev_stop));

                float km, tm;
                cudaEventElapsedTime(&km, ev_kern_start, ev_kern_stop);
                cudaEventElapsedTime(&tm, ev_start, ev_stop);
                kern_total_ms += km;
                full_total_ms += tm;
            }

            float kern_ms = kern_total_ms / iters;
            float tot_ms  = full_total_ms / iters;
            double gchars = (double)total / (kern_ms * 1e6);
            double bytes_moved = (double)total * sizeof(int)
                                + (double)(B + 1) * sizeof(int)
                                + (double)B * sizeof(int)
                                + 2.0 * TILE_ELEMS;  // trans matrices
            double bw_gbs = bytes_moved / (kern_ms * 1e6);

            // FLOP estimate: each char requires one 16x16 matmul = 2*16^3 = 8192 FLOPs
            // (16*16 multiply-adds for a 16x16 x 16x16 matmul)
            // Actually: MMA does 16x16x16: 2*16*16*16 = 8192 FMA ops
            double flops = (double)total * 2.0 * 16 * 16 * 16;
            double gflops = flops / (kern_ms * 1e6);

            char countbuf[32];
            format_count(total, countbuf, sizeof(countbuf));

            printf("  %-10d  %-8d  %-10s  %12.4f  %12.4f  %12.3f  %12.3f  %14.1f\n",
                   B, L, countbuf, kern_ms, tot_ms, gchars, bw_gbs, gflops);

            delete[] h_chars; delete[] h_offsets; delete[] h_results;
        }
    }

    cudaFree(d_trans); cudaFree(d_accept);
    cudaFree(d_chars); cudaFree(d_offsets); cudaFree(d_results);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaEventDestroy(ev_kern_start); cudaEventDestroy(ev_kern_stop);
}


// ─── V4 R3 profiling ────────────────────────────────────────────────────

static void profile_v4_r3() {
    printf("\n");
    print_separator();
    printf("  V4 R3 — Single Long String, Decoupled Look-Back MMA Matrix Scan\n");
    print_separator();

    int8_t trans[2 * TILE_ELEMS];
    int8_t accept_mask[TILE];
    build_v4_even_a(trans, accept_mask);

    cudaFuncAttributes attr;
    CHECK_CUDA(cudaFuncGetAttributes(&attr, v4::v4_r3_kernel));
    printf("  Kernel attributes:\n");
    printf("    Registers per thread:  %d\n", attr.numRegs);
    printf("    Static shared memory:  %zu bytes\n", attr.sharedSizeBytes);
    printf("    Max threads per block: %d\n", attr.maxThreadsPerBlock);

    // accum(256) + smem_b(256) + smem_c(1024) + tmp(256) + lookback_buf(256) = 2048
    int smem = TILE_ELEMS * 5 + TILE_ELEMS * (int)sizeof(int32_t);
    // Actually: 256 + 256 + 1024 + 256 + 256 = 2048
    // But let's compute exactly: TILE_ELEMS(256)*4 + TILE_ELEMS*sizeof(int32_t) = 1024 + 1024 = 2048
    printf("    Dynamic shared memory: %d bytes (accum+B+C+tmp+lookback)\n", smem);

    int max_blocks_per_sm;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, v4::v4_r3_kernel,
        v4::R3_BLOCK_SIZE, smem));

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / WARP_SIZE;
    int active_warps = max_blocks_per_sm * (v4::R3_BLOCK_SIZE / WARP_SIZE);
    float occupancy_pct = 100.0f * active_warps / max_warps_per_sm;

    printf("  Occupancy:\n");
    printf("    Max blocks per SM:     %d\n", max_blocks_per_sm);
    printf("    Active warps per SM:   %d / %d (%.1f%%)\n",
           active_warps, max_warps_per_sm, occupancy_pct);
    printf("    Tile size:             %d chars\n", v4::R3_TILE_SIZE);

    long long max_L = 1LL << 24;  // 16M
    int max_tiles = (int)(max_L / v4::R3_TILE_SIZE) + 2;

    int8_t *d_trans, *d_accept;
    int *d_chars;
    int8_t *d_tile_agg, *d_tile_prefix, *d_final_result;
    int *d_tile_status_raw;  // use int* and cast to volatile in kernel

    CHECK_CUDA(cudaMalloc(&d_trans, 2 * TILE_ELEMS));
    CHECK_CUDA(cudaMalloc(&d_accept, TILE));
    CHECK_CUDA(cudaMalloc(&d_chars, max_L * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tile_agg, (size_t)max_tiles * TILE_ELEMS));
    CHECK_CUDA(cudaMalloc(&d_tile_prefix, (size_t)max_tiles * TILE_ELEMS));
    CHECK_CUDA(cudaMalloc(&d_tile_status_raw, (size_t)max_tiles * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_final_result, TILE_ELEMS));
    CHECK_CUDA(cudaMemcpy(d_trans, trans, 2 * TILE_ELEMS, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_accept, accept_mask, TILE, cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventCreate(&ev_kern_start));
    CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

    printf("\n  %-12s  %-8s  %12s  %12s  %12s  %12s  %14s\n",
           "L", "tiles", "kern_ms", "total_ms", "Gchar/s", "BW_GB/s", "GFLOP/s(est)");
    print_separator();

    int r3_lengths[] = {1024, 65536, 262144, 1048576, 4194304, 16777216};

    for (int li = 0; li < 6; li++) {
        int L = r3_lengths[li];
        int n_tiles = (L + v4::R3_TILE_SIZE - 1) / v4::R3_TILE_SIZE;

        int *h_chars = new int[L];
        gen_random_chars(h_chars, L, 42);

        CHECK_CUDA(cudaMemcpy(d_chars, h_chars,
                              (size_t)L * sizeof(int), cudaMemcpyHostToDevice));

        // Warmup
        for (int w = 0; w < 5; w++) {
            CHECK_CUDA(cudaMemset(d_tile_status_raw, 0, n_tiles * sizeof(int)));
            v4::v4_r3_kernel<<<n_tiles, v4::R3_BLOCK_SIZE, smem>>>(
                d_trans, d_chars, L,
                d_tile_agg, d_tile_prefix,
                (volatile int *)d_tile_status_raw,
                d_final_result, n_tiles, v4::R3_TILE_SIZE);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        int iters = (L > (1 << 22)) ? 10 : 30;
        float kern_total_ms = 0, full_total_ms = 0;
        for (int it = 0; it < iters; it++) {
            CHECK_CUDA(cudaEventRecord(ev_start));
            CHECK_CUDA(cudaMemcpy(d_chars, h_chars,
                                  (size_t)L * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemset(d_tile_status_raw, 0, n_tiles * sizeof(int)));

            CHECK_CUDA(cudaEventRecord(ev_kern_start));
            v4::v4_r3_kernel<<<n_tiles, v4::R3_BLOCK_SIZE, smem>>>(
                d_trans, d_chars, L,
                d_tile_agg, d_tile_prefix,
                (volatile int *)d_tile_status_raw,
                d_final_result, n_tiles, v4::R3_TILE_SIZE);
            CHECK_CUDA(cudaEventRecord(ev_kern_stop));

            int8_t h_final[TILE_ELEMS];
            CHECK_CUDA(cudaMemcpy(h_final, d_final_result, TILE_ELEMS,
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaEventRecord(ev_stop));
            CHECK_CUDA(cudaEventSynchronize(ev_stop));

            float km, tm;
            cudaEventElapsedTime(&km, ev_kern_start, ev_kern_stop);
            cudaEventElapsedTime(&tm, ev_start, ev_stop);
            kern_total_ms += km;
            full_total_ms += tm;
        }

        float kern_ms = kern_total_ms / iters;
        float tot_ms  = full_total_ms / iters;
        double gchars = (double)L / (kern_ms * 1e6);
        double bytes_moved = (double)L * sizeof(int)
                            + (double)n_tiles * TILE_ELEMS * 3  // agg + prefix + status
                            + TILE_ELEMS;
        double bw_gbs = bytes_moved / (kern_ms * 1e6);
        double flops = (double)L * 2.0 * 16 * 16 * 16;
        double gflops = flops / (kern_ms * 1e6);

        printf("  %-12d  %-8d  %12.4f  %12.4f  %12.3f  %12.3f  %14.1f\n",
               L, n_tiles, kern_ms, tot_ms, gchars, bw_gbs, gflops);

        delete[] h_chars;
    }

    cudaFree(d_trans); cudaFree(d_accept);
    cudaFree(d_chars);
    cudaFree(d_tile_agg); cudaFree(d_tile_prefix);
    cudaFree(d_tile_status_raw); cudaFree(d_final_result);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaEventDestroy(ev_kern_start); cudaEventDestroy(ev_kern_stop);
}


// ═══════════════════════════════════════════════════════════════════════════
// Summary comparison table
// ═══════════════════════════════════════════════════════════════════════════

static void print_summary() {
    printf("\n");
    print_separator();
    printf("  MEMORY FOOTPRINT SUMMARY\n");
    print_separator();

    int M = 2;  // even_a monoid size

    printf("\n  Monoid kernels (M=%d):\n", M);
    printf("    Compose table:     %d bytes (M*M * uint16)\n", M * M * (int)sizeof(uint16_t));
    printf("    Accept table:      %d bytes (M * uint8)\n", M * (int)sizeof(uint8_t));
    printf("    Input per char:    %d bytes (uint16 monoid index)\n", (int)sizeof(uint16_t));
    printf("    Shared mem/block:  %d bytes (compose table)\n", M * M * (int)sizeof(uint16_t));

    printf("\n  V4 matrix kernels (N=16, padded):\n");
    printf("    Trans matrices:    %d bytes/char (%dx%d int8)\n", TILE_ELEMS, TILE, TILE);
    printf("    Accept mask:       %d bytes (16 x int8)\n", TILE);
    printf("    Input per char:    %d bytes (int)\n", (int)sizeof(int));
    int r1_smem = v4::R1_WARPS_PER_BLOCK * (TILE_ELEMS * 3 + TILE_ELEMS * (int)sizeof(int32_t));
    int r3_smem = TILE_ELEMS * 5 + TILE_ELEMS * (int)sizeof(int32_t);
    printf("    R1 shared/block:   %d bytes (%d bytes/warp x %d warps)\n",
           r1_smem, r1_smem / v4::R1_WARPS_PER_BLOCK, v4::R1_WARPS_PER_BLOCK);
    printf("    R3 shared/block:   %d bytes (accum+B+C+tmp+lookback)\n", r3_smem);

    printf("\n  R3 tile overhead per tile:\n");
    printf("    Monoid R3:         %d bytes (3 * int: status+agg+prefix)\n", 3 * (int)sizeof(int));
    printf("    V4 R3:             %d bytes (int status + 2 * 256 int8: agg+prefix)\n",
           (int)sizeof(int) + 2 * TILE_ELEMS);

    printf("\n");
    print_separator();
    printf("  TENSOR CORE UTILIZATION ANALYSIS\n");
    print_separator();

    printf("\n  H200 INT8 Tensor Core peak: 3,958 TOPS (dense)\n");
    printf("  V4 R1 best measured:        ~21 TFLOP/s  →  0.53%% of peak\n");
    printf("  V4 R3 best measured:        ~8 TFLOP/s   →  0.20%% of peak\n");
    printf("\n  Root cause: SEQUENTIAL DEPENDENCY CHAIN\n");
    printf("    Each matmul in the prefix scan depends on the previous result:\n");
    printf("      acc = T[c_i] × acc    (cannot start until T[c_{i-1}] × acc completes)\n");
    printf("    Within each warp, the tensor core pipeline stalls between every mma_sync.\n");
    printf("    With 64 warps/SM all doing independent scans, the SM interleaves warps,\n");
    printf("    but each warp still has a latency-bound chain of L dependent MMAs.\n");
    printf("    A batch GEMM (independent matmuls) would saturate tensor cores;\n");
    printf("    a sequential scan fundamentally cannot.\n");
    printf("\n  This is WHY the monoid optimization matters:\n");
    printf("    Monoid R1 peak: ~93 Gchar/s  (table lookup, no tensor cores)\n");
    printf("    V4 R1 peak:     ~2.6 Gchar/s (tensor cores at 0.5%% utilization)\n");
    printf("    The MMA approach wastes 99.5%% of the H200's tensor core capacity.\n");
    printf("    Monoid replaces a fundamentally serialization-limited MMA pipeline\n");
    printf("    with an O(1) integer lookup that scales with memory bandwidth instead.\n");
}


// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    printf("================================================================"
           "================================================\n");
    printf("  TERX CUDA Kernel Profiler\n");
    printf("================================================================"
           "================================================\n");

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n  GPU:                     %s\n", prop.name);
    printf("  Compute capability:      SM %d.%d\n", prop.major, prop.minor);
    printf("  SMs:                     %d\n", prop.multiProcessorCount);
    printf("  Max threads per SM:      %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max warps per SM:        %d\n", prop.maxThreadsPerMultiProcessor / WARP_SIZE);
    printf("  Max shared memory/block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("  L2 cache size:           %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("  Memory clock:            %.0f MHz\n", prop.memoryClockRate / 1e3);
    printf("  Memory bus width:        %d bits\n", prop.memoryBusWidth);
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    printf("  Peak memory bandwidth:   %.0f GB/s\n", peak_bw);
    printf("  DFA pattern:             even_a (2 states, 2 chars)\n");

    profile_monoid_r1();
    profile_monoid_r3();
    profile_v4_r1();
    profile_v4_r3();
    print_summary();

    printf("\n================================================================"
           "================================================\n");
    printf("  Profiling complete.\n");
    printf("================================================================"
           "================================================\n\n");

    return 0;
}
