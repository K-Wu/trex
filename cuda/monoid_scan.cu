/*
 * monoid_scan.cu — Monoid Scan CUDA Kernel
 *
 * Two execution regimes for DFA simulation via precomputed transition monoid:
 *   R1: Warp-per-string sequential scan (many short strings, L_max <= 4096)
 *   R3: Decoupled look-back for single long strings
 *   Adaptive dispatch selects regime based on (B, L_max)
 *
 * The monoid replaces O(N^3) matrix multiply with O(1) table lookups.
 * compose_table[i, j] = index of elements[i] @ elements[j]
 * i.e. element i is "newer" (applied after j in left-to-right reading).
 *
 * Kernel convention: compose[c_new * M + acc_old]
 * (newer character index as ROW, accumulated index as COLUMN)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ─── Configuration ─────────────────────────────────────────────────────────

constexpr int WARP_SIZE = 32;

// R1 config: 4 warps per block
constexpr int R1_WARPS_PER_BLOCK = 4;
constexpr int R1_BLOCK_SIZE = R1_WARPS_PER_BLOCK * WARP_SIZE;  // 128

// R3 config: tile size for decoupled look-back
constexpr int R3_TILE_SIZE = 256;   // positions per tile

// Decoupled look-back status flags
constexpr int STATUS_INVALID   = 0;
constexpr int STATUS_AGGREGATE = 1;
constexpr int STATUS_PREFIX    = 2;

// Threshold for adaptive dispatch
constexpr int ADAPTIVE_R1_THRESHOLD = 4096;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)


// ─── R1: Warp-per-string monoid scan ─────────────────────────────────────
//
// Each warp processes one string via sequential table lookups.
// Lane 0 does all the work; other lanes idle (needed for warp occupancy).
// Compose table (M*M uint16) is loaded into shared memory cooperatively.
//
// Shared memory layout:
//   compose[M * M * sizeof(uint16_t)]   — compose table (all warps share)
//
// Input: monoid indices per character (not raw chars), CSR offsets, accept table.
// Convention: compose[c_new * M + acc_old]

__global__ void monoid_r1_kernel(
    const uint16_t *__restrict__ compose,    // [M * M] compose table
    const uint16_t *__restrict__ char_indices, // [total_chars] monoid indices per character
    const int      *__restrict__ offsets,      // [B+1] CSR string offsets
    int            *__restrict__ results,       // [B] output: 1=accept, 0=reject
    const uint8_t  *__restrict__ accept,        // [M] accept table
    int M,           // monoid size
    int identity,    // identity element index
    int B            // number of strings
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int string_id = blockIdx.x * R1_WARPS_PER_BLOCK + warp_in_block;

    // Load compose table into shared memory cooperatively (all threads)
    extern __shared__ char smem_raw[];
    uint16_t *s_compose = (uint16_t *)smem_raw;

    int M2 = M * M;
    // All threads in block cooperate to load the compose table
    for (int i = threadIdx.x; i < M2; i += blockDim.x) {
        s_compose[i] = compose[i];
    }
    __syncthreads();

    if (string_id >= B) return;

    // Only lane 0 does the sequential scan
    if (lane != 0) return;

    int str_start = offsets[string_id];
    int str_end   = offsets[string_id + 1];
    int L = str_end - str_start;

    // Sequential scan: acc = identity, then for each char:
    //   acc = compose[c_new * M + acc]
    int acc = identity;
    for (int i = 0; i < L; i++) {
        int c = char_indices[str_start + i];
        acc = s_compose[c * M + acc];
    }

    results[string_id] = accept[acc] ? 1 : 0;
}


// ─── R3: Decoupled look-back for long strings ─────────────────────────────
//
// One block per tile (R3_TILE_SIZE=256 elements).
// Phase 1: sequential scan within tile -> tile aggregate (single monoid index)
// Phase 2: look-back protocol with STATUS_INVALID/AGGREGATE/PREFIX flags
// Phase 3: last tile outputs accept/reject via accept table
//
// Compose convention for tile scan (left-to-right within tile):
//   agg = compose[c_new * M + agg_old]   (newer on left)
//
// Look-back composition (later tiles on left, earlier predecessors on right):
//   lookback = compose[lookback * M + pred]
//
// Final prefix for a tile:
//   prefix = compose[agg * M + lookback]
//   (this tile's aggregate goes LEFT of everything before it)

__global__ void monoid_r3_kernel(
    const uint16_t *__restrict__ compose,       // [M * M]
    const uint16_t *__restrict__ char_indices,  // [L]
    int              L,
    volatile int    *__restrict__ tile_status,   // [n_tiles]
    volatile int    *__restrict__ tile_aggregates, // [n_tiles] single monoid index
    volatile int    *__restrict__ tile_prefixes,   // [n_tiles] inclusive prefix index
    int             *__restrict__ final_result,    // [1] accept/reject
    const uint8_t   *__restrict__ accept,          // [M]
    int M,
    int identity,
    int n_tiles
) {
    int tile_id = blockIdx.x;
    if (tile_id >= n_tiles) return;

    // Load compose table into shared memory cooperatively
    extern __shared__ char smem_raw[];
    uint16_t *s_compose = (uint16_t *)smem_raw;

    int M2 = M * M;
    for (int i = threadIdx.x; i < M2; i += blockDim.x) {
        s_compose[i] = compose[i];
    }
    __syncthreads();

    // Only thread 0 does the actual sequential scan work
    if (threadIdx.x != 0) return;

    // ── Phase 1: Compute tile aggregate via sequential scan ──
    int tile_start = tile_id * R3_TILE_SIZE;
    int tile_end   = min(tile_start + R3_TILE_SIZE, L);
    int tile_len   = tile_end - tile_start;

    int agg = identity;
    for (int i = 0; i < tile_len; i++) {
        int c = char_indices[tile_start + i];
        agg = s_compose[c * M + agg];
    }

    // Publish aggregate
    tile_aggregates[tile_id] = agg;
    __threadfence();
    tile_status[tile_id] = STATUS_AGGREGATE;

    // ── Phase 2: Decoupled look-back ──
    int prefix;
    if (tile_id == 0) {
        // First tile: prefix = aggregate
        prefix = agg;
        tile_prefixes[tile_id] = prefix;
        __threadfence();
        tile_status[tile_id] = STATUS_PREFIX;
    } else {
        // Look back to compute inclusive prefix.
        // lookback accumulates: compose[lookback * M + pred]
        // (later tiles on left, earlier predecessors on right)
        int lookback = identity;  // starts as identity (right side of our tile's agg)

        int look = tile_id - 1;
        while (look >= 0) {
            // Spin-wait for predecessor to publish
            int status;
            do {
                status = tile_status[look];
            } while (status == STATUS_INVALID);

            if (status == STATUS_PREFIX) {
                // Found an inclusive prefix: compose it on the right of lookback
                int pred = tile_prefixes[look];
                // lookback = compose[lookback * M + pred]
                lookback = s_compose[lookback * M + pred];
                break;  // done
            } else {
                // STATUS_AGGREGATE: accumulate this tile's aggregate
                int pred = tile_aggregates[look];
                // lookback = compose[lookback * M + pred]
                lookback = s_compose[lookback * M + pred];
                look--;
            }
        }

        // prefix[tile_id] = compose[agg * M + lookback]
        // this tile's aggregate goes LEFT of all predecessors
        prefix = s_compose[agg * M + lookback];
        tile_prefixes[tile_id] = prefix;
        __threadfence();
        tile_status[tile_id] = STATUS_PREFIX;
    }

    // ── Phase 3: Last tile outputs accept/reject ──
    if (tile_end >= L && tile_start < L) {
        *final_result = accept[prefix] ? 1 : 0;
    }
}


// ─── Host Engine ──────────────────────────────────────────────────────────

struct MonoidEngine {
    int  M;           // monoid size
    int  identity;    // identity element index

    // Host copies
    uint16_t *h_compose;   // [M * M]
    uint8_t  *h_accept;    // [M]

    // Device: persistent tables
    uint16_t *d_compose;   // [M * M]
    uint8_t  *d_accept;    // [M]

    // Device: I/O buffers
    uint16_t *d_char_indices; // [max_total_chars]
    int      *d_offsets;      // [max_B + 1]
    int      *d_results;      // [max_B]
    int       max_total_chars;
    int       max_B;

    // R3 buffers (for single long string)
    volatile int *d_tile_status;     // [max_tiles]
    volatile int *d_tile_aggregates; // [max_tiles]
    volatile int *d_tile_prefixes;   // [max_tiles]
    int          *d_final_result;    // [1]
    int           max_tiles;

    // Timing events
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;

    void init(int monoid_size, int identity_idx,
              const uint16_t *compose_table, const uint8_t *accept_table,
              int max_chars, int max_batch) {
        M         = monoid_size;
        identity  = identity_idx;
        max_total_chars = max_chars;
        max_B     = max_batch;

        // Host copies
        h_compose = new uint16_t[(size_t)M * M];
        h_accept  = new uint8_t[M];
        memcpy(h_compose, compose_table, (size_t)M * M * sizeof(uint16_t));
        memcpy(h_accept,  accept_table,  M * sizeof(uint8_t));

        // Device: tables
        CHECK_CUDA(cudaMalloc(&d_compose, (size_t)M * M * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc(&d_accept,  M * sizeof(uint8_t)));
        CHECK_CUDA(cudaMemcpy(d_compose, compose_table,
                              (size_t)M * M * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, accept_table,
                              M * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));

        // Device: I/O buffers
        CHECK_CUDA(cudaMalloc(&d_char_indices,
                              (size_t)max_chars * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc(&d_offsets,
                              (size_t)(max_batch + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_results,
                              (size_t)max_batch * sizeof(int)));

        // R3 buffers
        max_tiles = (max_chars + R3_TILE_SIZE - 1) / R3_TILE_SIZE + 1;
        CHECK_CUDA(cudaMalloc((void**)&d_tile_status,
                              (size_t)max_tiles * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&d_tile_aggregates,
                              (size_t)max_tiles * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&d_tile_prefixes,
                              (size_t)max_tiles * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_final_result, sizeof(int)));

        // Events
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));
    }

    void destroy() {
        delete[] h_compose;
        delete[] h_accept;
        cudaFree(d_compose);
        cudaFree(d_accept);
        cudaFree(d_char_indices);
        cudaFree(d_offsets);
        cudaFree(d_results);
        cudaFree((void*)d_tile_status);
        cudaFree((void*)d_tile_aggregates);
        cudaFree((void*)d_tile_prefixes);
        cudaFree(d_final_result);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start);
        cudaEventDestroy(ev_kern_stop);
    }

    // ── R1 dispatch: batch of short strings ──
    // char_indices[total_chars]: monoid index per character
    // offsets[B+1]: CSR string offsets
    // results[B]: output accept/reject
    void dispatch_r1(const uint16_t *h_char_indices,
                     const int *h_offsets,
                     int *h_results,
                     int B, int total_chars,
                     float *kernel_ms, float *total_ms) {
        CHECK_CUDA(cudaEventRecord(ev_start));

        // H2D
        CHECK_CUDA(cudaMemcpy(d_char_indices, h_char_indices,
                              (size_t)total_chars * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets,
                              (size_t)(B + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Shared memory: compose table
        int smem = (size_t)M * M * sizeof(uint16_t);

        int grid = (B + R1_WARPS_PER_BLOCK - 1) / R1_WARPS_PER_BLOCK;

        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        monoid_r1_kernel<<<grid, R1_BLOCK_SIZE, smem>>>(
            d_compose, d_char_indices, d_offsets, d_results,
            d_accept, M, identity, B);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        // D2H
        CHECK_CUDA(cudaMemcpy(h_results, d_results,
                              (size_t)B * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms,  ev_start,      ev_stop);
    }

    // ── R3 dispatch: single long string ──
    // Returns accept/reject (1/0).
    int dispatch_r3(const uint16_t *h_char_indices, int L,
                    float *kernel_ms, float *total_ms) {
        if (L == 0) return h_accept[identity] ? 1 : 0;

        int n_tiles = (L + R3_TILE_SIZE - 1) / R3_TILE_SIZE;

        CHECK_CUDA(cudaEventRecord(ev_start));

        // H2D
        CHECK_CUDA(cudaMemcpy(d_char_indices, h_char_indices,
                              (size_t)L * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));

        // Clear tile status
        CHECK_CUDA(cudaMemset((void*)d_tile_status, 0,
                              (size_t)n_tiles * sizeof(int)));

        // Shared memory: compose table
        int smem = (size_t)M * M * sizeof(uint16_t);

        // One block per tile, R1_BLOCK_SIZE threads for cooperative smem load
        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        monoid_r3_kernel<<<n_tiles, R1_BLOCK_SIZE, smem>>>(
            d_compose, d_char_indices, L,
            d_tile_status, d_tile_aggregates, d_tile_prefixes,
            d_final_result, d_accept,
            M, identity, n_tiles);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        // D2H: read final result
        int h_result;
        CHECK_CUDA(cudaMemcpy(&h_result, d_final_result, sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms,  ev_start,      ev_stop);

        return h_result;
    }

    // ── Adaptive dispatch: auto-selects R1 vs R3 ──
    void dispatch_adaptive(const uint16_t *h_char_indices,
                           const int *h_offsets,
                           int *h_results,
                           int B, int total_chars,
                           float *kernel_ms, float *total_ms) {
        // Find max length
        int L_max = 0;
        for (int i = 0; i < B; i++) {
            int len = h_offsets[i + 1] - h_offsets[i];
            if (len > L_max) L_max = len;
        }

        if (L_max <= ADAPTIVE_R1_THRESHOLD) {
            dispatch_r1(h_char_indices, h_offsets, h_results,
                        B, total_chars, kernel_ms, total_ms);
        } else if (B == 1) {
            // Single long string: R3
            int result = dispatch_r3(h_char_indices, L_max,
                                     kernel_ms, total_ms);
            h_results[0] = result;
        } else {
            // Mixed batch: use R3 per string
            CHECK_CUDA(cudaEventRecord(ev_start));
            CHECK_CUDA(cudaEventRecord(ev_kern_start));
            for (int i = 0; i < B; i++) {
                int off = h_offsets[i];
                int len = h_offsets[i + 1] - off;
                float km, tm;
                h_results[i] = dispatch_r3(h_char_indices + off, len,
                                           &km, &tm);
            }
            CHECK_CUDA(cudaEventRecord(ev_kern_stop));
            CHECK_CUDA(cudaEventRecord(ev_stop));
            CHECK_CUDA(cudaEventSynchronize(ev_stop));
            if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
            if (total_ms)  cudaEventElapsedTime(total_ms,  ev_start,      ev_stop);
        }
    }
};


// ─── Python/C Library API ─────────────────────────────────────────────────

#ifdef BUILD_LIB

extern "C" {

struct MonoidEngineHandle {
    MonoidEngine engine;
    bool initialized;
};

static MonoidEngineHandle g_engine = {.initialized = false};

int monoid_engine_init(int monoid_size, int identity_idx,
                       const uint16_t *compose_table,
                       const uint8_t  *accept_table,
                       int max_total_chars, int max_batch) {
    if (g_engine.initialized) g_engine.engine.destroy();
    g_engine.engine.init(monoid_size, identity_idx,
                         compose_table, accept_table,
                         max_total_chars, max_batch);
    g_engine.initialized = true;
    return 0;
}

void monoid_engine_destroy() {
    if (g_engine.initialized) {
        g_engine.engine.destroy();
        g_engine.initialized = false;
    }
}

// Dispatch batch of variable-length strings.
// chars[total_chars]: monoid indices (uint16_t)
// offsets[B+1]: CSR string offsets
// results[B]: output accept/reject
// Returns 0 on success, -1 if not initialized.
int monoid_engine_dispatch_batch(
    const uint16_t *chars,
    const int      *offsets,
    int            *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms
) {
    if (!g_engine.initialized) return -1;
    g_engine.engine.dispatch_adaptive(chars, offsets, results,
                                      B, total_chars,
                                      kernel_ms, total_ms);
    return 0;
}

int monoid_engine_device_check() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // Require at least SM 7.x
    if (prop.major < 7) return -2;
    return 0;
}

}  // extern "C"

#endif  // BUILD_LIB


// ─── Built-in Tests and Benchmarks ────────────────────────────────────────

#ifndef BUILD_LIB

// ── "Even number of a's" DFA with 2-element monoid ────────────────────────
//
// States:  0 = even (accept), 1 = odd (reject)
// Alphabet: 'a' (char 0), 'b' (char 1)
// Transitions: a: 0->1, 1->0;  b: 0->0, 1->1
//
// Monoid elements:
//   identity (idx 0): state unchanged
//   swap     (idx 1): 0<->1
//
// char_to_monoid: 'a' -> 1 (swap), 'b' -> 0 (identity)
//
// compose_table (newer on left):
//   compose[0,0]=0  compose[0,1]=1
//   compose[1,0]=1  compose[1,1]=0
//
// accept_table: accept[0]=1 (identity is even/start state), accept[1]=0

static void build_even_a_monoid(
    int &M, int &identity,
    uint16_t compose_table[4],  // [2*2]
    uint8_t  accept_table[2]
) {
    M        = 2;
    identity = 0;

    // compose[i*2 + j] = i XOR j
    compose_table[0*2 + 0] = 0;  // identity @ identity = identity
    compose_table[0*2 + 1] = 1;  // identity @ swap     = swap
    compose_table[1*2 + 0] = 1;  // swap     @ identity = swap
    compose_table[1*2 + 1] = 0;  // swap     @ swap     = identity

    accept_table[0] = 1;  // identity -> even # of a's -> accept
    accept_table[1] = 0;  // swap     -> odd  # of a's -> reject
}

// Reference: simulate "even a's" sequentially on host
// Input: raw chars (0='a', 1='b')
static bool even_a_sequential(const uint8_t *chars, int L) {
    int count_a = 0;
    for (int i = 0; i < L; i++) {
        if (chars[i] == 0) count_a++;
    }
    return (count_a % 2) == 0;
}

// Convert raw chars (0='a', 1='b') to monoid indices
// 'a'->1 (swap), 'b'->0 (identity)
static void chars_to_monoid_indices(const uint8_t *chars, uint16_t *out, int L) {
    for (int i = 0; i < L; i++) {
        out[i] = (chars[i] == 0) ? 1 : 0;  // 'a'->swap(1), 'b'->identity(0)
    }
}

static int g_tests = 0, g_pass = 0;

static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) {
        g_pass++;
        printf("  PASS: %s\n", name);
    } else {
        printf("  FAIL: %s\n", name);
    }
}

// ─── test_monoid_r1 ────────────────────────────────────────────────────────
static void test_monoid_r1() {
    printf("\n--- test_monoid_r1: Warp-per-string monoid scan ---\n");

    int M, identity;
    uint16_t compose[4];
    uint8_t  accept[2];
    build_even_a_monoid(M, identity, compose, accept);

    // Max params
    int max_chars = 1 << 23;
    int max_batch = 1 << 16;
    MonoidEngine eng;
    eng.init(M, identity, compose, accept, max_chars, max_batch);

    // Single string: empty -> even -> accept
    {
        uint16_t chars[1] = {0};  // unused
        int offsets[2] = {0, 0};
        int results[1];
        eng.dispatch_r1(chars, offsets, results, 1, 0, nullptr, nullptr);
        check("R1 empty string -> accept (0 a's)", results[0] == 1);
    }

    // Single string: "a" -> 1 a's -> reject
    {
        uint8_t raw[] = {0};
        uint16_t chars[1];
        chars_to_monoid_indices(raw, chars, 1);
        int offsets[2] = {0, 1};
        int results[1];
        eng.dispatch_r1(chars, offsets, results, 1, 1, nullptr, nullptr);
        check("R1 'a' -> reject (1 a)", results[0] == 0);
    }

    // Single string: "aa" -> 2 a's -> accept
    {
        uint8_t raw[] = {0, 0};
        uint16_t chars[2];
        chars_to_monoid_indices(raw, chars, 2);
        int offsets[2] = {0, 2};
        int results[1];
        eng.dispatch_r1(chars, offsets, results, 1, 2, nullptr, nullptr);
        check("R1 'aa' -> accept (2 a's)", results[0] == 1);
    }

    // Single string: "ab" -> 1 a -> reject
    {
        uint8_t raw[] = {0, 1};
        uint16_t chars[2];
        chars_to_monoid_indices(raw, chars, 2);
        int offsets[2] = {0, 2};
        int results[1];
        eng.dispatch_r1(chars, offsets, results, 1, 2, nullptr, nullptr);
        check("R1 'ab' -> reject (1 a)", results[0] == 0);
    }

    // Single string: "aab" -> 2 a's -> accept
    {
        uint8_t raw[] = {0, 0, 1};
        uint16_t chars[3];
        chars_to_monoid_indices(raw, chars, 3);
        int offsets[2] = {0, 3};
        int results[1];
        eng.dispatch_r1(chars, offsets, results, 1, 3, nullptr, nullptr);
        check("R1 'aab' -> accept (2 a's)", results[0] == 1);
    }

    // Batch of mixed strings
    {
        // "b"(accept: 0 a's), "a"(reject: 1 a), "aa"(accept: 2 a's), "aba"(accept: 2 a's)
        uint8_t raw[] = {1,  0,  0,0,  0,1,0};
        int L_total = 7;
        uint16_t chars[7];
        chars_to_monoid_indices(raw, chars, L_total);
        int offsets[5] = {0, 1, 2, 4, 7};
        int results[4];
        eng.dispatch_r1(chars, offsets, results, 4, L_total, nullptr, nullptr);
        check("R1 batch 'b' -> accept",   results[0] == 1);
        check("R1 batch 'a' -> reject",   results[1] == 0);
        check("R1 batch 'aa' -> accept",  results[2] == 1);
        check("R1 batch 'aba' -> accept", results[3] == 1);
    }

    // Large batch cross-validation against sequential (B=10000)
    {
        int B = 10000;
        int L = 50;
        int total = B * L;
        uint8_t  *raw_chars   = new uint8_t[total];
        uint16_t *char_indices = new uint16_t[total];
        int      *offsets = new int[B + 1];
        int      *results = new int[B];

        srand(42);
        for (int i = 0; i < total; i++) raw_chars[i] = rand() % 2;
        chars_to_monoid_indices(raw_chars, char_indices, total);
        for (int i = 0; i <= B; i++) offsets[i] = i * L;

        eng.dispatch_r1(char_indices, offsets, results, B, total, nullptr, nullptr);

        int mismatches = 0;
        for (int i = 0; i < B; i++) {
            bool expected = even_a_sequential(raw_chars + i * L, L);
            if (results[i] != (expected ? 1 : 0)) mismatches++;
        }
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "R1 batch B=%d L=%d cross-validate (%d mismatches)",
                 B, L, mismatches);
        check(msg, mismatches == 0);

        delete[] raw_chars;
        delete[] char_indices;
        delete[] offsets;
        delete[] results;
    }

    eng.destroy();
}

// ─── test_monoid_r3 ────────────────────────────────────────────────────────
static void test_monoid_r3() {
    printf("\n--- test_monoid_r3: Decoupled look-back monoid scan ---\n");

    int M, identity;
    uint16_t compose[4];
    uint8_t  accept[2];
    build_even_a_monoid(M, identity, compose, accept);

    int max_chars = 1 << 21;  // 2M chars
    MonoidEngine eng;
    eng.init(M, identity, compose, accept, max_chars, 1);

    // Test at various lengths
    int test_lengths[] = {100, 256, 1000, 4096, 10000, 100000, 1000000};
    int n_lengths = 7;

    srand(12345);
    for (int li = 0; li < n_lengths; li++) {
        int L = test_lengths[li];

        uint8_t  *raw_chars    = new uint8_t[L];
        uint16_t *char_indices = new uint16_t[L];
        for (int j = 0; j < L; j++) raw_chars[j] = rand() % 2;
        chars_to_monoid_indices(raw_chars, char_indices, L);

        bool expected = even_a_sequential(raw_chars, L);
        int  gpu_result = eng.dispatch_r3(char_indices, L, nullptr, nullptr);

        char msg[128];
        snprintf(msg, sizeof(msg),
                 "R3 L=%d expected=%d gpu=%d", L, expected ? 1 : 0, gpu_result);
        check(msg, (expected ? 1 : 0) == gpu_result);

        delete[] raw_chars;
        delete[] char_indices;
    }

    // Empty string
    {
        uint16_t dummy[1] = {0};
        int result = eng.dispatch_r3(dummy, 0, nullptr, nullptr);
        check("R3 empty string -> accept", result == 1);
    }

    // Force-even: even number of a's in each test
    {
        int L = 100000;
        uint8_t  *raw_chars    = new uint8_t[L];
        uint16_t *char_indices = new uint16_t[L];
        srand(99);
        for (int j = 0; j < L; j++) raw_chars[j] = rand() % 2;
        // Count a's and make even
        int count_a = 0;
        for (int j = 0; j < L; j++) if (raw_chars[j] == 0) count_a++;
        if (count_a % 2 != 0) raw_chars[L - 1] = 0;  // append/flip last to make even
        chars_to_monoid_indices(raw_chars, char_indices, L);
        int result = eng.dispatch_r3(char_indices, L, nullptr, nullptr);
        check("R3 forced-even L=100K -> accept", result == 1);
        delete[] raw_chars;
        delete[] char_indices;
    }

    eng.destroy();
}

// ─── bench_monoid ──────────────────────────────────────────────────────────
static void bench_monoid() {
    printf("\n=== Monoid Benchmark ===\n");

    int M, identity;
    uint16_t compose[4];
    uint8_t  accept[2];
    build_even_a_monoid(M, identity, compose, accept);

    // ── R1 throughput: varying B and L ──
    printf("\n-- R1: Warp-per-string --\n");
    printf("  %10s  %8s  %12s  %12s  %14s  %14s\n",
           "B", "L", "kern(ms)", "total(ms)", "kern Mchar/s", "total Mchar/s");

    {
        int max_chars = 1 << 24;
        int max_batch = 1 << 20;
        MonoidEngine eng;
        eng.init(M, identity, compose, accept, max_chars, max_batch);

        int batch_sizes[] = {1000, 10000, 100000, 1000000};
        int str_lengths[] = {10, 50, 100, 500, 1000, 4096};

        for (int bi = 0; bi < 4; bi++) {
            for (int li = 0; li < 6; li++) {
                int B = batch_sizes[bi];
                int L = str_lengths[li];
                long long total = (long long)B * L;
                if (total > (1 << 24)) continue;

                uint8_t  *raw_chars    = new uint8_t[total];
                uint16_t *char_indices = new uint16_t[total];
                int      *offsets = new int[B + 1];
                int      *results = new int[B];
                srand(42);
                for (long long i = 0; i < total; i++) raw_chars[i] = rand() % 2;
                chars_to_monoid_indices(raw_chars, char_indices, (int)total);
                for (int i = 0; i <= B; i++) offsets[i] = i * L;

                // Warmup
                for (int w = 0; w < 3; w++)
                    eng.dispatch_r1(char_indices, offsets, results, B, (int)total, nullptr, nullptr);

                int iters = 20;
                float kern_ms_total = 0, total_ms_total = 0;
                for (int it = 0; it < iters; it++) {
                    float km, tm;
                    eng.dispatch_r1(char_indices, offsets, results, B, (int)total, &km, &tm);
                    kern_ms_total += km;
                    total_ms_total += tm;
                }
                float kern_ms = kern_ms_total / iters;
                float tot_ms  = total_ms_total / iters;
                double kern_mcs  = (double)total / (kern_ms * 1e3);
                double total_mcs = (double)total / (tot_ms  * 1e3);

                printf("  %10d  %8d  %12.4f  %12.4f  %14.3f  %14.3f\n",
                       B, L, kern_ms, tot_ms, kern_mcs, total_mcs);

                delete[] raw_chars;
                delete[] char_indices;
                delete[] offsets;
                delete[] results;
            }
        }
        eng.destroy();
    }

    // ── R3 throughput: varying L ──
    printf("\n-- R3: Decoupled look-back --\n");
    printf("  %12s  %12s  %12s  %14s  %14s\n",
           "L", "kern(ms)", "total(ms)", "kern Mchar/s", "total Mchar/s");

    {
        int max_L = 1 << 21;
        MonoidEngine eng;
        eng.init(M, identity, compose, accept, max_L, 1);

        int r3_lengths[] = {256, 1000, 4096, 10000, 100000, 1000000, 1<<21};

        for (int li = 0; li < 7; li++) {
            int L = r3_lengths[li];
            if (L > max_L) break;

            uint8_t  *raw_chars    = new uint8_t[L];
            uint16_t *char_indices = new uint16_t[L];
            srand(42);
            for (int j = 0; j < L; j++) raw_chars[j] = rand() % 2;
            chars_to_monoid_indices(raw_chars, char_indices, L);

            // Warmup
            for (int w = 0; w < 3; w++)
                eng.dispatch_r3(char_indices, L, nullptr, nullptr);

            int iters = L > (1 << 19) ? 10 : 50;
            float kern_ms_total = 0, total_ms_total = 0;
            for (int it = 0; it < iters; it++) {
                float km, tm;
                eng.dispatch_r3(char_indices, L, &km, &tm);
                kern_ms_total += km;
                total_ms_total += tm;
            }
            float kern_ms = kern_ms_total / iters;
            float tot_ms  = total_ms_total / iters;
            double kern_mcs  = (double)L / (kern_ms * 1e3);
            double total_mcs = (double)L / (tot_ms  * 1e3);

            printf("  %12d  %12.4f  %12.4f  %14.3f  %14.3f\n",
                   L, kern_ms, tot_ms, kern_mcs, total_mcs);

            delete[] raw_chars;
            delete[] char_indices;
        }
        eng.destroy();
    }
}

int main() {
    printf("=== TERX Monoid Scan ===\n");
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    test_monoid_r1();
    test_monoid_r3();

    printf("\n=== Results: %d / %d passed ===\n", g_pass, g_tests);
    if (g_pass != g_tests) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }

    bench_monoid();

    return 0;
}

#endif  // BUILD_LIB
