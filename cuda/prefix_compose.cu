/*
 * prefix_compose.cu — Fused DFA Simulation Engine
 *
 * Each DFA transition is a map f: {0..N-1} → {0..N-1} stored as N bytes,
 * composed via O(N) smem gathers instead of O(N³) matmul.
 *
 * Three-component architecture:
 *   1. L2 prefetch kernel — warms L2 after H2D DMA (30x cold-cache fix)
 *   2. Thread kernel (high-B) — one thread per string, 1 smem lookup/char
 *   3. Warp kernel (low-B) — one warp per string, 16 lanes track full map
 *
 * Dispatch auto-selects kernel based on batch size threshold.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

constexpr int N_STATES = 16;
constexpr int BLOCK_THREADS = 256;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)


// ─── L2 Prefetch ─────────────────────────────────────────────────────────
// H2D memcpy via DMA bypasses L2 cache. Without prefetch, the main kernel
// hits cold L2 on every cache line (30x slower). This kernel reads one byte
// per 128-byte cache line to populate L2 before the main kernel.

static __global__ void prefetch_l2(const uint8_t * __restrict__ data, int n_bytes,
                                   int * __restrict__ dummy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int sum = 0;
    for (int i = tid * 128; i < n_bytes; i += stride * 128) {
        sum += data[i];
    }
    if (tid == 0) *dummy = sum;
}


// ─── Warp Kernel (low-B) ──────────────────────────────────────────────────
//
// One warp per string: lanes 0-15 track the full 16-entry function map.
// 16 smem lookups per char but 32x more threads than the thread kernel
// for the same B — better occupancy when B < ~8K.

constexpr int WARP_WARPS_PER_BLOCK = 8;
constexpr int WARP_BLOCK = WARP_WARPS_PER_BLOCK * 32;

__launch_bounds__(WARP_BLOCK, 8)
__global__ void prefix_warp_kernel(
    const uint8_t * __restrict__ raw_concat,
    const int     * __restrict__ offsets,
    const uint8_t * __restrict__ d_tmap,
    const uint8_t * __restrict__ d_accept,
    int start_state,
    int B,
    int * __restrict__ results)
{
    extern __shared__ uint8_t smem[];
    uint8_t *tmap_sh = smem;

    for (int i = threadIdx.x; i < 256 * N_STATES; i += WARP_BLOCK)
        tmap_sh[i] = d_tmap[i];
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_global = blockIdx.x * WARP_WARPS_PER_BLOCK + warp_id;
    const int total_warps = gridDim.x * WARP_WARPS_PER_BLOCK;

    if (lane >= N_STATES) return;

    for (int sid = warp_global; sid < B; sid += total_warps) {
        int str_start = offsets[sid];
        int str_len = offsets[sid + 1] - str_start;
        const uint8_t *ptr = raw_concat + str_start;

        uint8_t acc = (uint8_t)lane;

        int t = 0;
        int full4 = str_len & ~3;
        for (; t < full4; t += 4) {
            acc = tmap_sh[ptr[t]     * N_STATES + acc];
            acc = tmap_sh[ptr[t + 1] * N_STATES + acc];
            acc = tmap_sh[ptr[t + 2] * N_STATES + acc];
            acc = tmap_sh[ptr[t + 3] * N_STATES + acc];
        }
        for (; t < str_len; t++) {
            acc = tmap_sh[ptr[t] * N_STATES + acc];
        }

        int final_state = __shfl_sync(0x0000FFFF, (int)acc, start_state);
        if (lane == 0) {
            results[sid] = (int)d_accept[final_state];
        }
    }
}


// ─── Cooperative Split-and-Reduce Kernel ─────────────────────────────────
//
// 1 warp per string: all 32 lanes cooperatively load string data into
// shared memory (coalesced), then each lane processes len/32 chars
// building a partial function map. Binary reduction in smem composes
// partial maps. Eliminates global memory from the inner loop.
//
// smem layout: [tmap: 4096] [map_buf: 2048] [string_data: variable]

constexpr int COOP_WARPS_PER_BLOCK = 8;
constexpr int COOP_BLOCK = COOP_WARPS_PER_BLOCK * 32;
constexpr int MAP_BUF_SIZE = COOP_WARPS_PER_BLOCK * 32 * N_STATES;

__launch_bounds__(COOP_BLOCK, 4)
__global__ void prefix_coop_kernel(
    const uint8_t * __restrict__ raw_concat,
    const int     * __restrict__ offsets,
    const uint8_t * __restrict__ d_tmap,
    const uint8_t * __restrict__ d_accept,
    int start_state,
    int B,
    int * __restrict__ results)
{
    extern __shared__ uint8_t smem[];
    uint8_t *tmap_sh   = smem;
    uint8_t *map_buf   = tmap_sh + 256 * N_STATES;
    uint8_t *string_sh = map_buf + MAP_BUF_SIZE;

    for (int i = threadIdx.x; i < 256 * N_STATES; i += COOP_BLOCK)
        tmap_sh[i] = d_tmap[i];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    __shared__ int blk_off[COOP_WARPS_PER_BLOCK + 1];

    __syncthreads();

    for (int block_base = blockIdx.x * COOP_WARPS_PER_BLOCK; block_base < B;
         block_base += gridDim.x * COOP_WARPS_PER_BLOCK)
    {
        int block_count = min(COOP_WARPS_PER_BLOCK, B - block_base);

        if ((int)threadIdx.x <= block_count)
            blk_off[threadIdx.x] = offsets[block_base + threadIdx.x];
        __syncthreads();

        int load_start = blk_off[0];
        int load_bytes = blk_off[block_count] - load_start;
        for (int e = threadIdx.x; e < load_bytes; e += COOP_BLOCK)
            string_sh[e] = raw_concat[load_start + e];
        __syncthreads();

        if (warp_id < block_count) {
            int my_start = blk_off[warp_id] - load_start;
            int my_len = blk_off[warp_id + 1] - blk_off[warp_id];

            int chunk = (my_len + 31) / 32;
            int my_off = lane_id * chunk;
            int my_n = min(chunk, max(0, my_len - my_off));

            uint8_t map[N_STATES];
            #pragma unroll
            for (int s = 0; s < N_STATES; s++) map[s] = (uint8_t)s;

            for (int t = 0; t < my_n; t++) {
                uint8_t bv = string_sh[my_start + my_off + t];
                #pragma unroll
                for (int s = 0; s < N_STATES; s++)
                    map[s] = tmap_sh[(int)bv * N_STATES + map[s]];
            }

            uint8_t *my_slot = map_buf + (warp_id * 32 + lane_id) * N_STATES;
            #pragma unroll
            for (int s = 0; s < N_STATES; s++) my_slot[s] = map[s];
            __syncwarp();

            for (int step = 1; step < 32; step *= 2) {
                if ((lane_id & (2 * step - 1)) == 0 && lane_id + step < 32) {
                    uint8_t *earlier = map_buf + (warp_id * 32 + lane_id) * N_STATES;
                    uint8_t *later   = map_buf + (warp_id * 32 + lane_id + step) * N_STATES;
                    #pragma unroll
                    for (int s = 0; s < N_STATES; s++)
                        earlier[s] = later[earlier[s]];
                }
                __syncwarp();
            }

            if (lane_id == 0) {
                uint8_t *final_map = map_buf + warp_id * 32 * N_STATES;
                results[block_base + warp_id] = (int)d_accept[final_map[start_state]];
            }
        }
        __syncthreads();
    }
}


// ─── Thread Kernel (high-B) ──────────────────────────────────────────────
//
// Thread-per-string: each thread walks the DFA sequentially for one string.
// 1 smem lookup per character (vs 16 in warp kernel). Better when B is large
// enough for good occupancy (B >= ~8K).

__launch_bounds__(BLOCK_THREADS, 8)
__global__ void prefix_fused_kernel(
    const uint8_t * __restrict__ raw_concat,
    const int     * __restrict__ offsets,
    const uint8_t * __restrict__ d_tmap,
    const uint8_t * __restrict__ d_accept,
    int start_state,
    int B,
    int * __restrict__ results)
{
    extern __shared__ uint8_t smem[];
    uint8_t *tmap_sh = smem;

    for (int i = threadIdx.x; i < 256 * N_STATES; i += BLOCK_THREADS)
        tmap_sh[i] = d_tmap[i];
    __syncthreads();

    const int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    const int stride = gridDim.x * BLOCK_THREADS;

    for (int sid = tid; sid < B; sid += stride) {
        int str_start = offsets[sid];
        int str_len = offsets[sid + 1] - str_start;
        const uint8_t *ptr = raw_concat + str_start;

        uint8_t state = (uint8_t)start_state;

        int t = 0;
        int full4 = str_len & ~3;
        for (; t < full4; t += 4) {
            state = tmap_sh[ptr[t]     * N_STATES + state];
            state = tmap_sh[ptr[t + 1] * N_STATES + state];
            state = tmap_sh[ptr[t + 2] * N_STATES + state];
            state = tmap_sh[ptr[t + 3] * N_STATES + state];
        }
        for (; t < str_len; t++) {
            state = tmap_sh[ptr[t] * N_STATES + state];
        }

        results[sid] = (int)d_accept[state];
    }
}


// ─── Engine Struct ────────────────────────────────────────────────────────

struct PrefixComposeEngine {
    uint8_t *d_tmap;
    uint8_t *d_accept;
    int      start_state;
    int      N;

    uint8_t *d_raw_concat;
    int     *d_offsets;
    int     *d_results;
    int     *d_dummy;
    int      max_total_chars;
    int      max_batch;
    int      persistent_grid_thread;
    int      persistent_grid_warp;
    int      persistent_grid_coop;
    int      n_sms;

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;

    void init(const uint8_t *tmap, const uint8_t *accept,
              int _start_state, int _N,
              int max_chars, int max_b)
    {
        start_state = _start_state;
        N = _N;
        max_total_chars = max_chars;
        max_batch = max_b;

        CHECK_CUDA(cudaMalloc(&d_tmap,       256 * N_STATES));
        CHECK_CUDA(cudaMalloc(&d_accept,     N_STATES));
        CHECK_CUDA(cudaMalloc(&d_raw_concat, max_chars));
        CHECK_CUDA(cudaMalloc(&d_offsets,    (max_b + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_results,    max_b * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_dummy,      sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_tmap,   tmap,   256 * N_STATES, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, accept, N_STATES,       cudaMemcpyHostToDevice));

        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
        n_sms = prop.multiProcessorCount;

        int max_blocks_per_sm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, prefix_fused_kernel, BLOCK_THREADS, 256 * N_STATES));
        persistent_grid_thread = n_sms * max_blocks_per_sm;

        int max_warp_bpsm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_warp_bpsm, prefix_warp_kernel, WARP_BLOCK, 256 * N_STATES));
        persistent_grid_warp = n_sms * max_warp_bpsm;

        int coop_smem_min = 256 * N_STATES + MAP_BUF_SIZE;
        int max_coop_bpsm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_coop_bpsm, prefix_coop_kernel, COOP_BLOCK, coop_smem_min));
        persistent_grid_coop = n_sms * max_coop_bpsm;

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));
    }

    void destroy() {
        cudaFree(d_tmap);
        cudaFree(d_accept);
        cudaFree(d_raw_concat);
        cudaFree(d_offsets);
        cudaFree(d_results);
        cudaFree(d_dummy);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start);
        cudaEventDestroy(ev_kern_stop);
    }

    void dispatch(const uint8_t *h_raw_concat,
                  const int *h_offsets,
                  int *h_results,
                  int B, int total_chars,
                  float *kernel_ms, float *total_ms)
    {
        CHECK_CUDA(cudaEventRecord(ev_start));

        CHECK_CUDA(cudaMemcpy(d_raw_concat, h_raw_concat,
                              std::max(total_chars, 1), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets,
                              (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

        // Warm L2 cache: DMA bypasses L2, so the kernel would hit cold cache.
        int warm_grid = std::min((total_chars + 128 * 256 - 1) / (128 * 256), n_sms);
        if (warm_grid > 0) {
            prefetch_l2<<<warm_grid, 256>>>(d_raw_concat, total_chars, d_dummy);
        }

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        // Compute L_max for kernel selection
        int L_max = 0;
        for (int i = 0; i < B; i++) {
            int len = h_offsets[i + 1] - h_offsets[i];
            if (len > L_max) L_max = len;
        }

        // Coop kernel: replaces warp kernel for low-B, long-string configs.
        // Splits string across 32 lanes with cooperative smem loading.
        // Never used for high-B (thread kernel is better there).
        constexpr int B_THRESHOLD = 8192;
        int coop_smem = 256 * N_STATES + MAP_BUF_SIZE
                        + COOP_WARPS_PER_BLOCK * L_max;
        bool use_coop = (B < B_THRESHOLD && L_max >= 128 && coop_smem <= 49152);

        if (B >= B_THRESHOLD) {
            int smem = 256 * N_STATES;
            int needed = (B + BLOCK_THREADS - 1) / BLOCK_THREADS;
            int grid = std::min(needed, persistent_grid_thread);
            prefix_fused_kernel<<<grid, BLOCK_THREADS, smem>>>(
                d_raw_concat, d_offsets, d_tmap, d_accept,
                start_state, B, d_results
            );
        } else if (use_coop) {
            int needed = (B + COOP_WARPS_PER_BLOCK - 1) / COOP_WARPS_PER_BLOCK;
            int grid = std::min(needed, persistent_grid_coop);
            prefix_coop_kernel<<<grid, COOP_BLOCK, coop_smem>>>(
                d_raw_concat, d_offsets, d_tmap, d_accept,
                start_state, B, d_results
            );
        } else {
            int smem = 256 * N_STATES;
            int needed = (B + WARP_WARPS_PER_BLOCK - 1) / WARP_WARPS_PER_BLOCK;
            int grid = std::min(needed, persistent_grid_warp);
            prefix_warp_kernel<<<grid, WARP_BLOCK, smem>>>(
                d_raw_concat, d_offsets, d_tmap, d_accept,
                start_state, B, d_results
            );
        }

        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results,
                              B * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        CHECK_CUDA(cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop));
        CHECK_CUDA(cudaEventElapsedTime(total_ms, ev_start, ev_stop));
    }
};


// ─── C API ────────────────────────────────────────────────────────────────

#ifdef BUILD_LIB

static PrefixComposeEngine g_engine;
static bool g_initialized = false;

extern "C" {

int prefix_engine_device_check() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 7) return -2;
    return 0;
}

int prefix_engine_init(
    const uint8_t *tmap,
    const uint8_t *accept,
    int start_state,
    int N, int K,
    int max_total_chars,
    int max_batch)
{
    if (g_initialized) {
        g_engine.destroy();
        g_initialized = false;
    }
    g_engine.init(tmap, accept, start_state, N, max_total_chars, max_batch);
    g_initialized = true;
    return 0;
}

int prefix_engine_dispatch(
    const uint8_t *raw_concat,
    const int *offsets,
    int *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms)
{
    if (!g_initialized) return -1;
    g_engine.dispatch(raw_concat, offsets, results, B, total_chars,
                      kernel_ms, total_ms);
    return 0;
}

void prefix_engine_destroy() {
    if (g_initialized) {
        g_engine.destroy();
        g_initialized = false;
    }
}

}  // extern "C"

#else  // standalone test

// ─── CPU Reference ────────────────────────────────────────────────────────

static int tests_passed = 0, tests_failed = 0;
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

static void cpu_prefix_compose(
    const uint8_t *raw_concat,
    const int *offsets,
    const uint8_t *tmap,
    const uint8_t *accept,
    int start_state,
    int N, int B,
    int *results)
{
    for (int sid = 0; sid < B; sid++) {
        int start = offsets[sid];
        int len = offsets[sid + 1] - start;

        uint8_t map[16];
        for (int s = 0; s < N; s++) map[s] = (uint8_t)s;

        for (int t = 0; t < len; t++) {
            uint8_t byte_val = raw_concat[start + t];
            uint8_t new_map[16];
            for (int s = 0; s < N; s++) {
                new_map[s] = tmap[byte_val * N + map[s]];
            }
            memcpy(map, new_map, N);
        }

        int final_state = map[start_state];
        results[sid] = (int)accept[final_state];
    }
}

// (a|b)*a(a|b) — accepts strings where second-to-last char is 'a'
// 4-state DFA tracking (second_to_last, last):
//   q0: last != 'a' (or start), second-to-last irrelevant
//   q1: last = 'a', not accepting
//   q2: accept, last != 'a', second-to-last = 'a'
//   q3: accept, last = 'a', second-to-last = 'a'
static void build_test_tmap(uint8_t *tmap, uint8_t *accept, int *start_state) {
    int N = N_STATES;
    for (int b = 0; b < 256; b++)
        for (int s = 0; s < N; s++)
            tmap[b * N + s] = (uint8_t)s;

    tmap['a' * N + 0] = 1;
    tmap['a' * N + 1] = 3;
    tmap['a' * N + 2] = 1;
    tmap['a' * N + 3] = 3;

    tmap['b' * N + 0] = 0;
    tmap['b' * N + 1] = 2;
    tmap['b' * N + 2] = 0;
    tmap['b' * N + 3] = 2;

    memset(accept, 0, N);
    accept[2] = 1;
    accept[3] = 1;
    *start_state = 0;
}


void test_correctness_small() {
    printf("test_correctness_small... ");

    uint8_t tmap[256 * N_STATES], accept[N_STATES];
    int start_state;
    build_test_tmap(tmap, accept, &start_state);

    const char *test_strings[] = {
        "", "a", "b", "aa", "ab", "ba", "bb",
        "aab", "aba", "bab", "bba", "aabb", "abab"
    };
    int expected[] = {
        0, 0, 0, 1, 1, 0, 0,
        1, 0, 1, 0, 0, 1
    };
    int n_tests = 13;

    int total_chars = 0;
    for (int i = 0; i < n_tests; i++) total_chars += strlen(test_strings[i]);

    uint8_t *raw_concat = new uint8_t[std::max(total_chars, 1)];
    int *offsets = new int[n_tests + 1];
    offsets[0] = 0;
    for (int i = 0; i < n_tests; i++) {
        int len = strlen(test_strings[i]);
        memcpy(raw_concat + offsets[i], test_strings[i], len);
        offsets[i + 1] = offsets[i] + len;
    }

    int *cpu_results = new int[n_tests];
    cpu_prefix_compose(raw_concat, offsets, tmap, accept, start_state,
                       N_STATES, n_tests, cpu_results);

    int cpu_mismatch = 0;
    for (int i = 0; i < n_tests; i++) {
        if (cpu_results[i] != expected[i]) {
            fprintf(stderr, "\n  CPU '%s': got=%d exp=%d", test_strings[i], cpu_results[i], expected[i]);
            cpu_mismatch++;
        }
    }
    if (cpu_mismatch > 0) {
        fprintf(stderr, "\n  CPU reference itself is wrong (%d mismatches)\n", cpu_mismatch);
        tests_failed++;
        delete[] raw_concat; delete[] offsets; delete[] cpu_results;
        return;
    }

    int *gpu_results = new int[n_tests];
    float kern_ms, total_ms;

    PrefixComposeEngine engine;
    engine.init(tmap, accept, start_state, N_STATES, total_chars + 1, n_tests + 1);
    engine.dispatch(raw_concat, offsets, gpu_results, n_tests, total_chars, &kern_ms, &total_ms);

    int mismatches = 0;
    for (int i = 0; i < n_tests; i++) {
        if (gpu_results[i] != expected[i]) {
            fprintf(stderr, "\n  '%s': gpu=%d cpu=%d exp=%d",
                    test_strings[i], gpu_results[i], cpu_results[i], expected[i]);
            mismatches++;
        }
    }
    TEST_ASSERT(mismatches == 0, "small correctness");
    if (mismatches == 0) printf("PASS (kern=%.3fms)\n", kern_ms);

    delete[] raw_concat; delete[] offsets;
    delete[] gpu_results; delete[] cpu_results;
    engine.destroy();
}


void test_correctness_large_random() {
    printf("test_correctness_large_random... ");

    uint8_t tmap[256 * N_STATES], accept[N_STATES];
    int start_state;
    build_test_tmap(tmap, accept, &start_state);

    int B = 4096;
    srand(42);

    int *offsets = new int[B + 1];
    offsets[0] = 0;
    for (int i = 0; i < B; i++)
        offsets[i + 1] = offsets[i] + 1 + rand() % 256;
    int total_chars = offsets[B];

    uint8_t *raw_concat = new uint8_t[total_chars];
    for (int i = 0; i < total_chars; i++)
        raw_concat[i] = (rand() % 2) ? 'a' : 'b';

    PrefixComposeEngine engine;
    engine.init(tmap, accept, start_state, N_STATES, total_chars + 1, B + 1);

    int *gpu_results = new int[B];
    int *cpu_results = new int[B];
    float kern_ms, total_ms;

    engine.dispatch(raw_concat, offsets, gpu_results, B, total_chars, &kern_ms, &total_ms);
    cpu_prefix_compose(raw_concat, offsets, tmap, accept, start_state, N_STATES, B, cpu_results);

    int mismatches = 0;
    for (int i = 0; i < B; i++)
        if (gpu_results[i] != cpu_results[i]) mismatches++;

    TEST_ASSERT(mismatches == 0, "large random batch");
    if (mismatches == 0) printf("PASS (B=%d, kern=%.3fms)\n", B, kern_ms);

    delete[] raw_concat; delete[] offsets;
    delete[] gpu_results; delete[] cpu_results;
    engine.destroy();
}


void test_long_strings() {
    printf("test_long_strings... ");

    uint8_t tmap[256 * N_STATES], accept[N_STATES];
    int start_state;
    build_test_tmap(tmap, accept, &start_state);

    int B = 64;
    int L = 10000;
    srand(99);

    int *offsets = new int[B + 1];
    offsets[0] = 0;
    for (int i = 0; i < B; i++)
        offsets[i + 1] = offsets[i] + L;
    int total_chars = offsets[B];

    uint8_t *raw_concat = new uint8_t[total_chars];
    for (int i = 0; i < total_chars; i++)
        raw_concat[i] = (rand() % 2) ? 'a' : 'b';

    PrefixComposeEngine engine;
    engine.init(tmap, accept, start_state, N_STATES, total_chars + 1, B + 1);

    int *gpu_results = new int[B];
    int *cpu_results = new int[B];
    float kern_ms, total_ms;

    engine.dispatch(raw_concat, offsets, gpu_results, B, total_chars, &kern_ms, &total_ms);
    cpu_prefix_compose(raw_concat, offsets, tmap, accept, start_state, N_STATES, B, cpu_results);

    int mismatches = 0;
    for (int i = 0; i < B; i++)
        if (gpu_results[i] != cpu_results[i]) mismatches++;

    TEST_ASSERT(mismatches == 0, "long strings");
    if (mismatches == 0) printf("PASS (B=%d, L=%d, kern=%.3fms)\n", B, L, kern_ms);

    delete[] raw_concat; delete[] offsets;
    delete[] gpu_results; delete[] cpu_results;
    engine.destroy();
}


void test_benchmark() {
    printf("test_benchmark...\n");

    uint8_t tmap[256 * N_STATES], accept[N_STATES];
    int start_state;
    build_test_tmap(tmap, accept, &start_state);

    int configs[][2] = {
        {65536, 128}, {65536, 512}, {4096, 4096}, {1024, 32768},
        {131072, 64}, {131072, 256},
        {4096, 512}, {4096, 1024}, {2048, 2048}, {512, 4096}
    };

    for (auto &cfg : configs) {
        int B = cfg[0], L = cfg[1];
        srand(42);

        int *offsets = new int[B + 1];
        offsets[0] = 0;
        for (int i = 0; i < B; i++)
            offsets[i + 1] = offsets[i] + L;
        int total_chars = offsets[B];

        uint8_t *raw_concat = new uint8_t[total_chars];
        for (int i = 0; i < total_chars; i++)
            raw_concat[i] = (rand() % 2) ? 'a' : 'b';

        PrefixComposeEngine engine;
        engine.init(tmap, accept, start_state, N_STATES, total_chars + 1, B + 1);

        int *results = new int[B];
        float kern_ms, total_ms;

        // Warmup
        engine.dispatch(raw_concat, offsets, results, B, total_chars, &kern_ms, &total_ms);

        float best_kern = 1e9, best_total = 1e9;
        for (int rep = 0; rep < 10; rep++) {
            engine.dispatch(raw_concat, offsets, results, B, total_chars, &kern_ms, &total_ms);
            if (kern_ms < best_kern) best_kern = kern_ms;
            if (total_ms < best_total) best_total = total_ms;
        }

        double total_chars_d = (double)B * L;
        double gc_kern = total_chars_d / (best_kern * 1e-3) / 1e9;
        double gc_total = total_chars_d / (best_total * 1e-3) / 1e9;
        printf("  B=%7d L=%6d  chars=%10.0f  kern=%.3fms (%.0f Gc/s)  total=%.3fms (%.0f Gc/s)\n",
               B, L, total_chars_d, best_kern, gc_kern, best_total, gc_total);

        delete[] offsets; delete[] raw_concat; delete[] results;
        engine.destroy();
    }
}


int main() {
    printf("=== Prefix Compose Engine Tests ===\n");
    test_correctness_small();
    test_correctness_large_random();
    test_long_strings();
    test_benchmark();
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}

#endif
