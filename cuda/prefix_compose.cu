/*
 * prefix_compose.cu — Function Map Parallel Prefix Engine
 *
 * Warp-shuffle-based function map composition for DFA simulation.
 * Each transition is a map f: {0..N-1} → {0..N-1} (N bytes, not N² matrix).
 * Composition via shared memory gathers: O(N) per step instead of O(N³) matmul.
 *
 * Two-phase parallel prefix:
 *   Phase 1 (prefix_block_reduce_kernel): compose K chars into one map per block
 *   Phase 2 (prefix_scan_accept_kernel): serial compose of block products + accept
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

constexpr int N_STATES = 16;
constexpr int BLOCK_K = 32;
constexpr int PC_WARPS = 4;
constexpr int PC_BLOCK = PC_WARPS * 32;  // 128 threads

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)


// ─── Phase 1: Block Tree Reduce ───────────────────────────────────────────
//
// Each warp processes one K-character block. Lanes 0-15 each track the
// destination of their starting state through the block's character sequence.
// After K iterations, lanes hold the composed map for the entire block.
//
// Shared memory: tmap_sh[256 * 16] = 4 KB (shared across all warps in CUDA block)

__launch_bounds__(PC_BLOCK)
__global__ void prefix_block_reduce_kernel(
    const uint8_t * __restrict__ raw_concat,
    const int     * __restrict__ block_desc_string_id,
    const int     * __restrict__ block_desc_char_offset,
    const int     * __restrict__ string_offsets,
    const int     * __restrict__ string_lengths,
    const uint8_t * __restrict__ d_tmap,
    int total_blocks,
    uint8_t       * __restrict__ block_products)   // [total_blocks * 16]
{
    extern __shared__ uint8_t smem[];
    uint8_t *tmap_sh = smem;  // 256 * 16 = 4096 bytes

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Cooperative load of tmap into shared memory
    for (int i = threadIdx.x; i < 256 * N_STATES; i += PC_BLOCK)
        tmap_sh[i] = d_tmap[i];
    __syncthreads();

    int gid = blockIdx.x * PC_WARPS + warp_id;
    if (gid >= total_blocks) return;
    if (lane >= N_STATES) return;

    int sid = block_desc_string_id[gid];
    int block_off = block_desc_char_offset[gid];
    int str_start = string_offsets[sid];
    int str_len = string_lengths[sid];

    // Each of 16 lanes composes the block's map for its starting state.
    // acc starts as identity: lane j maps to j.
    // For each character, acc[j] = tmap[char][acc[j]] — follow the chain.
    uint8_t acc = (uint8_t)lane;

    for (int k = 0; k < BLOCK_K; k++) {
        int pos = block_off + k;
        if (pos < str_len) {
            uint8_t byte_val = raw_concat[str_start + pos];
            acc = tmap_sh[byte_val * N_STATES + acc];
        }
        // else: identity — acc unchanged
    }

    block_products[gid * N_STATES + lane] = acc;
}


// ─── Phase 2: Block Scan + Accept ─────────────────────────────────────────
//
// Each warp processes one string by serially composing its block products
// via warp shuffles. The composed map applied to start_state gives the
// final DFA state.

__launch_bounds__(PC_BLOCK)
__global__ void prefix_scan_accept_kernel(
    const uint8_t * __restrict__ block_products,    // [total_blocks * 16]
    const int     * __restrict__ string_n_blocks,   // n_blocks per string
    const int     * __restrict__ string_block_start, // first block index per string
    const uint8_t * __restrict__ d_accept,
    int start_state,
    int B,
    int * __restrict__ results)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    int sid = blockIdx.x * PC_WARPS + warp_id;
    if (sid >= B) return;
    if (lane >= N_STATES) return;

    int n_blk = string_n_blocks[sid];
    int blk_start = string_block_start[sid];

    // Serial compose of block products via shuffles.
    // acc[lane] = composed map so far, applied to state lane.
    uint8_t acc = (uint8_t)lane;  // identity

    for (int b = 0; b < n_blk; b++) {
        uint8_t bp = block_products[(blk_start + b) * N_STATES + lane];
        // Compose: new_acc[lane] = bp[acc[lane]]
        // __shfl_sync reads bp from the lane whose index == acc[lane]
        acc = (uint8_t)__shfl_sync(0x0000FFFF, (int)bp, (int)acc);
    }

    // All lanes participate in the shuffle; only lane 0 writes
    int final_state = __shfl_sync(0x0000FFFF, (int)acc, start_state);
    if (lane == 0) {
        results[sid] = (int)d_accept[final_state];
    }
}


// ─── Engine Struct ────────────────────────────────────────────────────────

struct PrefixComposeEngine {
    uint8_t *d_tmap;            // 256 * N_STATES
    uint8_t *d_accept;          // N_STATES
    int      start_state;
    int      N;

    uint8_t *d_raw_concat;
    int     *d_offsets;
    int     *d_results;
    int      max_total_chars;
    int      max_batch;

    int     *d_block_string_id;
    int     *d_block_char_offset;
    int     *d_string_n_blocks;
    int     *d_string_block_start;
    int     *d_string_lengths;
    uint8_t *d_block_products;
    int      max_total_blocks;

    int     *h_block_string_id;
    int     *h_block_char_offset;
    int     *h_string_n_blocks;
    int     *h_string_block_start;
    int     *h_string_lengths;

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;

    void init(const uint8_t *tmap, const uint8_t *accept,
              int _start_state, int _N,
              int max_chars, int max_b)
    {
        start_state = _start_state;
        N = _N;
        max_total_chars = max_chars;
        max_batch = max_b;
        max_total_blocks = max_chars / BLOCK_K + max_b + 1;

        CHECK_CUDA(cudaMalloc(&d_tmap,    256 * N_STATES));
        CHECK_CUDA(cudaMalloc(&d_accept,  N_STATES));
        CHECK_CUDA(cudaMalloc(&d_raw_concat, max_chars));
        CHECK_CUDA(cudaMalloc(&d_offsets, (max_b + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_results, max_b * sizeof(int)));

        CHECK_CUDA(cudaMalloc(&d_block_string_id,   max_total_blocks * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_block_char_offset,  max_total_blocks * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_string_n_blocks,    max_b * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_string_block_start, max_b * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_string_lengths,     max_b * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_block_products,     max_total_blocks * N_STATES));

        CHECK_CUDA(cudaMemcpy(d_tmap,   tmap,   256 * N_STATES, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, accept, N_STATES,       cudaMemcpyHostToDevice));

        h_block_string_id   = new int[max_total_blocks];
        h_block_char_offset = new int[max_total_blocks];
        h_string_n_blocks   = new int[max_b];
        h_string_block_start = new int[max_b];
        h_string_lengths    = new int[max_b];

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
        cudaFree(d_block_string_id);
        cudaFree(d_block_char_offset);
        cudaFree(d_string_n_blocks);
        cudaFree(d_string_block_start);
        cudaFree(d_string_lengths);
        cudaFree(d_block_products);
        delete[] h_block_string_id;
        delete[] h_block_char_offset;
        delete[] h_string_n_blocks;
        delete[] h_string_block_start;
        delete[] h_string_lengths;
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

        // Build block descriptors on CPU
        int total_blocks = 0;
        for (int i = 0; i < B; i++) {
            int len = h_offsets[i + 1] - h_offsets[i];
            h_string_lengths[i] = len;
            int n_blk = len > 0 ? (len + BLOCK_K - 1) / BLOCK_K : 1;
            h_string_n_blocks[i] = n_blk;
            h_string_block_start[i] = total_blocks;
            for (int b = 0; b < n_blk; b++) {
                h_block_string_id[total_blocks + b] = i;
                h_block_char_offset[total_blocks + b] = b * BLOCK_K;
            }
            total_blocks += n_blk;
        }

        // Copy to device
        CHECK_CUDA(cudaMemcpy(d_raw_concat, h_raw_concat,
                              std::max(total_chars, 1), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets,
                              (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_block_string_id,   h_block_string_id,
                              total_blocks * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_block_char_offset,  h_block_char_offset,
                              total_blocks * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_string_n_blocks,    h_string_n_blocks,
                              B * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_string_block_start, h_string_block_start,
                              B * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_string_lengths,     h_string_lengths,
                              B * sizeof(int), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        // Phase 1: block reduce
        int grid1 = (total_blocks + PC_WARPS - 1) / PC_WARPS;
        int smem1 = 256 * N_STATES;
        prefix_block_reduce_kernel<<<grid1, PC_BLOCK, smem1>>>(
            d_raw_concat,
            d_block_string_id,
            d_block_char_offset,
            d_offsets,
            d_string_lengths,
            d_tmap,
            total_blocks,
            d_block_products
        );

        // Phase 2: scan + accept
        int grid2 = (B + PC_WARPS - 1) / PC_WARPS;
        prefix_scan_accept_kernel<<<grid2, PC_BLOCK>>>(
            d_block_products,
            d_string_n_blocks,
            d_string_block_start,
            d_accept,
            start_state,
            B,
            d_results
        );

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

        // Build composed map: identity → apply each char's map
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

    // 'a': q0→q1, q1→q3, q2→q1, q3→q3
    tmap['a' * N + 0] = 1;
    tmap['a' * N + 1] = 3;
    tmap['a' * N + 2] = 1;
    tmap['a' * N + 3] = 3;

    // 'b': q0→q0, q1→q2, q2→q0, q3→q2
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
    // (a|b)*a(a|b): accepts iff second-to-last char is 'a'
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

    // Verify CPU reference against expected first
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

    // Now test GPU
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
        {65536, 128}, {65536, 512}, {4096, 4096}, {1024, 32768}
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

        // Timed run
        float best_kern = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            engine.dispatch(raw_concat, offsets, results, B, total_chars, &kern_ms, &total_ms);
            if (kern_ms < best_kern) best_kern = kern_ms;
        }

        double total_chars_d = (double)B * L;
        double gc_per_s = total_chars_d / (best_kern * 1e-3) / 1e9;
        printf("  B=%6d L=%6d  chars=%10.0f  kern=%.3fms  %.1f Gc/s\n",
               B, L, total_chars_d, best_kern, gc_per_s);

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
