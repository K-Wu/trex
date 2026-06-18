/*
 * monoid_batch.cu — Monoid Batch Pipeline
 *
 * Thread-per-string compose-table lookup kernels for small-monoid DFAs.
 * Replaces O(N³) tensor-core MMA with O(1) shared-memory table lookups.
 *
 * Kernels:
 *   monoid_batch_kernel:  1 thread per string, sequential compose
 *   monoid_prefix_kernel: parallel tree reduce for few long strings
 *
 * Shared memory layout (batch kernel):
 *   char_compose[M * sigma_ext]  — fused compose table (uint8)
 *   raw_char_map[256]            — raw byte → DFA char index (uint8)
 *   accept[M]                    — per-element acceptance (uint8)
 *
 * Inner loop: 2 shared memory reads + 1 register update per character.
 * Bottleneck: HBM bandwidth at ~4900 Gc/s. Target: 1500-2500 Gc/s.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

constexpr int MB_BLOCK_SIZE = 128;
constexpr int MP_BLOCK_SIZE = 256;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)


// ─── Batch Kernel ──────────────────────────────────────────────────────────

__launch_bounds__(MB_BLOCK_SIZE, 16)
__global__ void monoid_batch_kernel(
    const uint8_t * __restrict__ raw_concat,
    const int     * __restrict__ offsets,
    const uint8_t * __restrict__ d_char_compose,
    const uint8_t * __restrict__ d_raw_char_map,
    const uint8_t * __restrict__ d_accept,
    int B, int M, int sigma_ext,
    uint8_t identity,
    int * __restrict__ results)
{
    extern __shared__ uint8_t smem[];
    uint8_t *compose_sh = smem;
    uint8_t *charmap_sh = compose_sh + M * sigma_ext;
    uint8_t *accept_sh  = charmap_sh + 256;

    for (int e = threadIdx.x; e < M * sigma_ext; e += MB_BLOCK_SIZE)
        compose_sh[e] = d_char_compose[e];
    for (int e = threadIdx.x; e < 256; e += MB_BLOCK_SIZE)
        charmap_sh[e] = d_raw_char_map[e];
    for (int e = threadIdx.x; e < M; e += MB_BLOCK_SIZE)
        accept_sh[e] = d_accept[e];
    __syncthreads();

    int sid = blockIdx.x * MB_BLOCK_SIZE + threadIdx.x;
    if (sid >= B) return;

    int start = offsets[sid];
    int len   = offsets[sid + 1] - start;
    uint8_t curr = identity;

    for (int t = 0; t < len; t++) {
        uint8_t ch_idx = charmap_sh[raw_concat[start + t]];
        curr = compose_sh[curr * sigma_ext + ch_idx];
    }

    results[sid] = (int)accept_sh[curr];
}


// ─── Prefix Kernel ─────────────────────────────────────────────────────────
// One block per string, MP_BLOCK_SIZE threads.
// Phase 1: thread-local sequential reduce over chunk of characters.
// Phase 2: tree reduce in shared memory using M×M compose table.
// Phase 3: thread 0 writes accept result.

__launch_bounds__(MP_BLOCK_SIZE, 1)
__global__ void monoid_prefix_kernel(
    const uint8_t * __restrict__ raw_concat,
    const int     * __restrict__ offsets,
    const uint8_t * __restrict__ d_char_compose,
    const uint8_t * __restrict__ d_raw_char_map,
    const uint8_t * __restrict__ d_monoid_compose,
    const uint8_t * __restrict__ d_accept,
    int B, int M, int sigma_ext,
    uint8_t identity,
    int * __restrict__ results)
{
    int sid = blockIdx.x;
    if (sid >= B) return;

    extern __shared__ uint8_t smem[];
    uint8_t *compose_sh  = smem;
    uint8_t *charmap_sh  = compose_sh + M * sigma_ext;
    uint8_t *mcompose_sh = charmap_sh + 256;
    uint8_t *accept_sh   = mcompose_sh + M * M;
    uint8_t *reduce_sh   = accept_sh + M;

    int total_tables = M * sigma_ext + 256 + M * M + M;
    for (int e = threadIdx.x; e < total_tables; e += MP_BLOCK_SIZE) {
        if (e < M * sigma_ext)
            smem[e] = d_char_compose[e];
        else if (e < M * sigma_ext + 256)
            smem[e] = d_raw_char_map[e - M * sigma_ext];
        else if (e < M * sigma_ext + 256 + M * M)
            smem[e] = d_monoid_compose[e - M * sigma_ext - 256];
        else
            smem[e] = d_accept[e - M * sigma_ext - 256 - M * M];
    }
    __syncthreads();

    int start = offsets[sid];
    int len   = offsets[sid + 1] - start;

    // Phase 1: thread-local sequential reduce
    int chunk = (len + MP_BLOCK_SIZE - 1) / MP_BLOCK_SIZE;
    int my_offset = threadIdx.x * chunk;
    int my_len = min(chunk, max(0, len - my_offset));

    uint8_t local = identity;
    for (int t = 0; t < my_len; t++) {
        uint8_t ch_idx = charmap_sh[raw_concat[start + my_offset + t]];
        local = compose_sh[local * sigma_ext + ch_idx];
    }
    reduce_sh[threadIdx.x] = local;
    __syncthreads();

    // Phase 2: tree reduce — compose(newer, older)
    for (int stride = MP_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            uint8_t older = reduce_sh[threadIdx.x];
            uint8_t newer = reduce_sh[threadIdx.x + stride];
            reduce_sh[threadIdx.x] = mcompose_sh[newer * M + older];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        results[sid] = (int)accept_sh[reduce_sh[0]];
    }
}


// ─── Engine Struct ─────────────────────────────────────────────────────────

struct MonoidBatchEngine {
    int M;
    int sigma_ext;
    uint8_t identity;

    uint8_t *d_char_compose;
    uint8_t *d_raw_char_map;
    uint8_t *d_accept;
    uint8_t *d_monoid_compose;

    uint8_t *d_raw_concat;
    int     *d_offsets;
    int     *d_results;
    int      max_total_chars;
    int      max_batch;

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;

    void init(int _M, int _sigma_ext, int _identity,
              const uint8_t *char_compose,
              const uint8_t *raw_char_map,
              const uint8_t *accept,
              const uint8_t *monoid_compose,
              int max_chars, int max_b)
    {
        M = _M;
        sigma_ext = _sigma_ext;
        identity = (uint8_t)_identity;
        max_total_chars = max_chars;
        max_batch = max_b;

        CHECK_CUDA(cudaMalloc(&d_char_compose,   M * sigma_ext));
        CHECK_CUDA(cudaMalloc(&d_raw_char_map,   256));
        CHECK_CUDA(cudaMalloc(&d_accept,         M));
        CHECK_CUDA(cudaMalloc(&d_monoid_compose, M * M));
        CHECK_CUDA(cudaMalloc(&d_raw_concat,     max_chars));
        CHECK_CUDA(cudaMalloc(&d_offsets,        (max_b + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_results,        max_b * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_char_compose,   char_compose,   M * sigma_ext,  cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_raw_char_map,   raw_char_map,   256,            cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept,         accept,         M,              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_monoid_compose, monoid_compose, M * M,          cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));
    }

    void destroy() {
        cudaFree(d_char_compose);
        cudaFree(d_raw_char_map);
        cudaFree(d_accept);
        cudaFree(d_monoid_compose);
        cudaFree(d_raw_concat);
        cudaFree(d_offsets);
        cudaFree(d_results);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start);
        cudaEventDestroy(ev_kern_stop);
    }

    void dispatch_batch(const uint8_t *h_raw_concat,
                        const int *h_offsets,
                        int *h_results,
                        int B, int total_chars,
                        float *kernel_ms, float *total_ms)
    {
        CHECK_CUDA(cudaEventRecord(ev_start));

        CHECK_CUDA(cudaMemcpy(d_raw_concat, h_raw_concat, total_chars, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

        int grid = (B + MB_BLOCK_SIZE - 1) / MB_BLOCK_SIZE;
        int smem = M * sigma_ext + 256 + M;

        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        monoid_batch_kernel<<<grid, MB_BLOCK_SIZE, smem>>>(
            d_raw_concat, d_offsets,
            d_char_compose, d_raw_char_map, d_accept,
            B, M, sigma_ext, identity,
            d_results
        );
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results, B * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        CHECK_CUDA(cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop));
        CHECK_CUDA(cudaEventElapsedTime(total_ms, ev_start, ev_stop));
    }

    void dispatch_prefix(const uint8_t *h_raw_concat,
                         const int *h_offsets,
                         int *h_results,
                         int B, int total_chars,
                         float *kernel_ms, float *total_ms)
    {
        CHECK_CUDA(cudaEventRecord(ev_start));

        CHECK_CUDA(cudaMemcpy(d_raw_concat, h_raw_concat, total_chars, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

        int smem = M * sigma_ext + 256 + M * M + M + MP_BLOCK_SIZE;

        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        monoid_prefix_kernel<<<B, MP_BLOCK_SIZE, smem>>>(
            d_raw_concat, d_offsets,
            d_char_compose, d_raw_char_map, d_monoid_compose, d_accept,
            B, M, sigma_ext, identity,
            d_results
        );
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results, B * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        CHECK_CUDA(cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop));
        CHECK_CUDA(cudaEventElapsedTime(total_ms, ev_start, ev_stop));
    }
};


// ─── C API ─────────────────────────────────────────────────────────────────

#ifdef BUILD_LIB

static MonoidBatchEngine g_engine;
static bool g_initialized = false;

extern "C" {

int monoid_batch_engine_device_check() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 7) return -2;
    return 0;
}

int monoid_batch_engine_init(
    int M, int sigma_ext, int identity,
    const uint8_t *char_compose,
    const uint8_t *raw_char_map,
    const uint8_t *accept,
    const uint8_t *monoid_compose,
    int max_total_chars, int max_batch)
{
    if (g_initialized) {
        g_engine.destroy();
        g_initialized = false;
    }
    g_engine.init(M, sigma_ext, identity,
                  char_compose, raw_char_map, accept, monoid_compose,
                  max_total_chars, max_batch);
    g_initialized = true;
    return 0;
}

void monoid_batch_engine_destroy() {
    if (g_initialized) {
        g_engine.destroy();
        g_initialized = false;
    }
}

int monoid_batch_engine_dispatch(
    const uint8_t *raw_concat,
    const int *offsets,
    int *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms)
{
    if (!g_initialized) return -1;
    g_engine.dispatch_batch(raw_concat, offsets, results, B, total_chars,
                            kernel_ms, total_ms);
    return 0;
}

int monoid_batch_engine_dispatch_prefix(
    const uint8_t *raw_concat,
    const int *offsets,
    int *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms)
{
    if (!g_initialized) return -1;
    g_engine.dispatch_prefix(raw_concat, offsets, results, B, total_chars,
                             kernel_ms, total_ms);
    return 0;
}

}  // extern "C"

#else  // standalone test

// ─── CPU Reference ─────────────────────────────────────────────────────────

static void cpu_monoid_batch(
    const uint8_t *raw_concat,
    const int *offsets,
    const uint8_t *char_compose,
    const uint8_t *raw_char_map,
    const uint8_t *accept,
    int B, int M, int sigma_ext,
    uint8_t identity,
    int *results)
{
    for (int sid = 0; sid < B; sid++) {
        int start = offsets[sid];
        int len = offsets[sid + 1] - start;
        uint8_t curr = identity;
        for (int t = 0; t < len; t++) {
            uint8_t ch_idx = raw_char_map[raw_concat[start + t]];
            curr = char_compose[curr * sigma_ext + ch_idx];
        }
        results[sid] = (int)accept[curr];
    }
}


// ─── Standalone Tests ──────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_total = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_total++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        tests_passed++; \
    } \
} while(0)

// Even-A DFA: (b*ab*ab*)*b*
// Monoid: {I, T_a} (size 2). T_a swaps states, T_b = I.
// compose[newer, older]: compose[0,0]=0, compose[0,1]=1, compose[1,0]=1, compose[1,1]=0
static void build_even_a_tables(
    uint8_t *char_compose, uint8_t *raw_char_map, uint8_t *accept,
    uint8_t *monoid_compose,
    int *M_out, int *sigma_ext_out, uint8_t *identity_out)
{
    int M = 2, sigma = 2, sigma_ext = 3;

    char_compose[0 * sigma_ext + 0] = 1;  // curr=0, a -> T_a∘I = T_a
    char_compose[0 * sigma_ext + 1] = 0;  // curr=0, b -> I∘I = I
    char_compose[0 * sigma_ext + 2] = 0;  // curr=0, identity
    char_compose[1 * sigma_ext + 0] = 0;  // curr=1, a -> T_a∘T_a = I
    char_compose[1 * sigma_ext + 1] = 1;  // curr=1, b -> I∘T_a = T_a
    char_compose[1 * sigma_ext + 2] = 1;  // curr=1, identity

    memset(raw_char_map, sigma, 256);
    raw_char_map['a'] = 0;
    raw_char_map['b'] = 1;

    accept[0] = 1;
    accept[1] = 0;

    monoid_compose[0 * M + 0] = 0;
    monoid_compose[0 * M + 1] = 1;
    monoid_compose[1 * M + 0] = 1;
    monoid_compose[1 * M + 1] = 0;

    *M_out = M;
    *sigma_ext_out = sigma_ext;
    *identity_out = 0;
}


void test_batch_correctness() {
    printf("test_batch_correctness... ");

    uint8_t char_compose[6], raw_char_map[256], accept[2], monoid_compose[4];
    int M, sigma_ext; uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    MonoidBatchEngine engine;
    engine.init(M, sigma_ext, identity,
                char_compose, raw_char_map, accept, monoid_compose,
                1 << 20, 1 << 16);

    const char *test_strings[] = {"", "a", "aa", "b", "aab", "aabb", "abab",
                                  "aaaa", "aaaaa", "bbb", "aba", "bab"};
    int expected[] = {1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0};
    int n_tests = 12;

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

    int *gpu_results = new int[n_tests];
    int *cpu_results = new int[n_tests];
    float kern_ms, total_ms;

    engine.dispatch_batch(raw_concat, offsets, gpu_results,
                          n_tests, total_chars, &kern_ms, &total_ms);
    cpu_monoid_batch(raw_concat, offsets, char_compose, raw_char_map, accept,
                     n_tests, M, sigma_ext, identity, cpu_results);

    int mismatches = 0;
    for (int i = 0; i < n_tests; i++) {
        if (gpu_results[i] != expected[i] || cpu_results[i] != expected[i]) {
            fprintf(stderr, "\n  '%s': gpu=%d cpu=%d exp=%d",
                    test_strings[i], gpu_results[i], cpu_results[i], expected[i]);
            mismatches++;
        }
    }
    TEST_ASSERT(mismatches == 0, "batch correctness");
    if (mismatches == 0) printf("PASS (kern=%.3fms)\n", kern_ms);

    delete[] raw_concat; delete[] offsets;
    delete[] gpu_results; delete[] cpu_results;
    engine.destroy();
}


void test_batch_large_random() {
    printf("test_batch_large_random... ");

    uint8_t char_compose[6], raw_char_map[256], accept[2], monoid_compose[4];
    int M, sigma_ext; uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

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

    MonoidBatchEngine engine;
    engine.init(M, sigma_ext, identity,
                char_compose, raw_char_map, accept, monoid_compose,
                total_chars + 1, B + 1);

    int *gpu_results = new int[B];
    int *cpu_results = new int[B];
    float kern_ms, total_ms;

    engine.dispatch_batch(raw_concat, offsets, gpu_results,
                          B, total_chars, &kern_ms, &total_ms);
    cpu_monoid_batch(raw_concat, offsets, char_compose, raw_char_map, accept,
                     B, M, sigma_ext, identity, cpu_results);

    int mismatches = 0;
    for (int i = 0; i < B; i++)
        if (gpu_results[i] != cpu_results[i]) mismatches++;

    TEST_ASSERT(mismatches == 0, "large random batch");
    if (mismatches == 0) printf("PASS (B=%d, kern=%.3fms)\n", B, kern_ms);

    delete[] raw_concat; delete[] offsets;
    delete[] gpu_results; delete[] cpu_results;
    engine.destroy();
}


void test_prefix_correctness() {
    printf("test_prefix_correctness... ");

    uint8_t char_compose[6], raw_char_map[256], accept[2], monoid_compose[4];
    int M, sigma_ext; uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    int B = 1, L = 100000;
    uint8_t *raw_concat = new uint8_t[L];
    int offsets[2] = {0, L};
    srand(55);
    int count_a = 0;
    for (int i = 0; i < L; i++) {
        raw_concat[i] = (rand() % 2) ? 'a' : 'b';
        if (raw_concat[i] == 'a') count_a++;
    }

    MonoidBatchEngine engine;
    engine.init(M, sigma_ext, identity,
                char_compose, raw_char_map, accept, monoid_compose,
                L + 1, B + 1);

    int gpu_batch, gpu_prefix, cpu_result;
    float kern_ms, total_ms;

    engine.dispatch_batch(raw_concat, offsets, &gpu_batch,
                          B, L, &kern_ms, &total_ms);
    engine.dispatch_prefix(raw_concat, offsets, &gpu_prefix,
                           B, L, &kern_ms, &total_ms);
    cpu_monoid_batch(raw_concat, offsets, char_compose, raw_char_map, accept,
                     B, M, sigma_ext, identity, &cpu_result);

    bool ok = (gpu_batch == cpu_result) && (gpu_prefix == cpu_result);
    TEST_ASSERT(ok, "prefix correctness");
    if (ok) printf("PASS (L=%d, count_a=%d, result=%d)\n", L, count_a, cpu_result);
    else printf("FAIL (cpu=%d batch=%d prefix=%d)\n", cpu_result, gpu_batch, gpu_prefix);

    delete[] raw_concat;
    engine.destroy();
}


void test_prefix_multi_string() {
    printf("test_prefix_multi_string... ");

    uint8_t char_compose[6], raw_char_map[256], accept[2], monoid_compose[4];
    int M, sigma_ext; uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    int B = 8, L = 50000;
    int total_chars = B * L;
    uint8_t *raw_concat = new uint8_t[total_chars];
    int *offsets = new int[B + 1];
    srand(77);
    offsets[0] = 0;
    for (int i = 0; i < B; i++) offsets[i + 1] = offsets[i] + L;
    for (int i = 0; i < total_chars; i++)
        raw_concat[i] = (rand() % 2) ? 'a' : 'b';

    MonoidBatchEngine engine;
    engine.init(M, sigma_ext, identity,
                char_compose, raw_char_map, accept, monoid_compose,
                total_chars + 1, B + 1);

    int *gpu_prefix = new int[B];
    int *cpu_results = new int[B];
    float kern_ms, total_ms;

    engine.dispatch_prefix(raw_concat, offsets, gpu_prefix,
                           B, total_chars, &kern_ms, &total_ms);
    cpu_monoid_batch(raw_concat, offsets, char_compose, raw_char_map, accept,
                     B, M, sigma_ext, identity, cpu_results);

    int mismatches = 0;
    for (int i = 0; i < B; i++)
        if (gpu_prefix[i] != cpu_results[i]) mismatches++;

    TEST_ASSERT(mismatches == 0, "prefix multi-string");
    if (mismatches == 0) printf("PASS (B=%d, L=%d, kern=%.3fms)\n", B, L, kern_ms);

    delete[] raw_concat; delete[] offsets;
    delete[] gpu_prefix; delete[] cpu_results;
    engine.destroy();
}


void bench_throughput() {
    printf("\n=== Monoid Batch Throughput ===\n");
    printf("%7s  %5s  %10s  %10s\n", "B", "L", "kern(ms)", "Gc/s");
    printf("------  -----  ----------  ----------\n");

    uint8_t char_compose[6], raw_char_map[256], accept[2], monoid_compose[4];
    int M, sigma_ext; uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    int Bs[] = {1024, 4096, 16384, 65536, 262144};
    int Ls[] = {128, 512, 2048};

    for (int bi = 0; bi < 5; bi++) {
        for (int li = 0; li < 3; li++) {
            int B = Bs[bi], L = Ls[li];
            long long total_chars = (long long)B * L;

            uint8_t *raw_concat = new uint8_t[total_chars];
            int *offsets = new int[B + 1];
            srand(123);
            offsets[0] = 0;
            for (int i = 0; i < B; i++) offsets[i + 1] = offsets[i] + L;
            for (long long i = 0; i < total_chars; i++)
                raw_concat[i] = (rand() % 2) ? 'a' : 'b';

            MonoidBatchEngine eng;
            eng.init(M, sigma_ext, identity,
                     char_compose, raw_char_map, accept, monoid_compose,
                     (int)total_chars + 1, B + 1);

            int *results = new int[B];
            float kern_ms, total_ms_val;

            for (int w = 0; w < 3; w++)
                eng.dispatch_batch(raw_concat, offsets, results, B, (int)total_chars,
                                   &kern_ms, &total_ms_val);

            float sum_kern = 0;
            int runs = 20;
            for (int r = 0; r < runs; r++) {
                eng.dispatch_batch(raw_concat, offsets, results, B, (int)total_chars,
                                   &kern_ms, &total_ms_val);
                sum_kern += kern_ms;
            }
            float avg_kern = sum_kern / runs;
            double gcs = (double)total_chars / (avg_kern * 1e6);

            printf("%7d  %5d  %10.3f  %10.1f\n", B, L, avg_kern, gcs);

            delete[] raw_concat; delete[] offsets; delete[] results;
            eng.destroy();
        }
    }
}


void bench_prefix_throughput() {
    printf("\n=== Prefix Kernel Throughput ===\n");
    printf("%7s  %10s  %10s  %10s\n", "B", "L", "kern(ms)", "Gc/s");
    printf("------  ----------  ----------  ----------\n");

    uint8_t char_compose[6], raw_char_map[256], accept[2], monoid_compose[4];
    int M, sigma_ext; uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    int configs[][2] = {{1, 1000000}, {1, 10000000}, {4, 1000000}, {16, 500000}, {128, 100000}};
    int n_configs = 5;

    for (int ci = 0; ci < n_configs; ci++) {
        int B = configs[ci][0], L = configs[ci][1];
        long long total_chars = (long long)B * L;

        uint8_t *raw_concat = new uint8_t[total_chars];
        int *offsets = new int[B + 1];
        srand(123);
        offsets[0] = 0;
        for (int i = 0; i < B; i++) offsets[i + 1] = offsets[i] + L;
        for (long long i = 0; i < total_chars; i++)
            raw_concat[i] = (rand() % 2) ? 'a' : 'b';

        MonoidBatchEngine eng;
        eng.init(M, sigma_ext, identity,
                 char_compose, raw_char_map, accept, monoid_compose,
                 (int)total_chars + 1, B + 1);

        int *results = new int[B];
        float kern_ms, total_ms_val;

        for (int w = 0; w < 3; w++)
            eng.dispatch_prefix(raw_concat, offsets, results, B, (int)total_chars,
                                &kern_ms, &total_ms_val);

        float sum_kern = 0;
        int runs = 10;
        for (int r = 0; r < runs; r++) {
            eng.dispatch_prefix(raw_concat, offsets, results, B, (int)total_chars,
                                &kern_ms, &total_ms_val);
            sum_kern += kern_ms;
        }
        float avg_kern = sum_kern / runs;
        double gcs = (double)total_chars / (avg_kern * 1e6);

        printf("%7d  %10d  %10.3f  %10.1f\n", B, L, avg_kern, gcs);

        delete[] raw_concat; delete[] offsets; delete[] results;
        eng.destroy();
    }
}


int main() {
    printf("monoid_batch standalone tests\n");
    printf("=============================\n\n");

    test_batch_correctness();
    test_batch_large_random();
    test_prefix_correctness();
    test_prefix_multi_string();

    printf("\n%d/%d tests passed\n", tests_passed, tests_total);

    bench_throughput();
    bench_prefix_throughput();

    return (tests_passed == tests_total) ? 0 : 1;
}

#endif  // BUILD_LIB
