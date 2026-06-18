/*
 * fp16_evolution.cu -- FP16 Tensor Core State-Vector Evolution
 *
 * Processes B strings simultaneously using FP16 WMMA MMA operations.
 * Maintains a state matrix S[N][B] where each column is a DFA state vector.
 * At each string position, applies transition matrices via WMMA 16x16 FP16 MMA.
 *
 * Key advantage over INT8 evolution: DFA permutation matrices preserve the
 * {0.0, 1.0} invariant in FP16 — no threshold or INT32→INT8 conversion needed.
 * The select loop directly assigns accumulator values to the state tile.
 *
 * V1: store_matrix_sync → shared memory → select → S_sh (smem path)
 *
 * Threading model:
 *   Each warp handles 16 consecutive columns (strings) -- one WMMA 16x16 tile
 *   4 warps per block -> 64 strings per block (COLS_PER_BLOCK = 64)
 *   State tiles live in shared memory across the entire L-step loop
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace nvcuda;

// ---- Configuration --------------------------------------------------------

constexpr int TILE = 16;
constexpr int TILE_ELEMS = TILE * TILE;  // 256
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;  // 128 threads
constexpr int COLS_PER_BLOCK = WARPS_PER_BLOCK * TILE;   // 64 strings

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// ---- FP16 Binary Kernel (|Sigma|=2, N=16) --------------------------------
//
// Shared memory layout:
//   T0_sh[16×16]      512B   (half, transition matrix char 0)
//   T1_sh[16×16]      512B   (half, transition matrix char 1)
//   S_sh[4×16×16]     2048B  (half, state tiles for 4 warps, col-major)
//   acc0_sh[4×16×16]  2048B  (half, accumulator buffer 0, row-major)
//   acc1_sh[4×16×16]  2048B  (half, accumulator buffer 1, row-major)
//   Total:             7168B
//
// S is stored in COL-MAJOR layout in shared memory:
//   S_sh[col * 16 + row] = S[state_row][string_col]
// This allows direct loading as wmma::matrix_b with col_major, ldm=16.
// After MMA, the half accumulator is stored row-major, then selected back.

__global__ void fp16_evolution_binary_kernel(
    const half    *__restrict__ T0_global,     // [16*16] transition for char 0
    const half    *__restrict__ T1_global,     // [16*16] transition for char 1
    const uint8_t *__restrict__ input,         // [L][B_padded] position-contiguous
    const half    *__restrict__ accept_mask,   // [16]
    const half    *__restrict__ start_vec,     // [16] initial state (one-hot)
    int B, int B_padded, int L,
    int *__restrict__ results
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * COLS_PER_BLOCK;
    int warp_col_start = block_col_start + warp_in_block * TILE;

    extern __shared__ char smem_raw[];
    half *T0_sh   = (half *)smem_raw;
    half *T1_sh   = T0_sh + TILE_ELEMS;
    half *S_base  = T1_sh + TILE_ELEMS;
    half *acc0_base = S_base + WARPS_PER_BLOCK * TILE_ELEMS;
    half *acc1_base = acc0_base + WARPS_PER_BLOCK * TILE_ELEMS;

    half *S_sh    = S_base + warp_in_block * TILE_ELEMS;
    half *acc0_sh = acc0_base + warp_in_block * TILE_ELEMS;
    half *acc1_sh = acc1_base + warp_in_block * TILE_ELEMS;

    // Load T0, T1 into shared memory (all threads cooperate)
    for (int e = threadIdx.x; e < TILE_ELEMS; e += blockDim.x) {
        T0_sh[e] = T0_global[e];
        T1_sh[e] = T1_global[e];
    }

    // Initialize S to start vector (col-major: S_sh[col*16 + row])
    half h_zero = __float2half(0.0f);
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        S_sh[e] = start_vec[row];
    }
    __syncthreads();

    // Fragment declarations — ALL FP16
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_T0, frag_T1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_S;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc0, frag_acc1;

    // Load T fragments (constant across all positions)
    wmma::load_matrix_sync(frag_T0, T0_sh, TILE);
    wmma::load_matrix_sync(frag_T1, T1_sh, TILE);

    for (int t = 0; t < L; t++) {
        wmma::load_matrix_sync(frag_S, S_sh, TILE);

        // FP16 MMA: T0 × S → acc0, T1 × S → acc1
        wmma::fill_fragment(frag_acc0, h_zero);
        wmma::mma_sync(frag_acc0, frag_T0, frag_S, frag_acc0);
        wmma::fill_fragment(frag_acc1, h_zero);
        wmma::mma_sync(frag_acc1, frag_T1, frag_S, frag_acc1);

        // Store accumulators to smem (row-major)
        wmma::store_matrix_sync(acc0_sh, frag_acc0, TILE, wmma::mem_row_major);
        wmma::store_matrix_sync(acc1_sh, frag_acc1, TILE, wmma::mem_row_major);
        __syncwarp();

        // Per-column select: NO threshold, NO type conversion
        // acc is row-major: acc[row * 16 + col]
        // S_sh is col-major: S_sh[col * 16 + row]
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
            int col = e / TILE;
            int row = e % TILE;
            int string_id = warp_col_start + col;
            uint8_t ch = 2;
            if (string_id < B_padded)
                ch = input[t * B_padded + string_id];
            if (ch < 2) {
                S_sh[e] = (ch == 0) ? acc0_sh[row * TILE + col]
                                    : acc1_sh[row * TILE + col];
            }
        }
        __syncwarp();
    }

    // Accept check
    for (int col = lane; col < TILE; col += WARP_SIZE) {
        int string_id = warp_col_start + col;
        if (string_id >= B) continue;
        int accepted = 0;
        for (int r = 0; r < TILE; r++) {
            if (__hgt(S_sh[col * TILE + r], h_zero) &&
                __hgt(accept_mask[r], h_zero)) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}


// ---- Engine Struct --------------------------------------------------------

struct FP16Engine {
    int N;
    int sigma;
    int max_B;
    int max_L;
    int B_padded_max;

    // Device memory
    half    *d_T;          // [sigma * N * N] transition matrices
    half    *d_accept;     // [N]
    half    *d_start_vec;  // [N]
    uint8_t *d_input;      // [max_L * B_padded_max]
    int     *d_results;    // [max_B]

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;
    bool initialized;

    void init(const float *T_matrices, const float *accept_mask,
              const float *start_vec, int n, int sig, int maxB, int maxL) {
        N = n;
        sigma = sig;
        max_B = maxB;
        max_L = maxL;
        B_padded_max = ((maxB + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

        int T_count = sig * n * n;
        std::vector<half> h_T(T_count);
        for (int i = 0; i < T_count; i++)
            h_T[i] = __float2half(T_matrices[i]);

        std::vector<half> h_accept(n);
        for (int i = 0; i < n; i++)
            h_accept[i] = __float2half(accept_mask[i]);

        std::vector<half> h_start(n);
        for (int i = 0; i < n; i++)
            h_start[i] = __float2half(start_vec[i]);

        CHECK_CUDA(cudaMalloc(&d_T, T_count * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_accept, n * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_start_vec, n * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_input, (size_t)maxL * B_padded_max));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)maxB * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_T, h_T.data(), T_count * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, h_accept.data(), n * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_start_vec, h_start.data(), n * sizeof(half), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

        initialized = true;
    }

    void destroy() {
        if (!initialized) return;
        cudaFree(d_T);
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
        if (sigma != 2 || N != TILE) return -2;

        CHECK_CUDA(cudaEventRecord(ev_start));

        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        int n_blocks = B_padded / COLS_PER_BLOCK;
        // smem: T0(512) + T1(512) + S(2048) + acc0(2048) + acc1(2048) = 7168
        int smem = (2 + WARPS_PER_BLOCK * 3) * TILE_ELEMS * (int)sizeof(half);
        fp16_evolution_binary_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
            d_T, d_T + TILE_ELEMS,
            d_input, d_accept, d_start_vec,
            B, B_padded, L, d_results);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results, (size_t)B * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms) cudaEventElapsedTime(total_ms, ev_start, ev_stop);

        return 0;
    }
};

static FP16Engine g_fp16_engine = {};


// ---- C API ----------------------------------------------------------------

extern "C" {

int fp16_engine_device_check(void) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 7) return -2;
    return 0;
}

int fp16_engine_init(
    const float *T_matrices,
    const float *accept_mask,
    const float *start_vec,
    int N, int sigma,
    int max_B, int max_L)
{
    if (g_fp16_engine.initialized) g_fp16_engine.destroy();
    g_fp16_engine.init(T_matrices, accept_mask, start_vec, N, sigma, max_B, max_L);
    return 0;
}

int fp16_engine_dispatch(
    const uint8_t *raw_concat,
    const int     *offsets,
    int           *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms)
{
    if (!g_fp16_engine.initialized) return -1;

    int L = 0;
    for (int i = 0; i < B; i++) {
        int len = offsets[i + 1] - offsets[i];
        if (len > L) L = len;
    }
    if (L == 0) {
        // All empty strings: check if start state accepts
        for (int i = 0; i < B; i++) results[i] = 0;
        std::vector<half> h_accept(g_fp16_engine.N);
        CHECK_CUDA(cudaMemcpy(h_accept.data(), g_fp16_engine.d_accept,
                              g_fp16_engine.N * sizeof(half), cudaMemcpyDeviceToHost));
        std::vector<half> h_start(g_fp16_engine.N);
        CHECK_CUDA(cudaMemcpy(h_start.data(), g_fp16_engine.d_start_vec,
                              g_fp16_engine.N * sizeof(half), cudaMemcpyDeviceToHost));
        for (int i = 0; i < B; i++) {
            for (int r = 0; r < g_fp16_engine.N; r++) {
                if (__half2float(h_start[r]) > 0.0f && __half2float(h_accept[r]) > 0.0f) {
                    results[i] = 1;
                    break;
                }
            }
        }
        if (kernel_ms) *kernel_ms = 0.0f;
        if (total_ms) *total_ms = 0.0f;
        return 0;
    }

    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
    int identity_idx = g_fp16_engine.sigma;

    // Transpose raw_concat + offsets → position-contiguous layout
    size_t input_size = (size_t)L * B_padded;
    std::vector<uint8_t> input(input_size, (uint8_t)identity_idx);
    for (int b = 0; b < B; b++) {
        int str_start = offsets[b];
        int str_len = offsets[b + 1] - str_start;
        for (int t = 0; t < str_len && t < L; t++) {
            input[t * B_padded + b] = raw_concat[str_start + t];
        }
    }

    return g_fp16_engine.dispatch(input.data(), B, L, B_padded,
                                  results, kernel_ms, total_ms);
}

void fp16_engine_destroy(void) {
    g_fp16_engine.destroy();
}

}  // extern "C"


// ---- Built-in Tests and Benchmark -----------------------------------------

#ifndef BUILD_LIB

static int g_tests = 0, g_pass = 0;
static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) { g_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

// Sequential reference: apply transition matrices to a state vector
static bool simulate_sequential_fp16(
    int N, const float *start_vec, const float *accept_mask,
    const float *trans, int sigma,
    const uint8_t *chars, int L)
{
    std::vector<float> state(N), new_state(N);
    for (int i = 0; i < N; i++) state[i] = start_vec[i];

    for (int t = 0; t < L; t++) {
        int c = chars[t];
        if (c >= sigma) continue;  // identity
        const float *T = trans + c * N * N;
        std::fill(new_state.begin(), new_state.end(), 0.0f);
        for (int row = 0; row < N; row++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += T[row * N + k] * state[k];
            new_state[row] = (sum > 0.5f) ? 1.0f : 0.0f;
        }
        state = new_state;
    }

    for (int r = 0; r < N; r++)
        if (state[r] > 0.5f && accept_mask[r] > 0.5f) return true;
    return false;
}

static void test_basic_correctness() {
    printf("\n--- test_basic_correctness ---\n");

    int N = TILE;
    int sigma = 2;
    float trans[2 * TILE_ELEMS];
    float accept[TILE];
    float start_vec[TILE];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));

    accept[0] = 1.0f;
    start_vec[0] = 1.0f;

    // Padding states: self-loop
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1.0f;

    // 'a' (0): swap 0<->1
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1.0f;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1.0f;
    // 'b' (1): identity on 0,1
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1.0f;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1.0f;

    fp16_engine_init(trans, accept, start_vec, N, sigma, 64, 256);

    // Test strings via dispatch
    // "": accept (even a-count)
    // "a": reject (odd)
    // "aa": accept (even)
    // "b": accept (even)
    // "ab": reject (odd)
    // "ba": reject (odd)
    // "aab": accept (even)
    // "abba": accept (even)
    uint8_t all_chars[] = {0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0};
    int offsets[] = {0, 0, 1, 3, 4, 6, 8, 11, 15};
    bool expected[] = {true, false, true, true, false, false, true, true};
    const char *names[] = {
        "empty", "'a'", "'aa'", "'b'", "'ab'", "'ba'", "'aab'", "'abba'"
    };
    int B = 8;

    int results[8];
    float km, tm;
    int rc = fp16_engine_dispatch(all_chars, offsets, results, B, 15, &km, &tm);

    for (int i = 0; i < B; i++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s -> %s", names[i], expected[i] ? "accept" : "reject");
        check(msg, results[i] == (expected[i] ? 1 : 0));
    }
    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);

    fp16_engine_destroy();
}

static void test_large_random() {
    printf("\n--- test_large_random (4096 x 256) ---\n");

    int N = TILE;
    int sigma = 2;
    float trans[2 * TILE_ELEMS];
    float accept[TILE];
    float start_vec[TILE];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));

    accept[0] = 1.0f;
    start_vec[0] = 1.0f;
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1.0f;
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1.0f;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1.0f;
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1.0f;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1.0f;

    fp16_engine_init(trans, accept, start_vec, N, sigma, 4096, 256);

    int B = 4096, L = 256;
    srand(42);
    std::vector<uint8_t> chars_flat;
    std::vector<int> offsets(B + 1, 0);
    for (int b = 0; b < B; b++) {
        int len = 1 + rand() % L;
        offsets[b + 1] = offsets[b] + len;
        for (int t = 0; t < len; t++)
            chars_flat.push_back(rand() % 2);
    }
    int total = offsets[B];

    std::vector<int> results(B);
    float km, tm;
    fp16_engine_dispatch(chars_flat.data(), offsets.data(), results.data(),
                         B, total, &km, &tm);

    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        int len = offsets[b + 1] - offsets[b];
        bool seq = simulate_sequential_fp16(N, start_vec, accept, trans, sigma,
                                             chars_flat.data() + offsets[b], len);
        if (results[b] != (seq ? 1 : 0)) mismatches++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "large_random B=%d (%d mismatches)", B, mismatches);
    check(msg, mismatches == 0);
    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);

    fp16_engine_destroy();
}

static void test_fp16_invariant() {
    printf("\n--- test_fp16_invariant (64 x 100000) ---\n");

    int N = TILE;
    int sigma = 2;
    float trans[2 * TILE_ELEMS];
    float accept[TILE];
    float start_vec[TILE];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));

    accept[0] = 1.0f;
    start_vec[0] = 1.0f;
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1.0f;
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1.0f;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1.0f;
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1.0f;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1.0f;

    int B = 64, L = 100000;
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

    // Initialize engine via C API
    fp16_engine_init(trans, accept, start_vec, N, sigma, B_padded, L);

    srand(77);
    size_t input_size = (size_t)L * B_padded;
    std::vector<uint8_t> input(input_size, (uint8_t)sigma);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < L; t++)
            input[t * B_padded + b] = rand() % 2;

    // Run kernel via direct dispatch on the global engine
    std::vector<int> results(B);
    float km, tm;
    g_fp16_engine.dispatch(input.data(), B, L, B_padded, results.data(), &km, &tm);

    // Verify correctness = invariant holds (results match sequential)
    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        std::vector<uint8_t> str_chars(L);
        for (int t = 0; t < L; t++)
            str_chars[t] = input[t * B_padded + b];
        bool seq = simulate_sequential_fp16(N, start_vec, accept, trans, sigma,
                                             str_chars.data(), L);
        if (results[b] != (seq ? 1 : 0)) mismatches++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "fp16_invariant B=%d L=%d (%d mismatches)", B, L, mismatches);
    check(msg, mismatches == 0);
    printf("  Kernel: %.3f ms, Total: %.3f ms (%.1f Gc/s)\n",
           km, tm, (double)B * L / (km * 1e6));

    fp16_engine_destroy();
}

static void bench_throughput() {
    printf("\n=== FP16 TC Throughput Benchmark (binary, N=16) ===\n");

    int N = TILE;
    int sigma = 2;
    float trans[2 * TILE_ELEMS];
    float accept[TILE];
    float start_vec[TILE];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));
    accept[0] = 1.0f;
    start_vec[0] = 1.0f;
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1.0f;
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1.0f;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1.0f;
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1.0f;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1.0f;

    int batch_sizes[] = {1024, 4096, 16384, 65536, 262144};
    int lengths[]     = {128, 512, 2048};
    int n_batches = 5, n_lengths = 3;

    printf("  %8s  %6s  |  %8s  %8s\n", "B", "L", "Gc/s", "kern_ms");
    printf("  %s\n", "-------------------------------------");

    for (int bi = 0; bi < n_batches; bi++) {
        for (int li = 0; li < n_lengths; li++) {
            int B = batch_sizes[bi];
            int L = lengths[li];
            int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

            fp16_engine_init(trans, accept, start_vec, N, sigma, B_padded, L);

            srand(42);
            size_t input_size = (size_t)L * B_padded;
            std::vector<uint8_t> input(input_size, (uint8_t)sigma);
            for (int b = 0; b < B; b++)
                for (int t = 0; t < L; t++)
                    input[t * B_padded + b] = rand() % 2;

            std::vector<int> results(B);

            // Warmup
            for (int w = 0; w < 3; w++)
                g_fp16_engine.dispatch(input.data(), B, L, B_padded, results.data(), nullptr, nullptr);

            int iters = 20;
            float total_km = 0;
            for (int it = 0; it < iters; it++) {
                float km;
                g_fp16_engine.dispatch(input.data(), B, L, B_padded, results.data(), &km, nullptr);
                total_km += km;
            }
            float avg_km = total_km / iters;
            double gchs = (double)B * L / (avg_km * 1e6);

            printf("  %8d  %6d  |  %8.1f  %8.3f\n", B, L, gchs, avg_km);

            fp16_engine_destroy();
        }
    }
}


int main() {
    printf("=== FP16 TC Evolution Engine ===\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    int rc = fp16_engine_device_check();
    if (rc != 0) {
        printf("Device check failed: %d\n", rc);
        return 1;
    }

    test_basic_correctness();
    test_large_random();
    test_fp16_invariant();

    printf("\n=== Results: %d / %d passed ===\n", g_pass, g_tests);
    if (g_pass != g_tests) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }

    bench_throughput();

    return 0;
}

#endif  // BUILD_LIB
