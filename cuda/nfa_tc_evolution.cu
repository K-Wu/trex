/*
 * nfa_tc_evolution.cu -- NFA Tensor Core State-Vector Evolution (N=64)
 *
 * NFA transition matrices are general Boolean matrices (not permutations).
 * Composition is irreducibly matrix multiply — tensor cores are the right tool.
 *
 * Uses tiled WMMA for 64x64 matmul: 4x4 blocks of 16x16 MMA operations.
 * Per position: 32 MMA calls (4 row-tiles x 4 K-tiles x 2 chars).
 * V2 register-level select with NFA threshold (clamp to {0,1}).
 *
 * FP16 accumulator layout (probed on SM 9.0):
 *   row = lane/4 + ((i>>1)&1)*8
 *   col = (lane%4)*2 + (i&1) + (i>>2)*8
 *
 * Threading model:
 *   Each warp handles 16 columns (strings) — one 16-wide WMMA tile
 *   4 warps per block -> 64 strings per block
 *   State vectors are 64-dimensional (N=64)
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

constexpr int N = 64;
constexpr int NTILES = N / 16;           // 4 tiles per dimension
constexpr int TILE = 16;
constexpr int N_ELEMS = N * N;           // 4096
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;  // 128
constexpr int COLS_PER_BLOCK = WARPS_PER_BLOCK * TILE;   // 64 strings

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// ---- NFA Binary Kernel V2 (N=64, tiled WMMA, register select) -----------
//
// Shared memory layout:
//   T0_sh[64x64]     8192B  (half, transition matrix char 0, row-major)
//   T1_sh[64x64]     8192B  (half, transition matrix char 1, row-major)
//   S_sh[4x64x16]    8192B  (half, state tiles, 4 warps, col-major)
//   Total:           24576B  (24 KB)

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
nfa_tc_binary_v2_kernel(
    const half    *__restrict__ T0_global,
    const half    *__restrict__ T1_global,
    const uint8_t *__restrict__ input,
    const half    *__restrict__ accept_mask,
    const half    *__restrict__ start_vec,
    int B, int B_padded, int L,
    int *__restrict__ results
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * COLS_PER_BLOCK;
    int warp_col_start = block_col_start + warp_in_block * TILE;

    extern __shared__ char smem_raw[];
    half *T0_sh = (half *)smem_raw;
    half *T1_sh = T0_sh + N_ELEMS;
    half *S_base = T1_sh + N_ELEMS;
    half *S_sh = S_base + warp_in_block * N * TILE;

    for (int e = threadIdx.x; e < N_ELEMS; e += blockDim.x) {
        T0_sh[e] = T0_global[e];
        T1_sh[e] = T1_global[e];
    }
    for (int e = lane; e < N * TILE; e += WARP_SIZE) {
        int row = e % N;
        S_sh[e] = start_vec[row];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_T0_tile, frag_T1_tile;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_S[NTILES];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc0, frag_acc1;

    half h_zero = __float2half(0.0f);
    half h_one  = __float2half(1.0f);

    int col_lo = (lane & 3) * 2;
    int col_hi = col_lo + 8;
    int row_lo = lane >> 2;
    int row_hi = row_lo + 8;

    for (int t = 0; t < L; t++) {
        for (int kt = 0; kt < NTILES; kt++)
            wmma::load_matrix_sync(frag_S[kt], S_sh + kt * TILE, N);

        uint8_t my_ch = 2;
        if (lane < TILE) {
            int sid = warp_col_start + lane;
            if (sid < B_padded) my_ch = input[t * B_padded + sid];
        }
        uint8_t ch0 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_lo);
        uint8_t ch1 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_lo + 1);
        uint8_t ch2 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_hi);
        uint8_t ch3 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_hi + 1);

        for (int mt = 0; mt < NTILES; mt++) {
            wmma::fill_fragment(frag_acc0, h_zero);
            wmma::fill_fragment(frag_acc1, h_zero);

            for (int kt = 0; kt < NTILES; kt++) {
                wmma::load_matrix_sync(frag_T0_tile,
                    T0_sh + mt * TILE * N + kt * TILE, N);
                wmma::load_matrix_sync(frag_T1_tile,
                    T1_sh + mt * TILE * N + kt * TILE, N);
                wmma::mma_sync(frag_acc0, frag_T0_tile, frag_S[kt], frag_acc0);
                wmma::mma_sync(frag_acc1, frag_T1_tile, frag_S[kt], frag_acc1);
            }

            int row_off = mt * TILE;

            #define NFA_REGSEL(EI, COL, ROW, CH) \
                if ((CH) < 2) { \
                    half v = ((CH) == 0) ? frag_acc0.x[EI] : frag_acc1.x[EI]; \
                    S_sh[(COL) * N + (ROW)] = __hgt(v, h_zero) ? h_one : h_zero; \
                }

            NFA_REGSEL(0, col_lo,     row_off + row_lo, ch0)
            NFA_REGSEL(1, col_lo + 1, row_off + row_lo, ch1)
            NFA_REGSEL(2, col_lo,     row_off + row_hi, ch0)
            NFA_REGSEL(3, col_lo + 1, row_off + row_hi, ch1)
            NFA_REGSEL(4, col_hi,     row_off + row_lo, ch2)
            NFA_REGSEL(5, col_hi + 1, row_off + row_lo, ch3)
            NFA_REGSEL(6, col_hi,     row_off + row_hi, ch2)
            NFA_REGSEL(7, col_hi + 1, row_off + row_hi, ch3)

            #undef NFA_REGSEL
        }
        __syncwarp();
    }

    for (int col = lane; col < TILE; col += WARP_SIZE) {
        int string_id = warp_col_start + col;
        if (string_id >= B) continue;
        int accepted = 0;
        for (int r = 0; r < N; r++) {
            if (__hgt(S_sh[col * N + r], h_zero) &&
                __hgt(accept_mask[r], h_zero)) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}


// ---- NFA Binary Kernel V3 (T fragments cached in registers) --------------
//
// Same as V2 but loads all 32 T fragments (16 for T0, 16 for T1) into
// registers ONCE before the L-loop. Eliminates 32 smem loads per position
// at the cost of higher register pressure (~170 regs, limiting occupancy).
//
// Shared memory: T0(8192) + T1(8192) + S(8192) = 24 KB (same as V2,
// T stays in smem for the initial load but is only read once).

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
nfa_tc_binary_v3_kernel(
    const half    *__restrict__ T0_global,
    const half    *__restrict__ T1_global,
    const uint8_t *__restrict__ input,
    const half    *__restrict__ accept_mask,
    const half    *__restrict__ start_vec,
    int B, int B_padded, int L,
    int *__restrict__ results
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * COLS_PER_BLOCK;
    int warp_col_start = block_col_start + warp_in_block * TILE;

    extern __shared__ char smem_raw[];
    half *T0_sh = (half *)smem_raw;
    half *T1_sh = T0_sh + N_ELEMS;
    half *S_base = T1_sh + N_ELEMS;
    half *S_sh = S_base + warp_in_block * N * TILE;

    for (int e = threadIdx.x; e < N_ELEMS; e += blockDim.x) {
        T0_sh[e] = T0_global[e];
        T1_sh[e] = T1_global[e];
    }
    for (int e = lane; e < N * TILE; e += WARP_SIZE) {
        int row = e % N;
        S_sh[e] = start_vec[row];
    }
    __syncthreads();

    // Cache ALL T fragments in registers (32 total = 16 per T matrix)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_T0[NTILES][NTILES];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_T1[NTILES][NTILES];

    for (int mt = 0; mt < NTILES; mt++)
        for (int kt = 0; kt < NTILES; kt++) {
            wmma::load_matrix_sync(frag_T0[mt][kt], T0_sh + mt * TILE * N + kt * TILE, N);
            wmma::load_matrix_sync(frag_T1[mt][kt], T1_sh + mt * TILE * N + kt * TILE, N);
        }

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_S[NTILES];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc0, frag_acc1;

    half h_zero = __float2half(0.0f);
    half h_one  = __float2half(1.0f);

    int col_lo = (lane & 3) * 2;
    int col_hi = col_lo + 8;
    int row_lo = lane >> 2;
    int row_hi = row_lo + 8;

    for (int t = 0; t < L; t++) {
        for (int kt = 0; kt < NTILES; kt++)
            wmma::load_matrix_sync(frag_S[kt], S_sh + kt * TILE, N);

        uint8_t my_ch = 2;
        if (lane < TILE) {
            int sid = warp_col_start + lane;
            if (sid < B_padded) my_ch = input[t * B_padded + sid];
        }
        uint8_t ch0 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_lo);
        uint8_t ch1 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_lo + 1);
        uint8_t ch2 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_hi);
        uint8_t ch3 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_hi + 1);

        for (int mt = 0; mt < NTILES; mt++) {
            wmma::fill_fragment(frag_acc0, h_zero);
            wmma::fill_fragment(frag_acc1, h_zero);

            for (int kt = 0; kt < NTILES; kt++) {
                wmma::mma_sync(frag_acc0, frag_T0[mt][kt], frag_S[kt], frag_acc0);
                wmma::mma_sync(frag_acc1, frag_T1[mt][kt], frag_S[kt], frag_acc1);
            }

            int row_off = mt * TILE;

            #define NFA_REGSEL3(EI, COL, ROW, CH) \
                if ((CH) < 2) { \
                    half v = ((CH) == 0) ? frag_acc0.x[EI] : frag_acc1.x[EI]; \
                    S_sh[(COL) * N + (ROW)] = __hgt(v, h_zero) ? h_one : h_zero; \
                }

            NFA_REGSEL3(0, col_lo,     row_off + row_lo, ch0)
            NFA_REGSEL3(1, col_lo + 1, row_off + row_lo, ch1)
            NFA_REGSEL3(2, col_lo,     row_off + row_hi, ch0)
            NFA_REGSEL3(3, col_lo + 1, row_off + row_hi, ch1)
            NFA_REGSEL3(4, col_hi,     row_off + row_lo, ch2)
            NFA_REGSEL3(5, col_hi + 1, row_off + row_lo, ch3)
            NFA_REGSEL3(6, col_hi,     row_off + row_hi, ch2)
            NFA_REGSEL3(7, col_hi + 1, row_off + row_hi, ch3)

            #undef NFA_REGSEL3
        }
        __syncwarp();
    }

    for (int col = lane; col < TILE; col += WARP_SIZE) {
        int string_id = warp_col_start + col;
        if (string_id >= B) continue;
        int accepted = 0;
        for (int r = 0; r < N; r++) {
            if (__hgt(S_sh[col * N + r], h_zero) &&
                __hgt(accept_mask[r], h_zero)) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}


// ---- Engine Struct --------------------------------------------------------

struct NFATCEngine {
    int n_states;
    int sigma;
    int max_B;
    int max_L;
    int B_padded_max;
    int kernel_variant;    // 2=V2(reload T), 3=V3(T in registers)

    half    *d_T;
    half    *d_accept;
    half    *d_start_vec;
    uint8_t *d_input;
    int     *d_results;

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;
    bool initialized;

    void init(const float *T_matrices, const float *accept_mask,
              const float *start_vec, int n, int sig, int maxB, int maxL) {
        n_states = n;
        sigma = sig;
        max_B = maxB;
        max_L = maxL;
        kernel_variant = 3;
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

    int dispatch(const uint8_t *h_input, int B, int L_val, int B_padded,
                 int *h_results, float *kernel_ms, float *total_ms) {
        if (!initialized) return -1;
        if (sigma != 2 || n_states != N) return -2;

        CHECK_CUDA(cudaEventRecord(ev_start));

        size_t input_bytes = (size_t)L_val * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        int n_blocks = B_padded / COLS_PER_BLOCK;
        int smem = (2 * N_ELEMS + WARPS_PER_BLOCK * N * TILE) * (int)sizeof(half);

        if (kernel_variant == 3) {
            nfa_tc_binary_v3_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
                d_T, d_T + N_ELEMS,
                d_input, d_accept, d_start_vec,
                B, B_padded, L_val, d_results);
        } else {
            nfa_tc_binary_v2_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
                d_T, d_T + N_ELEMS,
                d_input, d_accept, d_start_vec,
                B, B_padded, L_val, d_results);
        }
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

static NFATCEngine g_nfa_engine = {};


// ---- C API ----------------------------------------------------------------

extern "C" {

int nfa_tc_engine_init(
    const float *T_matrices,
    const float *accept_mask,
    const float *start_vec,
    int n, int sigma,
    int max_B, int max_L)
{
    if (g_nfa_engine.initialized) g_nfa_engine.destroy();
    g_nfa_engine.init(T_matrices, accept_mask, start_vec, n, sigma, max_B, max_L);
    return 0;
}

int nfa_tc_engine_dispatch(
    const uint8_t *raw_concat,
    const int     *offsets,
    int           *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms)
{
    if (!g_nfa_engine.initialized) return -1;

    int L = 0;
    for (int i = 0; i < B; i++) {
        int len = offsets[i + 1] - offsets[i];
        if (len > L) L = len;
    }
    if (L == 0) {
        std::vector<half> h_accept(g_nfa_engine.n_states);
        CHECK_CUDA(cudaMemcpy(h_accept.data(), g_nfa_engine.d_accept,
                              g_nfa_engine.n_states * sizeof(half), cudaMemcpyDeviceToHost));
        std::vector<half> h_start(g_nfa_engine.n_states);
        CHECK_CUDA(cudaMemcpy(h_start.data(), g_nfa_engine.d_start_vec,
                              g_nfa_engine.n_states * sizeof(half), cudaMemcpyDeviceToHost));
        for (int i = 0; i < B; i++) {
            results[i] = 0;
            for (int r = 0; r < g_nfa_engine.n_states; r++) {
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
    int identity_idx = g_nfa_engine.sigma;

    size_t input_size = (size_t)L * B_padded;
    std::vector<uint8_t> input(input_size, (uint8_t)identity_idx);
    for (int b = 0; b < B; b++) {
        int str_start = offsets[b];
        int str_len = offsets[b + 1] - str_start;
        for (int t = 0; t < str_len && t < L; t++)
            input[t * B_padded + b] = raw_concat[str_start + t];
    }

    return g_nfa_engine.dispatch(input.data(), B, L, B_padded,
                                  results, kernel_ms, total_ms);
}

void nfa_tc_engine_destroy(void) {
    g_nfa_engine.destroy();
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

static bool simulate_nfa_sequential(
    int n, const float *start_vec, const float *accept_mask,
    const float *trans, int sigma,
    const uint8_t *chars, int L_val)
{
    std::vector<float> state(n), new_state(n);
    for (int i = 0; i < n; i++) state[i] = start_vec[i];

    for (int t = 0; t < L_val; t++) {
        int c = chars[t];
        if (c >= sigma) continue;
        const float *T = trans + c * n * n;
        std::fill(new_state.begin(), new_state.end(), 0.0f);
        for (int row = 0; row < n; row++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += T[row * n + k] * state[k];
            new_state[row] = (sum > 0.5f) ? 1.0f : 0.0f;
        }
        state = new_state;
    }

    for (int r = 0; r < n; r++)
        if (state[r] > 0.5f && accept_mask[r] > 0.5f) return true;
    return false;
}

// NFA: "contains >= 3 a's"
// States 0-3 active (padded to 64), start=0, accept=3
// 'a' (0): state i -> {i, i+1} for i<3, state 3 -> {3}
// 'b' (1): identity
static void build_nfa_3a(float *trans, float *accept, float *start_vec) {
    memset(trans, 0, 2 * N * N * sizeof(float));
    memset(accept, 0, N * sizeof(float));
    memset(start_vec, 0, N * sizeof(float));

    start_vec[0] = 1.0f;
    accept[3] = 1.0f;

    float *Ta = trans;
    float *Tb = trans + N_ELEMS;

    // 'a': i -> {i, i+1} for i in [0,2]; 3 -> {3}
    for (int i = 0; i < 3; i++) {
        Ta[i * N + i] = 1.0f;         // i -> i
        Ta[(i + 1) * N + i] = 1.0f;   // i -> i+1
    }
    Ta[3 * N + 3] = 1.0f;

    // 'b': identity for active states
    for (int i = 0; i < 4; i++)
        Tb[i * N + i] = 1.0f;

    // Padding states 4-63: self-loop on both chars
    for (int s = 4; s < N; s++) {
        Ta[s * N + s] = 1.0f;
        Tb[s * N + s] = 1.0f;
    }
}

static void test_nfa_basic() {
    printf("\n--- test_nfa_basic (>=3 a's NFA) ---\n");

    float trans[2 * N_ELEMS], accept[N], start_vec[N];
    build_nfa_3a(trans, accept, start_vec);

    nfa_tc_engine_init(trans, accept, start_vec, N, 2, 64, 256);

    uint8_t all_chars[] = {
        0,                // "a"     [0]
        0, 0,             // "aa"    [1-2]
        0, 0, 0,          // "aaa"   [3-5]
        1,                // "b"     [6]
        1, 0, 1,          // "bab"   [7-9]
        0, 0, 0, 1,       // "aaab"  [10-13]
        1, 0, 0, 0,       // "baaa"  [14-17]
        0, 1, 0, 1,       // "abab"  [18-21]
        0, 0, 1, 0, 0     // "aabaa" [22-26]
    };
    int offsets[] = {0, 0, 1, 3, 6, 7, 10, 14, 18, 22, 27};
    bool expected[] = {
        false,  // ""     -> {0},         accept={3} -> reject
        false,  // "a"    -> {0,1}                   -> reject
        false,  // "aa"   -> {0,1,2}                 -> reject
        true,   // "aaa"  -> {0,1,2,3}               -> accept
        false,  // "b"    -> {0}                      -> reject
        false,  // "bab"  -> {0,1}                    -> reject
        true,   // "aaab" -> {0,1,2,3}               -> accept
        true,   // "baaa" -> {0,1,2,3}               -> accept
        false,  // "abab" -> {0,1,2}                 -> reject
        true,   // "aabaa"-> {0,1,2,3}               -> accept
    };
    const char *names[] = {
        "empty", "a", "aa", "aaa", "b", "bab", "aaab", "baaa", "abab", "aabaa"
    };
    int B = 10;

    int results[10];
    float km, tm;
    nfa_tc_engine_dispatch(all_chars, offsets, results, B, 27, &km, &tm);

    for (int i = 0; i < B; i++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s -> %s", names[i], expected[i] ? "accept" : "reject");
        check(msg, results[i] == (expected[i] ? 1 : 0));
    }
    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);

    nfa_tc_engine_destroy();
}

static void test_nfa_random() {
    printf("\n--- test_nfa_random (4096 x 256, random NFA) ---\n");

    float trans[2 * N_ELEMS], accept[N], start_vec[N];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));

    start_vec[0] = 1.0f;
    srand(42);

    // Random accept states
    for (int s = 0; s < N; s++)
        if (rand() % 4 == 0) accept[s] = 1.0f;

    // Random NFA: each state has 1-3 successors per character
    for (int c = 0; c < 2; c++) {
        float *T = trans + c * N_ELEMS;
        for (int src = 0; src < N; src++) {
            int n_succ = 1 + rand() % 3;
            for (int s = 0; s < n_succ; s++) {
                int dst = rand() % N;
                T[dst * N + src] = 1.0f;
            }
        }
    }

    nfa_tc_engine_init(trans, accept, start_vec, N, 2, 4096, 256);

    int B = 4096, L_val = 256;
    srand(77);
    std::vector<uint8_t> chars_flat;
    std::vector<int> offsets(B + 1, 0);
    for (int b = 0; b < B; b++) {
        int len = 1 + rand() % L_val;
        offsets[b + 1] = offsets[b] + len;
        for (int t = 0; t < len; t++)
            chars_flat.push_back(rand() % 2);
    }
    int total = offsets[B];

    std::vector<int> results(B);
    float km, tm;
    nfa_tc_engine_dispatch(chars_flat.data(), offsets.data(), results.data(),
                           B, total, &km, &tm);

    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        int len = offsets[b + 1] - offsets[b];
        bool seq = simulate_nfa_sequential(N, start_vec, accept, trans, 2,
                                            chars_flat.data() + offsets[b], len);
        if (results[b] != (seq ? 1 : 0)) mismatches++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "random_nfa B=%d (%d mismatches)", B, mismatches);
    check(msg, mismatches == 0);
    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);

    nfa_tc_engine_destroy();
}

static void test_nfa_long() {
    printf("\n--- test_nfa_long (64 x 50000, NFA invariant) ---\n");

    float trans[2 * N_ELEMS], accept[N], start_vec[N];
    build_nfa_3a(trans, accept, start_vec);

    int B = 64, L_val = 50000;
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

    nfa_tc_engine_init(trans, accept, start_vec, N, 2, B_padded, L_val);

    srand(99);
    size_t input_size = (size_t)L_val * B_padded;
    std::vector<uint8_t> input(input_size, 2);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < L_val; t++)
            input[t * B_padded + b] = rand() % 2;

    std::vector<int> results(B);
    float km, tm;
    g_nfa_engine.dispatch(input.data(), B, L_val, B_padded, results.data(), &km, &tm);

    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        std::vector<uint8_t> str_chars(L_val);
        for (int t = 0; t < L_val; t++)
            str_chars[t] = input[t * B_padded + b];
        bool seq = simulate_nfa_sequential(N, start_vec, accept, trans, 2,
                                            str_chars.data(), L_val);
        if (results[b] != (seq ? 1 : 0)) mismatches++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "nfa_long B=%d L=%d (%d mismatches)", B, L_val, mismatches);
    check(msg, mismatches == 0);
    printf("  Kernel: %.3f ms, Total: %.3f ms (%.1f Gc/s)\n",
           km, tm, (double)B * L_val / (km * 1e6));

    nfa_tc_engine_destroy();
}

static void bench_throughput() {

    float trans[2 * N_ELEMS], accept[N], start_vec[N];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));
    start_vec[0] = 1.0f;
    accept[N - 1] = 1.0f;

    // Random sparse NFA for benchmark
    srand(42);
    for (int c = 0; c < 2; c++) {
        float *T = trans + c * N_ELEMS;
        for (int src = 0; src < N; src++) {
            int n_succ = 1 + rand() % 3;
            for (int s = 0; s < n_succ; s++)
                T[(rand() % N) * N + src] = 1.0f;
        }
    }

    int batch_sizes[] = {1024, 4096, 16384, 65536, 262144};
    int lengths[]     = {128, 512, 2048};
    int n_batches = 5, n_lengths = 3;

    for (int variant = 2; variant <= 3; variant++) {
        printf("\n=== NFA TC N=64 V%d Throughput Benchmark (binary) ===\n", variant);
        printf("  %8s  %6s  |  %8s  %8s  %8s\n", "B", "L", "Gc/s", "TFLOPS", "kern_ms");
        printf("  %s\n", "---------------------------------------------------");

        for (int bi = 0; bi < n_batches; bi++) {
            for (int li = 0; li < n_lengths; li++) {
                int B = batch_sizes[bi];
                int L_val = lengths[li];
                int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

                nfa_tc_engine_init(trans, accept, start_vec, N, 2, B_padded, L_val);
                g_nfa_engine.kernel_variant = variant;

                srand(42);
                size_t input_size = (size_t)L_val * B_padded;
                std::vector<uint8_t> input(input_size, 2);
                for (int b = 0; b < B; b++)
                    for (int t = 0; t < L_val; t++)
                        input[t * B_padded + b] = rand() % 2;

                std::vector<int> results(B);

                for (int w = 0; w < 3; w++)
                    g_nfa_engine.dispatch(input.data(), B, L_val, B_padded,
                                          results.data(), nullptr, nullptr);

                int iters = 10;
                float total_km = 0;
                for (int it = 0; it < iters; it++) {
                    float km;
                    g_nfa_engine.dispatch(input.data(), B, L_val, B_padded,
                                          results.data(), &km, nullptr);
                    total_km += km;
                }
                float avg_km = total_km / iters;
                double gchs = (double)B * L_val / (avg_km * 1e6);
                double mma_per_pos = 32.0;
                double flops_per_mma = 16.0 * 16.0 * 16.0 * 2.0;
                double total_flops = (double)B_padded / TILE * mma_per_pos * flops_per_mma * L_val;
                double tflops = total_flops / (avg_km * 1e9);

                printf("  %8d  %6d  |  %8.1f  %8.1f  %8.3f\n",
                       B, L_val, gchs, tflops, avg_km);

                nfa_tc_engine_destroy();
            }
        }
    }
}


int main() {
    printf("=== NFA TC Evolution Engine (N=64) ===\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 7) {
        printf("Need SM >= 7.0 for WMMA\n");
        return 1;
    }

    test_nfa_basic();
    test_nfa_random();
    test_nfa_long();

    // V2 vs V3 cross-validation
    {
        printf("\n--- test_v2_v3_cross (4096 x 256, random NFA) ---\n");
        float tr2[2 * N_ELEMS], ac2[N], sv2[N];
        memset(tr2, 0, sizeof(tr2)); memset(ac2, 0, sizeof(ac2)); memset(sv2, 0, sizeof(sv2));
        sv2[0] = 1.0f;
        srand(55);
        for (int s = 0; s < N; s++) if (rand() % 4 == 0) ac2[s] = 1.0f;
        for (int c = 0; c < 2; c++) {
            float *T = tr2 + c * N_ELEMS;
            for (int src = 0; src < N; src++)
                for (int ns = 0; ns < 1 + rand() % 3; ns++)
                    T[(rand() % N) * N + src] = 1.0f;
        }
        int B2 = 4096, L2 = 256;
        int B2_p = ((B2 + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
        srand(66);
        size_t isz = (size_t)L2 * B2_p;
        std::vector<uint8_t> inp(isz, 2);
        for (int b = 0; b < B2; b++)
            for (int t = 0; t < L2; t++)
                inp[t * B2_p + b] = rand() % 2;
        std::vector<int> r2(B2), r3(B2);
        nfa_tc_engine_init(tr2, ac2, sv2, N, 2, B2_p, L2);
        g_nfa_engine.kernel_variant = 2;
        g_nfa_engine.dispatch(inp.data(), B2, L2, B2_p, r2.data(), nullptr, nullptr);
        g_nfa_engine.kernel_variant = 3;
        g_nfa_engine.dispatch(inp.data(), B2, L2, B2_p, r3.data(), nullptr, nullptr);
        int mm = 0;
        for (int b = 0; b < B2; b++) if (r2[b] != r3[b]) mm++;
        char msg[128];
        snprintf(msg, sizeof(msg), "v2_v3_cross B=%d L=%d (%d mismatches)", B2, L2, mm);
        check(msg, mm == 0);
        nfa_tc_engine_destroy();
    }

    printf("\n=== Results: %d / %d passed ===\n", g_pass, g_tests);
    if (g_pass != g_tests) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }

    bench_throughput();

    return 0;
}

#endif  // BUILD_LIB
