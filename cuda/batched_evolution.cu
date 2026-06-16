/*
 * batched_evolution.cu -- Batched State-Vector Evolution via WMMA
 *
 * Processes B strings simultaneously using tensor-core MMA operations.
 * Maintains a state matrix S[N][B] where each column is a DFA state vector.
 * At each string position, applies transition matrices via WMMA 16x16 int8 MMA.
 *
 * Two kernel variants:
 *   Binary (|Sigma|=2): Two independent MMAs per step, select by input char
 *   General (|Sigma|>2): Per-character iteration with ballot-based skip
 *
 * Threading model:
 *   Each warp handles 16 consecutive columns (strings) -- one WMMA 16x16 tile
 *   4 warps per block -> 64 strings per block (COLS_PER_BLOCK = 64)
 *   State tiles live in shared memory across the entire L-step loop
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

// ---- Binary Kernel (|Sigma|=2) -------------------------------------------
//
// Shared memory layout:
//   T0_sh[16][16]           256 bytes   -- transition matrix for char 0
//   T1_sh[16][16]           256 bytes   -- transition matrix for char 1
//   S_sh[4][16][16]         1024 bytes  -- state tiles (col-major per warp)
//   acc0_sh[4][16][16]      4096 bytes  -- int32 accumulators for T0*S
//   acc1_sh[4][16][16]      4096 bytes  -- int32 accumulators for T1*S
//   Total:                  9728 bytes
//
// S is stored in COL-MAJOR layout in shared memory:
//   S_sh[col * 16 + row] = S[state_row][string_col]
// This allows direct loading as wmma::matrix_b with col_major, ldm=16.
// After MMA, the int32 accumulator is stored row-major, then converted back.

__global__ void batched_evolution_binary_kernel(
    const int8_t  *__restrict__ T0_global,     // [16*16] transition for char 0
    const int8_t  *__restrict__ T1_global,     // [16*16] transition for char 1
    const uint8_t *__restrict__ input,          // [L][B_padded] position-contiguous
    const int8_t  *__restrict__ accept_mask,   // [16]
    int start_state,
    int B,          // actual number of strings
    int B_padded,   // B rounded up to multiple of COLS_PER_BLOCK
    int L,          // max string length
    int *__restrict__ results                   // [B] output
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * COLS_PER_BLOCK;
    int warp_col_start = block_col_start + warp_in_block * TILE;

    // Shared memory
    extern __shared__ char smem_raw[];
    int8_t  *T0_sh  = (int8_t *)smem_raw;                            // 256
    int8_t  *T1_sh  = T0_sh + TILE_ELEMS;                            // 256
    int8_t  *S_base = T1_sh + TILE_ELEMS;                            // 4 * 256 = 1024
    int32_t *acc0_base = (int32_t *)(S_base + WARPS_PER_BLOCK * TILE_ELEMS);  // 4 * 1024 = 4096
    int32_t *acc1_base = acc0_base + WARPS_PER_BLOCK * TILE_ELEMS;            // 4 * 1024 = 4096

    int8_t  *S_sh   = S_base + warp_in_block * TILE_ELEMS;
    int32_t *acc0_sh = acc0_base + warp_in_block * TILE_ELEMS;
    int32_t *acc1_sh = acc1_base + warp_in_block * TILE_ELEMS;

    // Load T0 and T1 into shared memory (all threads cooperate)
    for (int e = threadIdx.x; e < TILE_ELEMS; e += blockDim.x) {
        T0_sh[e] = T0_global[e];
        T1_sh[e] = T1_global[e];
    }

    // Initialize S to start state vector (col-major):
    // S[row][col] stored at S_sh[col * 16 + row]
    // start_state row = 1, all others = 0
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        S_sh[e] = (row == start_state) ? (int8_t)1 : (int8_t)0;
    }
    __syncthreads();

    // Fragment declarations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_T0, frag_T1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_S;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_acc0, frag_acc1;

    // Load T0 and T1 fragments (they don't change)
    wmma::load_matrix_sync(frag_T0, T0_sh, TILE);
    wmma::load_matrix_sync(frag_T1, T1_sh, TILE);

    // Main loop over string positions
    for (int t = 0; t < L; t++) {
        // Load S fragment from shared memory (col-major)
        wmma::load_matrix_sync(frag_S, S_sh, TILE);

        // Compute acc0 = T0 * S and acc1 = T1 * S
        wmma::fill_fragment(frag_acc0, 0);
        wmma::mma_sync(frag_acc0, frag_T0, frag_S, frag_acc0);

        wmma::fill_fragment(frag_acc1, 0);
        wmma::mma_sync(frag_acc1, frag_T1, frag_S, frag_acc1);

        // Store accumulators to shared memory (row-major)
        wmma::store_matrix_sync(acc0_sh, frag_acc0, TILE, wmma::mem_row_major);
        wmma::store_matrix_sync(acc1_sh, frag_acc1, TILE, wmma::mem_row_major);
        __syncwarp();

        // Per-column: select acc0 or acc1 based on input char, threshold, write to S_sh (col-major)
        // input[t * B_padded + string_col] gives the char index for string string_col at position t
        // acc is stored row-major: acc[row * 16 + col]
        // S_sh is col-major: S_sh[col * 16 + row]
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
            int col = e / TILE;  // col-major index
            int row = e % TILE;
            int string_id = warp_col_start + col;
            uint8_t ch = 0;
            if (string_id < B_padded) {
                ch = input[t * B_padded + string_id];
            }
            // acc row-major index: row * 16 + col
            int32_t val;
            if (ch == 0)
                val = acc0_sh[row * TILE + col];
            else
                val = acc1_sh[row * TILE + col];
            S_sh[e] = (int8_t)(val > 0 ? 1 : 0);
        }
        __syncwarp();
    }

    // Extract results: for each string in this warp, check accept
    // S_sh[col * 16 + row] is the state vector for string (warp_col_start + col)
    // A string accepts if any row r with accept_mask[r] != 0 has S_sh[col*16+r] > 0
    for (int col = lane; col < TILE; col += WARP_SIZE) {
        int string_id = warp_col_start + col;
        if (string_id >= B) continue;
        int accepted = 0;
        for (int r = 0; r < TILE; r++) {
            if (S_sh[col * TILE + r] > 0 && accept_mask[r] != 0) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}


// ---- General Kernel (|Sigma| > 2) ----------------------------------------
//
// Shared memory layout:
//   T_sh[16][16]            256 bytes   -- transition matrix workspace
//   S_sh[4][16][16]         1024 bytes  -- state tiles (col-major per warp)
//   acc_sh[4][16][16]       4096 bytes  -- int32 accumulators
//   S_tmp[4][16][16]        1024 bytes  -- temp state for accumulation
//   Total:                  6400 bytes

__global__ void batched_evolution_general_kernel(
    const int8_t  *__restrict__ trans_matrices, // [sigma][16][16] row-major per matrix
    const uint8_t *__restrict__ input,           // [L][B_padded]
    const int8_t  *__restrict__ accept_mask,    // [16]
    int start_state,
    int sigma,       // alphabet size
    int B,
    int B_padded,
    int L,
    int identity_idx, // char index that means "identity / padding"
    int *__restrict__ results
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * COLS_PER_BLOCK;
    int warp_col_start = block_col_start + warp_in_block * TILE;

    extern __shared__ char smem_raw[];
    int8_t  *T_sh    = (int8_t *)smem_raw;                                      // 256
    int8_t  *S_base  = T_sh + TILE_ELEMS;                                       // 4 * 256
    int32_t *acc_base = (int32_t *)(S_base + WARPS_PER_BLOCK * TILE_ELEMS);     // 4 * 1024
    int8_t  *Stmp_base = (int8_t *)(acc_base + WARPS_PER_BLOCK * TILE_ELEMS);   // 4 * 256

    int8_t  *S_sh    = S_base + warp_in_block * TILE_ELEMS;
    int32_t *acc_sh  = acc_base + warp_in_block * TILE_ELEMS;
    int8_t  *S_tmp   = Stmp_base + warp_in_block * TILE_ELEMS;

    // Initialize S to start state vector (col-major)
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        S_sh[e] = (row == start_state) ? (int8_t)1 : (int8_t)0;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_T;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_S;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_acc;

    for (int t = 0; t < L; t++) {
        // Initialize S_tmp to zeros (will be filled per-character)
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
            S_tmp[e] = 0;
        }
        __syncwarp();

        // Gather which chars are present in this warp's 16 columns
        // Each lane reads chars for its assigned columns
        // We need per-column char info. Since we have 32 lanes and 16 columns,
        // lanes 0..15 each handle one column.
        uint8_t col_char = 255;  // invalid default
        if (lane < TILE) {
            int string_id = warp_col_start + lane;
            if (string_id < B_padded)
                col_char = input[t * B_padded + string_id];
            else
                col_char = (uint8_t)identity_idx;
        }

        for (int c = 0; c < sigma; c++) {
            if (c == identity_idx) continue;  // identity handled separately

            // Check if any column in this warp has char c
            unsigned mask = __ballot_sync(0xFFFFFFFF, (lane < TILE) && (col_char == c));
            if (mask == 0) continue;

            // Load T[c] into shared memory (all threads in warp cooperate)
            const int8_t *T_c = trans_matrices + c * TILE_ELEMS;
            for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
                T_sh[e] = T_c[e];
            }
            __syncwarp();

            // Compute T[c] * S via WMMA
            wmma::load_matrix_sync(frag_T, T_sh, TILE);
            wmma::load_matrix_sync(frag_S, S_sh, TILE);
            wmma::fill_fragment(frag_acc, 0);
            wmma::mma_sync(frag_acc, frag_T, frag_S, frag_acc);
            wmma::store_matrix_sync(acc_sh, frag_acc, TILE, wmma::mem_row_major);
            __syncwarp();

            // For columns matching char c, copy thresholded result to S_tmp (col-major)
            for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
                int col = e / TILE;
                int row = e % TILE;
                // Check if this column has char c
                // We need to broadcast col_char for this col
                uint8_t this_col_char = __shfl_sync(0xFFFFFFFF, col_char, col);
                if (this_col_char == c) {
                    int32_t val = acc_sh[row * TILE + col];
                    S_tmp[e] = (int8_t)(val > 0 ? 1 : 0);
                }
            }
            __syncwarp();
        }

        // Handle identity (padding): copy S unchanged for identity columns
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
            int col = e / TILE;
            uint8_t this_col_char = __shfl_sync(0xFFFFFFFF, col_char, col);
            if (this_col_char == (uint8_t)identity_idx) {
                S_tmp[e] = S_sh[e];
            }
        }
        __syncwarp();

        // Copy S_tmp -> S_sh
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
            S_sh[e] = S_tmp[e];
        }
        __syncwarp();
    }

    // Extract results
    for (int col = lane; col < TILE; col += WARP_SIZE) {
        int string_id = warp_col_start + col;
        if (string_id >= B) continue;
        int accepted = 0;
        for (int r = 0; r < TILE; r++) {
            if (S_sh[col * TILE + r] > 0 && accept_mask[r] != 0) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}


// ---- Engine Struct --------------------------------------------------------

struct BatchedEngine {
    int N;           // number of DFA states (padded to 16)
    int sigma;       // alphabet size
    int start_state;
    int max_B;
    int max_L;

    // Host copies
    int8_t *h_accept_mask;     // [16]
    int8_t *h_trans_matrices;  // [sigma * 256]

    // Device memory
    int8_t  *d_trans;          // [sigma * 256]
    int8_t  *d_accept;         // [16]
    uint8_t *d_input;          // [max_L * B_padded_max]
    int     *d_results;        // [max_B]

    int B_padded_max;          // max_B rounded up to COLS_PER_BLOCK

    // Timing
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;

    bool initialized;

    void init(int n, int sig, const int8_t *trans, const int8_t *accept,
              int start_st, int maxB, int maxL) {
        N = n;
        sigma = sig;
        start_state = start_st;
        max_B = maxB;
        max_L = maxL;
        B_padded_max = ((maxB + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

        h_accept_mask = new int8_t[TILE];
        h_trans_matrices = new int8_t[sig * TILE_ELEMS];
        memcpy(h_accept_mask, accept, TILE);
        memcpy(h_trans_matrices, trans, sig * TILE_ELEMS);

        CHECK_CUDA(cudaMalloc(&d_trans, sig * TILE_ELEMS));
        CHECK_CUDA(cudaMalloc(&d_accept, TILE));
        CHECK_CUDA(cudaMalloc(&d_input, (size_t)maxL * B_padded_max));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)maxB * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_trans, trans, sig * TILE_ELEMS, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, accept, TILE, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

        initialized = true;
    }

    void destroy() {
        if (!initialized) return;
        delete[] h_accept_mask;
        delete[] h_trans_matrices;
        cudaFree(d_trans);
        cudaFree(d_accept);
        cudaFree(d_input);
        cudaFree(d_results);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start);
        cudaEventDestroy(ev_kern_stop);
        initialized = false;
    }

    // Dispatch: input is [L][B_padded] position-contiguous, already on host
    // B_padded must be multiple of COLS_PER_BLOCK
    int dispatch(const uint8_t *h_input, int B, int L, int B_padded,
                 int *h_results, float *kernel_ms, float *total_ms) {
        if (!initialized) return -1;

        CHECK_CUDA(cudaEventRecord(ev_start));

        // H2D: copy input
        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        int n_blocks = B_padded / COLS_PER_BLOCK;

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        if (sigma == 2) {
            // Binary kernel
            // Shared memory: 256 (T0) + 256 (T1) + 4*256 (S) + 4*1024 (acc0) + 4*1024 (acc1)
            int smem = 2 * TILE_ELEMS + WARPS_PER_BLOCK * TILE_ELEMS
                       + 2 * WARPS_PER_BLOCK * TILE_ELEMS * (int)sizeof(int32_t);
            batched_evolution_binary_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
                d_trans,                      // T0
                d_trans + TILE_ELEMS,         // T1
                d_input,
                d_accept,
                start_state,
                B, B_padded, L,
                d_results);
        } else {
            // General kernel
            // Shared memory: 256 (T) + 4*256 (S) + 4*1024 (acc) + 4*256 (S_tmp)
            // identity_idx = sigma (one past last valid char)
            int identity_idx = sigma;
            int smem = TILE_ELEMS + WARPS_PER_BLOCK * TILE_ELEMS
                       + WARPS_PER_BLOCK * TILE_ELEMS * (int)sizeof(int32_t)
                       + WARPS_PER_BLOCK * TILE_ELEMS;
            batched_evolution_general_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
                d_trans,
                d_input,
                d_accept,
                start_state,
                sigma,
                B, B_padded, L,
                identity_idx,
                d_results);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        // D2H: results
        CHECK_CUDA(cudaMemcpy(h_results, d_results, (size_t)B * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);

        return 0;
    }
};


// ---- Global Engine (C API pattern) ----------------------------------------

static BatchedEngine g_engine = {.initialized = false};


// ---- C API ----------------------------------------------------------------

extern "C" {

int batched_engine_device_check() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // Need SM >= 7.0 for WMMA int8
    if (prop.major < 7) return -2;
    return 0;
}

int batched_engine_init(int N, int sigma,
                        const int8_t *trans_matrices,  // [sigma, N, N] row-major
                        const int8_t *accept_mask,     // [N]
                        int start_state,
                        int max_B, int max_L) {
    if (g_engine.initialized) g_engine.destroy();
    g_engine.init(N, sigma, trans_matrices, accept_mask, start_state, max_B, max_L);
    return 0;
}

void batched_engine_destroy() {
    g_engine.destroy();
}

int batched_engine_dispatch(const uint8_t *input,  // [L, B_padded]
                            int B, int L,
                            int *results,          // [B]
                            float *kernel_ms, float *total_ms) {
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
    return g_engine.dispatch(input, B, L, B_padded, results, kernel_ms, total_ms);
}

// C-accelerated batch preparation: transpose from [string][position] to [position][B_padded]
void batched_prepare_input(
    const char *strings_concat,
    const int *offsets,        // [B+1] CSR
    uint8_t *output,           // [L][B_padded] position-contiguous
    int B, int B_padded, int L,
    const int *char_to_idx,    // [256] char -> index, -1 = identity
    int identity_idx)
{
    // Zero-fill output (padding gets identity_idx)
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
        // Positions beyond string length: already set to identity_idx by memset
    }
}

}  // extern "C"


// ---- Built-in Tests (standalone build) ------------------------------------

#ifndef BUILD_LIB

static int g_tests = 0, g_pass = 0;
static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) { g_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

// Sequential reference simulation for a single string
static bool simulate_sequential(
    int n_states, int start_state,
    const int8_t *accept_mask, const int8_t *trans_matrices,
    const uint8_t *chars, int L, int identity_idx
) {
    if (L == 0) return accept_mask[start_state] != 0;

    // State vector
    int8_t state[TILE];
    memset(state, 0, TILE);
    state[start_state] = 1;

    for (int t = 0; t < L; t++) {
        int c = chars[t];
        if (c == identity_idx) continue;  // identity: no change

        const int8_t *T = trans_matrices + c * TILE_ELEMS;
        int8_t new_state[TILE];
        memset(new_state, 0, TILE);
        for (int row = 0; row < TILE; row++) {
            int32_t sum = 0;
            for (int k = 0; k < TILE; k++) {
                sum += (int32_t)T[row * TILE + k] * (int32_t)state[k];
            }
            new_state[row] = (int8_t)(sum > 0 ? 1 : 0);
        }
        memcpy(state, new_state, TILE);
    }

    for (int r = 0; r < TILE; r++)
        if (state[r] > 0 && accept_mask[r] != 0) return true;
    return false;
}

// Even-A DFA: 2 states, binary alphabet
// State 0 = even count of 'a' (accept), State 1 = odd count
// T[0]='a': swaps states 0<->1
// T[1]='b': identity on states 0,1
struct EvenADFA {
    int8_t accept[TILE];
    int8_t trans[2 * TILE_ELEMS];
    EvenADFA() {
        memset(accept, 0, sizeof(accept));
        memset(trans, 0, sizeof(trans));
        accept[0] = 1;  // state 0 = even count -> accept

        // Padding states: self-loop for both chars
        for (int c = 0; c < 2; c++)
            for (int s = 2; s < TILE; s++)
                trans[c * TILE_ELEMS + s * TILE + s] = 1;

        // 'a' (index 0): swap states 0<->1
        trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1;  // 0->1
        trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1;  // 1->0

        // 'b' (index 1): identity on states 0,1
        trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1;  // 0->0
        trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1;  // 1->1
    }
};


static void test_binary_even_a() {
    printf("\n--- test_binary_even_a ---\n");
    EvenADFA dfa;

    int max_B = 64;
    int max_L = 256;
    BatchedEngine eng;
    eng.init(2, 2, dfa.trans, dfa.accept, 0, max_B, max_L);

    // identity_idx = 2 (for binary, sigma=2)
    int identity_idx = 2;

    // 8 test cases: chars stored flat, with lengths
    // case 0: empty (even=accept)
    // case 1: "a" (odd=reject)
    // case 2: "aa" (even=accept)
    // case 3: "b" (even=accept)
    // case 4: "ab" (odd=reject)
    // case 5: "ba" (odd=reject)
    // case 6: "aab" (even=accept)
    // case 7: "abba" (even=accept)
    const char *names[] = {
        "empty (even=accept)", "'a' (odd=reject)", "'aa' (even=accept)",
        "'b' (even=accept)", "'ab' (odd=reject)", "'ba' (odd=reject)",
        "'aab' (even=accept)", "'abba' (even=accept)"
    };
    uint8_t all_chars[] = {
        /* case 1 */ 0,
        /* case 2 */ 0, 0,
        /* case 3 */ 1,
        /* case 4 */ 0, 1,
        /* case 5 */ 1, 0,
        /* case 6 */ 0, 0, 1,
        /* case 7 */ 0, 1, 1, 0
    };
    int lengths[] = {0, 1, 2, 1, 2, 2, 3, 4};
    bool expected[] = {true, false, true, true, false, false, true, true};
    int offsets[] = {0, 0, 1, 3, 4, 6, 8, 11, 15};  // CSR into all_chars
    int n_cases = 8;

    int L = 4;  // max length
    int B = n_cases;
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

    // Build position-contiguous input [L][B_padded]
    size_t input_size = (size_t)L * B_padded;
    uint8_t *input = new uint8_t[input_size];
    memset(input, (uint8_t)identity_idx, input_size);

    for (int i = 0; i < n_cases; i++) {
        for (int t = 0; t < lengths[i]; t++) {
            input[t * B_padded + i] = all_chars[offsets[i] + t];
        }
    }

    int *results = new int[B];
    float km, tm;
    eng.dispatch(input, B, L, B_padded, results, &km, &tm);

    for (int i = 0; i < n_cases; i++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "even_a %s -> %s",
                 names[i], expected[i] ? "accept" : "reject");
        check(msg, results[i] == (expected[i] ? 1 : 0));
    }

    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);
    delete[] input;
    delete[] results;
    eng.destroy();
}


static void test_large_batch_binary() {
    printf("\n--- test_large_batch_binary (1024 strings x 128 chars) ---\n");
    EvenADFA dfa;

    int B = 1024;
    int L = 128;
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
    int identity_idx = 2;

    BatchedEngine eng;
    eng.init(2, 2, dfa.trans, dfa.accept, 0, B_padded, L);

    // Generate random binary input
    srand(42);
    size_t input_size = (size_t)L * B_padded;
    uint8_t *input = new uint8_t[input_size];
    memset(input, (uint8_t)identity_idx, input_size);
    int *a_counts = new int[B];
    memset(a_counts, 0, B * sizeof(int));

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < L; t++) {
            uint8_t ch = rand() % 2;
            input[t * B_padded + b] = ch;
            if (ch == 0) a_counts[b]++;
        }
    }

    int *results = new int[B];
    float km, tm;
    eng.dispatch(input, B, L, B_padded, results, &km, &tm);

    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        bool expected = (a_counts[b] % 2 == 0);  // even count -> accept
        if (results[b] != (expected ? 1 : 0)) mismatches++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "large_batch B=%d L=%d (%d mismatches)", B, L, mismatches);
    check(msg, mismatches == 0);
    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);

    // Also cross-validate with sequential reference
    uint8_t *str_chars = new uint8_t[L];
    int seq_mismatches = 0;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < L; t++)
            str_chars[t] = input[t * B_padded + b];
        bool seq = simulate_sequential(2, 0, dfa.accept, dfa.trans,
                                        str_chars, L, identity_idx);
        if (results[b] != (seq ? 1 : 0)) seq_mismatches++;
    }
    snprintf(msg, sizeof(msg), "large_batch seq cross-validate (%d mismatches)", seq_mismatches);
    check(msg, seq_mismatches == 0);

    delete[] input;
    delete[] a_counts;
    delete[] results;
    delete[] str_chars;
    eng.destroy();
}


static void test_general_kernel() {
    printf("\n--- test_general_kernel (3-char alphabet) ---\n");

    // 3-state DFA over alphabet {a=0, b=1, c=2}
    // States: 0 (start), 1, 2 (accept)
    // Transitions:
    //   a: 0->1, 1->2, 2->0
    //   b: 0->0, 1->1, 2->2
    //   c: 0->2, 1->0, 2->1
    int sigma = 3;
    int8_t accept[TILE];
    int8_t trans[3 * TILE_ELEMS];
    memset(accept, 0, TILE);
    memset(trans, 0, sizeof(trans));
    accept[2] = 1;

    // Padding states: self-loop
    for (int c = 0; c < sigma; c++)
        for (int s = 3; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1;

    // a (0): 0->1, 1->2, 2->0
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1;
    trans[0 * TILE_ELEMS + 2 * TILE + 1] = 1;
    trans[0 * TILE_ELEMS + 0 * TILE + 2] = 1;

    // b (1): identity on 0,1,2
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1;
    trans[1 * TILE_ELEMS + 2 * TILE + 2] = 1;

    // c (2): 0->2, 1->0, 2->1
    trans[2 * TILE_ELEMS + 2 * TILE + 0] = 1;
    trans[2 * TILE_ELEMS + 0 * TILE + 1] = 1;
    trans[2 * TILE_ELEMS + 1 * TILE + 2] = 1;

    int identity_idx = sigma;  // 3
    int B = 64;
    int L = 16;
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

    BatchedEngine eng;
    eng.init(3, sigma, trans, accept, 0, B_padded, L);

    // Generate random input
    srand(99);
    size_t input_size = (size_t)L * B_padded;
    uint8_t *input = new uint8_t[input_size];
    memset(input, (uint8_t)identity_idx, input_size);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < L; t++) {
            input[t * B_padded + b] = rand() % sigma;
        }
    }

    int *results = new int[B];
    float km, tm;
    eng.dispatch(input, B, L, B_padded, results, &km, &tm);

    // Cross-validate with sequential
    uint8_t *str_chars = new uint8_t[L];
    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < L; t++)
            str_chars[t] = input[t * B_padded + b];
        bool seq = simulate_sequential(3, 0, accept, trans,
                                        str_chars, L, identity_idx);
        if (results[b] != (seq ? 1 : 0)) mismatches++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "general_kernel B=%d L=%d sigma=%d (%d mismatches)",
             B, L, sigma, mismatches);
    check(msg, mismatches == 0);
    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);

    delete[] input;
    delete[] results;
    delete[] str_chars;
    eng.destroy();
}


static void test_prepare_input() {
    printf("\n--- test_prepare_input ---\n");

    const char concat[] = "ababba";  // "aba" + "bb" + "a"
    int offsets[] = {0, 3, 5, 6};
    int B = 3;
    int B_padded = COLS_PER_BLOCK;  // 64
    int L = 3;  // max length
    int identity_idx = 2;
    int char_to_idx[256];
    memset(char_to_idx, -1, sizeof(char_to_idx));
    char_to_idx['a'] = 0;
    char_to_idx['b'] = 1;

    uint8_t *output = new uint8_t[L * B_padded];
    batched_prepare_input(concat, offsets, output, B, B_padded, L,
                          char_to_idx, identity_idx);

    // Check: position 0: a=0, b=1, a=0
    check("prepare[0][0]='a'=0", output[0 * B_padded + 0] == 0);
    check("prepare[0][1]='b'=1", output[0 * B_padded + 1] == 1);
    check("prepare[0][2]='a'=0", output[0 * B_padded + 2] == 0);

    // position 1: b=1, b=1, identity (string 2 has length 1)
    check("prepare[1][0]='b'=1", output[1 * B_padded + 0] == 1);
    check("prepare[1][1]='b'=1", output[1 * B_padded + 1] == 1);
    check("prepare[1][2]=identity", output[1 * B_padded + 2] == identity_idx);

    // position 2: a=0, identity, identity
    check("prepare[2][0]='a'=0", output[2 * B_padded + 0] == 0);
    check("prepare[2][1]=identity", output[2 * B_padded + 1] == identity_idx);

    // padding positions beyond B should be identity
    check("prepare[0][63]=identity", output[0 * B_padded + 63] == identity_idx);

    delete[] output;
}


static void bench_throughput() {
    printf("\n=== Throughput Benchmark (binary kernel) ===\n");
    EvenADFA dfa;

    int batch_sizes[] = {64, 256, 1024, 4096, 16384, 65536};
    int lengths[]     = {32, 128, 512, 2048};
    int n_batches = 6, n_lengths = 4;
    int identity_idx = 2;

    printf("  %8s  %8s  %12s  %12s  %12s  %12s\n",
           "B", "L", "kern(ms)", "total(ms)", "kern Gch/s", "total Gch/s");

    for (int bi = 0; bi < n_batches; bi++) {
        for (int li = 0; li < n_lengths; li++) {
            int B = batch_sizes[bi];
            int L = lengths[li];
            int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

            BatchedEngine eng;
            eng.init(2, 2, dfa.trans, dfa.accept, 0, B_padded, L);

            // Generate random input
            srand(42);
            size_t input_size = (size_t)L * B_padded;
            uint8_t *input = new uint8_t[input_size];
            memset(input, (uint8_t)identity_idx, input_size);
            for (int b = 0; b < B; b++)
                for (int t = 0; t < L; t++)
                    input[t * B_padded + b] = rand() % 2;

            int *results = new int[B];

            // Warmup
            for (int w = 0; w < 3; w++)
                eng.dispatch(input, B, L, B_padded, results, nullptr, nullptr);

            // Measure
            int iters = 20;
            float kern_total = 0, total_total = 0;
            for (int it = 0; it < iters; it++) {
                float km, tm;
                eng.dispatch(input, B, L, B_padded, results, &km, &tm);
                kern_total += km;
                total_total += tm;
            }
            float kern_ms = kern_total / iters;
            float tot_ms = total_total / iters;
            double chars = (double)B * L;
            double kern_gchs = chars / (kern_ms * 1e6);
            double total_gchs = chars / (tot_ms * 1e6);

            printf("  %8d  %8d  %12.4f  %12.4f  %12.3f  %12.3f\n",
                   B, L, kern_ms, tot_ms, kern_gchs, total_gchs);

            delete[] input;
            delete[] results;
            eng.destroy();
        }
    }
}


int main() {
    printf("=== TERX Batched State-Vector Evolution ===\n");

    // Device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    int devcheck = batched_engine_device_check();
    if (devcheck != 0) {
        printf("Device check failed: %d\n", devcheck);
        return 1;
    }

    // Correctness tests
    test_binary_even_a();
    test_large_batch_binary();
    test_general_kernel();
    test_prepare_input();

    printf("\n=== Results: %d / %d passed ===\n", g_pass, g_tests);
    if (g_pass != g_tests) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }

    // Benchmarks
    bench_throughput();

    return 0;
}

#endif  // BUILD_LIB
