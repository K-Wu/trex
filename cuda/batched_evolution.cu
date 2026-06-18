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
    const int8_t *__restrict__ start_vec,      // [16] initial state vector
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
        S_sh[e] = start_vec[row];
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
        // Identity (ch >= 2): leave S_sh unchanged (string already ended or padding)
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
            int col = e / TILE;  // col-major index
            int row = e % TILE;
            int string_id = warp_col_start + col;
            uint8_t ch = 2;  // default to identity for out-of-bounds columns
            if (string_id < B_padded) {
                ch = input[t * B_padded + string_id];
            }
            if (ch < 2) {
                // acc row-major index: row * 16 + col
                int32_t val;
                if (ch == 0)
                    val = acc0_sh[row * TILE + col];
                else
                    val = acc1_sh[row * TILE + col];
                S_sh[e] = (int8_t)(val > 0 ? 1 : 0);
            }
            // else: identity — S_sh[e] stays unchanged
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


// ---- Binary Kernel V2 (register-level select, no acc shared memory) --------
//
// Same algorithm as the binary kernel above, but eliminates the
// store_matrix_sync calls for accumulators. After MMA, each thread directly
// selects from frag_acc0.x[i] or frag_acc1.x[i] based on the input character,
// thresholds, and writes to S_sh.
//
// Relies on the empirically probed WMMA accumulator fragment layout:
//   row = lane/4 + ((i>>1)&1)*8
//   col = (lane%4)*2 + (i&1) + (i>>2)*8
//
// Shared memory: T0_sh(256) + T1_sh(256) + S_sh(4×256) = 1536 bytes
// (vs 9728 bytes for V1)

__global__ void __launch_bounds__(BLOCK_SIZE, 16)
batched_evolution_binary_v2_kernel(
    const int8_t  *__restrict__ T0_global,
    const int8_t  *__restrict__ T1_global,
    const uint8_t *__restrict__ input,
    const int8_t  *__restrict__ accept_mask,
    const int8_t  *__restrict__ start_vec,
    int B, int B_padded, int L,
    int *__restrict__ results
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * COLS_PER_BLOCK;
    int warp_col_start = block_col_start + warp_in_block * TILE;

    extern __shared__ char smem_raw[];
    int8_t *T0_sh = (int8_t *)smem_raw;
    int8_t *T1_sh = T0_sh + TILE_ELEMS;
    int8_t *S_base = T1_sh + TILE_ELEMS;
    int8_t *S_sh = S_base + warp_in_block * TILE_ELEMS;

    for (int e = threadIdx.x; e < TILE_ELEMS; e += blockDim.x) {
        T0_sh[e] = T0_global[e];
        T1_sh[e] = T1_global[e];
    }
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        S_sh[e] = start_vec[row];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_T0, frag_T1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_S;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_acc0, frag_acc1;

    wmma::load_matrix_sync(frag_T0, T0_sh, TILE);
    wmma::load_matrix_sync(frag_T1, T1_sh, TILE);

    // Pre-compute per-thread fragment geometry
    int col_lo = (lane & 3) * 2;       // columns for elements 0-3
    int col_hi = col_lo + 8;           // columns for elements 4-7
    int row_lo = lane >> 2;            // row for even-indexed elements
    int row_hi = row_lo + 8;           // row for odd-pair elements

    for (int t = 0; t < L; t++) {
        wmma::load_matrix_sync(frag_S, S_sh, TILE);

        wmma::fill_fragment(frag_acc0, 0);
        wmma::mma_sync(frag_acc0, frag_T0, frag_S, frag_acc0);
        wmma::fill_fragment(frag_acc1, 0);
        wmma::mma_sync(frag_acc1, frag_T1, frag_S, frag_acc1);

        // Read input chars: lanes 0-15 each read one column
        uint8_t my_ch = 2;
        if (lane < TILE) {
            int sid = warp_col_start + lane;
            if (sid < B_padded) my_ch = input[t * B_padded + sid];
        }

        // Get chars for the 4 columns this thread owns
        uint8_t ch0 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_lo);
        uint8_t ch1 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_lo + 1);
        uint8_t ch2 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_hi);
        uint8_t ch3 = (uint8_t)__shfl_sync(0xFFFFFFFF, my_ch, col_hi + 1);

        // Select from registers, threshold, write directly to S_sh (col-major)
        // Element layout: i -> (row, col, char_idx)
        //   0: (row_lo, col_lo,   ch0)    1: (row_lo, col_lo+1, ch1)
        //   2: (row_hi, col_lo,   ch0)    3: (row_hi, col_lo+1, ch1)
        //   4: (row_lo, col_hi,   ch2)    5: (row_lo, col_hi+1, ch3)
        //   6: (row_hi, col_hi,   ch2)    7: (row_hi, col_hi+1, ch3)

        #define REGSEL(EI, COL, ROW, CH) \
            if ((CH) < 2) { \
                int32_t v = ((CH) == 0) ? frag_acc0.x[EI] : frag_acc1.x[EI]; \
                S_sh[(COL) * TILE + (ROW)] = (int8_t)(v > 0 ? 1 : 0); \
            }

        REGSEL(0, col_lo,     row_lo, ch0)
        REGSEL(1, col_lo + 1, row_lo, ch1)
        REGSEL(2, col_lo,     row_hi, ch0)
        REGSEL(3, col_lo + 1, row_hi, ch1)
        REGSEL(4, col_hi,     row_lo, ch2)
        REGSEL(5, col_hi + 1, row_lo, ch3)
        REGSEL(6, col_hi,     row_hi, ch2)
        REGSEL(7, col_hi + 1, row_hi, ch3)

        #undef REGSEL

        __syncwarp();
    }

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
//   T_sh[4][16][16]         1024 bytes  -- transition matrix workspace (per warp)
//   S_sh[4][16][16]         1024 bytes  -- state tiles (col-major per warp)
//   acc_sh[4][16][16]       4096 bytes  -- int32 accumulators
//   S_tmp[4][16][16]        1024 bytes  -- temp state for accumulation
//   Total:                  7168 bytes

__global__ void batched_evolution_general_kernel(
    const int8_t  *__restrict__ trans_matrices, // [sigma][16][16] row-major per matrix
    const uint8_t *__restrict__ input,           // [L][B_padded]
    const int8_t  *__restrict__ accept_mask,    // [16]
    const int8_t  *__restrict__ start_vec,     // [16] initial state vector
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
    int8_t  *T_base  = (int8_t *)smem_raw;                                      // 4 * 256
    int8_t  *S_base  = T_base + WARPS_PER_BLOCK * TILE_ELEMS;                   // 4 * 256
    int32_t *acc_base = (int32_t *)(S_base + WARPS_PER_BLOCK * TILE_ELEMS);     // 4 * 1024
    int8_t  *Stmp_base = (int8_t *)(acc_base + WARPS_PER_BLOCK * TILE_ELEMS);   // 4 * 256

    int8_t  *T_sh    = T_base + warp_in_block * TILE_ELEMS;
    int8_t  *S_sh    = S_base + warp_in_block * TILE_ELEMS;
    int32_t *acc_sh  = acc_base + warp_in_block * TILE_ELEMS;
    int8_t  *S_tmp   = Stmp_base + warp_in_block * TILE_ELEMS;

    // Initialize S from start vector (col-major)
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        S_sh[e] = start_vec[row];
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


// ---- Multi-Tile Kernel (NP > 16) -----------------------------------------
//
// For packed engines with NP = k*16 (k > 1) state dimensions.
// Each block handles 16 columns (one column-tile of S).
// NP_tiles warps per block, each warp computes one 16-row block of the output.
// S is stored col-major with leading dimension NP: S[col * NP + row].
// Double-buffered S (S_A / S_B) to avoid a copy step.
//
// Shared memory layout:
//   S_A[16 * NP]              -- state buffer A (col-major, ldm=NP)
//   S_B[16 * NP]              -- state buffer B
//   acc[NP_tiles * 256]       -- int32 accumulators (one 16x16 per warp)
//   Total: 32*NP + (NP/16)*1024 = 96*NP bytes

__global__ void batched_evolution_multitile_kernel(
    const int8_t  *__restrict__ T_all,         // [sigma, NP, NP] row-major per matrix
    const uint8_t *__restrict__ input,         // [L, B_padded]
    const int8_t  *__restrict__ accept_mask,   // [NP] (used only when states_out == NULL)
    const int8_t  *__restrict__ start_vec,     // [NP]
    int *__restrict__ results,                 // [B] (used only when states_out == NULL)
    int8_t *__restrict__ states_out,           // [NP, B_padded] or NULL
    int B, int B_padded, int L, int NP, int sigma, int NP_tiles
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int col_start = blockIdx.x * TILE;
    int tpb = NP_tiles * WARP_SIZE;

    extern __shared__ char smem_mt[];
    int8_t  *S_A = (int8_t *)smem_mt;
    int8_t  *S_B = S_A + 16 * NP;
    int32_t *acc_base = (int32_t *)(S_B + 16 * NP);
    int32_t *my_acc = acc_base + warp_id * TILE_ELEMS;

    int my_row = warp_id * TILE;

    // Initialize S_A from start_vec (all threads cooperate)
    for (int e = threadIdx.x; e < 16 * NP; e += tpb) {
        int row = e % NP;
        S_A[e] = start_vec[row];
    }
    __syncthreads();

    int8_t *S_cur = S_A;
    int8_t *S_nxt = S_B;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_T;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_S;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_acc;

    for (int t = 0; t < L; t++) {
        // Read column chars (lanes 0..15 each get one column's char)
        uint8_t col_char = (uint8_t)sigma;
        if (lane < TILE) {
            int sid = col_start + lane;
            if (sid < B_padded)
                col_char = input[t * B_padded + sid];
        }

        // Clear this warp's row-block portion of S_nxt
        for (int e = lane; e < 16 * TILE; e += WARP_SIZE) {
            int local_col = e / TILE;
            int local_row = e % TILE;
            S_nxt[local_col * NP + my_row + local_row] = 0;
        }
        __syncwarp();

        for (int c = 0; c < sigma; c++) {
            unsigned has_c = __ballot_sync(0xFFFFFFFF, (lane < TILE) && (col_char == c));
            if (has_c == 0) continue;

            // output[my_row_block] = sum_k T[c][warp_id][k] * S_cur[k]
            wmma::fill_fragment(frag_acc, 0);
            for (int k = 0; k < NP_tiles; k++) {
                const int8_t *T_block = T_all + c * NP * NP + my_row * NP + k * TILE;
                wmma::load_matrix_sync(frag_T, T_block, NP);
                wmma::load_matrix_sync(frag_S, &S_cur[k * TILE], NP);
                wmma::mma_sync(frag_acc, frag_T, frag_S, frag_acc);
            }

            wmma::store_matrix_sync(my_acc, frag_acc, TILE, wmma::mem_row_major);
            __syncwarp();

            // Threshold and write to S_nxt for columns matching char c
            for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
                int local_row = e / TILE;
                int local_col = e % TILE;
                uint8_t this_char = __shfl_sync(0xFFFFFFFF, col_char, local_col);
                if (this_char == (uint8_t)c) {
                    int32_t val = my_acc[e];
                    S_nxt[local_col * NP + my_row + local_row] = (int8_t)(val > 0 ? 1 : 0);
                }
            }
            __syncwarp();
        }

        // Identity columns: copy from S_cur
        for (int e = lane; e < 16 * TILE; e += WARP_SIZE) {
            int local_col = e / TILE;
            int local_row = e % TILE;
            uint8_t this_char = __shfl_sync(0xFFFFFFFF, col_char, local_col);
            if (this_char >= (uint8_t)sigma) {
                S_nxt[local_col * NP + my_row + local_row] =
                    S_cur[local_col * NP + my_row + local_row];
            }
        }
        __syncthreads();

        // Double-buffer swap
        int8_t *tmp = S_cur; S_cur = S_nxt; S_nxt = tmp;
    }

    if (states_out != nullptr) {
        // Write final state to global memory: states_out[row * B_padded + sid]
        for (int e = lane; e < 16 * TILE; e += WARP_SIZE) {
            int local_col = e / TILE;
            int local_row = e % TILE;
            int sid = col_start + local_col;
            if (sid < B_padded) {
                states_out[(my_row + local_row) * B_padded + sid] =
                    S_cur[local_col * NP + my_row + local_row];
            }
        }
    } else {
        // Single-pattern accept check
        if (warp_id == 0) {
            for (int col = lane; col < TILE; col += WARP_SIZE) {
                int sid = col_start + col;
                if (sid >= B) continue;
                int accepted = 0;
                for (int r = 0; r < NP; r++) {
                    if (S_cur[col * NP + r] > 0 && accept_mask[r] != 0) {
                        accepted = 1;
                        break;
                    }
                }
                results[sid] = accepted;
            }
        }
    }
}


// ---- Multi-Pattern Accept Check Kernel -----------------------------------

__global__ void multi_accept_check_kernel(
    const int8_t *__restrict__ states,        // [NP, B_padded]
    const int8_t *__restrict__ accept_masks,  // [n_patterns, NP]
    int *__restrict__ results,                // [n_patterns * B]
    int B, int B_padded, int NP, int n_patterns
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= B) return;
    for (int p = 0; p < n_patterns; p++) {
        int accepted = 0;
        for (int r = 0; r < NP; r++) {
            if (states[r * B_padded + sid] > 0 && accept_masks[p * NP + r] != 0) {
                accepted = 1;
                break;
            }
        }
        results[p * B + sid] = accepted;
    }
}


// ---- Sparse Block-Diagonal Kernel (|Sigma|=2) ---------------------------
//
// Each warp handles one pattern's independent 16x16 diagonal block.
// Grid: dim3(ceil(B_padded / COLS_PER_BLOCK), P)
// blockIdx.y selects the pattern.
//
// Shared memory layout identical to binary kernel (9728 bytes):
//   T0_sh[16][16]           256 bytes
//   T1_sh[16][16]           256 bytes
//   S_sh[4][16][16]         1024 bytes
//   acc0_sh[4][16][16]      4096 bytes
//   acc1_sh[4][16][16]      4096 bytes

__global__ void batched_evolution_sparse_blockdiag_kernel(
    const int8_t  *__restrict__ T_diag,       // [P, 2, TILE, TILE] = [P * 2 * 256]
    const uint8_t *__restrict__ input,         // [L, B_padded]
    const int8_t  *__restrict__ accept_masks,  // [P, TILE] = [P * 16]
    const int8_t  *__restrict__ start_vecs,    // [P, TILE] = [P * 16]
    int *__restrict__ results,                 // [P * B]
    int B, int B_padded, int L, int P
) {
    int pattern_id = blockIdx.y;
    if (pattern_id >= P) return;

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

    // Load T0 and T1 from this pattern's diagonal block
    const int8_t *T_pat = T_diag + pattern_id * 2 * TILE_ELEMS;
    for (int e = threadIdx.x; e < TILE_ELEMS; e += blockDim.x) {
        T0_sh[e] = T_pat[e];
        T1_sh[e] = T_pat[TILE_ELEMS + e];
    }

    // Initialize S from this pattern's start vector (col-major)
    const int8_t *sv = start_vecs + pattern_id * TILE;
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        S_sh[e] = sv[row];
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
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
            int col = e / TILE;
            int row = e % TILE;
            int string_id = warp_col_start + col;
            uint8_t ch = 2;  // default to identity for out-of-bounds columns
            if (string_id < B_padded) {
                ch = input[t * B_padded + string_id];
            }
            if (ch < 2) {
                int32_t val;
                if (ch == 0)
                    val = acc0_sh[row * TILE + col];
                else
                    val = acc1_sh[row * TILE + col];
                S_sh[e] = (int8_t)(val > 0 ? 1 : 0);
            }
            // else: identity -- S_sh[e] stays unchanged
        }
        __syncwarp();
    }

    // Extract results: write to results[pattern_id * B + string_id]
    const int8_t *accept_mask = accept_masks + pattern_id * TILE;
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
        results[pattern_id * B + string_id] = accepted;
    }
}


// ---- Engine Struct --------------------------------------------------------

struct BatchedEngine {
    int N;           // number of DFA states (padded to multiple of 16)
    int sigma;       // alphabet size
    int max_B;
    int max_L;

    // Host copies
    int8_t *h_accept_mask;     // [N]
    int8_t *h_trans_matrices;  // [sigma * N * N]
    int8_t *h_start_vec;       // [N]

    // Device memory
    int8_t  *d_trans;          // [sigma * N * N]
    int8_t  *d_accept;         // [N]
    int8_t  *d_start_vec;      // [N]
    uint8_t *d_input;          // [max_L * B_padded_max]
    int     *d_results;        // [max_B] (or larger for multi-pattern)
    int8_t  *d_states;         // [N * B_padded] for multi-pattern state output
    size_t   d_states_bytes;   // current d_states allocation size
    int      max_results;      // current d_results capacity

    int B_padded_max;          // max_B rounded up to COLS_PER_BLOCK

    // Timing
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;

    bool initialized;

    // Sparse block-diagonal mode
    int8_t  *d_T_diag;            // [P * sigma * TILE * TILE] diagonal blocks
    int8_t  *d_start_vecs_sp;     // [P * TILE]
    int8_t  *d_accept_masks_sp;   // [P * TILE]
    int     *d_results_sp;        // [P * max_B]
    int      P_sparse;
    int      sigma_sparse;
    bool     sparse_initialized;

    void init(int n, int sig, const int8_t *trans, const int8_t *accept,
              const int8_t *start_vec, int maxB, int maxL) {
        N = n;
        sigma = sig;
        max_B = maxB;
        max_L = maxL;
        B_padded_max = ((maxB + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

        h_accept_mask = new int8_t[N];
        h_trans_matrices = new int8_t[sig * N * N];
        h_start_vec = new int8_t[N];
        memcpy(h_accept_mask, accept, N);
        memcpy(h_trans_matrices, trans, sig * N * N);
        memcpy(h_start_vec, start_vec, N);

        CHECK_CUDA(cudaMalloc(&d_trans, sig * N * N));
        CHECK_CUDA(cudaMalloc(&d_accept, N));
        CHECK_CUDA(cudaMalloc(&d_start_vec, N));
        CHECK_CUDA(cudaMalloc(&d_input, (size_t)maxL * B_padded_max));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)maxB * sizeof(int)));
        d_states = nullptr;
        d_states_bytes = 0;
        max_results = maxB;

        CHECK_CUDA(cudaMemcpy(d_trans, trans, sig * N * N, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, accept, N, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_start_vec, start_vec, N, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

        initialized = true;

        // Initialize sparse fields to safe defaults
        d_T_diag = nullptr;
        d_start_vecs_sp = nullptr;
        d_accept_masks_sp = nullptr;
        d_results_sp = nullptr;
        P_sparse = 0;
        sigma_sparse = 0;
        sparse_initialized = false;
    }

    void destroy_sparse() {
        if (!sparse_initialized) return;
        if (d_T_diag) { cudaFree(d_T_diag); d_T_diag = nullptr; }
        if (d_start_vecs_sp) { cudaFree(d_start_vecs_sp); d_start_vecs_sp = nullptr; }
        if (d_accept_masks_sp) { cudaFree(d_accept_masks_sp); d_accept_masks_sp = nullptr; }
        if (d_results_sp) { cudaFree(d_results_sp); d_results_sp = nullptr; }
        sparse_initialized = false;
    }

    void destroy() {
        destroy_sparse();
        if (!initialized) return;
        delete[] h_accept_mask;
        delete[] h_trans_matrices;
        delete[] h_start_vec;
        cudaFree(d_trans);
        cudaFree(d_accept);
        cudaFree(d_start_vec);
        cudaFree(d_input);
        cudaFree(d_results);
        if (d_states) cudaFree(d_states);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start);
        cudaEventDestroy(ev_kern_stop);
        initialized = false;
    }

    void init_sparse(int P, int sig, const int8_t *T_diag,
                     const int8_t *start_vecs, const int8_t *accept_masks,
                     int maxB, int maxL) {
        destroy_sparse();
        P_sparse = P;
        sigma_sparse = sig;

        // Allocate and copy diagonal blocks: [P * sigma * TILE * TILE]
        size_t T_bytes = (size_t)P * sig * TILE_ELEMS;
        CHECK_CUDA(cudaMalloc(&d_T_diag, T_bytes));
        CHECK_CUDA(cudaMemcpy(d_T_diag, T_diag, T_bytes, cudaMemcpyHostToDevice));

        // Allocate and copy start vectors: [P * TILE]
        size_t sv_bytes = (size_t)P * TILE;
        CHECK_CUDA(cudaMalloc(&d_start_vecs_sp, sv_bytes));
        CHECK_CUDA(cudaMemcpy(d_start_vecs_sp, start_vecs, sv_bytes, cudaMemcpyHostToDevice));

        // Allocate and copy accept masks: [P * TILE]
        size_t am_bytes = (size_t)P * TILE;
        CHECK_CUDA(cudaMalloc(&d_accept_masks_sp, am_bytes));
        CHECK_CUDA(cudaMemcpy(d_accept_masks_sp, accept_masks, am_bytes, cudaMemcpyHostToDevice));

        // Allocate results: [P * maxB]
        CHECK_CUDA(cudaMalloc(&d_results_sp, (size_t)P * maxB * sizeof(int)));

        // Ensure d_input is allocated (reuse from main init, or allocate here)
        if (!initialized) {
            max_B = maxB;
            max_L = maxL;
            B_padded_max = ((maxB + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
            CHECK_CUDA(cudaMalloc(&d_input, (size_t)maxL * B_padded_max));
            CHECK_CUDA(cudaEventCreate(&ev_start));
            CHECK_CUDA(cudaEventCreate(&ev_stop));
            CHECK_CUDA(cudaEventCreate(&ev_kern_start));
            CHECK_CUDA(cudaEventCreate(&ev_kern_stop));
        } else {
            // Main engine already initialized -- ensure input buffer is large enough
            int new_B_padded_max = ((maxB + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
            size_t needed = (size_t)maxL * new_B_padded_max;
            size_t existing = (size_t)max_L * B_padded_max;
            if (needed > existing) {
                cudaFree(d_input);
                CHECK_CUDA(cudaMalloc(&d_input, needed));
                if (maxB > max_B) max_B = maxB;
                if (maxL > max_L) max_L = maxL;
                B_padded_max = new_B_padded_max;
            }
        }

        sparse_initialized = true;
    }

    int dispatch_sparse(const uint8_t *h_input, int B, int L, int B_padded,
                        int *h_results, float *kernel_ms, float *total_ms) {
        if (!sparse_initialized) return -1;
        if (sigma_sparse != 2) return -2;  // sparse kernel supports sigma=2 only

        CHECK_CUDA(cudaEventRecord(ev_start));

        // H2D: copy input
        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        // Launch sparse block-diagonal kernel with 2D grid
        int blocks_x = (B_padded + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
        dim3 grid(blocks_x, P_sparse);
        int smem = 2 * TILE_ELEMS + WARPS_PER_BLOCK * TILE_ELEMS
                   + 2 * WARPS_PER_BLOCK * TILE_ELEMS * (int)sizeof(int32_t);
        batched_evolution_sparse_blockdiag_kernel<<<grid, BLOCK_SIZE, smem>>>(
            d_T_diag, d_input, d_accept_masks_sp, d_start_vecs_sp,
            d_results_sp, B, B_padded, L, P_sparse);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        // D2H: results [P_sparse * B]
        CHECK_CUDA(cudaMemcpy(h_results, d_results_sp,
                              (size_t)P_sparse * B * sizeof(int),
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
        if (sigma != 2 || N > TILE) return -2;

        CHECK_CUDA(cudaEventRecord(ev_start));
        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        int n_blocks = B_padded / COLS_PER_BLOCK;
        int smem = 2 * TILE_ELEMS + WARPS_PER_BLOCK * TILE_ELEMS;
        batched_evolution_binary_v2_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
            d_trans, d_trans + TILE_ELEMS,
            d_input, d_accept, d_start_vec,
            B, B_padded, L, d_results);
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

    // Dispatch: input is [L][B_padded] position-contiguous, already on host
    // B_padded must be multiple of COLS_PER_BLOCK
    int dispatch(const uint8_t *h_input, int B, int L, int B_padded,
                 int *h_results, float *kernel_ms, float *total_ms) {
        if (!initialized) return -1;

        CHECK_CUDA(cudaEventRecord(ev_start));

        // H2D: copy input
        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        if (N <= TILE) {
            // Single-tile kernels (N=16)
            int n_blocks = B_padded / COLS_PER_BLOCK;

            if (sigma == 2) {
                int smem = 2 * TILE_ELEMS + WARPS_PER_BLOCK * TILE_ELEMS
                           + 2 * WARPS_PER_BLOCK * TILE_ELEMS * (int)sizeof(int32_t);
                batched_evolution_binary_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
                    d_trans, d_trans + TILE_ELEMS,
                    d_input, d_accept, d_start_vec,
                    B, B_padded, L, d_results);
            } else {
                int smem = WARPS_PER_BLOCK * TILE_ELEMS
                           + WARPS_PER_BLOCK * TILE_ELEMS
                           + WARPS_PER_BLOCK * TILE_ELEMS * (int)sizeof(int32_t)
                           + WARPS_PER_BLOCK * TILE_ELEMS;
                batched_evolution_general_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
                    d_trans, d_input, d_accept, d_start_vec,
                    sigma, B, B_padded, L, sigma, d_results);
            }
        } else {
            // Multi-tile kernel (NP > 16)
            int NP_tiles = N / TILE;
            int threads = NP_tiles * WARP_SIZE;
            int n_blocks_mt = B_padded / TILE;
            int smem = 32 * N + NP_tiles * TILE_ELEMS * (int)sizeof(int32_t);
            batched_evolution_multitile_kernel<<<n_blocks_mt, threads, smem>>>(
                d_trans, d_input, d_accept, d_start_vec,
                d_results, nullptr, B, B_padded, L, N, sigma, NP_tiles);
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

    // Multi-pattern dispatch: evolves state, then checks P accept masks
    int dispatch_multi(const uint8_t *h_input, int B, int L, int B_padded,
                       const int8_t *h_accept_masks, int n_patterns,
                       int *h_results, float *kernel_ms, float *total_ms) {
        if (!initialized) return -1;

        CHECK_CUDA(cudaEventRecord(ev_start));

        // H2D: input
        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        // Ensure d_results is large enough for n_patterns * B
        int needed = n_patterns * B;
        if (needed > max_results) {
            cudaFree(d_results);
            CHECK_CUDA(cudaMalloc(&d_results, (size_t)needed * sizeof(int)));
            max_results = needed;
        }

        // Allocate/reallocate d_states if needed
        size_t states_bytes = (size_t)N * B_padded;
        if (states_bytes > d_states_bytes) {
            if (d_states) cudaFree(d_states);
            CHECK_CUDA(cudaMalloc(&d_states, states_bytes));
            d_states_bytes = states_bytes;
        }

        // Upload accept masks
        int8_t *d_accept_masks;
        CHECK_CUDA(cudaMalloc(&d_accept_masks, (size_t)n_patterns * N));
        CHECK_CUDA(cudaMemcpy(d_accept_masks, h_accept_masks,
                              n_patterns * N, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        // Always use multi-tile kernel for multi-pattern dispatch
        int NP_tiles = N / TILE;
        int threads = NP_tiles * WARP_SIZE;
        int n_blocks_mt = B_padded / TILE;
        int smem = 32 * N + NP_tiles * TILE_ELEMS * (int)sizeof(int32_t);
        batched_evolution_multitile_kernel<<<n_blocks_mt, threads, smem>>>(
            d_trans, d_input, d_accept, d_start_vec,
            d_results, d_states, B, B_padded, L, N, sigma, NP_tiles);
        CHECK_CUDA(cudaGetLastError());

        // Accept check kernel
        int acc_threads = 256;
        int acc_blocks = (B + acc_threads - 1) / acc_threads;
        multi_accept_check_kernel<<<acc_blocks, acc_threads>>>(
            d_states, d_accept_masks, d_results,
            B, B_padded, N, n_patterns);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        // D2H: results
        CHECK_CUDA(cudaMemcpy(h_results, d_results,
                              (size_t)n_patterns * B * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        cudaFree(d_accept_masks);

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);

        return 0;
    }
};


// ---- Global Engine (C API pattern) ----------------------------------------

static BatchedEngine g_engine = {};


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
                        const int8_t *start_vec,       // [N] initial state vector
                        int max_B, int max_L) {
    if (g_engine.initialized) g_engine.destroy();
    g_engine.init(N, sigma, trans_matrices, accept_mask, start_vec, max_B, max_L);
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

int batched_engine_dispatch_v2(const uint8_t *input,
                               int B, int L,
                               int *results,
                               float *kernel_ms, float *total_ms) {
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
    return g_engine.dispatch_v2(input, B, L, B_padded, results, kernel_ms, total_ms);
}

int batched_engine_dispatch_multi(
    const uint8_t *input,              // [L, B_padded]
    int B, int L,
    const int8_t *accept_masks,        // [n_patterns, N]
    int n_patterns,
    int *results,                      // [n_patterns * B]
    float *kernel_ms, float *total_ms)
{
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
    return g_engine.dispatch_multi(input, B, L, B_padded,
                                   accept_masks, n_patterns,
                                   results, kernel_ms, total_ms);
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

int batched_engine_init_sparse(int P, int sigma,
                               const int8_t *T_diag,
                               const int8_t *start_vecs,
                               const int8_t *accept_masks,
                               int max_B, int max_L) {
    g_engine.init_sparse(P, sigma, T_diag, start_vecs, accept_masks, max_B, max_L);
    return 0;
}

int batched_engine_dispatch_sparse(const uint8_t *input, int B, int L,
                                   int *results,
                                   float *kernel_ms, float *total_ms) {
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
    return g_engine.dispatch_sparse(input, B, L, B_padded, results, kernel_ms, total_ms);
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
    int N, const int8_t *start_vec,
    const int8_t *accept_mask, const int8_t *trans_matrices,
    const uint8_t *chars, int L, int identity_idx
) {
    std::vector<int8_t> state(N), new_state(N);
    memcpy(state.data(), start_vec, N);

    if (L == 0) {
        for (int r = 0; r < N; r++)
            if (state[r] > 0 && accept_mask[r] != 0) return true;
        return false;
    }

    for (int t = 0; t < L; t++) {
        int c = chars[t];
        if (c == identity_idx) continue;

        const int8_t *T = trans_matrices + c * N * N;
        memset(new_state.data(), 0, N);
        for (int row = 0; row < N; row++) {
            int32_t sum = 0;
            for (int k = 0; k < N; k++) {
                sum += (int32_t)T[row * N + k] * (int32_t)state[k];
            }
            new_state[row] = (int8_t)(sum > 0 ? 1 : 0);
        }
        memcpy(state.data(), new_state.data(), N);
    }

    for (int r = 0; r < N; r++)
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

    // Build start_vec: state 0 = 1, rest = 0
    int8_t start_vec[TILE];
    memset(start_vec, 0, TILE);
    start_vec[0] = 1;

    int max_B = 64;
    int max_L = 256;
    BatchedEngine eng;
    eng.init(TILE, 2, dfa.trans, dfa.accept, start_vec, max_B, max_L);

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
    int8_t start_vec[TILE];
    memset(start_vec, 0, TILE);
    start_vec[0] = 1;

    int B = 1024;
    int L = 128;
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
    int identity_idx = 2;

    BatchedEngine eng;
    eng.init(TILE, 2, dfa.trans, dfa.accept, start_vec, B_padded, L);

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
        bool seq = simulate_sequential(TILE, start_vec, dfa.accept, dfa.trans,
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

    int8_t start_vec[TILE];
    memset(start_vec, 0, TILE);
    start_vec[0] = 1;

    int identity_idx = sigma;  // 3
    int B = 64;
    int L = 16;
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

    BatchedEngine eng;
    eng.init(TILE, sigma, trans, accept, start_vec, B_padded, L);

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
        bool seq = simulate_sequential(TILE, start_vec, accept, trans,
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
    printf("\n=== Throughput Benchmark: V1 vs V2 (binary kernel) ===\n");
    EvenADFA dfa;
    int8_t start_vec[TILE];
    memset(start_vec, 0, TILE);
    start_vec[0] = 1;

    int batch_sizes[] = {1024, 4096, 16384, 65536, 262144};
    int lengths[]     = {128, 512, 2048};
    int n_batches = 5, n_lengths = 3;
    int identity_idx = 2;

    printf("  %8s  %8s  |  %10s  %10s  |  %10s  %10s  | speedup\n",
           "B", "L", "V1 kern ms", "V1 Gc/s", "V2 kern ms", "V2 Gc/s");
    printf("  %s\n", "--------------------------------------------------------------------------");

    for (int bi = 0; bi < n_batches; bi++) {
        for (int li = 0; li < n_lengths; li++) {
            int B = batch_sizes[bi];
            int L = lengths[li];
            int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

            BatchedEngine eng;
            eng.init(TILE, 2, dfa.trans, dfa.accept, start_vec, B_padded, L);

            srand(42);
            size_t input_size = (size_t)L * B_padded;
            uint8_t *input = new uint8_t[input_size];
            memset(input, (uint8_t)identity_idx, input_size);
            for (int b = 0; b < B; b++)
                for (int t = 0; t < L; t++)
                    input[t * B_padded + b] = rand() % 2;

            int *results = new int[B];

            // Warmup both
            for (int w = 0; w < 3; w++) {
                eng.dispatch(input, B, L, B_padded, results, nullptr, nullptr);
                eng.dispatch_v2(input, B, L, B_padded, results, nullptr, nullptr);
            }

            int iters = 20;
            float v1_kern = 0, v2_kern = 0;
            for (int it = 0; it < iters; it++) {
                float km;
                eng.dispatch(input, B, L, B_padded, results, &km, nullptr);
                v1_kern += km;
            }
            for (int it = 0; it < iters; it++) {
                float km;
                eng.dispatch_v2(input, B, L, B_padded, results, &km, nullptr);
                v2_kern += km;
            }
            v1_kern /= iters;
            v2_kern /= iters;
            double chars = (double)B * L;
            double v1_gchs = chars / (v1_kern * 1e6);
            double v2_gchs = chars / (v2_kern * 1e6);

            printf("  %8d  %8d  |  %10.4f  %10.3f  |  %10.4f  %10.3f  |  %.2fx\n",
                   B, L, v1_kern, v1_gchs, v2_kern, v2_gchs, v1_kern / v2_kern);

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

    // V2 correctness: compare against V1 on large random batch
    {
        printf("\n--- test_v2_correctness (V2 vs V1 cross-validate) ---\n");
        EvenADFA dfa;
        int8_t sv[TILE];
        memset(sv, 0, TILE);
        sv[0] = 1;

        int B = 4096, L = 256;
        int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
        int identity_idx = 2;

        BatchedEngine eng;
        eng.init(TILE, 2, dfa.trans, dfa.accept, sv, B_padded, L);

        srand(77);
        size_t input_size = (size_t)L * B_padded;
        uint8_t *inp = new uint8_t[input_size];
        memset(inp, (uint8_t)identity_idx, input_size);
        for (int b = 0; b < B; b++)
            for (int t = 0; t < L; t++)
                inp[t * B_padded + b] = rand() % 2;

        int *res_v1 = new int[B], *res_v2 = new int[B];
        float km;
        eng.dispatch(inp, B, L, B_padded, res_v1, &km, nullptr);
        eng.dispatch_v2(inp, B, L, B_padded, res_v2, &km, nullptr);

        int mismatches = 0;
        for (int b = 0; b < B; b++)
            if (res_v1[b] != res_v2[b]) mismatches++;

        char msg[128];
        snprintf(msg, sizeof(msg), "V2 vs V1 B=%d L=%d (%d mismatches)", B, L, mismatches);
        check(msg, mismatches == 0);

        delete[] inp; delete[] res_v1; delete[] res_v2;
        eng.destroy();
    }

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
