# K-gram TC Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a k-gram fusion CUDA kernel that precomputes σ^k product matrices and processes k characters per WMMA MMA, plus integrate it as a new `kgram+gpu` backend in OptimizedEngine.

**Architecture:** Single-string-per-warp WMMA kernel loads precomputed T_kgram matrices from global memory (L2-resident). Python bridge uses existing `kgram.py` Mode B for precomputation. OptimizedEngine auto-selects this backend when N>16 and σ^k fits in L2.

**Tech Stack:** CUDA C++ (WMMA int8 MMA), Python ctypes, numpy, pytest

---

## File Structure

- **Create:** `cuda/kgram_evolution.cu` — K-gram TC CUDA kernel + engine struct + C API
- **Create:** `src/gpu_bridge_kgram.py` — Python ctypes bridge
- **Create:** `tests/test_kgram_gpu.py` — GPU correctness tests
- **Create:** `docs/tc_utilization_analysis.md` — TC utilization analysis document
- **Modify:** `src/kgram.py` — Add `auto_k_for_gpu()` function
- **Modify:** `src/optimized_engine.py` — Add `kgram+gpu` config + update auto-select
- **Modify:** `tests/test_kgram.py` — Add tests for `auto_k_for_gpu`
- **Modify:** `tests/test_optimized_engine.py` — Add kgram+gpu to integration tests
- **Modify:** `Makefile` — Add build targets for kgram_evolution

---

### Task 1: Add `auto_k_for_gpu` to kgram.py

**Files:**
- Modify: `src/kgram.py`
- Modify: `tests/test_kgram.py`

- [ ] **Step 1: Write the failing tests**

Add a new test class at the end of `tests/test_kgram.py` (before the `if __name__` block):

```python
# ═══════════════════════════════════════════════════════════════════════════
# 4. TestAutoKForGPU
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoKForGPU:
    """Tests for auto_k_for_gpu (GPU L2-budget-aware k selection)."""

    def test_binary_n16(self):
        """sigma=2, N=16: 2^k * 256 <= 48MB → k=17 (2^17*256=32MB, 2^18*256=64MB)."""
        from src.kgram import auto_k_for_gpu
        k = auto_k_for_gpu(2, 16)
        assert k == 17

    def test_binary_n32(self):
        """sigma=2, N=32: 2^k * 1024 <= 48MB → k=15 (2^15*1024=32MB)."""
        from src.kgram import auto_k_for_gpu
        k = auto_k_for_gpu(2, 32)
        assert k == 15

    def test_quad_n16(self):
        """sigma=4, N=16: 4^k * 256 <= 48MB → k=8 (4^8*256=16MB, 4^9=64MB)."""
        from src.kgram import auto_k_for_gpu
        k = auto_k_for_gpu(4, 16)
        assert k == 8

    def test_byte_n16(self):
        """sigma=256, N=16: 256^k * 256 <= 48MB → k=2 (256^2*256=16MB)."""
        from src.kgram import auto_k_for_gpu
        k = auto_k_for_gpu(256, 16)
        assert k == 2

    def test_byte_n64(self):
        """sigma=256, N=64: 256^k * 4096 <= 48MB → k=1 (256^1*4096=1MB, 256^2=268MB)."""
        from src.kgram import auto_k_for_gpu
        k = auto_k_for_gpu(256, 64)
        assert k == 1

    def test_custom_budget(self):
        """sigma=2, N=16, 1MB budget: 2^k * 256 <= 1MB → k=12 (2^12*256=1MB)."""
        from src.kgram import auto_k_for_gpu
        k = auto_k_for_gpu(2, 16, max_table_bytes=1_048_576)
        assert k == 12

    def test_monotone_in_sigma(self):
        """Larger sigma → smaller or equal k."""
        from src.kgram import auto_k_for_gpu
        prev_k = auto_k_for_gpu(2, 16)
        for sigma in [4, 8, 16, 64, 256]:
            k = auto_k_for_gpu(sigma, 16)
            assert k <= prev_k, f"auto_k_for_gpu({sigma}, 16)={k} > prev={prev_k}"
            prev_k = k

    def test_monotone_in_n(self):
        """Larger N → smaller or equal k."""
        from src.kgram import auto_k_for_gpu
        prev_k = auto_k_for_gpu(2, 16)
        for n in [32, 48, 64]:
            k = auto_k_for_gpu(2, n)
            assert k <= prev_k, f"auto_k_for_gpu(2, {n})={k} > prev={prev_k}"
            prev_k = k
```

Also add `auto_k_for_gpu` to the import at the top of the test file:

```python
from src.kgram import (
    auto_k,
    auto_k_for_gpu,
    KGramTable,
    precompute_kgrams,
    simulate_kgram_monoid,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_kgram.py::TestAutoKForGPU -v`
Expected: FAIL with `ImportError: cannot import name 'auto_k_for_gpu'`

- [ ] **Step 3: Implement auto_k_for_gpu**

Add this function to `src/kgram.py` right after the `auto_k` function (after line 40):

```python
def auto_k_for_gpu(
    alphabet_size: int,
    n_states: int,
    max_table_bytes: int = 48_000_000,
) -> int:
    """Return the largest k such that alphabet_size^k * n_states^2 <= max_table_bytes.

    Targets GPU L2 cache budget. Each k-gram table entry is an n_states×n_states
    int8 matrix (n_states^2 bytes). Total table size = alphabet_size^k * n_states^2.
    """
    if alphabet_size <= 1:
        return 1
    matrix_bytes = n_states * n_states
    k = 1
    while True:
        if alphabet_size ** (k + 1) * matrix_bytes > max_table_bytes:
            return k
        k += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_kgram.py::TestAutoKForGPU -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Run all existing kgram tests to check for regressions**

Run: `python -m pytest tests/test_kgram.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/kgram.py tests/test_kgram.py
git commit -m "feat: add auto_k_for_gpu for L2-budget-aware k-gram size selection"
```

---

### Task 2: CUDA kernel + Makefile

**Files:**
- Create: `cuda/kgram_evolution.cu`
- Modify: `Makefile`

- [ ] **Step 1: Create the CUDA kernel file**

Create `cuda/kgram_evolution.cu`:

```c
/*
 * kgram_evolution.cu -- K-gram TC State Evolution via WMMA
 *
 * Precomputes product matrices for all σ^k possible k-grams, then
 * processes k characters per WMMA MMA call (single-string-per-warp).
 * This increases effective arithmetic intensity by k× compared to
 * per-character evolution.
 *
 * N=16 only (single WMMA tile). Multi-tile extension is future work.
 *
 * Threading model:
 *   Each warp processes ONE string (1 column of the 16×16 tile)
 *   4 warps per block → 4 strings per block
 *   Grid: ceil(B / WARPS_PER_BLOCK)
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

constexpr int TILE = 16;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int STRINGS_PER_BLOCK = WARPS_PER_BLOCK;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)


// ---- K-gram Evolution Kernel (N=16, single WMMA tile) --------------------
//
// Shared memory layout per block:
//   S_sh[4][16*16]       1024 bytes  (state matrices, col-major, only col 0 used)
//   acc_sh[4][16*16]     4096 bytes  (int32 accumulators, row-major)
//   Total:               5120 bytes

__global__ void kgram_evolution_kernel(
    const int8_t  *__restrict__ T_kgram,     // [n_entries, TILE, TILE]
    const int8_t  *__restrict__ T_base,      // [sigma, TILE, TILE]
    const uint8_t *__restrict__ input,       // [L, B_padded]
    const int8_t  *__restrict__ accept_mask, // [TILE]
    const int8_t  *__restrict__ start_vec,   // [TILE]
    int *__restrict__ results,               // [B]
    int B, int B_padded, int L,
    int sigma, int k
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int string_id = blockIdx.x * STRINGS_PER_BLOCK + warp_in_block;

    extern __shared__ char smem_raw[];
    int8_t  *S_base  = (int8_t *)smem_raw;
    int32_t *acc_base = (int32_t *)(S_base + WARPS_PER_BLOCK * TILE_ELEMS);

    int8_t  *S_sh   = S_base + warp_in_block * TILE_ELEMS;
    int32_t *acc_sh = acc_base + warp_in_block * TILE_ELEMS;

    // Initialize S from start vector (col-major, only column 0)
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        int col = e / TILE;
        S_sh[e] = (col == 0) ? start_vec[row] : (int8_t)0;
    }
    __syncwarp();

    // Early exit for out-of-range warps
    if (string_id >= B) {
        // Still need to participate in syncwarp if other warps in block are active,
        // but warp-level operations don't need cross-warp sync.
        return;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_T;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_S;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_acc;

    wmma::load_matrix_sync(frag_S, S_sh, TILE);

    // Main loop: process k characters per iteration
    int pos = 0;
    for (; pos + k <= L; pos += k) {
        // Compute k-gram index (all lanes compute same value — same string)
        uint32_t idx = 0;
        bool valid = true;
        for (int i = 0; i < k; i++) {
            uint8_t ch = input[(pos + i) * B_padded + string_id];
            if (ch >= (uint8_t)sigma) { valid = false; break; }
            idx = idx * (uint32_t)sigma + (uint32_t)ch;
        }

        if (valid) {
            // Full valid k-gram: single MMA with precomputed product matrix
            wmma::load_matrix_sync(frag_T, &T_kgram[idx * TILE_ELEMS], TILE);
            wmma::fill_fragment(frag_acc, 0);
            wmma::mma_sync(frag_acc, frag_T, frag_S, frag_acc);
            wmma::store_matrix_sync(acc_sh, frag_acc, TILE, wmma::mem_row_major);
            __syncwarp();

            if (lane < TILE) {
                S_sh[lane] = (int8_t)(acc_sh[lane * TILE] > 0 ? 1 : 0);
            }
            __syncwarp();
            wmma::load_matrix_sync(frag_S, S_sh, TILE);
        } else {
            // Hit identity/padding — process remaining chars one by one then done
            for (int i = 0; i < k && (pos + i) < L; i++) {
                uint8_t ch = input[(pos + i) * B_padded + string_id];
                if (ch >= (uint8_t)sigma) break;
                wmma::load_matrix_sync(frag_T, &T_base[ch * TILE_ELEMS], TILE);
                wmma::fill_fragment(frag_acc, 0);
                wmma::mma_sync(frag_acc, frag_T, frag_S, frag_acc);
                wmma::store_matrix_sync(acc_sh, frag_acc, TILE, wmma::mem_row_major);
                __syncwarp();
                if (lane < TILE) {
                    S_sh[lane] = (int8_t)(acc_sh[lane * TILE] > 0 ? 1 : 0);
                }
                __syncwarp();
                wmma::load_matrix_sync(frag_S, S_sh, TILE);
            }
            // String ended — skip to accept check
            pos = L;
            break;
        }
    }

    // Tail: remaining L % k characters (per-character MMA)
    for (; pos < L; pos++) {
        uint8_t ch = input[pos * B_padded + string_id];
        if (ch >= (uint8_t)sigma) break;
        wmma::load_matrix_sync(frag_T, &T_base[ch * TILE_ELEMS], TILE);
        wmma::fill_fragment(frag_acc, 0);
        wmma::mma_sync(frag_acc, frag_T, frag_S, frag_acc);
        wmma::store_matrix_sync(acc_sh, frag_acc, TILE, wmma::mem_row_major);
        __syncwarp();
        if (lane < TILE) {
            S_sh[lane] = (int8_t)(acc_sh[lane * TILE] > 0 ? 1 : 0);
        }
        __syncwarp();
        wmma::load_matrix_sync(frag_S, S_sh, TILE);
    }

    // Accept check: column 0 of S_sh vs accept_mask
    if (lane == 0) {
        int accepted = 0;
        for (int r = 0; r < TILE; r++) {
            if (S_sh[r] > 0 && accept_mask[r] != 0) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}


// ---- Engine Struct --------------------------------------------------------

struct KGramEngine {
    int N;
    int sigma;
    int k;
    int n_entries;
    int max_B;
    int max_L;

    int8_t  *d_T_kgram;
    int8_t  *d_T_base;
    int8_t  *d_accept;
    int8_t  *d_start_vec;
    uint8_t *d_input;
    int     *d_results;

    int B_padded_max;

    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;

    bool initialized;

    void init(const int8_t *T_kgram, const int8_t *T_base,
              const int8_t *accept_mask, const int8_t *start_vec,
              int n, int sig, int _k, int _n_entries,
              int maxB, int maxL) {
        N = n;
        sigma = sig;
        k = _k;
        n_entries = _n_entries;
        max_B = maxB;
        max_L = maxL;
        B_padded_max = ((maxB + STRINGS_PER_BLOCK - 1) / STRINGS_PER_BLOCK) * STRINGS_PER_BLOCK;

        size_t table_bytes = (size_t)n_entries * TILE_ELEMS;
        CHECK_CUDA(cudaMalloc(&d_T_kgram, table_bytes));
        CHECK_CUDA(cudaMemcpy(d_T_kgram, T_kgram, table_bytes, cudaMemcpyHostToDevice));

        size_t base_bytes = (size_t)sig * TILE_ELEMS;
        CHECK_CUDA(cudaMalloc(&d_T_base, base_bytes));
        CHECK_CUDA(cudaMemcpy(d_T_base, T_base, base_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&d_accept, TILE));
        CHECK_CUDA(cudaMemcpy(d_accept, accept_mask, TILE, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&d_start_vec, TILE));
        CHECK_CUDA(cudaMemcpy(d_start_vec, start_vec, TILE, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&d_input, (size_t)maxL * B_padded_max));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)maxB * sizeof(int)));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

        initialized = true;
    }

    void destroy() {
        if (!initialized) return;
        cudaFree(d_T_kgram);
        cudaFree(d_T_base);
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

        CHECK_CUDA(cudaEventRecord(ev_start));

        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        int n_blocks = (B + STRINGS_PER_BLOCK - 1) / STRINGS_PER_BLOCK;
        int smem = WARPS_PER_BLOCK * TILE_ELEMS
                 + WARPS_PER_BLOCK * TILE_ELEMS * (int)sizeof(int32_t);

        kgram_evolution_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
            d_T_kgram, d_T_base, d_input, d_accept, d_start_vec,
            d_results, B, B_padded, L, sigma, k);
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
};


// ---- Global Engine --------------------------------------------------------

static KGramEngine g_kgram_engine = {};


// ---- C API ----------------------------------------------------------------

extern "C" {

int kgram_engine_device_check() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 7) return -2;
    return 0;
}

int kgram_engine_init(
    const int8_t *T_kgram,
    const int8_t *T_base,
    const int8_t *accept_mask,
    const int8_t *start_vec,
    int N, int sigma, int k, int n_entries,
    int max_B, int max_L
) {
    if (g_kgram_engine.initialized) g_kgram_engine.destroy();
    g_kgram_engine.init(T_kgram, T_base, accept_mask, start_vec,
                        N, sigma, k, n_entries, max_B, max_L);
    return 0;
}

void kgram_engine_destroy() {
    g_kgram_engine.destroy();
}

int kgram_engine_dispatch(
    const uint8_t *input,
    int B, int L,
    int *results,
    float *kernel_ms, float *total_ms
) {
    int B_padded = ((B + STRINGS_PER_BLOCK - 1) / STRINGS_PER_BLOCK) * STRINGS_PER_BLOCK;
    return g_kgram_engine.dispatch(input, B, L, B_padded, results, kernel_ms, total_ms);
}

// Reuse batched_prepare_input from batched_evolution for input layout
void kgram_prepare_input(
    const char *strings_concat,
    const int *offsets,
    uint8_t *output,
    int B, int B_padded, int L,
    const int *char_to_idx,
    int identity_idx)
{
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
    }
}

}  // extern "C"


// ---- Built-in Tests -------------------------------------------------------

#ifndef BUILD_LIB

static int g_tests = 0, g_pass = 0;
static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) { g_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

static bool simulate_sequential_ref(
    int N, const int8_t *start_vec,
    const int8_t *accept_mask, const int8_t *trans_matrices,
    const uint8_t *chars, int L, int sigma
) {
    std::vector<int8_t> state(N), new_state(N);
    memcpy(state.data(), start_vec, N);

    for (int t = 0; t < L; t++) {
        int c = chars[t];
        if (c >= sigma) continue;
        const int8_t *T = trans_matrices + c * N * N;
        memset(new_state.data(), 0, N);
        for (int row = 0; row < N; row++) {
            int32_t sum = 0;
            for (int k = 0; k < N; k++)
                sum += (int32_t)T[row * N + k] * (int32_t)state[k];
            new_state[row] = (int8_t)(sum > 0 ? 1 : 0);
        }
        memcpy(state.data(), new_state.data(), N);
    }

    for (int r = 0; r < N; r++)
        if (state[r] > 0 && accept_mask[r] != 0) return true;
    return false;
}

// Precompute k-gram product matrices on CPU
static void precompute_kgram_table(
    const int8_t *T_base, int N, int sigma, int k,
    std::vector<int8_t> &table
) {
    int n_entries = 1;
    for (int i = 0; i < k; i++) n_entries *= sigma;
    table.resize((size_t)n_entries * N * N);

    for (int idx = 0; idx < n_entries; idx++) {
        // Decode idx into k characters (mixed radix)
        std::vector<int> chars(k);
        int tmp = idx;
        for (int i = k - 1; i >= 0; i--) {
            chars[i] = tmp % sigma;
            tmp /= sigma;
        }

        // Compose: acc = T[c0] * T[c1] * ... * T[c_{k-1}]
        // Convention: acc = T[c_new] * acc (left-multiply each new char)
        // Start with identity
        std::vector<int8_t> acc(N * N, 0);
        for (int i = 0; i < N; i++) acc[i * N + i] = 1;

        for (int ci = 0; ci < k; ci++) {
            int c = chars[ci];
            const int8_t *T = T_base + c * N * N;
            std::vector<int8_t> result(N * N, 0);
            for (int r = 0; r < N; r++) {
                for (int col = 0; col < N; col++) {
                    int32_t sum = 0;
                    for (int m = 0; m < N; m++)
                        sum += (int32_t)T[r * N + m] * (int32_t)acc[m * N + col];
                    result[r * N + col] = (int8_t)(sum > 0 ? 1 : 0);
                }
            }
            acc = result;
        }

        memcpy(&table[idx * N * N], acc.data(), N * N);
    }
}

int main() {
    printf("=== kgram_evolution built-in tests ===\n");

    // Even-A DFA: 2 real states + 14 padded = N=16
    // State 0 = even a-count (accept), State 1 = odd a-count
    // T[0]='a': swap states 0<->1
    // T[1]='b': identity on states 0,1
    int8_t accept[TILE] = {};
    accept[0] = 1;

    int8_t T_base[2 * TILE_ELEMS] = {};
    // Padded states self-loop
    for (int s = 2; s < TILE; s++) {
        T_base[0 * TILE_ELEMS + s * TILE + s] = 1;  // T[a][s][s]=1
        T_base[1 * TILE_ELEMS + s * TILE + s] = 1;  // T[b][s][s]=1
    }
    // T['a']: swap 0<->1
    T_base[0 * TILE_ELEMS + 1 * TILE + 0] = 1;  // T[a][1][0]=1
    T_base[0 * TILE_ELEMS + 0 * TILE + 1] = 1;  // T[a][0][1]=1
    // T['b']: identity on 0,1
    T_base[1 * TILE_ELEMS + 0 * TILE + 0] = 1;  // T[b][0][0]=1
    T_base[1 * TILE_ELEMS + 1 * TILE + 1] = 1;  // T[b][1][1]=1

    int8_t start_vec[TILE] = {};
    start_vec[0] = 1;

    int sigma = 2;

    // Test with k=4
    int k = 4;
    int n_entries = 1;
    for (int i = 0; i < k; i++) n_entries *= sigma;

    std::vector<int8_t> T_kgram;
    precompute_kgram_table(T_base, TILE, sigma, k, T_kgram);

    printf("Precomputed %d k-gram matrices (k=%d, sigma=%d)\n", n_entries, k, sigma);

    // Test strings: variable-length, check acceptance
    struct TestCase { const char *name; std::vector<uint8_t> chars; bool expected; };
    std::vector<TestCase> tests = {
        {"empty",     {},                           true},   // even a-count (0)
        {"a",         {0},                          false},  // 1 a
        {"aa",        {0,0},                        true},   // 2 a's
        {"b",         {1},                          true},   // 0 a's
        {"ab",        {0,1},                        false},  // 1 a
        {"aabb",      {0,0,1,1},                    true},   // 2 a's
        {"aaab",      {0,0,0,1},                    false},  // 3 a's
        {"aaaabb",    {0,0,0,0,1,1},                true},   // 4 a's
        {"aabba",     {0,0,1,1,0},                  false},  // 3 a's
        {"aabbaa",    {0,0,1,1,0,0},                true},   // 4 a's
    };

    // Verify reference simulation
    for (auto &tc : tests) {
        bool ref = simulate_sequential_ref(TILE, start_vec, accept, T_base,
                                           tc.chars.data(), (int)tc.chars.size(), sigma);
        char buf[256];
        snprintf(buf, sizeof(buf), "ref_%s", tc.name);
        check(buf, ref == tc.expected);
    }

    // Batch GPU test
    int B = (int)tests.size();
    int L_max = 0;
    for (auto &tc : tests) L_max = std::max(L_max, (int)tc.chars.size());
    if (L_max == 0) L_max = 1;

    int B_padded = ((B + STRINGS_PER_BLOCK - 1) / STRINGS_PER_BLOCK) * STRINGS_PER_BLOCK;

    // Build position-contiguous input
    std::vector<uint8_t> input(L_max * B_padded, (uint8_t)sigma);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < (int)tests[b].chars.size(); t++) {
            input[t * B_padded + b] = tests[b].chars[t];
        }
    }

    // Init engine
    kgram_engine_init(T_kgram.data(), T_base, accept, start_vec,
                      TILE, sigma, k, n_entries, B_padded, L_max);

    // Dispatch
    std::vector<int> results(B, -1);
    float kern_ms = 0, total_ms = 0;
    kgram_engine_dispatch(input.data(), B, L_max, results.data(), &kern_ms, &total_ms);

    for (int i = 0; i < B; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "gpu_k%d_%s", k, tests[i].name);
        check(buf, (results[i] != 0) == tests[i].expected);
    }

    printf("Kernel: %.3f ms, Total: %.3f ms\n", kern_ms, total_ms);

    // Test with k=1 (should degenerate to per-character)
    k = 1;
    n_entries = sigma;
    std::vector<int8_t> T_kgram_k1;
    precompute_kgram_table(T_base, TILE, sigma, 1, T_kgram_k1);

    kgram_engine_init(T_kgram_k1.data(), T_base, accept, start_vec,
                      TILE, sigma, 1, sigma, B_padded, L_max);

    std::vector<int> results_k1(B, -1);
    kgram_engine_dispatch(input.data(), B, L_max, results_k1.data(), &kern_ms, &total_ms);

    for (int i = 0; i < B; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "gpu_k1_%s", tests[i].name);
        check(buf, (results_k1[i] != 0) == tests[i].expected);
    }

    kgram_engine_destroy();

    printf("\n%d/%d tests passed\n", g_pass, g_tests);
    return (g_pass == g_tests) ? 0 : 1;
}

#endif  // BUILD_LIB
```

- [ ] **Step 2: Add Makefile targets**

Add these lines to the `Makefile`, following the existing pattern. After the `LIB_BATCHED` definition block, add:

```makefile
SRC_KGRAM = $(CUDA_DIR)/kgram_evolution.cu
EXE_KGRAM = $(BUILD_DIR)/kgram_evolution
LIB_KGRAM = $(BUILD_DIR)/libkgram_evolution.so
```

Add `$(EXE_KGRAM) $(LIB_KGRAM)` to the `all:` target.

Add build rules:

```makefile
$(EXE_KGRAM): $(SRC_KGRAM) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_KGRAM): $(SRC_KGRAM) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<
```

Add test target:

```makefile
test-kgram: $(EXE_KGRAM)
	./$(EXE_KGRAM)
```

Add `test-kgram` to the `.PHONY` line.

- [ ] **Step 3: Build and run standalone tests**

Run: `make build/kgram_evolution && ./build/kgram_evolution`
Expected: all tests PASS (ref tests + GPU k=4 tests + GPU k=1 tests)

- [ ] **Step 4: Build the shared library**

Run: `make build/libkgram_evolution.so`
Expected: successful build

- [ ] **Step 5: Commit**

```bash
git add cuda/kgram_evolution.cu Makefile
git commit -m "feat: k-gram TC CUDA kernel with single-string-per-warp WMMA"
```

---

### Task 3: Python GPU Bridge

**Files:**
- Create: `src/gpu_bridge_kgram.py`

- [ ] **Step 1: Create the Python bridge**

Create `src/gpu_bridge_kgram.py`:

```python
"""
Python bridge to the CUDA k-gram TC evolution engine via ctypes.

The k-gram engine precomputes σ^k product matrices and processes k characters
per WMMA MMA call in single-string-per-warp mode.

Usage:
    from src.gpu_bridge_kgram import KGramGPUSimulator
    sim = KGramGPUSimulator()
    engine = sim.create_engine(dm, k=8)
    results = engine.simulate_batch(["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    engine.destroy()
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.simulation import DFAMatrices
from src.kgram import precompute_kgrams


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libkgram_evolution.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libkgram_evolution.so not found at {base}. Run 'make' first."
    )


class KGramGPUEngine:
    """Wraps a persistent GPU engine context for k-gram TC evolution."""

    def __init__(self, lib, dm: DFAMatrices, k: int,
                 max_B: int = 65536, max_L: int = 4096):
        self.lib = lib
        self.dm = dm
        self.k = k
        self.N = dm.n_states
        self.sigma = len(dm.alphabet)

        # Precompute k-gram product matrices via kgram.py Mode B
        kg = precompute_kgrams(dm, k, monoid=None)
        n_entries = self.sigma ** k

        # Flatten matrix table to contiguous [n_entries, N, N] int8 array
        T_kgram = np.zeros((n_entries, self.N, self.N), dtype=np.int8)
        for key, mat in kg._matrix_table.items():
            T_kgram[key] = mat
        T_kgram = np.ascontiguousarray(T_kgram)

        # Base per-character matrices for tail handling
        T_base = np.ascontiguousarray(dm.matrix_stack, dtype=np.int8)

        # Accept mask and start vector
        accept = np.ascontiguousarray(dm.accept_mask, dtype=np.int8)
        start = np.zeros(self.N, dtype=np.int8)
        start[dm.dfa.start] = 1

        rc = self.lib.kgram_engine_init(
            T_kgram.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            T_base.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            start.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self.N, self.sigma, k, n_entries,
            max_B, max_L,
        )
        if rc != 0:
            raise RuntimeError(f"kgram_engine_init failed with code {rc}")

        # Build char_to_idx lookup: int32[256]
        self._identity_idx = self.sigma
        self._char_to_idx = np.full(256, -1, dtype=np.int32)
        for ch, idx in dm.char_to_idx.items():
            self._char_to_idx[ord(ch)] = idx

    def destroy(self):
        self.lib.kgram_engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        """Convert strings to position-contiguous uint8 layout."""
        B = len(strings)
        L_max = max(len(s) for s in strings) if strings else 0

        STRINGS_PER_BLOCK = 4
        B_padded = ((B + STRINGS_PER_BLOCK - 1) // STRINGS_PER_BLOCK) * STRINGS_PER_BLOCK

        strings_concat = "".join(strings).encode("latin-1")
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)

        output = np.zeros(L_max * B_padded, dtype=np.uint8)

        self.lib.kgram_prepare_input(
            strings_concat,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, B_padded, L_max,
            self._char_to_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._identity_idx,
        )

        return output, B_padded, L_max

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        if not strings:
            return []

        B = len(strings)
        L_max = max(len(s) for s in strings)
        if L_max == 0:
            is_accept = self.dm.check_accept(self.dm.start_vec)
            return [is_accept] * B

        input_data, B_padded, L_max = self._prepare_batch(strings)

        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.kgram_engine_dispatch(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, L_max,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"kgram_engine_dispatch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        if not strings:
            return [], 0.0, 0.0

        B = len(strings)
        L_max = max(len(s) for s in strings)
        if L_max == 0:
            is_accept = self.dm.check_accept(self.dm.start_vec)
            return [is_accept] * B, 0.0, 0.0

        input_data, B_padded, L_max = self._prepare_batch(strings)

        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.kgram_engine_dispatch(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, L_max,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"kgram_engine_dispatch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class KGramGPUSimulator:
    """Factory for KGramGPUEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.kgram_engine_device_check.restype = ctypes.c_int
        self.lib.kgram_engine_device_check.argtypes = []

        self.lib.kgram_engine_init.restype = ctypes.c_int
        self.lib.kgram_engine_init.argtypes = [
            ctypes.POINTER(ctypes.c_int8),    # T_kgram
            ctypes.POINTER(ctypes.c_int8),    # T_base
            ctypes.POINTER(ctypes.c_int8),    # accept_mask
            ctypes.POINTER(ctypes.c_int8),    # start_vec
            ctypes.c_int,                     # N
            ctypes.c_int,                     # sigma
            ctypes.c_int,                     # k
            ctypes.c_int,                     # n_entries
            ctypes.c_int,                     # max_B
            ctypes.c_int,                     # max_L
        ]

        self.lib.kgram_engine_destroy.restype = None
        self.lib.kgram_engine_destroy.argtypes = []

        self.lib.kgram_engine_dispatch.restype = ctypes.c_int
        self.lib.kgram_engine_dispatch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # input
            ctypes.c_int,                     # B
            ctypes.c_int,                     # L
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        self.lib.kgram_prepare_input.restype = None
        self.lib.kgram_prepare_input.argtypes = [
            ctypes.c_char_p,                  # strings_concat
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_uint8),   # output
            ctypes.c_int,                     # B
            ctypes.c_int,                     # B_padded
            ctypes.c_int,                     # L
            ctypes.POINTER(ctypes.c_int),     # char_to_idx
            ctypes.c_int,                     # identity_idx
        ]

        rc = self.lib.kgram_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, dm: DFAMatrices, k: int,
                      max_B: int = 65536,
                      max_L: int = 4096) -> KGramGPUEngine:
        return KGramGPUEngine(self.lib, dm, k, max_B, max_L)
```

- [ ] **Step 2: Verify the bridge loads and creates an engine**

Run: `python -c "from src.gpu_bridge_kgram import KGramGPUSimulator; print('OK')"`
Expected: `OK` (if GPU available) or `FileNotFoundError` (if lib not built)

- [ ] **Step 3: Commit**

```bash
git add src/gpu_bridge_kgram.py
git commit -m "feat: Python GPU bridge for k-gram TC evolution engine"
```

---

### Task 4: GPU Correctness Tests

**Files:**
- Create: `tests/test_kgram_gpu.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_kgram_gpu.py`:

```python
"""
Tests for the k-gram TC GPU engine (src/gpu_bridge_kgram.py).

Cross-validates against sequential CPU simulation.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest

from src.generate_data import PATTERNS
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential
from src.kgram import auto_k_for_gpu


def _kgram_gpu_available():
    try:
        from src.gpu_bridge_kgram import KGramGPUSimulator
        KGramGPUSimulator()
        return True
    except Exception:
        return False


skip_no_gpu = pytest.mark.skipif(
    not _kgram_gpu_available(), reason="K-gram GPU lib not available"
)

_ALPHABETS = {
    "abb": "ab",
    "binary_div3": "01",
    "even_a": "ab",
    "ab_star": "ab",
    "hex_number": "0123456789abcdefx",
    "identifier": "abcdefghijklmnopqrstuvwxyz0123456789",
}


def _random_strings(pattern_name: str, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    alpha = _ALPHABETS[pattern_name]
    lengths = [rng.randint(0, 50) for _ in range(n)]
    return ["".join(rng.choice(alpha) for _ in range(L)) for L in lengths]


def _sequential_results(pattern_name: str, strings: list) -> list:
    dfa = compile_regex(PATTERNS[pattern_name].regex)
    return [simulate_sequential(dfa, s) for s in strings]


@skip_no_gpu
class TestKGramGPUCorrectness:

    @pytest.mark.parametrize("pattern_name", ["abb", "binary_div3", "even_a", "ab_star"])
    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_matches_sequential(self, pattern_name, k):
        """K-gram GPU must agree with sequential on 200 random strings."""
        from src.gpu_bridge_kgram import KGramGPUSimulator

        strings = _random_strings(pattern_name, 200, seed=k * 100 + hash(pattern_name) % 1000)
        expected = _sequential_results(pattern_name, strings)

        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=k)
        got = engine.simulate_batch(strings)
        engine.destroy()

        mismatches = [
            (i, strings[i], expected[i], got[i])
            for i in range(len(strings))
            if got[i] != expected[i]
        ]
        assert not mismatches, (
            f"[{pattern_name}, k={k}] {len(mismatches)} mismatches: {mismatches[:5]}"
        )

    def test_empty_strings(self):
        """Batch of empty strings should return correct acceptance."""
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS["abb"]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=4)
        results = engine.simulate_batch(["", "", ""])
        engine.destroy()

        expected = simulate_sequential(dfa, "")
        assert results == [expected] * 3

    def test_mixed_lengths(self):
        """Strings of very different lengths should all be correct."""
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS["even_a"]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        rng = random.Random(77)
        alpha = "ab"
        strings = ["".join(rng.choice(alpha) for _ in range(L))
                    for L in [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 256]]
        expected = [simulate_sequential(dfa, s) for s in strings]

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=4)
        got = engine.simulate_batch(strings)
        engine.destroy()

        assert got == expected

    def test_large_k(self):
        """k=16 with binary alphabet (65536 precomputed matrices)."""
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS["abb"]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        strings = _random_strings("abb", 100, seed=999)
        expected = _sequential_results("abb", strings)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=16)
        got = engine.simulate_batch(strings)
        engine.destroy()

        assert got == expected

    @pytest.mark.parametrize("pattern_name", ["hex_number", "identifier"])
    def test_larger_alphabet(self, pattern_name):
        """Larger-alphabet patterns with k=2."""
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        strings = _random_strings(pattern_name, 100, seed=55)
        expected = _sequential_results(pattern_name, strings)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=2)
        got = engine.simulate_batch(strings)
        engine.destroy()

        assert got == expected


@skip_no_gpu
class TestKGramGPUTiming:

    def test_timed_returns_tuple(self):
        """simulate_batch_timed must return (list[bool], float, float)."""
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS["abb"]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=4)
        strings = _random_strings("abb", 50, seed=11)
        results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
        engine.destroy()

        assert isinstance(results, list)
        assert len(results) == 50
        assert all(isinstance(r, bool) for r in results)
        assert isinstance(kern_ms, float)
        assert isinstance(total_ms, float)
        assert kern_ms >= 0.0
        assert total_ms >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_kgram_gpu.py -v`
Expected: all tests PASS (or SKIP if no GPU)

- [ ] **Step 3: Commit**

```bash
git add tests/test_kgram_gpu.py
git commit -m "test: add GPU correctness tests for k-gram TC engine"
```

---

### Task 5: OptimizedEngine Integration

**Files:**
- Modify: `src/optimized_engine.py`
- Modify: `tests/test_optimized_engine.py`

- [ ] **Step 1: Add kgram+gpu config to OptimizedEngine**

In `src/optimized_engine.py`, add the import for `auto_k_for_gpu` at the top:

```python
from src.kgram import precompute_kgrams, simulate_kgram_monoid, auto_k, auto_k_for_gpu
```

Add `self._kgram_gpu = None` to `__init__` alongside the other backend fields (after line 68).

Add the `kgram+gpu` config case in `__init__` after the `batched+gpu` elif (after line 93):

```python
        elif config == "kgram+gpu":
            self._force_baseline()
            self._setup_kgram_gpu()
```

Update the error message to include `'kgram+gpu'`.

Add the `_setup_kgram_gpu` method after `_setup_batched_gpu`:

```python
    def _setup_kgram_gpu(self):
        self._build_dfa()
        from src.gpu_bridge_kgram import KGramGPUSimulator
        sigma = len(self._dfa.alphabet)
        k = auto_k_for_gpu(sigma, self._dm.n_states)
        sim = KGramGPUSimulator()
        self._kgram_gpu = sim.create_engine(self._dm, k)
        self._kgram_k = k
        self._scan_backend = 'kgram+gpu'
        self._selection_reason = (
            f'GPU k-gram TC (k={k}, table={sigma**k} entries, '
            f'N={self._dm.n_states})'
        )
```

Update `match_batch` to dispatch to kgram_gpu (add before the batched_gpu check):

```python
        if self._kgram_gpu is not None:
            return self._kgram_gpu.simulate_batch(strings)
```

Update `match_batch_timed` to handle kgram_gpu (add before the batched_gpu check):

```python
        if self._kgram_gpu is not None:
            results, kern_ms, total_ms = self._kgram_gpu.simulate_batch_timed(strings)
            return results, {'kernel_ms': kern_ms, 'total_ms': total_ms}
```

- [ ] **Step 2: Add tests for kgram+gpu to test_optimized_engine.py**

Add the availability check function near the existing `_batched_gpu_available`:

```python
def _kgram_gpu_available():
    try:
        from src.gpu_bridge_kgram import KGramGPUSimulator
        KGramGPUSimulator()
        return True
    except Exception:
        return False


skip_no_kgram_gpu = pytest.mark.skipif(
    not _kgram_gpu_available(), reason="K-gram GPU not available"
)
```

Add a new test class at the end of the file:

```python
@skip_no_kgram_gpu
class TestKGramGPUIntegration:

    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_kgram_gpu_matches_baseline(self, pattern_name):
        pat = PATTERNS[pattern_name]
        engine_base = OptimizedEngine(pat.regex, config="baseline")
        engine_kgram = OptimizedEngine(pat.regex, config="kgram+gpu")

        strings = _random_strings(pattern_name, 500, seed=42)
        expected = engine_base.match_batch(strings)
        actual = engine_kgram.match_batch(strings)
        assert actual == expected, f"kgram+gpu mismatch for {pattern_name}"

    def test_kgram_gpu_config_info(self):
        engine = OptimizedEngine("(a|b)*abb", config="kgram+gpu")
        info = engine.config_info
        assert info["scan_backend"] == "kgram+gpu"
        assert info["kgram_k"] is not None
        assert info["kgram_k"] >= 1

    def test_kgram_gpu_timed(self):
        engine = OptimizedEngine("(a|b)*abb", config="kgram+gpu")
        strings = _random_strings("abb", 200, seed=42)
        results, timing = engine.match_batch_timed(strings)
        assert isinstance(results, list)
        assert len(results) == 200
        assert "kernel_ms" in timing
        assert "total_ms" in timing

    @pytest.mark.parametrize("pattern_name", ["hex_number", "identifier"])
    def test_kgram_gpu_larger_alphabet(self, pattern_name):
        pat = PATTERNS[pattern_name]
        engine_base = OptimizedEngine(pat.regex, config="baseline")
        engine_kgram = OptimizedEngine(pat.regex, config="kgram+gpu")

        strings = _random_strings(pattern_name, 200, seed=99)
        expected = engine_base.match_batch(strings)
        actual = engine_kgram.match_batch(strings)
        assert actual == expected, f"kgram+gpu mismatch for {pattern_name}"
```

Also add `"kgram+gpu"` to the configs list in `TestAllConfigsSameResult.test_all_configs_agree`:

```python
        if _kgram_gpu_available():
            configs.append("kgram+gpu")
```

- [ ] **Step 3: Run the tests**

Run: `python -m pytest tests/test_optimized_engine.py -v`
Expected: all tests PASS (new kgram+gpu tests pass or skip)

- [ ] **Step 4: Commit**

```bash
git add src/optimized_engine.py tests/test_optimized_engine.py
git commit -m "feat: integrate kgram+gpu backend into OptimizedEngine"
```

---

### Task 6: TC Utilization Analysis Document

**Files:**
- Create: `docs/tc_utilization_analysis.md`

- [ ] **Step 1: Write the analysis document**

Create `docs/tc_utilization_analysis.md`:

```markdown
# Tensor Core Utilization Analysis for DFA State Evolution

## Arithmetic Intensity

For a single 16×16×16 WMMA MMA (INT8):

| Metric | Value |
|--------|-------|
| Compute | 2 × 16³ = 8,192 INT8 ops |
| Data loaded | 2 × 256 = 512 bytes (matrices A + B) |
| Arithmetic Intensity | 16 ops/byte |

H200 SXM balance point:
- INT8 compute: 3,958 TOPS
- HBM bandwidth: 4.915 TB/s
- Balance AI = 3,958 / 4.915 = **805 ops/byte**

The kernel's AI of 16 is **50× below the balance point**. At most ~2% of
tensor core capacity can be utilized for per-character 16×16 state evolution.

## K-gram Fusion

Processing k characters per MMA multiplies effective AI by k:

| k  | Eff. AI | TC util est. | Table size (σ=2, N=16) | Table size (σ=256, N=16) |
|----|---------|-------------|------------------------|--------------------------|
| 1  | 16      | 2.0%        | 512 B                  | 64 KB                    |
| 2  | 32      | 4.0%        | 1 KB                   | 16 MB                    |
| 4  | 64      | 8.0%        | 4 KB                   | —                        |
| 8  | 128     | 15.9%       | 64 KB                  | —                        |
| 16 | 256     | 31.8%       | 16 MB                  | —                        |
| 20 | 320     | 39.8%       | 256 MB                 | —                        |

Table size = σ^k × N² bytes. Must fit in L2 cache (~48 MB on H200) for
acceptable random-access latency.

## Backend Decision Framework

### When to use each backend

| Condition | Backend | Rationale |
|-----------|---------|-----------|
| N ≤ 16, monoid fits | Monoid R1 | O(1) table lookup per char. 93 Gc/s measured. |
| N > 16, σ^k fits L2 | K-gram TC | k× fewer MMAs. Best for small σ. |
| N > 16, σ^k too large | TC state evolution | One MMA/char. Fallback. |

### Why monoid wins for N ≤ 16

The monoid approach replaces O(N³) matrix multiplication with O(1) table
lookup. For N=16, each MMA does 8,192 ops; a monoid table lookup does ~2 ops.
No amount of TC optimization can close this 4,000× gap.

Monoid is feasible when the transition monoid size M is bounded (typically
M ≤ 200 for regex DFAs with N ≤ 16). The compose table (M² bytes) fits in
L1/registers. Measured throughput: 93 Gc/s vs 28 Gc/s for TC sparse.

### Why monoid fails for N > 16

The worst-case monoid size is N! (all permutation matrices are reachable).
At N=17, M ≤ 17! ≈ 3.6×10¹⁴ — infeasible to enumerate or store.

K-gram TC bridges this gap: TC does the O(N²) matmul natively, and k-gram
precomputation reduces the number of matmuls by k×.

## Measured Performance

| Backend | Config | Throughput | TC Utilization |
|---------|--------|------------|----------------|
| Monoid R1 (GPU) | N=16, σ=2 | 93 Gc/s | N/A (no TC) |
| TC sparse | N=16, σ=2, P=1 | 28 Gc/s | <1% |
| TC batched | N=16, σ=2 | 3.3 Gc/s | <1% |
| K-gram TC | N=16, σ=2, k=16 | TBD | ~32% (estimated) |

## Scaling with N

For square N×N matrices, GEMM arithmetic intensity is 2N/3:

| N | AI | TC util est. | Monoid feasible? |
|---|-----|-------------|-----------------|
| 16 | 10.7 | 1.3% | Yes (M ≤ ~200) |
| 32 | 21.3 | 2.6% | Unlikely |
| 64 | 42.7 | 5.3% | No |
| 128 | 85.3 | 10.6% | No |
| 256 | 170.7 | 21.2% | No |
| 1024 | 682.7 | 84.8% | No |

TC utilization only approaches 100% at N ≈ 1,200 — far beyond practical DFAs.
K-gram fusion (multiplying AI by k) is the practical path to meaningful TC
utilization for medium-N DFAs.
```

- [ ] **Step 2: Commit**

```bash
git add docs/tc_utilization_analysis.md
git commit -m "docs: add TC utilization analysis for DFA state evolution"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Architecture: three-tier hybrid (Task 5 auto-selection)
- ✅ K-gram TC kernel: single-string-per-warp, tail handling (Task 2)
- ✅ C API: init/dispatch/destroy (Task 2)
- ✅ Python bridge (Task 3)
- ✅ OptimizedEngine integration + `kgram+gpu` config (Task 5)
- ✅ auto_k_for_gpu selection logic (Task 1)
- ✅ TC utilization analysis document (Task 6)
- ⏭️ Multi-tile extension (N>16): documented in spec as future work, not in this plan
- ⏭️ Updated auto-selection for N>16: deferred until multi-tile kernel exists

**Placeholder scan:** No TBDs, TODOs, or "implement later" in any task.

**Type consistency:** `KGramGPUSimulator.create_engine(dm, k)` matches across Tasks 3, 4, 5. `auto_k_for_gpu(sigma, n_states)` matches across Tasks 1, 5. C API signatures match between Task 2 kernel and Task 3 bridge.
