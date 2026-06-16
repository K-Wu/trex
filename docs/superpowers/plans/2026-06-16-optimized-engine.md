# Optimized DFA/NFA Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four composable optimizations (transition monoid, k-gram, binary encoding, NFA path) with auto-selection, compounding to eliminate O(N³) matrix math from the DFA scan hot path.

**Architecture:** Each optimization is an independent, testable layer. The transition monoid replaces matrix multiplication with O(1) table lookups. k-gram precomputation reduces scan length by factor k. Binary encoding cuts memory 8x via WMMA b1. NFA path skips DFA construction entirely. An `OptimizedEngine` auto-selects the best combination per regex.

**Tech Stack:** Python 3, numpy, CUDA C++ (WMMA int8 + b1), ctypes, pytest

---

### Task 1: Transition Monoid — Python Precompute

**Files:**
- Create: `src/monoid.py`
- Test: `tests/test_monoid.py`

- [ ] **Step 1: Write failing tests for monoid computation**

```python
# tests/test_monoid.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, _matmul_int8, simulate_sequential
from src.generate_data import PATTERNS


class TestMonoidCompute:
    def _get_monoid(self, pattern_name):
        from src.monoid import compute_monoid
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        return compute_monoid(dm), dm, dfa

    def test_abb_monoid_size(self):
        """(a|b)*abb has 5 states, monoid should be small (< 100 elements)."""
        md, dm, dfa = self._get_monoid('abb')
        assert md.size > 0
        assert md.size < 100

    def test_even_a_monoid_size(self):
        md, dm, dfa = self._get_monoid('even_a')
        assert md.size > 0
        assert md.size < 50

    def test_closure_property(self):
        """For all pairs in the monoid, their product is also in the monoid."""
        md, dm, dfa = self._get_monoid('abb')
        for i in range(md.size):
            for j in range(md.size):
                composed = md.compose_table[i, j]
                assert 0 <= composed < md.size, (
                    f"compose[{i}][{j}]={composed} out of range [0, {md.size})")

    def test_compose_matches_matmul(self):
        """compose[i][j] corresponds to the matrix product of elements i and j."""
        md, dm, dfa = self._get_monoid('abb')
        for i in range(md.size):
            for j in range(md.size):
                product = _matmul_int8(md.elements[i], md.elements[j])
                k = md.compose_table[i, j]
                np.testing.assert_array_equal(
                    product, md.elements[k],
                    err_msg=f"compose[{i}][{j}]={k} does not match matmul"
                )

    def test_char_to_monoid_mapping(self):
        """Each character maps to a valid monoid index whose matrix matches."""
        md, dm, dfa = self._get_monoid('abb')
        for ch in dm.alphabet:
            idx = md.char_to_monoid[ch]
            assert 0 <= idx < md.size
            np.testing.assert_array_equal(
                dm.matrices[ch], md.elements[idx])

    def test_accept_table(self):
        """accept_table[m] matches applying element m to start_vec."""
        md, dm, dfa = self._get_monoid('abb')
        for i in range(md.size):
            state = md.elements[i].astype(np.int32) @ dm.start_vec.astype(np.int32)
            expected = dm.check_accept(state.astype(np.int8))
            assert md.accept_table[i] == expected, (
                f"accept_table[{i}]={md.accept_table[i]} != {expected}")

    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_identity_element(self, pattern_name):
        """Identity matrix should be in the monoid (it's the product of zero elements)."""
        md, dm, dfa = self._get_monoid(pattern_name)
        identity = np.eye(dm.n_states, dtype=np.int8)
        found = False
        for i in range(md.size):
            if np.array_equal(md.elements[i], identity):
                found = True
                assert md.identity_idx == i
                break
        assert found, "Identity matrix not found in monoid"


class TestMonoidSimulate:
    """Cross-validate monoid-based simulation against sequential DFA simulation."""

    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_known_strings(self, pattern_name):
        from src.monoid import compute_monoid, simulate_monoid
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)

        import random
        random.seed(42)
        alpha = sorted(dfa.alphabet)
        for _ in range(50):
            L = random.randint(0, 200)
            s = ''.join(random.choice(alpha) for _ in range(L))
            expected = simulate_sequential(dfa, s)
            got = simulate_monoid(md, dm, s)
            assert got == expected, (
                f"{pattern_name} L={L} s={s[:20]}... expected={expected} got={got}")

    @pytest.mark.parametrize("pattern_name", ['hex_number', 'identifier'])
    def test_larger_alphabet(self, pattern_name):
        from src.monoid import compute_monoid, simulate_monoid
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)

        import random
        random.seed(123)
        alpha = sorted(dfa.alphabet)
        for _ in range(30):
            L = random.randint(0, 100)
            s = ''.join(random.choice(alpha) for _ in range(L))
            expected = simulate_sequential(dfa, s)
            got = simulate_monoid(md, dm, s)
            assert got == expected


class TestMonoidSizeGuard:
    def test_cap_respected(self):
        """compute_monoid with a low cap should return None when monoid is too large."""
        from src.monoid import compute_monoid
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        result = compute_monoid(dm, max_size=2)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_monoid.py -v 2>&1 | head -30`
Expected: ImportError — `src.monoid` does not exist yet.

- [ ] **Step 3: Implement monoid computation**

```python
# src/monoid.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from src.simulation import DFAMatrices, _matmul_int8


@dataclass
class MonoidData:
    elements: list[np.ndarray]     # list of (N, N) int8 matrices
    compose_table: np.ndarray      # (M, M) uint16 — compose_table[i][j] = index of elements[i] @ elements[j]
    char_to_monoid: dict[str, int] # character → monoid index
    accept_table: np.ndarray       # (M,) bool — accept_table[i] = does elements[i] @ start_vec land in accept?
    identity_idx: int              # index of identity matrix in elements
    size: int                      # len(elements)


def _matrix_key(m: np.ndarray) -> bytes:
    return m.tobytes()


def compute_monoid(dm: DFAMatrices, max_size: int = 65536) -> MonoidData | None:
    N = dm.n_states
    identity = np.eye(N, dtype=np.int8)

    elements = [identity]
    key_to_idx = {_matrix_key(identity): 0}
    identity_idx = 0

    # Add per-character matrices
    char_to_monoid = {}
    for ch in dm.alphabet:
        mat = dm.matrices[ch]
        key = _matrix_key(mat)
        if key not in key_to_idx:
            key_to_idx[key] = len(elements)
            elements.append(mat.copy())
        char_to_monoid[ch] = key_to_idx[key]

    # BFS closure
    queue_start = 0
    while queue_start < len(elements):
        if len(elements) > max_size:
            return None
        batch_end = len(elements)
        for i in range(queue_start, batch_end):
            for j in range(len(elements)):
                product = _matmul_int8(elements[i], elements[j])
                key = _matrix_key(product)
                if key not in key_to_idx:
                    if len(elements) >= max_size:
                        return None
                    key_to_idx[key] = len(elements)
                    elements.append(product)
                product = _matmul_int8(elements[j], elements[i])
                key = _matrix_key(product)
                if key not in key_to_idx:
                    if len(elements) >= max_size:
                        return None
                    key_to_idx[key] = len(elements)
                    elements.append(product)
        queue_start = batch_end

    M = len(elements)

    # Build composition table
    compose_table = np.zeros((M, M), dtype=np.uint16)
    for i in range(M):
        for j in range(M):
            product = _matmul_int8(elements[i], elements[j])
            compose_table[i, j] = key_to_idx[_matrix_key(product)]

    # Build accept table
    accept_table = np.zeros(M, dtype=bool)
    for i in range(M):
        state = elements[i].astype(np.int32) @ dm.start_vec.astype(np.int32)
        accept_table[i] = dm.check_accept(state.astype(np.int8))

    return MonoidData(
        elements=elements,
        compose_table=compose_table,
        char_to_monoid=char_to_monoid,
        accept_table=accept_table,
        identity_idx=identity_idx,
        size=M,
    )


def simulate_monoid(md: MonoidData, dm: DFAMatrices, input_str: str) -> bool:
    if not input_str:
        return md.accept_table[md.identity_idx]

    idx = md.char_to_monoid[input_str[0]]
    for ch in input_str[1:]:
        c_idx = md.char_to_monoid[ch]
        idx = md.compose_table[c_idx, idx]

    return bool(md.accept_table[idx])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_monoid.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/monoid.py tests/test_monoid.py
git commit -m "feat: add transition monoid precomputation with CPU simulation"
```

---

### Task 2: Monoid Scan CUDA Kernel

**Files:**
- Create: `cuda/monoid_scan.cu`
- Modify: `Makefile`

- [ ] **Step 1: Write the CUDA monoid scan kernel with built-in tests**

```c
// cuda/monoid_scan.cu — Integer prefix scan with table-lookup composition
//
// Two execution regimes:
//   R1: Warp-per-string sequential scan (many short strings)
//   R3: Decoupled look-back (long strings)
//
// The compose table is loaded into shared memory for O(1) per-step lookups.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

constexpr int WARP_SIZE = 32;
constexpr int R1_WARPS_PER_BLOCK = 4;
constexpr int R1_BLOCK_SIZE = R1_WARPS_PER_BLOCK * WARP_SIZE;
constexpr int R3_TILE_SIZE = 256;
constexpr int R3_BLOCK_SIZE = 32;
constexpr int MAX_MONOID_SIZE = 1024;

constexpr int STATUS_INVALID = 0;
constexpr int STATUS_AGGREGATE = 1;
constexpr int STATUS_PREFIX = 2;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// R1: warp-per-string sequential monoid scan
__global__ void monoid_r1_kernel(
    const uint16_t *__restrict__ compose,    // [M * M] row-major
    const int      *__restrict__ all_chars,  // [total_chars] monoid indices
    const int      *__restrict__ offsets,    // [B+1]
    const int8_t   *__restrict__ accept,     // [M]
    int            *__restrict__ results,    // [B]
    int             identity_idx,
    int             M,
    int             B
) {
    extern __shared__ uint16_t s_compose[];

    // Cooperatively load compose table into shared memory
    int total_entries = M * M;
    for (int i = threadIdx.x; i < total_entries; i += blockDim.x)
        s_compose[i] = compose[i];
    __syncthreads();

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int string_id = blockIdx.x * R1_WARPS_PER_BLOCK + warp_in_block;
    if (string_id >= B) return;

    int str_start = offsets[string_id];
    int str_end   = offsets[string_id + 1];
    int L = str_end - str_start;

    if (lane != 0) return;

    if (L == 0) {
        results[string_id] = accept[identity_idx] ? 1 : 0;
        return;
    }

    // Sequential scan: accumulate monoid index
    uint16_t acc = (uint16_t)all_chars[str_start];
    for (int i = 1; i < L; i++) {
        uint16_t c_idx = (uint16_t)all_chars[str_start + i];
        // compose[c_idx][acc] = compose[c_idx * M + acc]
        // Convention: c_idx is newer (left), acc is older (right)
        acc = s_compose[c_idx * M + acc];
    }
    results[string_id] = accept[acc] ? 1 : 0;
}


// R3: decoupled look-back for long strings
__global__ void monoid_r3_kernel(
    const uint16_t *__restrict__ compose,
    const int      *__restrict__ chars,       // [L] monoid indices
    const int8_t   *__restrict__ accept,      // [M]
    int             L,
    int             M,
    int             identity_idx,
    volatile int   *__restrict__ tile_status,  // [n_tiles]
    uint16_t       *__restrict__ tile_agg,     // [n_tiles]
    uint16_t       *__restrict__ tile_prefix,  // [n_tiles]
    int            *__restrict__ result,       // [1]
    int             n_tiles,
    int             tile_size
) {
    extern __shared__ uint16_t s_compose[];

    // Load compose table
    int total_entries = M * M;
    for (int i = threadIdx.x; i < total_entries; i += blockDim.x)
        s_compose[i] = compose[i];
    __syncthreads();

    int tile_id = blockIdx.x;
    if (tile_id >= n_tiles) return;
    int lane = threadIdx.x;
    if (lane != 0) return;

    int tile_start = tile_id * tile_size;
    int tile_end = min(tile_start + tile_size, L);
    int tile_len = tile_end - tile_start;

    // Phase 1: tile aggregate
    uint16_t agg;
    if (tile_len == 0) {
        agg = (uint16_t)identity_idx;
    } else {
        agg = (uint16_t)chars[tile_start];
        for (int i = 1; i < tile_len; i++) {
            uint16_t c = (uint16_t)chars[tile_start + i];
            agg = s_compose[c * M + agg];
        }
    }
    tile_agg[tile_id] = agg;
    __threadfence();
    tile_status[tile_id] = STATUS_AGGREGATE;

    // Phase 2: look-back
    if (tile_id == 0) {
        tile_prefix[tile_id] = agg;
        __threadfence();
        tile_status[tile_id] = STATUS_PREFIX;
    } else {
        uint16_t lookback = (uint16_t)identity_idx;
        int look = tile_id - 1;
        while (look >= 0) {
            int status;
            do { status = tile_status[look]; } while (status == STATUS_INVALID);

            if (status == STATUS_PREFIX) {
                uint16_t pred = tile_prefix[look];
                lookback = s_compose[lookback * M + pred];
                break;
            } else {
                uint16_t pred = tile_agg[look];
                lookback = s_compose[lookback * M + pred];
                look--;
            }
        }
        uint16_t my_prefix = s_compose[agg * M + lookback];
        tile_prefix[tile_id] = my_prefix;
        __threadfence();
        tile_status[tile_id] = STATUS_PREFIX;
    }

    // Phase 3: last tile outputs result
    if (tile_end >= L && tile_start < L) {
        result[0] = accept[tile_prefix[tile_id]] ? 1 : 0;
    }
}


// Host engine
struct MonoidEngine {
    int M;
    int identity_idx;

    uint16_t *d_compose;   // [M * M]
    int8_t   *d_accept;    // [M]
    int      *d_chars;     // [max_total_chars]
    int      *d_offsets;   // [max_B + 1]
    int      *d_results;   // [max_B]

    // R3 buffers
    volatile int *d_tile_status;
    uint16_t     *d_tile_agg;
    uint16_t     *d_tile_prefix;
    int          *d_result_single;
    int           max_tiles;

    int max_total_chars;
    int max_B;

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;

    void init(int monoid_size, int identity,
              const uint16_t *compose, const int8_t *accept_tbl,
              int max_chars, int max_batch) {
        M = monoid_size;
        identity_idx = identity;
        max_total_chars = max_chars;
        max_B = max_batch;

        CHECK_CUDA(cudaMalloc(&d_compose, (size_t)M * M * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc(&d_accept, (size_t)M));
        CHECK_CUDA(cudaMalloc(&d_chars, (size_t)max_chars * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_offsets, (size_t)(max_batch + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)max_batch * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_compose, compose, (size_t)M * M * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, accept_tbl, (size_t)M,
                              cudaMemcpyHostToDevice));

        max_tiles = (max_chars + R3_TILE_SIZE - 1) / R3_TILE_SIZE + 1;
        CHECK_CUDA(cudaMalloc(&d_tile_status, (size_t)max_tiles * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_tile_agg, (size_t)max_tiles * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc(&d_tile_prefix, (size_t)max_tiles * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc(&d_result_single, sizeof(int)));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));
    }

    void destroy() {
        cudaFree(d_compose); cudaFree(d_accept);
        cudaFree(d_chars); cudaFree(d_offsets); cudaFree(d_results);
        cudaFree((void*)d_tile_status); cudaFree(d_tile_agg);
        cudaFree(d_tile_prefix); cudaFree(d_result_single);
        cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start); cudaEventDestroy(ev_kern_stop);
    }

    void dispatch_r1(const int *h_chars, const int *h_offsets,
                     int *h_results, int B, int total_chars,
                     float *kernel_ms, float *total_ms) {
        CHECK_CUDA(cudaEventRecord(ev_start));
        CHECK_CUDA(cudaMemcpy(d_chars, h_chars, (size_t)total_chars * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (size_t)(B + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

        int grid = (B + R1_WARPS_PER_BLOCK - 1) / R1_WARPS_PER_BLOCK;
        int smem = M * M * sizeof(uint16_t);

        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        monoid_r1_kernel<<<grid, R1_BLOCK_SIZE, smem>>>(
            d_compose, d_chars, d_offsets, d_accept, d_results,
            identity_idx, M, B);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results, (size_t)B * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);
    }

    bool dispatch_r3(const int *h_chars, int L,
                     float *kernel_ms, float *total_ms) {
        if (L == 0) return d_accept[identity_idx] != 0;

        int n_tiles = (L + R3_TILE_SIZE - 1) / R3_TILE_SIZE;

        CHECK_CUDA(cudaEventRecord(ev_start));
        CHECK_CUDA(cudaMemcpy(d_chars, h_chars, (size_t)L * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset((void*)d_tile_status, 0, (size_t)n_tiles * sizeof(int)));

        int smem = M * M * sizeof(uint16_t);

        CHECK_CUDA(cudaEventRecord(ev_kern_start));
        monoid_r3_kernel<<<n_tiles, R3_BLOCK_SIZE, smem>>>(
            d_compose, d_chars, d_accept, L, M, identity_idx,
            d_tile_status, d_tile_agg, d_tile_prefix,
            d_result_single, n_tiles, R3_TILE_SIZE);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        int h_result;
        CHECK_CUDA(cudaMemcpy(&h_result, d_result_single, sizeof(int),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);
        return h_result != 0;
    }

    void dispatch_adaptive(const int *h_chars, const int *h_offsets,
                           int *h_results, int B, int total_chars,
                           float *kernel_ms, float *total_ms) {
        int L_max = 0;
        for (int i = 0; i < B; i++) {
            int len = h_offsets[i + 1] - h_offsets[i];
            if (len > L_max) L_max = len;
        }

        if (B >= 4 && L_max <= 4096) {
            dispatch_r1(h_chars, h_offsets, h_results, B, total_chars,
                        kernel_ms, total_ms);
        } else if (B == 1) {
            bool r = dispatch_r3(h_chars, h_offsets[1] - h_offsets[0],
                                  kernel_ms, total_ms);
            h_results[0] = r ? 1 : 0;
        } else {
            float avg_len = (float)total_chars / B;
            if (avg_len <= 4096) {
                dispatch_r1(h_chars, h_offsets, h_results, B, total_chars,
                            kernel_ms, total_ms);
            } else {
                CHECK_CUDA(cudaEventRecord(ev_start));
                CHECK_CUDA(cudaEventRecord(ev_kern_start));
                for (int i = 0; i < B; i++) {
                    int off = h_offsets[i];
                    int len = h_offsets[i + 1] - off;
                    float km, tm;
                    bool r = dispatch_r3(h_chars + off, len, &km, &tm);
                    h_results[i] = r ? 1 : 0;
                }
                CHECK_CUDA(cudaEventRecord(ev_kern_stop));
                CHECK_CUDA(cudaEventRecord(ev_stop));
                CHECK_CUDA(cudaEventSynchronize(ev_stop));
                if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
                if (total_ms)  cudaEventElapsedTime(total_ms, ev_start, ev_stop);
            }
        }
    }
};


#ifdef BUILD_LIB

extern "C" {

static MonoidEngine g_monoid_engine;
static bool g_monoid_initialized = false;

int monoid_engine_init(int monoid_size, int identity_idx,
                       const uint16_t *compose, const int8_t *accept_tbl,
                       int max_total_chars, int max_batch) {
    if (g_monoid_initialized) g_monoid_engine.destroy();
    g_monoid_engine.init(monoid_size, identity_idx, compose, accept_tbl,
                         max_total_chars, max_batch);
    g_monoid_initialized = true;
    return 0;
}

void monoid_engine_destroy() {
    if (g_monoid_initialized) {
        g_monoid_engine.destroy();
        g_monoid_initialized = false;
    }
}

int monoid_engine_dispatch_batch(
    const int *chars, const int *offsets, int *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms
) {
    if (!g_monoid_initialized) return -1;
    g_monoid_engine.dispatch_adaptive(chars, offsets, results, B, total_chars,
                                      kernel_ms, total_ms);
    return 0;
}

int monoid_engine_device_check() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    return 0;
}

}  // extern "C"

#endif  // BUILD_LIB


#ifndef BUILD_LIB

// ─── Built-in tests ──────────────────────────────────────────────────────

static int g_tests = 0, g_pass = 0;
static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) { g_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

// Simple 2-state DFA: even number of 'a's over {a, b}
// Monoid: identity, T[a], T[b], T[a]@T[a], T[a]@T[b], T[b]@T[a]
// T[b] = identity for states, T[a] swaps states 0↔1
// Monoid has 2 elements: identity and swap
static void build_even_a_monoid(
    uint16_t *compose, int8_t *accept, int &M, int &identity_idx,
    int *char_to_monoid
) {
    M = 2;
    identity_idx = 0;
    // Element 0: identity, Element 1: swap
    // compose[0][0] = 0, compose[0][1] = 1, compose[1][0] = 1, compose[1][1] = 0
    compose[0 * M + 0] = 0; compose[0 * M + 1] = 1;
    compose[1 * M + 0] = 1; compose[1 * M + 1] = 0;
    // Accept: element 0 (identity → state 0 is accept) = true
    //         element 1 (swap → state 1) = false
    accept[0] = 1; accept[1] = 0;
    // char 'a' (0) → monoid 1 (swap), char 'b' (1) → monoid 0 (identity)
    char_to_monoid[0] = 1;  // 'a'
    char_to_monoid[1] = 0;  // 'b'
}

static bool simulate_even_a_sequential(const int *chars, int L) {
    int count_a = 0;
    for (int i = 0; i < L; i++)
        if (chars[i] == 0) count_a++;  // 'a' is char index 0
    return (count_a % 2) == 0;
}

static void test_monoid_r1() {
    printf("\n--- Monoid R1: Basic correctness ---\n");
    uint16_t compose[4];
    int8_t accept[2];
    int M, identity_idx;
    int char_map[2];
    build_even_a_monoid(compose, accept, M, identity_idx, char_map);

    MonoidEngine eng;
    eng.init(M, identity_idx, compose, accept, 1 << 20, 1 << 16);

    // Single strings
    {
        int chars[] = {1};  // monoid idx for 'b' = 0
        int offsets[] = {0, 1};
        int results[1];
        eng.dispatch_r1(chars, offsets, results, 1, 1, nullptr, nullptr);
        check("R1 'b' (even_a) -> accept", results[0] == 1);
    }
    {
        int chars[] = {1, 1};  // monoid idx for 'a','a'
        int c_mon[] = {char_map[0], char_map[0]};
        int offsets[] = {0, 2};
        int results[1];
        eng.dispatch_r1(c_mon, offsets, results, 1, 2, nullptr, nullptr);
        check("R1 'aa' (even_a) -> accept", results[0] == 1);
    }
    {
        int c_mon[] = {char_map[0]};  // single 'a'
        int offsets[] = {0, 1};
        int results[1];
        eng.dispatch_r1(c_mon, offsets, results, 1, 1, nullptr, nullptr);
        check("R1 'a' (even_a) -> reject", results[0] == 0);
    }

    // Large batch cross-validation
    {
        int B = 10000;
        int L = 50;
        int total = B * L;
        int *raw_chars = new int[total];
        int *mon_chars = new int[total];
        int *offsets = new int[B + 1];
        int *results = new int[B];
        srand(42);
        for (int i = 0; i < total; i++) {
            raw_chars[i] = rand() % 2;
            mon_chars[i] = char_map[raw_chars[i]];
        }
        for (int i = 0; i <= B; i++) offsets[i] = i * L;

        eng.dispatch_r1(mon_chars, offsets, results, B, total, nullptr, nullptr);

        int mismatches = 0;
        for (int i = 0; i < B; i++) {
            bool expected = simulate_even_a_sequential(raw_chars + offsets[i], L);
            if (results[i] != (expected ? 1 : 0)) mismatches++;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "R1 batch B=%d (%d mismatches)", B, mismatches);
        check(msg, mismatches == 0);

        delete[] raw_chars; delete[] mon_chars; delete[] offsets; delete[] results;
    }

    eng.destroy();
}

static void test_monoid_r3() {
    printf("\n--- Monoid R3: Long string correctness ---\n");
    uint16_t compose[4];
    int8_t accept[2];
    int M, identity_idx;
    int char_map[2];
    build_even_a_monoid(compose, accept, M, identity_idx, char_map);

    MonoidEngine eng;
    eng.init(M, identity_idx, compose, accept, 1 << 22, 4);

    int lengths[] = {100, 256, 1000, 4096, 10000, 100000, 1000000};
    for (int li = 0; li < 7; li++) {
        int L = lengths[li];
        int *raw = new int[L];
        int *mon = new int[L];
        srand(L);
        for (int i = 0; i < L; i++) {
            raw[i] = rand() % 2;
            mon[i] = char_map[raw[i]];
        }
        bool expected = simulate_even_a_sequential(raw, L);
        bool got = eng.dispatch_r3(mon, L, nullptr, nullptr);
        char msg[128];
        snprintf(msg, sizeof(msg), "R3 L=%d expected=%d got=%d", L, expected, got);
        check(msg, got == expected);
        delete[] raw; delete[] mon;
    }

    eng.destroy();
}

static void bench_monoid() {
    printf("\n--- Monoid benchmarks ---\n");
    uint16_t compose[4];
    int8_t accept[2];
    int M, identity_idx;
    int char_map[2];
    build_even_a_monoid(compose, accept, M, identity_idx, char_map);

    // R1 benchmark
    {
        MonoidEngine eng;
        eng.init(M, identity_idx, compose, accept, 1 << 23, 1 << 18);

        int B_vals[] = {1000, 10000, 100000};
        int L_vals[] = {32, 128, 512, 1024, 4096};
        for (int bi = 0; bi < 3; bi++) {
            for (int li = 0; li < 5; li++) {
                int B = B_vals[bi], L = L_vals[li];
                int total = B * L;
                if (total > (1 << 23)) continue;
                int *chars = new int[total];
                int *offsets = new int[B + 1];
                int *results = new int[B];
                srand(42);
                for (int i = 0; i < total; i++) chars[i] = rand() % M;
                for (int i = 0; i <= B; i++) offsets[i] = i * L;

                float kern_ms, total_ms;
                // Warmup
                eng.dispatch_r1(chars, offsets, results, B, total, &kern_ms, &total_ms);
                // Measure
                eng.dispatch_r1(chars, offsets, results, B, total, &kern_ms, &total_ms);

                double gb = (double)total * sizeof(int) / 1e9;
                printf("  R1 B=%6d L=%5d  kern=%.3f ms  total=%.3f ms  kern_tput=%.2f GB/s\n",
                       B, L, kern_ms, total_ms, gb / (kern_ms / 1e3));

                delete[] chars; delete[] offsets; delete[] results;
            }
        }
        eng.destroy();
    }

    // R3 benchmark
    {
        MonoidEngine eng;
        eng.init(M, identity_idx, compose, accept, 1 << 24, 4);

        int L_vals[] = {1024, 4096, 65536, 262144, 1048576, 4194304, 16777216};
        for (int li = 0; li < 7; li++) {
            int L = L_vals[li];
            int *chars = new int[L];
            srand(42);
            for (int i = 0; i < L; i++) chars[i] = rand() % M;

            float kern_ms, total_ms;
            eng.dispatch_r3(chars, L, &kern_ms, &total_ms);
            eng.dispatch_r3(chars, L, &kern_ms, &total_ms);

            double gb = (double)L * sizeof(int) / 1e9;
            printf("  R3 L=%9d  kern=%.3f ms  total=%.3f ms  kern_tput=%.2f GB/s\n",
                   L, kern_ms, total_ms, gb / (kern_ms / 1e3));
            delete[] chars;
        }
        eng.destroy();
    }
}

int main() {
    test_monoid_r1();
    test_monoid_r3();
    printf("\n=== Tests: %d/%d passed ===\n", g_pass, g_tests);
    if (g_pass == g_tests) {
        bench_monoid();
    }
    return (g_pass == g_tests) ? 0 : 1;
}

#endif  // !BUILD_LIB
```

- [ ] **Step 2: Add Makefile targets**

Add to `Makefile` after the v4 entries:

```makefile
SRC_MONOID = $(CUDA_DIR)/monoid_scan.cu
EXE_MONOID = $(BUILD_DIR)/monoid_scan
LIB_MONOID = $(BUILD_DIR)/libmonoid_scan.so
```

Update the `all` target to include `$(EXE_MONOID) $(LIB_MONOID)`.

Add build rules:

```makefile
$(EXE_MONOID): $(SRC_MONOID) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_MONOID): $(SRC_MONOID) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

test-monoid: $(EXE_MONOID)
	./$(EXE_MONOID)
```

Add `test-monoid` to `.PHONY`.

- [ ] **Step 3: Build and run CUDA tests**

Run: `make test-monoid`
Expected: All CUDA tests PASS, then benchmarks print.

- [ ] **Step 4: Commit**

```bash
git add cuda/monoid_scan.cu Makefile
git commit -m "feat: add monoid scan CUDA kernel with R1/R3 dispatch"
```

---

### Task 3: Monoid GPU Bridge (Python ctypes)

**Files:**
- Create: `src/gpu_bridge_monoid.py`
- Test: `tests/test_monoid.py` (extend)

- [ ] **Step 1: Add GPU cross-validation tests to test_monoid.py**

Append to `tests/test_monoid.py`:

```python
def _monoid_gpu_available():
    try:
        from src.gpu_bridge_monoid import MonoidGPUSimulator
        MonoidGPUSimulator()
        return True
    except Exception:
        return False

skip_no_monoid_gpu = pytest.mark.skipif(
    not _monoid_gpu_available(),
    reason="monoid GPU engine not available"
)

@pytest.fixture(scope="module")
def monoid_simulator():
    from src.gpu_bridge_monoid import MonoidGPUSimulator
    return MonoidGPUSimulator()


@skip_no_monoid_gpu
class TestMonoidGPU:
    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_batch_cross_validate(self, monoid_simulator, pattern_name):
        from src.monoid import compute_monoid
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        assert md is not None

        engine = monoid_simulator.create_engine(md, dm)

        random.seed(42)
        alpha = sorted(dfa.alphabet)
        strings = [''.join(random.choice(alpha) for _ in range(random.randint(0, 200)))
                    for _ in range(500)]
        gpu_results = engine.simulate_batch(strings)
        cpu_results = [simulate_sequential(dfa, s) for s in strings]
        assert gpu_results == cpu_results, (
            f"{pattern_name}: {sum(a!=b for a,b in zip(gpu_results, cpu_results))} mismatches")
        engine.destroy()

    def test_long_string(self, monoid_simulator):
        from src.monoid import compute_monoid
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)

        engine = monoid_simulator.create_engine(md, dm, max_total_chars=1 << 22)

        random.seed(77)
        alpha = sorted(dfa.alphabet)
        for L in [10000, 100000, 1000000]:
            s = ''.join(random.choice(alpha) for _ in range(L))
            gpu = engine.simulate_batch([s])[0]
            cpu = simulate_sequential(dfa, s)
            assert gpu == cpu, f"L={L}"
        engine.destroy()

    def test_timing(self, monoid_simulator):
        from src.monoid import compute_monoid
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        engine = monoid_simulator.create_engine(md, dm)

        random.seed(42)
        alpha = sorted(dfa.alphabet)
        strings = [''.join(random.choice(alpha) for _ in range(100)) for _ in range(1000)]
        results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
        assert len(results) == 1000
        assert kern_ms > 0
        assert total_ms >= kern_ms
        engine.destroy()
```

- [ ] **Step 2: Run to verify tests fail**

Run: `python -m pytest tests/test_monoid.py::TestMonoidGPU -v 2>&1 | head -10`
Expected: ImportError — `src.gpu_bridge_monoid` does not exist.

- [ ] **Step 3: Implement the Python bridge**

```python
# src/gpu_bridge_monoid.py
from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.simulation import DFAMatrices
from src.monoid import MonoidData


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libmonoid_scan.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libmonoid_scan.so not found at {base}. Run 'make' first."
    )


class MonoidEngine:
    def __init__(self, lib, md: MonoidData, dm: DFAMatrices,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.md = md
        self.dm = dm

        compose_flat = np.ascontiguousarray(md.compose_table.astype(np.uint16).reshape(-1))
        accept_arr = md.accept_table.astype(np.int8)

        rc = self.lib.monoid_engine_init(
            md.size,
            md.identity_idx,
            compose_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            accept_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            max_total_chars,
            max_batch,
        )
        if rc != 0:
            raise RuntimeError(f"monoid_engine_init failed with code {rc}")

    def destroy(self):
        self.lib.monoid_engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        B = len(strings)
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)
        total_chars = int(offsets[B])

        chars = np.zeros(total_chars, dtype=np.int32)
        c2m = self.md.char_to_monoid
        pos = 0
        for s in strings:
            for ch in s:
                chars[pos] = c2m[ch]
                pos += 1

        return chars, offsets, total_chars

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        if not strings:
            return []

        B = len(strings)
        chars, offsets, total_chars = self._prepare_batch(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.monoid_engine_dispatch_batch(
            chars.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"monoid_engine_dispatch_batch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        if not strings:
            return [], 0.0, 0.0

        B = len(strings)
        chars, offsets, total_chars = self._prepare_batch(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.monoid_engine_dispatch_batch(
            chars.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"monoid_engine_dispatch_batch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class MonoidGPUSimulator:
    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.monoid_engine_init.restype = ctypes.c_int
        self.lib.monoid_engine_init.argtypes = [
            ctypes.c_int,                     # monoid_size
            ctypes.c_int,                     # identity_idx
            ctypes.POINTER(ctypes.c_uint16),  # compose
            ctypes.POINTER(ctypes.c_int8),    # accept_tbl
            ctypes.c_int,                     # max_total_chars
            ctypes.c_int,                     # max_batch
        ]

        self.lib.monoid_engine_destroy.restype = None
        self.lib.monoid_engine_destroy.argtypes = []

        self.lib.monoid_engine_dispatch_batch.restype = ctypes.c_int
        self.lib.monoid_engine_dispatch_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # chars (monoid indices)
            ctypes.POINTER(ctypes.c_int),   # offsets
            ctypes.POINTER(ctypes.c_int),   # results
            ctypes.c_int,                   # B
            ctypes.c_int,                   # total_chars
            ctypes.POINTER(ctypes.c_float), # kernel_ms
            ctypes.POINTER(ctypes.c_float), # total_ms
        ]

        self.lib.monoid_engine_device_check.restype = ctypes.c_int
        rc = self.lib.monoid_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")

    def create_engine(self, md: MonoidData, dm: DFAMatrices,
                      max_total_chars: int = 1 << 22,
                      max_batch: int = 1 << 18) -> MonoidEngine:
        return MonoidEngine(self.lib, md, dm, max_total_chars, max_batch)
```

- [ ] **Step 4: Build the shared library and run all monoid tests**

Run: `make && python -m pytest tests/test_monoid.py -v`
Expected: All tests PASS (CPU + GPU).

- [ ] **Step 5: Commit**

```bash
git add src/gpu_bridge_monoid.py tests/test_monoid.py
git commit -m "feat: add monoid GPU bridge with Python ctypes interface"
```

---

### Task 4: k-Gram Precomputation

**Files:**
- Create: `src/kgram.py`
- Test: `tests/test_kgram.py`

- [ ] **Step 1: Write failing tests for k-gram precomputation**

```python
# tests/test_kgram.py
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, _matmul_int8, simulate_sequential
from src.generate_data import PATTERNS


class TestKGramMonoidMode:
    """k-gram tables in monoid mode: k-gram → monoid index."""

    def _setup(self, pattern_name, k):
        from src.monoid import compute_monoid
        from src.kgram import precompute_kgrams
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        kg = precompute_kgrams(dm, k, monoid=md)
        return kg, md, dm, dfa

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_kgram_matches_sequential_compose(self, k):
        """Each k-gram's monoid index should match composing k individual chars."""
        from src.monoid import compute_monoid
        from src.kgram import precompute_kgrams
        pat = PATTERNS['abb']  # binary alphabet
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        if len(dm.alphabet) ** k > 65536:
            pytest.skip("table too large")
        kg = precompute_kgrams(dm, k, monoid=md)

        alpha = sorted(dfa.alphabet)
        # Check a sample of k-grams
        random.seed(42)
        for _ in range(min(1000, len(alpha) ** k)):
            gram = tuple(random.choice(alpha) for _ in range(k))
            idx = kg.lookup(gram)
            # Manually compose
            acc = md.char_to_monoid[gram[0]]
            for ch in gram[1:]:
                c = md.char_to_monoid[ch]
                acc = md.compose_table[c, acc]
            assert idx == acc, f"k-gram {gram}: lookup={idx} expected={acc}"

    @pytest.mark.parametrize("k", [2, 4])
    def test_simulate_with_kgram(self, k):
        """Full simulation using k-gram chunking matches sequential."""
        from src.monoid import compute_monoid
        from src.kgram import precompute_kgrams, simulate_kgram_monoid
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        kg = precompute_kgrams(dm, k, monoid=md)

        random.seed(42)
        alpha = sorted(dfa.alphabet)
        for _ in range(100):
            L = random.randint(0, 300)
            s = ''.join(random.choice(alpha) for _ in range(L))
            expected = simulate_sequential(dfa, s)
            got = simulate_kgram_monoid(kg, md, dm, s)
            assert got == expected, f"k={k} L={L}"

    @pytest.mark.parametrize("tail_len", [0, 1, 2, 3])
    def test_tail_handling(self, tail_len):
        """Strings whose length is not a multiple of k."""
        from src.monoid import compute_monoid
        from src.kgram import precompute_kgrams, simulate_kgram_monoid
        k = 4
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        kg = precompute_kgrams(dm, k, monoid=md)

        random.seed(tail_len)
        alpha = sorted(dfa.alphabet)
        L = 20 + tail_len  # 20 = 5 * k, so tail = tail_len
        s = ''.join(random.choice(alpha) for _ in range(L))
        expected = simulate_sequential(dfa, s)
        got = simulate_kgram_monoid(kg, md, dm, s)
        assert got == expected


class TestKGramMatrixMode:
    """k-gram tables in matrix mode: k-gram → composed matrix."""

    @pytest.mark.parametrize("k", [2, 4])
    def test_kgram_matrix_matches_composition(self, k):
        from src.kgram import precompute_kgrams
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        kg = precompute_kgrams(dm, k, monoid=None)

        alpha = sorted(dfa.alphabet)
        random.seed(42)
        for _ in range(200):
            gram = tuple(random.choice(alpha) for _ in range(k))
            got_matrix = kg.lookup_matrix(gram)
            # Manual composition: T[c_k] @ ... @ T[c_1] (rightmost applied first)
            acc = dm.matrices[gram[0]].copy()
            for ch in gram[1:]:
                acc = _matmul_int8(dm.matrices[ch], acc)
            np.testing.assert_array_equal(got_matrix, acc)


class TestKGramAutoK:
    def test_auto_k_binary(self):
        from src.kgram import auto_k
        assert auto_k(2) == 16

    def test_auto_k_byte(self):
        from src.kgram import auto_k
        assert auto_k(256) == 2

    def test_auto_k_small(self):
        from src.kgram import auto_k
        assert auto_k(16) == 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_kgram.py -v 2>&1 | head -10`
Expected: ImportError — `src.kgram` does not exist.

- [ ] **Step 3: Implement k-gram precomputation**

```python
# src/kgram.py
from __future__ import annotations
import math
import numpy as np
from itertools import product as iter_product
from src.simulation import DFAMatrices, _matmul_int8
from src.monoid import MonoidData


def auto_k(alphabet_size: int, max_entries: int = 65536) -> int:
    if alphabet_size <= 1:
        return 1
    k = int(math.log(max_entries) / math.log(alphabet_size))
    return max(1, k)


class KGramTable:
    def __init__(self, k: int, alphabet: list[str],
                 monoid_table: dict | None = None,
                 matrix_table: dict | None = None):
        self.k = k
        self.alphabet = alphabet
        self.char_to_idx = {c: i for i, c in enumerate(alphabet)}
        self._monoid_table = monoid_table
        self._matrix_table = matrix_table

    def _gram_key(self, gram: tuple[str, ...]) -> int:
        sigma = len(self.alphabet)
        key = 0
        for ch in gram:
            key = key * sigma + self.char_to_idx[ch]
        return key

    def lookup(self, gram: tuple[str, ...]) -> int:
        return self._monoid_table[self._gram_key(gram)]

    def lookup_matrix(self, gram: tuple[str, ...]) -> np.ndarray:
        return self._matrix_table[self._gram_key(gram)]


def precompute_kgrams(dm: DFAMatrices, k: int,
                      monoid: MonoidData | None = None) -> KGramTable:
    alpha = sorted(dm.dfa.alphabet)
    sigma = len(alpha)
    c2i = {c: i for i, c in enumerate(alpha)}

    if monoid is not None:
        table = {}
        for gram in iter_product(range(sigma), repeat=k):
            chars = [alpha[i] for i in gram]
            acc = monoid.char_to_monoid[chars[0]]
            for ch in chars[1:]:
                c = monoid.char_to_monoid[ch]
                acc = monoid.compose_table[c, acc]
            key = 0
            for i in gram:
                key = key * sigma + i
            table[key] = int(acc)
        return KGramTable(k, alpha, monoid_table=table)
    else:
        table = {}
        for gram in iter_product(range(sigma), repeat=k):
            chars = [alpha[i] for i in gram]
            acc = dm.matrices[chars[0]].copy()
            for ch in chars[1:]:
                acc = _matmul_int8(dm.matrices[ch], acc)
            key = 0
            for i in gram:
                key = key * sigma + i
            table[key] = acc
        return KGramTable(k, alpha, matrix_table=table)


def simulate_kgram_monoid(kg: KGramTable, md: MonoidData,
                          dm: DFAMatrices, input_str: str) -> bool:
    if not input_str:
        return md.accept_table[md.identity_idx]

    k = kg.k
    alpha = kg.alphabet
    sigma = len(alpha)
    c2i = kg.char_to_idx

    # Chunk into k-grams
    L = len(input_str)
    n_full = L // k
    tail_len = L % k

    acc = md.identity_idx

    for i in range(n_full):
        gram = tuple(input_str[i * k + j] for j in range(k))
        g_idx = kg.lookup(gram)
        if acc == md.identity_idx:
            acc = g_idx
        else:
            acc = md.compose_table[g_idx, acc]

    # Handle tail
    for j in range(n_full * k, L):
        c = md.char_to_monoid[input_str[j]]
        if acc == md.identity_idx:
            acc = c
        else:
            acc = md.compose_table[c, acc]

    return bool(md.accept_table[acc])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_kgram.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kgram.py tests/test_kgram.py
git commit -m "feat: add k-gram precomputation with monoid and matrix modes"
```

---

### Task 5: NFA Path — Matrix Export

**Files:**
- Create: `src/nfa_matrices.py`
- Test: `tests/test_nfa.py`

- [ ] **Step 1: Write failing tests for NFA matrix construction**

```python
# tests/test_nfa.py
import sys, os, random, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.regex_to_dfa import compile_regex, RegexParser, epsilon_closure
from src.simulation import simulate_sequential


class TestNFAMatrices:
    def test_simple_pattern(self):
        from src.nfa_matrices import compile_nfa_matrices
        nm = compile_nfa_matrices('(a|b)*abb')
        assert nm.n_states > 0
        assert 'a' in nm.alphabet
        assert 'b' in nm.alphabet
        # Each matrix should be square, n_states x n_states
        for ch in nm.alphabet:
            assert nm.matrices[ch].shape == (nm.n_states, nm.n_states)

    def test_nfa_matrices_are_boolean(self):
        from src.nfa_matrices import compile_nfa_matrices
        nm = compile_nfa_matrices('(a|b)*abb')
        for ch in nm.alphabet:
            vals = set(nm.matrices[ch].flatten().tolist())
            assert vals <= {0, 1}, f"Non-boolean values in NFA matrix for '{ch}': {vals}"

    def test_nfa_simulate_matches_dfa(self):
        from src.nfa_matrices import compile_nfa_matrices, simulate_nfa
        patterns = ['(a|b)*abb', '(ab)*', '(b*ab*ab*)*b*']
        for regex in patterns:
            dfa = compile_regex(regex)
            nm = compile_nfa_matrices(regex)

            random.seed(42)
            alpha = sorted(dfa.alphabet)
            for _ in range(50):
                L = random.randint(0, 100)
                s = ''.join(random.choice(alpha) for _ in range(L))
                dfa_result = simulate_sequential(dfa, s)
                nfa_result = simulate_nfa(nm, s)
                assert nfa_result == dfa_result, (
                    f"regex={regex} s={s[:20]}... dfa={dfa_result} nfa={nfa_result}")

    def test_nfa_state_count_linear(self):
        """NFA state count should be O(|pattern|), not exponential."""
        from src.nfa_matrices import compile_nfa_matrices
        nm = compile_nfa_matrices('(a|b)*abb')
        assert nm.n_states < 50

    @pytest.mark.parametrize("regex", [
        '(a|b)*abb',
        '[a-z]+',
        '(ab)*',
        'a(b|c)*d',
    ])
    def test_cross_validate_python_re(self, regex):
        from src.nfa_matrices import compile_nfa_matrices, simulate_nfa
        nm = compile_nfa_matrices(regex)
        dfa = compile_regex(regex)
        alpha = sorted(dfa.alphabet)

        random.seed(42)
        for _ in range(30):
            L = random.randint(0, 50)
            s = ''.join(random.choice(alpha) for _ in range(L))
            nfa_result = simulate_nfa(nm, s)
            py_result = bool(re.fullmatch(regex, s))
            assert nfa_result == py_result, (
                f"regex={regex} s={s!r} nfa={nfa_result} re={py_result}")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_nfa.py -v 2>&1 | head -10`
Expected: ImportError — `src.nfa_matrices` does not exist.

- [ ] **Step 3: Implement NFA matrix export and simulation**

```python
# src/nfa_matrices.py
from __future__ import annotations
import numpy as np
from typing import Optional
from src.regex_to_dfa import RegexParser, NFA, EPSILON


def _epsilon_closure_set(nfa: NFA, states: set[int]) -> set[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for dst in nfa.states[s].transitions.get(EPSILON, []):
            if dst not in closure:
                closure.add(dst)
                stack.append(dst)
    return closure


class NFAMatrices:
    def __init__(self, nfa: NFA, pad_to: Optional[int] = None):
        self.nfa = nfa
        self.alphabet = sorted(nfa.alphabet)
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.n_states_raw = len(nfa.states)

        if pad_to is not None:
            self.n_states = max(pad_to, self.n_states_raw)
        else:
            self.n_states = ((self.n_states_raw + 15) // 16) * 16

        self._build_matrices()
        self._build_state_vectors()

    def _build_matrices(self):
        N = self.n_states
        nfa = self.nfa
        self.matrices = {}

        for ch in self.alphabet:
            T = np.zeros((N, N), dtype=np.int8)
            for src_id, src_state in nfa.states.items():
                if src_id >= N:
                    continue
                dsts = src_state.transitions.get(ch, [])
                for dst in dsts:
                    reachable = _epsilon_closure_set(nfa, {dst})
                    for r in reachable:
                        if r < N:
                            T[r, src_id] = 1
            self.matrices[ch] = T

        self.matrix_stack = np.stack(
            [self.matrices[ch] for ch in self.alphabet], axis=0
        )

    def _build_state_vectors(self):
        N = self.n_states
        start_states = _epsilon_closure_set(self.nfa, {self.nfa.start})
        self.start_vec = np.zeros(N, dtype=np.int8)
        for s in start_states:
            if s < N:
                self.start_vec[s] = 1

        self.accept_mask = np.zeros(N, dtype=np.int8)
        accept_id = self.nfa.accept
        accept_closure = _epsilon_closure_set(self.nfa, {accept_id})
        for sid, state in self.nfa.states.items():
            if state.is_accept and sid < N:
                self.accept_mask[sid] = 1

    def check_accept(self, state_vec: np.ndarray) -> bool:
        return bool(np.any(
            (state_vec[:self.n_states_raw] > 0) &
            (self.accept_mask[:self.n_states_raw] > 0)
        ))


def compile_nfa_matrices(regex: str, pad_to: Optional[int] = None) -> NFAMatrices:
    parser = RegexParser(regex)
    nfa = parser.parse()
    return NFAMatrices(nfa, pad_to=pad_to)


def simulate_nfa(nm: NFAMatrices, input_str: str) -> bool:
    if not input_str:
        return nm.check_accept(nm.start_vec)

    state = nm.start_vec.astype(np.int32)
    for ch in input_str:
        T = nm.matrices.get(ch)
        if T is None:
            return False
        state = T.astype(np.int32) @ state
        state = np.minimum(state, 1)
    return nm.check_accept(state.astype(np.int8))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nfa.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nfa_matrices.py tests/test_nfa.py
git commit -m "feat: add NFA matrix export with Boolean matrix simulation"
```

---

### Task 6: Unified OptimizedEngine API

**Files:**
- Create: `src/optimized_engine.py`
- Test: `tests/test_optimized_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_optimized_engine.py
import sys, os, random, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.regex_to_dfa import compile_regex
from src.simulation import simulate_sequential
from src.generate_data import PATTERNS


class TestOptimizedEngineAutoSelect:
    def test_small_dfa_selects_monoid(self):
        from src.optimized_engine import OptimizedEngine
        engine = OptimizedEngine('(a|b)*abb')
        info = engine.config_info
        assert info['representation'] == 'dfa'
        assert info['scan_backend'] == 'monoid'
        assert info['monoid_size'] > 0

    def test_config_info_has_required_fields(self):
        from src.optimized_engine import OptimizedEngine
        engine = OptimizedEngine('(a|b)*abb')
        info = engine.config_info
        for key in ['representation', 'scan_backend', 'alphabet_size',
                     'kgram_k', 'selection_reason']:
            assert key in info, f"Missing key: {key}"


class TestOptimizedEngineCorrectness:
    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_auto_matches_sequential(self, pattern_name):
        from src.optimized_engine import OptimizedEngine
        pat = PATTERNS[pattern_name]
        engine = OptimizedEngine(pat.regex)
        dfa = compile_regex(pat.regex)

        random.seed(42)
        alpha = sorted(dfa.alphabet)
        strings = [''.join(random.choice(alpha) for _ in range(random.randint(0, 200)))
                    for _ in range(100)]
        results = engine.match_batch(strings)
        expected = [simulate_sequential(dfa, s) for s in strings]
        assert results == expected

    @pytest.mark.parametrize("config", [
        "monoid", "monoid+kgram", "baseline"])
    def test_forced_config_matches_sequential(self, config):
        from src.optimized_engine import OptimizedEngine
        regex = '(a|b)*abb'
        engine = OptimizedEngine(regex, config=config)
        dfa = compile_regex(regex)

        random.seed(42)
        alpha = sorted(dfa.alphabet)
        strings = [''.join(random.choice(alpha) for _ in range(random.randint(0, 200)))
                    for _ in range(100)]
        results = engine.match_batch(strings)
        expected = [simulate_sequential(dfa, s) for s in strings]
        assert results == expected, f"config={config}"

    def test_nfa_config_matches_sequential(self):
        from src.optimized_engine import OptimizedEngine
        regex = '(a|b)*abb'
        engine = OptimizedEngine(regex, config="nfa")
        dfa = compile_regex(regex)

        random.seed(42)
        alpha = sorted(dfa.alphabet)
        strings = [''.join(random.choice(alpha) for _ in range(random.randint(0, 50)))
                    for _ in range(50)]
        results = engine.match_batch(strings)
        expected = [simulate_sequential(dfa, s) for s in strings]
        assert results == expected

    @pytest.mark.parametrize("pattern_name", ['hex_number', 'identifier'])
    def test_larger_alphabet(self, pattern_name):
        from src.optimized_engine import OptimizedEngine
        pat = PATTERNS[pattern_name]
        engine = OptimizedEngine(pat.regex)
        dfa = compile_regex(pat.regex)

        random.seed(42)
        alpha = sorted(dfa.alphabet)
        strings = [''.join(random.choice(alpha) for _ in range(random.randint(0, 100)))
                    for _ in range(50)]
        results = engine.match_batch(strings)
        expected = [simulate_sequential(dfa, s) for s in strings]
        assert results == expected


class TestOptimizedEngineTiming:
    def test_timed_returns_dict(self):
        from src.optimized_engine import OptimizedEngine
        engine = OptimizedEngine('(a|b)*abb')

        random.seed(42)
        strings = [''.join(random.choice('ab') for _ in range(100)) for _ in range(100)]
        results, timing = engine.match_batch_timed(strings)
        assert len(results) == 100
        assert 'precompute_ms' in timing or 'scan_ms' in timing


class TestAllConfigsSameResult:
    def test_all_configs_agree(self):
        from src.optimized_engine import OptimizedEngine
        regex = '(a|b)*abb'
        configs = [None, "monoid", "monoid+kgram", "baseline", "nfa"]

        random.seed(42)
        strings = [''.join(random.choice('ab') for _ in range(random.randint(0, 100)))
                    for _ in range(200)]

        all_results = {}
        for cfg in configs:
            engine = OptimizedEngine(regex, config=cfg)
            all_results[str(cfg)] = engine.match_batch(strings)

        ref = all_results[str(configs[0])]
        for cfg in configs[1:]:
            assert all_results[str(cfg)] == ref, (
                f"Config {cfg} disagrees with auto-select")
```

- [ ] **Step 2: Run to verify tests fail**

Run: `python -m pytest tests/test_optimized_engine.py -v 2>&1 | head -10`
Expected: ImportError — `src.optimized_engine` does not exist.

- [ ] **Step 3: Implement the OptimizedEngine**

```python
# src/optimized_engine.py
from __future__ import annotations
import time
from src.regex_to_dfa import compile_regex, DFA
from src.simulation import DFAMatrices, simulate_sequential
from src.monoid import compute_monoid, simulate_monoid, MonoidData
from src.kgram import precompute_kgrams, simulate_kgram_monoid, auto_k, KGramTable
from src.nfa_matrices import compile_nfa_matrices, simulate_nfa, NFAMatrices


class OptimizedEngine:
    def __init__(self, regex: str, config: str | None = None,
                 dfa_state_cap: int = 64, monoid_cap: int = 65536):
        self._regex = regex
        self._config = config
        self._dfa: DFA | None = None
        self._dm: DFAMatrices | None = None
        self._nm: NFAMatrices | None = None
        self._md: MonoidData | None = None
        self._kg: KGramTable | None = None
        self._info: dict = {}

        if config == "nfa":
            self._setup_nfa()
        elif config == "baseline":
            self._setup_baseline()
        elif config == "monoid":
            self._setup_dfa()
            self._setup_monoid()
        elif config == "monoid+kgram":
            self._setup_dfa()
            self._setup_monoid()
            self._setup_kgram()
        elif config is None:
            self._auto_select(dfa_state_cap, monoid_cap)
        else:
            raise ValueError(f"Unknown config: {config}")

    def _setup_dfa(self):
        self._dfa = compile_regex(self._regex)
        self._dm = DFAMatrices(self._dfa)
        self._info['representation'] = 'dfa'
        self._info['dfa_states'] = self._dfa.n_states
        self._info['alphabet_size'] = len(self._dm.alphabet)

    def _setup_baseline(self):
        self._setup_dfa()
        self._info['scan_backend'] = 'matrix'
        self._info['kgram_k'] = 0
        self._info['selection_reason'] = 'Forced baseline config'

    def _setup_nfa(self):
        self._nm = compile_nfa_matrices(self._regex)
        self._info['representation'] = 'nfa'
        self._info['nfa_states'] = self._nm.n_states_raw
        self._info['scan_backend'] = 'nfa_matrix'
        self._info['alphabet_size'] = len(self._nm.alphabet)
        self._info['kgram_k'] = 0
        self._info['selection_reason'] = 'Forced NFA config'

    def _setup_monoid(self):
        if self._dm is None:
            return
        self._md = compute_monoid(self._dm)
        if self._md is not None:
            self._info['scan_backend'] = 'monoid'
            self._info['monoid_size'] = self._md.size
        else:
            self._info['scan_backend'] = 'matrix'
            self._info['monoid_size'] = -1

    def _setup_kgram(self):
        if self._dm is None:
            return
        k = auto_k(len(self._dm.alphabet))
        if k > 1:
            self._kg = precompute_kgrams(self._dm, k, monoid=self._md)
            self._info['kgram_k'] = k
        else:
            self._info['kgram_k'] = 0

    def _auto_select(self, dfa_state_cap: int, monoid_cap: int):
        try:
            self._setup_dfa()
            if self._dfa.n_states > dfa_state_cap:
                self._setup_nfa()
                self._info['selection_reason'] = (
                    f"DFA too large ({self._dfa.n_states} states > cap {dfa_state_cap}), using NFA")
                self._dfa = None
                self._dm = None
                return
        except Exception:
            self._setup_nfa()
            self._info['selection_reason'] = 'DFA construction failed, using NFA'
            return

        self._md = compute_monoid(self._dm, max_size=monoid_cap)
        if self._md is not None:
            self._info['scan_backend'] = 'monoid'
            self._info['monoid_size'] = self._md.size
            self._setup_kgram()
            self._info['selection_reason'] = (
                f"DFA succeeded ({self._dfa.n_states} states), "
                f"monoid small ({self._md.size} elements), "
                f"k-gram k={self._info.get('kgram_k', 0)}")
        else:
            self._info['scan_backend'] = 'matrix'
            self._info['monoid_size'] = -1
            self._info['kgram_k'] = 0
            self._info['selection_reason'] = (
                f"DFA succeeded ({self._dfa.n_states} states), "
                f"monoid too large, using matrix scan")

    @property
    def config_info(self) -> dict:
        return self._info.copy()

    def match_batch(self, strings: list[str]) -> list[bool]:
        if self._nm is not None:
            return [simulate_nfa(self._nm, s) for s in strings]

        if self._md is not None and self._kg is not None:
            return [simulate_kgram_monoid(self._kg, self._md, self._dm, s)
                    for s in strings]

        if self._md is not None:
            return [simulate_monoid(self._md, self._dm, s) for s in strings]

        return [simulate_sequential(self._dfa, s) for s in strings]

    def match_batch_timed(self, strings: list[str]) -> tuple[list[bool], dict]:
        t0 = time.perf_counter()
        results = self.match_batch(strings)
        t1 = time.perf_counter()
        timing = {'scan_ms': (t1 - t0) * 1000}
        return results, timing
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_optimized_engine.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/optimized_engine.py tests/test_optimized_engine.py
git commit -m "feat: add OptimizedEngine with auto-selection across monoid/kgram/nfa backends"
```

---

### Task 7: Integration — Wire GPU Monoid Into OptimizedEngine

**Files:**
- Modify: `src/optimized_engine.py`
- Test: `tests/test_optimized_engine.py` (extend)

- [ ] **Step 1: Add GPU integration tests**

Append to `tests/test_optimized_engine.py`:

```python
def _gpu_available():
    try:
        from src.gpu_bridge_monoid import MonoidGPUSimulator
        MonoidGPUSimulator()
        return True
    except Exception:
        return False

skip_no_gpu = pytest.mark.skipif(not _gpu_available(), reason="GPU not available")


@skip_no_gpu
class TestOptimizedEngineGPU:
    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_gpu_monoid_matches_cpu(self, pattern_name):
        from src.optimized_engine import OptimizedEngine
        pat = PATTERNS[pattern_name]

        cpu_engine = OptimizedEngine(pat.regex, config="monoid")
        gpu_engine = OptimizedEngine(pat.regex, config="monoid+gpu")

        random.seed(42)
        alpha = sorted(compile_regex(pat.regex).alphabet)
        strings = [''.join(random.choice(alpha) for _ in range(random.randint(0, 200)))
                    for _ in range(200)]

        cpu_results = cpu_engine.match_batch(strings)
        gpu_results = gpu_engine.match_batch(strings)
        assert gpu_results == cpu_results

    def test_gpu_long_string(self):
        from src.optimized_engine import OptimizedEngine
        engine = OptimizedEngine('(a|b)*abb', config="monoid+gpu")

        random.seed(77)
        s = ''.join(random.choice('ab') for _ in range(100000))
        dfa = compile_regex('(a|b)*abb')
        expected = simulate_sequential(dfa, s)
        got = engine.match_batch([s])[0]
        assert got == expected
```

- [ ] **Step 2: Add GPU monoid backend to OptimizedEngine**

Modify `src/optimized_engine.py` to add a `"monoid+gpu"` config:

In `__init__`, add a new elif branch:

```python
        elif config == "monoid+gpu":
            self._setup_dfa()
            self._setup_monoid()
            self._setup_gpu_monoid()
```

Add the `_setup_gpu_monoid` method:

```python
    def _setup_gpu_monoid(self):
        if self._md is None:
            raise RuntimeError("Monoid computation failed; cannot use GPU monoid")
        from src.gpu_bridge_monoid import MonoidGPUSimulator
        sim = MonoidGPUSimulator()
        self._gpu_engine = sim.create_engine(self._md, self._dm)
        self._info['scan_backend'] = 'monoid+gpu'
        self._info['selection_reason'] = 'GPU monoid scan'
```

Update `match_batch` to check for GPU engine first:

```python
    def match_batch(self, strings: list[str]) -> list[bool]:
        if hasattr(self, '_gpu_engine') and self._gpu_engine is not None:
            return self._gpu_engine.simulate_batch(strings)
        # ... rest unchanged
```

Update `match_batch_timed` similarly:

```python
    def match_batch_timed(self, strings: list[str]) -> tuple[list[bool], dict]:
        if hasattr(self, '_gpu_engine') and self._gpu_engine is not None:
            results, kern_ms, total_ms = self._gpu_engine.simulate_batch_timed(strings)
            return results, {'kernel_ms': kern_ms, 'total_ms': total_ms}
        t0 = time.perf_counter()
        results = self.match_batch(strings)
        t1 = time.perf_counter()
        return results, {'scan_ms': (t1 - t0) * 1000}
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/test_optimized_engine.py tests/test_monoid.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/optimized_engine.py tests/test_optimized_engine.py
git commit -m "feat: wire GPU monoid scan into OptimizedEngine"
```

---

### Task 8: Comparative Benchmarks

**Files:**
- Create: `bench/benchmark_optimized.py`

- [ ] **Step 1: Write benchmark script**

```python
# bench/benchmark_optimized.py
import sys, os, time, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential
from src.monoid import compute_monoid, simulate_monoid
from src.kgram import precompute_kgrams, simulate_kgram_monoid, auto_k
from src.generate_data import PATTERNS


def bench_cpu_backends():
    results = []
    patterns = ['abb', 'even_a', 'binary_div3', 'ab_star']
    B_vals = [100, 1000, 10000]
    L_vals = [32, 128, 512, 2048]

    for pname in patterns:
        pat = PATTERNS[pname]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        alpha = sorted(dfa.alphabet)
        k = auto_k(len(alpha))
        kg = precompute_kgrams(dm, k, monoid=md) if md else None

        print(f"\n=== Pattern: {pname} (states={dfa.n_states}, monoid={md.size if md else 'N/A'}, k={k}) ===")

        for B in B_vals:
            for L in L_vals:
                random.seed(42)
                strings = [''.join(random.choice(alpha) for _ in range(L))
                           for _ in range(B)]

                # Sequential baseline
                t0 = time.perf_counter()
                seq_results = [simulate_sequential(dfa, s) for s in strings]
                t_seq = time.perf_counter() - t0

                # Monoid
                if md:
                    t0 = time.perf_counter()
                    mon_results = [simulate_monoid(md, dm, s) for s in strings]
                    t_mon = time.perf_counter() - t0
                    assert mon_results == seq_results
                else:
                    t_mon = float('inf')

                # Monoid + k-gram
                if kg:
                    t0 = time.perf_counter()
                    kg_results = [simulate_kgram_monoid(kg, md, dm, s)
                                  for s in strings]
                    t_kg = time.perf_counter() - t0
                    assert kg_results == seq_results
                else:
                    t_kg = float('inf')

                total_chars = B * L
                seq_tput = total_chars / t_seq / 1e6 if t_seq > 0 else 0
                mon_tput = total_chars / t_mon / 1e6 if t_mon > 0 else 0
                kg_tput = total_chars / t_kg / 1e6 if t_kg > 0 else 0

                print(f"  B={B:5d} L={L:5d}  "
                      f"seq={t_seq*1e3:.1f}ms ({seq_tput:.1f} Mchar/s)  "
                      f"monoid={t_mon*1e3:.1f}ms ({mon_tput:.1f} Mchar/s)  "
                      f"monoid+k{k}gram={t_kg*1e3:.1f}ms ({kg_tput:.1f} Mchar/s)")

                results.append({
                    'pattern': pname, 'B': B, 'L': L,
                    'seq_ms': t_seq * 1e3, 'monoid_ms': t_mon * 1e3,
                    'kgram_ms': t_kg * 1e3, 'k': k,
                    'seq_mchars': seq_tput, 'monoid_mchars': mon_tput,
                    'kgram_mchars': kg_tput,
                })

    return results


def bench_gpu_monoid():
    """Compare GPU monoid scan vs GPU matrix scan (v4 baseline)."""
    results = []
    try:
        from src.gpu_bridge_monoid import MonoidGPUSimulator
        from src.gpu_bridge_v4 import ParallelGPUSimulator
        monoid_sim = MonoidGPUSimulator()
        v4_sim = ParallelGPUSimulator()
    except Exception as e:
        print(f"GPU not available: {e}")
        return results

    patterns = ['abb', 'even_a']
    B_vals = [1000, 10000, 100000]
    L_vals = [32, 128, 512]

    for pname in patterns:
        pat = PATTERNS[pname]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        alpha = sorted(dfa.alphabet)

        print(f"\n=== GPU: {pname} (monoid={md.size}) ===")

        for B in B_vals:
            for L in L_vals:
                total = B * L
                if total > (1 << 23):
                    continue

                random.seed(42)
                strings = [''.join(random.choice(alpha) for _ in range(L))
                           for _ in range(B)]

                # GPU monoid
                m_engine = monoid_sim.create_engine(md, dm, max_total_chars=total + 1,
                                                     max_batch=B + 1)
                _, mk, mt = m_engine.simulate_batch_timed(strings)
                _, mk, mt = m_engine.simulate_batch_timed(strings)
                m_engine.destroy()

                # GPU v4 baseline
                v4_engine = v4_sim.create_engine(dm, max_total_chars=total + 1,
                                                  max_batch=B + 1)
                _, vk, vt = v4_engine.simulate_batch_timed(strings)
                _, vk, vt = v4_engine.simulate_batch_timed(strings)
                v4_engine.destroy()

                gb = total / 1e9
                print(f"  B={B:6d} L={L:4d}  "
                      f"v4_kern={vk:.3f}ms ({gb/vk*1e3:.2f} GB/s)  "
                      f"monoid_kern={mk:.3f}ms ({gb/mk*1e3:.2f} GB/s)  "
                      f"speedup={vk/mk:.1f}x")

                results.append({
                    'pattern': pname, 'B': B, 'L': L,
                    'v4_kern_ms': vk, 'monoid_kern_ms': mk,
                    'speedup': vk / mk if mk > 0 else 0,
                })

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("CPU Backend Comparison")
    print("=" * 60)
    cpu_results = bench_cpu_backends()

    print("\n" + "=" * 60)
    print("GPU Monoid vs v4 Matrix Scan")
    print("=" * 60)
    gpu_results = bench_gpu_monoid()

    os.makedirs('results', exist_ok=True)
    with open('results/optimized_benchmarks.json', 'w') as f:
        json.dump({'cpu': cpu_results, 'gpu': gpu_results}, f, indent=2)
    print("\nResults saved to results/optimized_benchmarks.json")
```

- [ ] **Step 2: Run benchmarks**

Run: `python bench/benchmark_optimized.py`
Expected: Prints CPU comparison tables and GPU comparison tables. Monoid should show significant speedup over sequential.

- [ ] **Step 3: Commit**

```bash
git add bench/benchmark_optimized.py
git commit -m "feat: add comparative benchmarks for optimized engine backends"
```

---

### Task 9: Final Integration — Run All Tests

**Files:** None (verification only)

- [ ] **Step 1: Run all existing tests to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: All tests pass — original test_correctness.py, test_parallel_engine.py, plus new test_monoid.py, test_kgram.py, test_nfa.py, test_optimized_engine.py.

- [ ] **Step 2: Run CUDA built-in tests**

Run: `make test-v4 && make test-monoid`
Expected: Both test suites pass.

- [ ] **Step 3: Commit any fixups if needed**

Only if previous steps revealed issues.

- [ ] **Step 4: Final commit with all files**

```bash
git add -A
git status
git commit -m "chore: final integration — all optimized engine tests passing"
```
