# Function Map Parallel Prefix Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a warp-shuffle-based function map parallel prefix engine that composes DFA transitions as N-entry maps instead of N×N matrices, achieving ~700–1000 Gc/s for N=16 and unlocking M>255 DFAs that the monoid batch engine cannot handle.

**Architecture:** Two-phase parallel prefix over function maps. Phase 1: each warp tree-reduces a K-character block into a single 16-entry map via `__shfl_sync`. Phase 2: each warp serially composes block products for one string and checks acceptance. A fused transition map LUT (256×16 = 4 KB) maps raw bytes directly to per-state destinations.

**Tech Stack:** CUDA C++17, Python ctypes, numpy

**Spec:** `docs/superpowers/specs/2026-06-18-function-map-prefix-design.md`

---

### Task 1: Tmap Precomputation in Python

**Files:**
- Modify: `src/simulation.py` (add `precompute_tmap` function)
- Test: `tests/test_tmap.py` (new)

- [ ] **Step 1: Write failing test for precompute_tmap**

Create `tests/test_tmap.py`:

```python
import numpy as np
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, precompute_tmap


def test_tmap_even_a():
    """Even-A DFA: (b*ab*ab*)*b* — 2 real states, padded to 16."""
    dfa = compile_regex("(a|b)*a(a|b)")
    dm = DFAMatrices(dfa)
    N = dm.n_states  # padded to 16

    tmap = precompute_tmap(dm)
    assert tmap.dtype == np.uint8
    assert tmap.shape == (256 * N,)

    # For each character in the alphabet, verify against the transition matrix
    for ch in dm.alphabet:
        byte_val = ord(ch)
        T = dm.matrices[ch]
        for s in range(dm.n_states_raw):
            # Find destination: column s of T has a single 1
            dst = int(np.argmax(T[:, s]))
            assert tmap[byte_val * N + s] == dst, \
                f"tmap mismatch: char={ch!r}, state={s}, got={tmap[byte_val * N + s]}, expected={dst}"

    # Unmapped characters should be identity (state maps to itself)
    unmapped_byte = 0  # null byte, not in any DFA alphabet
    if chr(unmapped_byte) not in dm.char_to_idx:
        for s in range(N):
            assert tmap[unmapped_byte * N + s] == s, \
                f"unmapped char not identity: state={s}, got={tmap[unmapped_byte * N + s]}"


def test_tmap_padded_states_identity():
    """Padded states (beyond n_states_raw) should self-loop."""
    dfa = compile_regex("ab")
    dm = DFAMatrices(dfa)
    N = dm.n_states

    tmap = precompute_tmap(dm)

    for byte_val in range(256):
        for s in range(dm.n_states_raw, N):
            assert tmap[byte_val * N + s] == s, \
                f"padded state {s} not identity for byte {byte_val}"


def test_tmap_compose_matches_matrix():
    """Composing two tmaps should match matrix multiplication."""
    dfa = compile_regex("(a|b)*abb")
    dm = DFAMatrices(dfa)
    N = dm.n_states

    tmap = precompute_tmap(dm)

    # Compose f_a then f_b: for each state s, apply f_a then f_b
    a_byte = ord('a')
    b_byte = ord('b')
    for s in range(dm.n_states_raw):
        intermediate = tmap[a_byte * N + s]
        composed = tmap[b_byte * N + intermediate]
        # Compare with matrix product T_b @ T_a
        T_ab = dm.matrices['b'].astype(np.int32) @ dm.matrices['a'].astype(np.int32)
        expected = int(np.argmax(T_ab[:, s]))
        assert composed == expected, \
            f"compose mismatch: state={s}, got={composed}, expected={expected}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tmap.py -v`
Expected: FAIL with `ImportError: cannot import name 'precompute_tmap'`

- [ ] **Step 3: Implement precompute_tmap**

Add to `src/simulation.py` at the end (before the `if __name__` block):

```python
def precompute_tmap(dm: DFAMatrices) -> np.ndarray:
    """Build fused transition map: tmap[byte_val * N + state] = dest_state.

    For each raw byte value (0-255), maps each DFA state to its destination.
    Unmapped characters act as identity (state maps to itself).
    Table size: 256 * N bytes (4 KB for N=16).
    """
    N = dm.n_states
    tmap = np.zeros(256 * N, dtype=np.uint8)

    # Default: identity for all bytes and all states
    for s in range(N):
        for byte_val in range(256):
            tmap[byte_val * N + s] = s

    # Overwrite with actual transitions for alphabet characters
    for ch_name in dm.alphabet:
        byte_val = ord(ch_name)
        T = dm.matrices[ch_name]
        for s in range(dm.n_states_raw):
            dst = int(np.argmax(T[:, s]))
            tmap[byte_val * N + s] = dst

    return tmap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tmap.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/simulation.py tests/test_tmap.py
git commit -m "feat: add precompute_tmap for function map prefix engine"
```

---

### Task 2: CUDA Kernel — Phase 1 Block Tree Reduce

**Files:**
- Create: `cuda/prefix_compose.cu`
- No test file yet (built-in tests come in Task 5)

This task builds the CUDA file with the Phase 1 kernel, the engine struct shell, and a CPU reference. Phase 2 kernel is Task 3. The file follows the `monoid_batch.cu` pattern: kernels at top, engine struct, C API under `#ifdef BUILD_LIB`, built-in tests under `#else`.

- [ ] **Step 1: Create cuda/prefix_compose.cu with Phase 1 kernel**

```cuda
/*
 * prefix_compose.cu — Function Map Parallel Prefix Engine
 *
 * Warp-shuffle-based function map composition for DFA simulation.
 * Each transition is a map f: {0..N-1} → {0..N-1} (N bytes, not N² matrix).
 * Composition is O(N) gathers via __shfl_sync instead of O(N³) matmul.
 *
 * Two-phase parallel prefix:
 *   Phase 1 (prefix_block_reduce_kernel): tree-reduce K chars per block via shuffles
 *   Phase 2 (prefix_scan_accept_kernel): serial compose of block products + accept
 *
 * N is fixed at 16 (one map entry per lane in a half-warp).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

constexpr int N_STATES = 16;
constexpr int BLOCK_K = 32;       // characters per block
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
// Each warp processes one K-character block.
// Lanes 0-15 hold the map entry for their state index.
// Tree reduce: compose K maps into 1 via log2(K) levels of __shfl_sync.
//
// Shared memory layout per CUDA block:
//   tmap_sh[256 * 16]          — fused transition map LUT (4 KB, shared)
//   maps_sh[PC_WARPS][BLOCK_K][16]  — workspace for tree reduce intermediates

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
    uint8_t *tmap_sh = smem;                                // 256 * 16 = 4096 bytes
    // Per-warp workspace: maps_sh[warp_id][BLOCK_K][16]
    uint8_t *maps_base = smem + 256 * N_STATES;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Cooperative load of tmap into shared memory
    for (int i = threadIdx.x; i < 256 * N_STATES; i += PC_BLOCK)
        tmap_sh[i] = d_tmap[i];
    __syncthreads();

    // Each warp handles one block
    int global_block_id = blockIdx.x * PC_WARPS + warp_id;
    if (global_block_id >= total_blocks) return;

    int sid = block_desc_string_id[global_block_id];
    int block_char_off = block_desc_char_offset[global_block_id];
    int str_start = string_offsets[sid];
    int str_len = string_lengths[sid];

    // Pointer to this warp's workspace: [BLOCK_K][16]
    uint8_t *my_maps = maps_base + warp_id * BLOCK_K * N_STATES;

    // Only lanes 0-15 participate in map composition
    if (lane < N_STATES) {
        // Step 1: Load K characters, map each to a transition map entry for this lane
        for (int k = 0; k < BLOCK_K; k++) {
            int char_pos = block_char_off + k;
            if (char_pos < str_len) {
                uint8_t byte_val = raw_concat[str_start + char_pos];
                my_maps[k * N_STATES + lane] = tmap_sh[byte_val * N_STATES + lane];
            } else {
                my_maps[k * N_STATES + lane] = (uint8_t)lane;  // identity
            }
        }
        __syncwarp(0x0000FFFF);

        // Step 2: Tree reduce BLOCK_K maps into 1
        // Level 0: 32 → 16, Level 1: 16 → 8, ... Level 4: 2 → 1
        int count = BLOCK_K;
        while (count > 1) {
            int half = count / 2;
            for (int i = lane < N_STATES ? 0 : half; i < half; i++) {
                // Compose: newer = maps[2*i+1], older = maps[2*i]
                // result[lane] = newer[older[lane]]
                uint8_t older_val = my_maps[(2 * i) * N_STATES + lane];
                uint8_t newer_val = my_maps[(2 * i + 1) * N_STATES + lane];
                uint8_t composed = (uint8_t)__shfl_sync(0x0000FFFF, (int)newer_val, (int)older_val);
                my_maps[i * N_STATES + lane] = composed;
            }
            // Handle odd element: copy down
            if (count % 2 == 1 && lane < N_STATES) {
                my_maps[half * N_STATES + lane] = my_maps[count - 1 + (count - 1) * N_STATES + lane];
                // Actually for odd count, just copy the last one
            }
            count = half + (count % 2);
            __syncwarp(0x0000FFFF);
        }

        // Write block product to global memory
        block_products[global_block_id * N_STATES + lane] = my_maps[lane];
    }
}
```

Wait — the tree reduce with odd elements is tricky. Let me use a cleaner approach: pad BLOCK_K to a power of 2 (it already is: 32). The loop becomes simpler.

Actually, let me reconsider the kernel design. With BLOCK_K=32 (already power of 2), the tree reduce is clean: 5 levels, no odd handling. But writing to/from shared memory for each level is expensive. A better approach: **serial compose in registers for the first pass, then shuffle-reduce.**

Here's the refined design:
- Each of 16 lanes loads BLOCK_K map values from smem (one per character)
- Each lane serially composes its BLOCK_K values via shuffles: `acc = shfl(map[k], acc)`
- This gives the full block product in one pass, no tree needed

This is simpler and faster. Each lane tracks `acc[lane]` = the composed map applied to state `lane`. For each character, `acc = tmap[char][acc]` — but we need the *composed map*, not the evaluated result for one starting state.

The composition `(f_K ∘ ... ∘ f_1)[lane]` can be computed as:
```
acc = lane  (identity)
for k = 0 to K-1:
    acc = tmap[char_k * 16 + acc]   // apply f_k to acc
```

But this requires each lane to read `tmap[char_k * 16 + acc]` where `acc` differs per lane. This is a shared-memory gather — each lane reads a different address. That works fine with smem.

Actually, that IS the serial compose. No tree needed. Each lane independently computes `f_{K-1}(...(f_1(f_0(lane))))` via K shared memory reads. For K=32 and 16 lanes, that's 32 iterations of 1 smem read each.

Let me revise the kernel to use this much simpler approach.

- [ ] **Step 1: Create cuda/prefix_compose.cu with Phase 1 kernel (serial compose per warp)**

```cuda
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
constexpr int PC_BLOCK = PC_WARPS * 32;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)


__launch_bounds__(PC_BLOCK)
__global__ void prefix_block_reduce_kernel(
    const uint8_t * __restrict__ raw_concat,
    const int     * __restrict__ block_desc_string_id,
    const int     * __restrict__ block_desc_char_offset,
    const int     * __restrict__ string_offsets,
    const int     * __restrict__ string_lengths,
    const uint8_t * __restrict__ d_tmap,
    int total_blocks,
    uint8_t       * __restrict__ block_products)
{
    extern __shared__ uint8_t smem[];
    uint8_t *tmap_sh = smem;  // 256 * 16 = 4096 bytes

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

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

    // Each of 16 lanes composes the block's map for its starting state
    uint8_t acc = (uint8_t)lane;  // identity map

    for (int k = 0; k < BLOCK_K; k++) {
        int pos = block_off + k;
        if (pos < str_len) {
            uint8_t byte_val = raw_concat[str_start + pos];
            acc = tmap_sh[byte_val * N_STATES + acc];
        }
    }

    block_products[gid * N_STATES + lane] = acc;
}
```

- [ ] **Step 2: Verify it compiles**

Run: `nvcc -O3 -arch=sm_90 --use_fast_math -std=c++17 -c cuda/prefix_compose.cu -o /dev/null`
Expected: Compiles with no errors

- [ ] **Step 3: Commit**

```bash
git add cuda/prefix_compose.cu
git commit -m "feat: prefix_compose Phase 1 block reduce kernel"
```

---

### Task 3: CUDA Kernel — Phase 2 Scan + Accept, Engine Struct, C API

**Files:**
- Modify: `cuda/prefix_compose.cu`

- [ ] **Step 1: Add Phase 2 kernel to prefix_compose.cu**

Append after the Phase 1 kernel:

```cuda
__launch_bounds__(PC_BLOCK)
__global__ void prefix_scan_accept_kernel(
    const uint8_t * __restrict__ block_products,  // [total_blocks * 16]
    const int     * __restrict__ string_n_blocks,  // n_blocks per string
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

    // Serial compose of block products via shuffles
    uint8_t acc = (uint8_t)lane;  // identity

    for (int b = 0; b < n_blk; b++) {
        uint8_t bp = block_products[(blk_start + b) * N_STATES + lane];
        // Compose: acc = bp[acc[lane]] for each lane
        // acc holds the composed map so far; bp is the next block's map
        // new_acc[lane] = bp[acc[lane]]
        acc = (uint8_t)__shfl_sync(0x0000FFFF, (int)bp, (int)acc);
    }

    // Lane start_state holds the final state
    if (lane == 0) {
        int final_state = __shfl_sync(0x0000FFFF, (int)acc, start_state);
        results[sid] = (int)d_accept[final_state];
    }
}
```

- [ ] **Step 2: Add the PrefixComposeEngine struct**

Append after the Phase 2 kernel:

```cuda
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

    // Block descriptor arrays
    int     *d_block_string_id;
    int     *d_block_char_offset;
    int     *d_string_n_blocks;
    int     *d_string_block_start;
    int     *d_string_lengths;
    uint8_t *d_block_products;
    int      max_total_blocks;

    // Host-side scratch for building block descriptors
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
        // Estimate max blocks: max_chars / BLOCK_K + max_batch
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
        CHECK_CUDA(cudaMemcpy(d_raw_concat, h_raw_concat, total_chars, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_block_string_id,   h_block_string_id,   total_blocks * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_block_char_offset,  h_block_char_offset, total_blocks * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_string_n_blocks,    h_string_n_blocks,   B * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_string_block_start, h_string_block_start, B * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_string_lengths,     h_string_lengths,    B * sizeof(int), cudaMemcpyHostToDevice));

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

        CHECK_CUDA(cudaMemcpy(h_results, d_results, B * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        CHECK_CUDA(cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop));
        CHECK_CUDA(cudaEventElapsedTime(total_ms, ev_start, ev_stop));
    }
};
```

- [ ] **Step 3: Add C API under #ifdef BUILD_LIB**

```cuda
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

#else  // standalone test — built-in tests go here (Task 5)

#endif
```

- [ ] **Step 4: Verify it compiles as library**

Run: `nvcc -O3 -arch=sm_90 --use_fast_math -std=c++17 -DBUILD_LIB -shared -Xcompiler -fPIC -o build/libprefix_compose.so cuda/prefix_compose.cu`
Expected: Compiles with no errors

- [ ] **Step 5: Commit**

```bash
git add cuda/prefix_compose.cu
git commit -m "feat: prefix_compose Phase 2 kernel, engine struct, C API"
```

---

### Task 4: Makefile Targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add prefix_compose targets to Makefile**

Add these variable definitions after the `SRC_MONOID_BATCH` block (around line 42):

```makefile
SRC_PREFIX = $(CUDA_DIR)/prefix_compose.cu
EXE_PREFIX = $(BUILD_DIR)/prefix_compose
LIB_PREFIX = $(BUILD_DIR)/libprefix_compose.so
```

Add `$(EXE_PREFIX) $(LIB_PREFIX)` to the `all:` target list.

Add build rules after the monoid_batch rules:

```makefile
$(EXE_PREFIX): $(SRC_PREFIX) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_PREFIX): $(SRC_PREFIX) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<
```

Add test target:

```makefile
test-prefix: $(EXE_PREFIX)
	./$(EXE_PREFIX)
```

Add `test-prefix` to the `.PHONY` list.

- [ ] **Step 2: Verify build**

Run: `make build/libprefix_compose.so`
Expected: Builds successfully

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add prefix_compose Makefile targets"
```

---

### Task 5: Built-in CUDA Tests

**Files:**
- Modify: `cuda/prefix_compose.cu` (add `#else` test block)

- [ ] **Step 1: Add CPU reference and built-in tests**

Replace the `#else  // standalone test` placeholder with:

```cuda
#else  // standalone test

static int tests_passed = 0, tests_failed = 0;
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

// CPU reference: serial function map compose
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

        // Build composed map
        uint8_t map[16];
        for (int s = 0; s < N; s++) map[s] = (uint8_t)s;  // identity

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

// Even-A DFA: (a|b)*a(a|b) — accepts if second-to-last char is 'a'
// States: 0 (start), 1 (saw a), 2 (accept, saw a then anything)
// Padded to 16 states.
static void build_even_a_tmap(uint8_t *tmap, uint8_t *accept, int *start_state) {
    int N = N_STATES;
    // Default: identity
    for (int b = 0; b < 256; b++)
        for (int s = 0; s < N; s++)
            tmap[b * N + s] = (uint8_t)s;

    // 'a' transitions: 0→1, 1→1, 2→1
    int a = 'a';
    tmap[a * N + 0] = 1;
    tmap[a * N + 1] = 1;
    tmap[a * N + 2] = 1;

    // 'b' transitions: 0→0, 1→2, 2→0
    int b = 'b';
    tmap[b * N + 0] = 0;
    tmap[b * N + 1] = 2;
    tmap[b * N + 2] = 0;

    memset(accept, 0, N);
    accept[2] = 1;
    *start_state = 0;
}

void test_correctness_small() {
    printf("test_correctness_small... ");

    uint8_t tmap[256 * N_STATES], accept[N_STATES];
    int start_state;
    build_even_a_tmap(tmap, accept, &start_state);

    // Test strings: (a|b)*a(a|b) accepts strings where second-to-last is 'a'
    const char *test_strings[] = {
        "", "a", "b", "aa", "ab", "ba", "bb",
        "aab", "aba", "bab", "bba", "aabb", "abab"
    };
    // Expected: len<2 → reject (except "aa","ab" accept)
    int expected[] = {
        0, 0, 0, 1, 1, 0, 0,
        1, 1, 1, 0, 0, 1
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

    int *gpu_results = new int[n_tests];
    int *cpu_results = new int[n_tests];
    float kern_ms, total_ms;

    PrefixComposeEngine engine;
    engine.init(tmap, accept, start_state, N_STATES, total_chars + 1, n_tests + 1);
    engine.dispatch(raw_concat, offsets, gpu_results, n_tests, total_chars, &kern_ms, &total_ms);

    cpu_prefix_compose(raw_concat, offsets, tmap, accept, start_state, N_STATES, n_tests, cpu_results);

    int mismatches = 0;
    for (int i = 0; i < n_tests; i++) {
        if (gpu_results[i] != expected[i] || cpu_results[i] != expected[i]) {
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
    build_even_a_tmap(tmap, accept, &start_state);

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
    build_even_a_tmap(tmap, accept, &start_state);

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
    build_even_a_tmap(tmap, accept, &start_state);

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
```

- [ ] **Step 2: Build and run standalone tests**

Run: `make build/prefix_compose && ./build/prefix_compose`
Expected: All correctness tests PASS, benchmark prints Gc/s numbers

- [ ] **Step 3: Commit**

```bash
git add cuda/prefix_compose.cu
git commit -m "test: built-in CUDA tests for prefix compose engine"
```

---

### Task 6: Python ctypes Bridge

**Files:**
- Create: `src/gpu_bridge_prefix_compose.py`
- Test: `tests/test_gpu_bridge_prefix.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/test_gpu_bridge_prefix.py`:

```python
import pytest
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, precompute_tmap
from src.gpu_bridge_prefix_compose import PrefixComposeGPUSimulator


@pytest.fixture
def prefix_engine():
    dfa = compile_regex("(a|b)*abb")
    dm = DFAMatrices(dfa)
    sim = PrefixComposeGPUSimulator()
    engine = sim.create_engine(dm)
    yield engine, dfa
    engine.destroy()


def test_basic_correctness(prefix_engine):
    engine, dfa = prefix_engine
    strings = ["abb", "aabb", "babb", "ab", "ba", ""]
    expected = [dfa.simulate(s) for s in strings]
    results = engine.simulate_batch(strings)
    assert results == expected


def test_long_strings(prefix_engine):
    engine, dfa = prefix_engine
    import random
    random.seed(42)
    strings = ["".join(random.choice("ab") for _ in range(1000)) for _ in range(100)]
    expected = [dfa.simulate(s) for s in strings]
    results = engine.simulate_batch(strings)
    assert results == expected


def test_timed_dispatch(prefix_engine):
    engine, dfa = prefix_engine
    strings = ["abb", "aabb", "babb"]
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    expected = [dfa.simulate(s) for s in strings]
    assert results == expected
    assert kern_ms >= 0
    assert total_ms >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_bridge_prefix.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement gpu_bridge_prefix_compose.py**

Create `src/gpu_bridge_prefix_compose.py`:

```python
"""
Python bridge to the prefix compose GPU engine via ctypes.

The prefix compose engine uses warp-shuffle-based function map composition
instead of matrix multiplication. Each DFA transition is an N-entry map,
composed via O(N) gathers instead of O(N³) matmul.

Usage:
    from src.gpu_bridge_prefix_compose import PrefixComposeGPUSimulator
    sim = PrefixComposeGPUSimulator()
    engine = sim.create_engine(dm)
    results = engine.simulate_batch(["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    engine.destroy()
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.simulation import DFAMatrices, precompute_tmap


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libprefix_compose.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libprefix_compose.so not found at {base}. Run 'make' first."
    )


class PrefixComposeEngine:
    """Wraps a persistent GPU engine context for prefix compose dispatch."""

    def __init__(self, lib, dm: DFAMatrices,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.dm = dm

        tmap = np.ascontiguousarray(precompute_tmap(dm))
        accept = np.zeros(dm.n_states, dtype=np.uint8)
        for s in dm.dfa.accept_states:
            accept[s] = 1

        rc = self.lib.prefix_engine_init(
            tmap.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            dm.dfa.start,
            dm.n_states,
            32,  # K = BLOCK_K
            max_total_chars,
            max_batch,
        )
        if rc != 0:
            raise RuntimeError(f"prefix_engine_init failed with code {rc}")

    def destroy(self):
        self.lib.prefix_engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        B = len(strings)
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)
        total_chars = int(offsets[B])
        if total_chars > 0:
            raw_concat = np.frombuffer(
                "".join(strings).encode("latin-1"), dtype=np.uint8
            ).copy()
        else:
            raw_concat = np.zeros(1, dtype=np.uint8)
        return raw_concat, offsets, total_chars

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        if not strings:
            return []

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        B = len(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.prefix_engine_dispatch(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"prefix_engine dispatch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        if not strings:
            return [], 0.0, 0.0

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        B = len(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.prefix_engine_dispatch(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"prefix_engine dispatch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class PrefixComposeGPUSimulator:
    """Factory for PrefixComposeEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.prefix_engine_device_check.restype = ctypes.c_int
        self.lib.prefix_engine_device_check.argtypes = []

        self.lib.prefix_engine_init.restype = ctypes.c_int
        self.lib.prefix_engine_init.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # tmap
            ctypes.POINTER(ctypes.c_uint8),   # accept
            ctypes.c_int,                     # start_state
            ctypes.c_int,                     # N
            ctypes.c_int,                     # K
            ctypes.c_int,                     # max_total_chars
            ctypes.c_int,                     # max_batch
        ]

        self.lib.prefix_engine_destroy.restype = None
        self.lib.prefix_engine_destroy.argtypes = []

        self.lib.prefix_engine_dispatch.restype = ctypes.c_int
        self.lib.prefix_engine_dispatch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # raw_concat
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.c_int,                     # B
            ctypes.c_int,                     # total_chars
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        rc = self.lib.prefix_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, dm: DFAMatrices,
                      max_total_chars: int = 1 << 22,
                      max_batch: int = 1 << 18) -> PrefixComposeEngine:
        return PrefixComposeEngine(self.lib, dm,
                                   max_total_chars, max_batch)
```

- [ ] **Step 4: Build library and run tests**

Run: `make build/libprefix_compose.so && python -m pytest tests/test_gpu_bridge_prefix.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gpu_bridge_prefix_compose.py tests/test_gpu_bridge_prefix.py
git commit -m "feat: Python ctypes bridge for prefix compose engine"
```

---

### Task 7: OptimizedEngine Integration

**Files:**
- Modify: `src/optimized_engine.py`
- Test: `tests/test_optimized_engine.py` (add new test)

- [ ] **Step 1: Write failing test**

Add to `tests/test_optimized_engine.py`:

```python
def test_prefix_gpu_config():
    """Test explicit prefix+gpu config."""
    engine = OptimizedEngine("(a|b)*abb", config="prefix+gpu")
    assert engine.config_info["scan_backend"] == "prefix+gpu"

    strings = ["abb", "aabb", "babb", "ab", "ba", ""]
    results = engine.match_batch(strings)
    expected = [True, True, True, False, False, False]
    assert results == expected


def test_prefix_gpu_timed():
    engine = OptimizedEngine("(a|b)*abb", config="prefix+gpu")
    strings = ["abb", "aabb"]
    results, timing = engine.match_batch_timed(strings)
    assert results == [True, True]
    assert "kernel_ms" in timing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_optimized_engine.py::test_prefix_gpu_config -v`
Expected: FAIL with `ValueError: Unknown config: 'prefix+gpu'`

- [ ] **Step 3: Add prefix+gpu to OptimizedEngine**

In `src/optimized_engine.py`, add the `_prefix_compose_gpu` attribute in `__init__` alongside the other GPU engine attributes:

```python
self._prefix_compose_gpu = None  # PrefixComposeEngine (GPU)
```

Add the config case in the `__init__` elif chain (after the `monoid_batch+gpu` case):

```python
elif config == "prefix+gpu":
    self._force_baseline()
    self._setup_prefix_gpu()
```

Update the error message to include `'prefix+gpu'`:

```python
raise ValueError(f"Unknown config: {config!r}. "
                 f"Choose from None, 'monoid', 'monoid+kgram', 'baseline', 'nfa', "
                 f"'monoid+gpu', 'batched+gpu', 'kgram+gpu', 'monoid_batch+gpu', "
                 f"'prefix+gpu'.")
```

Add the setup method:

```python
def _setup_prefix_gpu(self):
    self._build_dfa()
    from src.gpu_bridge_prefix_compose import PrefixComposeGPUSimulator
    sim = PrefixComposeGPUSimulator()
    self._prefix_compose_gpu = sim.create_engine(
        self._dm,
        max_total_chars=1 << 29,
        max_batch=1 << 19,
    )
    self._scan_backend = 'prefix+gpu'
    self._selection_reason = (
        f'GPU prefix compose (N={self._dm.n_states})'
    )
```

Add dispatch in `_match_one` (before the monoid_batch_gpu check):

```python
if self._prefix_compose_gpu is not None:
    return self._prefix_compose_gpu.simulate_batch([s])[0]
```

Add dispatch in `match_batch` (before monoid_batch_gpu):

```python
if self._prefix_compose_gpu is not None:
    return self._prefix_compose_gpu.simulate_batch(strings)
```

Add dispatch in `match_batch_timed` (before monoid_batch_gpu):

```python
if self._prefix_compose_gpu is not None:
    results, kern_ms, total_ms = self._prefix_compose_gpu.simulate_batch_timed(strings)
    return results, {'kernel_ms': kern_ms, 'total_ms': total_ms}
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_optimized_engine.py::test_prefix_gpu_config tests/test_optimized_engine.py::test_prefix_gpu_timed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimized_engine.py tests/test_optimized_engine.py
git commit -m "feat: integrate prefix+gpu config into OptimizedEngine"
```

---

### Task 8: Auto-Selection for M > 255

**Files:**
- Modify: `src/optimized_engine.py` (update `_auto_select`)

- [ ] **Step 1: Update _auto_select to use prefix+gpu when monoid is too large for monoid_batch**

In the `_auto_select` method, after the monoid batch GPU attempt and the monoid+kgram fallback, add a prefix+gpu fallback for the case when the monoid is too large (M > 255) or monoid computation fails:

Change the `_auto_select` method. After the existing `if md is not None:` block with the `md.size <= 255` check, add a branch where M > 255 tries prefix+gpu:

```python
if md is not None:
    self._md = md
    if md.size <= 255:
        try:
            self._setup_monoid_batch_gpu()
            self._representation = "dfa"
            self._selection_reason = (
                f"DFA has {n_states} states; monoid size {md.size} ≤ 255; "
                f"auto-selected monoid_batch+gpu"
            )
            return
        except Exception:
            pass
    else:
        # M > 255: monoid batch can't handle this, try prefix compose
        try:
            self._setup_prefix_gpu()
            self._representation = "dfa"
            self._selection_reason = (
                f"DFA has {n_states} states; monoid size {md.size} > 255; "
                f"auto-selected prefix+gpu"
            )
            return
        except Exception:
            pass
    # ... existing kgram fallback continues
```

Also add a prefix+gpu attempt in the `else` block where monoid computation fails entirely (md is None):

```python
else:
    # Monoid too large — try prefix compose, fall back to sequential
    try:
        self._setup_prefix_gpu()
        self._representation = "dfa"
        self._selection_reason = (
            f"DFA has {n_states} states; monoid exceeds cap; "
            f"auto-selected prefix+gpu"
        )
        return
    except Exception:
        pass
    self._representation = "dfa"
    self._scan_backend = "sequential"
    self._selection_reason = (
        f"DFA has {n_states} states; monoid exceeds cap {self._monoid_cap}; "
        f"using sequential"
    )
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `python -m pytest tests/test_optimized_engine.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/optimized_engine.py
git commit -m "feat: auto-select prefix+gpu for M>255 DFAs"
```

---

### Task 9: Integration Test + Benchmark

**Files:**
- Modify: `tests/test_gpu_bridge_prefix.py` (add cross-engine validation)

- [ ] **Step 1: Add cross-engine validation test**

Add to `tests/test_gpu_bridge_prefix.py`:

```python
def test_cross_engine_validation():
    """Compare prefix compose results against sequential DFA simulation."""
    regexes = [
        "(a|b)*abb",
        "(a|b)*a(a|b)",
        "a*b*",
        "(ab|ba)*",
        "(a|b|c)*abc",
    ]
    import random
    random.seed(123)

    for regex in regexes:
        dfa = compile_regex(regex)
        dm = DFAMatrices(dfa)
        sim = PrefixComposeGPUSimulator()
        engine = sim.create_engine(dm)

        # Mix of short and long strings
        alphabet = list(dfa.alphabet)
        strings = []
        for _ in range(200):
            length = random.randint(0, 500)
            strings.append("".join(random.choice(alphabet) for _ in range(length)))

        gpu_results = engine.simulate_batch(strings)
        cpu_results = [dfa.simulate(s) for s in strings]

        mismatches = sum(g != c for g, c in zip(gpu_results, cpu_results))
        assert mismatches == 0, \
            f"Regex {regex!r}: {mismatches}/{len(strings)} mismatches"

        engine.destroy()
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/test_gpu_bridge_prefix.py tests/test_tmap.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_gpu_bridge_prefix.py
git commit -m "test: cross-engine validation for prefix compose"
```
