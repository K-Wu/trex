# Monoid Batch Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TC MMA with O(1) monoid compose-table lookups for small-monoid DFAs, targeting 1500-2500 Gc/s (13-21x over current 116 Gc/s TC V3).

**Architecture:** One CUDA thread per string reads raw bytes from a concatenated buffer via CSR offsets and looks up a precomputed `char_compose[M × sigma_ext]` table in shared memory. A parallel prefix variant handles few very long strings (B ≤ 128, L > 100K). Auto-selection in OptimizedEngine routes to the fastest available kernel.

**Tech Stack:** CUDA C++ (SM ≥ 7.0), Python 3.10+, ctypes, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-06-18-monoid-batch-pipeline-design.md`

---

## File Structure

### New files
- `cuda/monoid_batch.cu` — Monoid batch kernel, parallel prefix kernel, engine struct, C API, standalone tests
- `src/gpu_bridge_monoid_batch.py` — Python ctypes bridge to libmonoid_batch.so
- `tests/test_monoid_batch_gpu.py` — GPU correctness tests (cross-validate vs CPU)

### Modified files
- `src/monoid.py` — Add `precompute_batch_tables()` function
- `src/optimized_engine.py` — Add `monoid_batch+gpu` config and auto-selection routing
- `Makefile` — Add build targets for monoid_batch.cu
- `tests/test_monoid.py` — Add tests for precompute_batch_tables

---

### Task 1: Precompute Batch Tables (Python)

**Files:**
- Modify: `src/monoid.py` (append new function after `simulate_monoid`)
- Modify: `tests/test_monoid.py` (append new test class)

This task adds a Python function that transforms the existing `MonoidData` (M×M uint16 compose table) into GPU-friendly tables: a fused `char_compose[M × sigma_ext]` uint8 table (monoid element × DFA char index → next monoid element), a `raw_char_map[256]` uint8 table (raw byte → DFA char index), and a `monoid_compose[M × M]` uint8 table (for the parallel prefix tree reduce).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_monoid.py`:

```python
# ═══════════════════════════════════════════════════════════════════════════
# 5. TestPrecomputeBatchTables — validate GPU-friendly tables
# ═══════════════════════════════════════════════════════════════════════════

class TestPrecomputeBatchTables:
    """Validate precompute_batch_tables against simulate_monoid."""

    @pytest.mark.parametrize("pattern_name", ["abb", "even_a", "identifier"])
    def test_tables_reproduce_simulate_monoid(self, pattern_name):
        """Step through a string using the batch tables and compare to simulate_monoid."""
        from src.monoid import precompute_batch_tables
        dm, dfa = _make_dm(pattern_name)
        md = compute_monoid(dm)
        assert md is not None

        tables = precompute_batch_tables(md, dm)
        char_compose = tables['char_compose']
        raw_char_map = tables['raw_char_map']
        accept = tables['accept']
        M = tables['M']
        sigma_ext = tables['sigma_ext']
        identity = tables['identity_idx']

        alphabet = _get_alphabet(pattern_name)
        rng = random.Random(42)
        for _ in range(200):
            length = rng.randint(0, 100)
            s = "".join(rng.choice(alphabet) for _ in range(length))

            # Simulate using batch tables (mimics GPU kernel logic)
            curr = identity
            for ch in s:
                ch_idx = raw_char_map[ord(ch)]
                curr = char_compose[curr * sigma_ext + ch_idx]
            batch_accept = bool(accept[curr])

            # Reference
            ref_accept = simulate_monoid(md, dm, s)
            assert batch_accept == ref_accept, (
                f"Mismatch for '{s[:40]}': batch={batch_accept}, ref={ref_accept}"
            )

    @pytest.mark.parametrize("pattern_name", ["abb", "even_a"])
    def test_identity_column(self, pattern_name):
        """Unmapped characters should not change the monoid state."""
        from src.monoid import precompute_batch_tables
        dm, _ = _make_dm(pattern_name)
        md = compute_monoid(dm)
        tables = precompute_batch_tables(md, dm)
        char_compose = tables['char_compose']
        raw_char_map = tables['raw_char_map']
        sigma_ext = tables['sigma_ext']

        # 'z' is not in {a,b} alphabet — should map to identity column
        z_idx = raw_char_map[ord('z')]
        assert z_idx == sigma_ext - 1, "unmapped char should map to identity column"

        # Composing with identity column should not change state
        for m in range(tables['M']):
            assert char_compose[m * sigma_ext + z_idx] == m, (
                f"identity column broken for monoid element {m}"
            )

    @pytest.mark.parametrize("pattern_name", ["abb", "even_a"])
    def test_monoid_compose_table(self, pattern_name):
        """monoid_compose[i*M+j] should match compose_table[i,j] (cast to uint8)."""
        from src.monoid import precompute_batch_tables
        dm, _ = _make_dm(pattern_name)
        md = compute_monoid(dm)
        tables = precompute_batch_tables(md, dm)
        monoid_compose = tables['monoid_compose']
        M = tables['M']

        for i in range(M):
            for j in range(M):
                expected = int(md.compose_table[i, j])
                got = int(monoid_compose[i * M + j])
                assert got == expected, f"compose mismatch at [{i},{j}]: {got} vs {expected}"

    def test_shapes(self):
        """Table shapes and dtypes are correct."""
        from src.monoid import precompute_batch_tables
        dm, _ = _make_dm("abb")
        md = compute_monoid(dm)
        tables = precompute_batch_tables(md, dm)

        M = tables['M']
        sigma_ext = tables['sigma_ext']
        assert tables['char_compose'].shape == (M * sigma_ext,)
        assert tables['char_compose'].dtype == np.uint8
        assert tables['raw_char_map'].shape == (256,)
        assert tables['raw_char_map'].dtype == np.uint8
        assert tables['accept'].shape == (M,)
        assert tables['accept'].dtype == np.uint8
        assert tables['monoid_compose'].shape == (M * M,)
        assert tables['monoid_compose'].dtype == np.uint8
        assert sigma_ext == len(dm.alphabet) + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_monoid.py::TestPrecomputeBatchTables -v`
Expected: ImportError or AttributeError (precompute_batch_tables not defined yet)

- [ ] **Step 3: Implement precompute_batch_tables**

Add to `src/monoid.py` after the `simulate_monoid` function:

```python
def precompute_batch_tables(md: MonoidData, dm: DFAMatrices) -> dict:
    """Build GPU-friendly tables for the monoid batch kernel.

    Returns a dict with keys:
        char_compose   (M * sigma_ext,) uint8 — fused compose table
                       char_compose[curr * sigma_ext + ch_idx] = next monoid element
        raw_char_map   (256,) uint8 — maps raw ASCII byte to DFA char index
                       unmapped bytes map to sigma (identity column)
        accept         (M,) uint8 — 1 if monoid element is accepting
        monoid_compose (M * M,) uint8 — M×M compose table for tree reduce
        M              int — monoid size (must be ≤ 255 for uint8)
        sigma_ext      int — len(alphabet) + 1 (extra identity column)
        identity_idx   int — index of identity monoid element
    """
    M = md.size
    if M > 255:
        raise ValueError(f"Monoid size {M} exceeds uint8 limit (255)")

    sigma = len(dm.alphabet)
    sigma_ext = sigma + 1

    # char_compose[curr, ch_dfa_idx] = compose_table[char_monoid, curr]
    char_compose = np.zeros((M, sigma_ext), dtype=np.uint8)
    for ch_name, ch_dfa_idx in dm.char_to_idx.items():
        ch_monoid = md.char_to_monoid[ch_name]
        for m in range(M):
            char_compose[m, ch_dfa_idx] = int(md.compose_table[ch_monoid, m])
    for m in range(M):
        char_compose[m, sigma] = m

    raw_char_map = np.full(256, sigma, dtype=np.uint8)
    for ch_name, ch_dfa_idx in dm.char_to_idx.items():
        raw_char_map[ord(ch_name)] = ch_dfa_idx

    accept = np.ascontiguousarray(md.accept_table.astype(np.uint8))

    monoid_compose = np.ascontiguousarray(
        md.compose_table.astype(np.uint8).reshape(-1)
    )

    return {
        'char_compose': np.ascontiguousarray(char_compose.reshape(-1)),
        'raw_char_map': raw_char_map,
        'accept': accept,
        'monoid_compose': monoid_compose,
        'M': M,
        'sigma_ext': sigma_ext,
        'identity_idx': md.identity_idx,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_monoid.py::TestPrecomputeBatchTables -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/monoid.py tests/test_monoid.py
git commit -m "feat: add precompute_batch_tables for monoid batch GPU kernel"
```

---

### Task 2: Makefile + CUDA Monoid Batch Kernel

**Files:**
- Modify: `Makefile` (add monoid_batch build targets)
- Create: `cuda/monoid_batch.cu` (kernel + engine struct + C API + standalone tests)

This task creates the complete CUDA file with the monoid batch kernel (1 thread per string), engine struct (init/destroy/dispatch), C API, and a standalone test that cross-validates against a CPU reference.

- [ ] **Step 1: Add build targets to Makefile**

Add after the kgram block (line 38) in `Makefile`:

```makefile
SRC_MONOID_BATCH = $(CUDA_DIR)/monoid_batch.cu
EXE_MONOID_BATCH = $(BUILD_DIR)/monoid_batch
LIB_MONOID_BATCH = $(BUILD_DIR)/libmonoid_batch.so
```

Update the `all` target (line 40) to include `$(EXE_MONOID_BATCH) $(LIB_MONOID_BATCH)`.

Add build rules (after the kgram rules, line 73):

```makefile
$(EXE_MONOID_BATCH): $(SRC_MONOID_BATCH) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_MONOID_BATCH): $(SRC_MONOID_BATCH) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<
```

Add test target (after test-kgram, line 100):

```makefile
test-monoid-batch: $(EXE_MONOID_BATCH)
	./$(EXE_MONOID_BATCH)
```

Update `.PHONY` to include `test-monoid-batch`.

- [ ] **Step 2: Create cuda/monoid_batch.cu**

```cuda
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
//
// 1 thread per string. Each thread reads its string's raw bytes sequentially
// from the concatenated buffer, maps through char_compose in shared memory.

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


// ─── Prefix Kernel (stub — implemented in Task 5) ─────────────────────────

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
    // Placeholder — filled in Task 5
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
// States: 0 (start, accept), 1 (odd a's)
// Transitions: a: 0->1, 1->0; b: 0->0, 1->1
// Monoid: {I, T_a, T_b, T_aa, T_ab, T_ba, T_bb} but for Even-A it's smaller
// Let's build it explicitly for testing.
static void build_even_a_tables(
    uint8_t *char_compose, uint8_t *raw_char_map, uint8_t *accept,
    uint8_t *monoid_compose,
    int *M_out, int *sigma_ext_out, uint8_t *identity_out)
{
    // Even-A: 2 real states, 2 characters (a=0, b=1)
    // N=2, sigma=2. The transition matrices are 2×2:
    //   T_a = [[0,1],[1,0]] (swap states)
    //   T_b = [[1,0],[0,1]] = I (identity)
    // Monoid elements: {I, T_a} — size 2
    //   I @ I = I, I @ T_a = T_a, T_a @ I = T_a, T_a @ T_a = I
    // compose_table (convention: compose[newer, older]):
    //   compose[0,0]=0  compose[0,1]=1  (I∘I=I, I∘T_a=T_a)
    //   compose[1,0]=1  compose[1,1]=0  (T_a∘I=T_a, T_a∘T_a=I)
    // char_to_monoid: a->1 (T_a), b->0 (I)
    // accept_table: element 0 (I) -> accept (even a's), element 1 (T_a) -> reject
    // DFA char indices: a=0, b=1

    int M = 2;
    int sigma = 2;
    int sigma_ext = sigma + 1;  // identity column at index 2

    // char_compose[curr * sigma_ext + ch_dfa_idx]
    // For ch_dfa_idx=0 (a), monoid index is 1 (T_a):
    //   char_compose[curr, 0] = compose[1, curr] = compose_table[1][curr]
    //   curr=0: compose[1,0]=1; curr=1: compose[1,1]=0
    // For ch_dfa_idx=1 (b), monoid index is 0 (I):
    //   char_compose[curr, 1] = compose[0, curr] = compose_table[0][curr]
    //   curr=0: compose[0,0]=0; curr=1: compose[0,1]=1
    // Identity column (ch_dfa_idx=2): char_compose[curr, 2] = curr

    // Row 0 (curr=0): [1, 0, 0]  (a->1, b->0, id->0)
    // Row 1 (curr=1): [0, 1, 1]  (a->0, b->1, id->1)
    char_compose[0 * sigma_ext + 0] = 1;  // curr=0, a
    char_compose[0 * sigma_ext + 1] = 0;  // curr=0, b
    char_compose[0 * sigma_ext + 2] = 0;  // curr=0, identity
    char_compose[1 * sigma_ext + 0] = 0;  // curr=1, a
    char_compose[1 * sigma_ext + 1] = 1;  // curr=1, b
    char_compose[1 * sigma_ext + 2] = 1;  // curr=1, identity

    // raw_char_map: 'a'(97)->0, 'b'(98)->1, everything else->2 (identity)
    memset(raw_char_map, sigma, 256);  // default = identity column
    raw_char_map['a'] = 0;
    raw_char_map['b'] = 1;

    accept[0] = 1;  // I = accept (even a's including 0)
    accept[1] = 0;  // T_a = reject

    // monoid_compose[i*M+j] = compose_table[i,j]
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

    uint8_t char_compose[2 * 3];
    uint8_t raw_char_map[256];
    uint8_t accept[2];
    uint8_t monoid_compose[4];
    int M, sigma_ext;
    uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    MonoidBatchEngine engine;
    engine.init(M, sigma_ext, identity,
                char_compose, raw_char_map, accept, monoid_compose,
                1 << 20, 1 << 16);

    // Test strings: "" (accept), "a" (reject), "aa" (accept), "b" (accept),
    // "aab" (reject), "aabb" (accept), "abab" (accept)
    const char *test_strings[] = {"", "a", "aa", "b", "aab", "aabb", "abab",
                                  "aaaa", "aaaaa", "bbb", "aba", "bab"};
    int expected[] = {1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0};
    int n_tests = 12;

    // Build raw_concat and offsets
    int total_chars = 0;
    for (int i = 0; i < n_tests; i++) total_chars += strlen(test_strings[i]);

    uint8_t *raw_concat = new uint8_t[total_chars > 0 ? total_chars : 1];
    int *offsets = new int[n_tests + 1];
    offsets[0] = 0;
    for (int i = 0; i < n_tests; i++) {
        int len = strlen(test_strings[i]);
        memcpy(raw_concat + offsets[i], test_strings[i], len);
        offsets[i + 1] = offsets[i] + len;
    }

    // GPU dispatch
    int *gpu_results = new int[n_tests];
    float kern_ms, total_ms;
    engine.dispatch_batch(raw_concat, offsets, gpu_results,
                          n_tests, total_chars, &kern_ms, &total_ms);

    // CPU reference
    int *cpu_results = new int[n_tests];
    cpu_monoid_batch(raw_concat, offsets, char_compose, raw_char_map, accept,
                     n_tests, M, sigma_ext, identity, cpu_results);

    int mismatches = 0;
    for (int i = 0; i < n_tests; i++) {
        if (gpu_results[i] != expected[i] || cpu_results[i] != expected[i]) {
            fprintf(stderr, "\n  string '%s': gpu=%d cpu=%d expected=%d",
                    test_strings[i], gpu_results[i], cpu_results[i], expected[i]);
            mismatches++;
        }
    }
    TEST_ASSERT(mismatches == 0, "batch correctness");
    if (mismatches == 0) printf("PASS (kern=%.3fms)\n", kern_ms);

    delete[] raw_concat;
    delete[] offsets;
    delete[] gpu_results;
    delete[] cpu_results;
    engine.destroy();
}


void test_batch_large_random() {
    printf("test_batch_large_random... ");

    uint8_t char_compose[2 * 3];
    uint8_t raw_char_map[256];
    uint8_t accept[2];
    uint8_t monoid_compose[4];
    int M, sigma_ext;
    uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    int B = 4096;
    int L = 256;
    srand(42);

    // Generate random strings
    int total_chars = B * L;
    uint8_t *raw_concat = new uint8_t[total_chars];
    int *offsets = new int[B + 1];
    offsets[0] = 0;
    for (int i = 0; i < B; i++) {
        int len = 1 + rand() % L;
        offsets[i + 1] = offsets[i] + len;
    }
    total_chars = offsets[B];
    delete[] raw_concat;
    raw_concat = new uint8_t[total_chars];
    for (int i = 0; i < total_chars; i++) {
        raw_concat[i] = (rand() % 2) ? 'a' : 'b';
    }

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
    for (int i = 0; i < B; i++) {
        if (gpu_results[i] != cpu_results[i]) mismatches++;
    }
    TEST_ASSERT(mismatches == 0, "large random batch");
    if (mismatches == 0) printf("PASS (B=%d, kern=%.3fms)\n", B, kern_ms);

    delete[] raw_concat;
    delete[] offsets;
    delete[] gpu_results;
    delete[] cpu_results;
    engine.destroy();
}


void bench_throughput() {
    printf("\n=== Monoid Batch Throughput ===\n");

    uint8_t char_compose[2 * 3];
    uint8_t raw_char_map[256];
    uint8_t accept[2];
    uint8_t monoid_compose[4];
    int M, sigma_ext;
    uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    int Bs[] = {1024, 4096, 16384, 65536, 262144};
    int Ls[] = {128, 512, 2048};

    for (int bi = 0; bi < 5; bi++) {
        for (int li = 0; li < 3; li++) {
            int B = Bs[bi];
            int L = Ls[li];
            int total_chars = B * L;

            uint8_t *raw_concat = new uint8_t[total_chars];
            int *offsets = new int[B + 1];
            srand(123);
            offsets[0] = 0;
            for (int i = 0; i < B; i++) {
                offsets[i + 1] = offsets[i] + L;
            }
            for (int i = 0; i < total_chars; i++) {
                raw_concat[i] = (rand() % 2) ? 'a' : 'b';
            }

            MonoidBatchEngine eng;
            eng.init(M, sigma_ext, identity,
                     char_compose, raw_char_map, accept, monoid_compose,
                     total_chars + 1, B + 1);

            int *results = new int[B];
            float kern_ms, total_ms;

            // Warmup
            for (int w = 0; w < 3; w++)
                eng.dispatch_batch(raw_concat, offsets, results, B, total_chars,
                                   &kern_ms, &total_ms);

            // Benchmark: average of 20 runs
            float sum_kern = 0;
            int runs = 20;
            for (int r = 0; r < runs; r++) {
                eng.dispatch_batch(raw_concat, offsets, results, B, total_chars,
                                   &kern_ms, &total_ms);
                sum_kern += kern_ms;
            }
            float avg_kern = sum_kern / runs;
            double gcs = (double)total_chars / (avg_kern * 1e6);

            printf("  B=%6d  L=%5d  kern=%.3fms  %.1f Gc/s\n",
                   B, L, avg_kern, gcs);

            delete[] raw_concat;
            delete[] offsets;
            delete[] results;
            eng.destroy();
        }
    }
}


int main() {
    printf("monoid_batch standalone tests\n");
    printf("=============================\n\n");

    test_batch_correctness();
    test_batch_large_random();

    printf("\n%d/%d tests passed\n", tests_passed, tests_total);

    bench_throughput();

    return (tests_passed == tests_total) ? 0 : 1;
}

#endif  // BUILD_LIB
```

- [ ] **Step 3: Build and run standalone tests**

Run: `make build/monoid_batch && ./build/monoid_batch`
Expected: 2/2 tests passed, plus throughput numbers

- [ ] **Step 4: Build the shared library**

Run: `make build/libmonoid_batch.so`
Expected: Builds without errors

- [ ] **Step 5: Commit**

```bash
git add cuda/monoid_batch.cu Makefile
git commit -m "feat: monoid batch kernel — 1 thread per string, compose-table lookups"
```

---

### Task 3: Python GPU Bridge

**Files:**
- Create: `src/gpu_bridge_monoid_batch.py`

This task creates the Python ctypes bridge following the same pattern as `src/gpu_bridge_monoid.py`, but sends raw bytes instead of pre-mapped monoid indices.

- [ ] **Step 1: Create src/gpu_bridge_monoid_batch.py**

```python
"""
Python bridge to the monoid batch GPU engine via ctypes.

The monoid batch engine processes strings by sending raw bytes to the GPU,
where a fused compose-table lookup replaces O(N³) matrix multiplication
with O(1) shared-memory reads per character.

Usage:
    from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator
    sim = MonoidBatchGPUSimulator()
    engine = sim.create_engine(md, dm)
    results = engine.simulate_batch(["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    engine.destroy()
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.monoid import MonoidData, precompute_batch_tables
from src.simulation import DFAMatrices


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libmonoid_batch.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libmonoid_batch.so not found at {base}. Run 'make' first."
    )


class MonoidBatchEngine:
    """Wraps a persistent GPU engine context for monoid batch dispatch."""

    def __init__(self, lib, md: MonoidData, dm: DFAMatrices,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.md = md

        tables = precompute_batch_tables(md, dm)
        self._M = tables['M']
        self._sigma_ext = tables['sigma_ext']
        self._identity = tables['identity_idx']

        char_compose = np.ascontiguousarray(tables['char_compose'])
        raw_char_map = np.ascontiguousarray(tables['raw_char_map'])
        accept = np.ascontiguousarray(tables['accept'])
        monoid_compose = np.ascontiguousarray(tables['monoid_compose'])

        rc = self.lib.monoid_batch_engine_init(
            self._M,
            self._sigma_ext,
            self._identity,
            char_compose.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            raw_char_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            monoid_compose.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            max_total_chars,
            max_batch,
        )
        if rc != 0:
            raise RuntimeError(f"monoid_batch_engine_init failed with code {rc}")

    def destroy(self):
        self.lib.monoid_batch_engine_destroy()

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

        B = len(strings)
        L_max = max((len(s) for s in strings), default=0)
        if L_max == 0:
            from src.monoid import simulate_monoid
            is_accept = bool(self.md.accept_table[self.md.identity_idx])
            return [is_accept] * B

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        use_prefix = (B <= 128 and L_max > 100_000)
        dispatch_fn = (self.lib.monoid_batch_engine_dispatch_prefix
                       if use_prefix
                       else self.lib.monoid_batch_engine_dispatch)

        rc = dispatch_fn(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"monoid_batch dispatch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        if not strings:
            return [], 0.0, 0.0

        B = len(strings)
        L_max = max((len(s) for s in strings), default=0)
        if L_max == 0:
            is_accept = bool(self.md.accept_table[self.md.identity_idx])
            return [is_accept] * B, 0.0, 0.0

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        use_prefix = (B <= 128 and L_max > 100_000)
        dispatch_fn = (self.lib.monoid_batch_engine_dispatch_prefix
                       if use_prefix
                       else self.lib.monoid_batch_engine_dispatch)

        rc = dispatch_fn(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"monoid_batch dispatch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class MonoidBatchGPUSimulator:
    """Factory for MonoidBatchEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.monoid_batch_engine_device_check.restype = ctypes.c_int
        self.lib.monoid_batch_engine_device_check.argtypes = []

        self.lib.monoid_batch_engine_init.restype = ctypes.c_int
        self.lib.monoid_batch_engine_init.argtypes = [
            ctypes.c_int,                     # M
            ctypes.c_int,                     # sigma_ext
            ctypes.c_int,                     # identity
            ctypes.POINTER(ctypes.c_uint8),   # char_compose
            ctypes.POINTER(ctypes.c_uint8),   # raw_char_map
            ctypes.POINTER(ctypes.c_uint8),   # accept
            ctypes.POINTER(ctypes.c_uint8),   # monoid_compose
            ctypes.c_int,                     # max_total_chars
            ctypes.c_int,                     # max_batch
        ]

        self.lib.monoid_batch_engine_destroy.restype = None
        self.lib.monoid_batch_engine_destroy.argtypes = []

        self.lib.monoid_batch_engine_dispatch.restype = ctypes.c_int
        self.lib.monoid_batch_engine_dispatch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # raw_concat
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.c_int,                     # B
            ctypes.c_int,                     # total_chars
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        self.lib.monoid_batch_engine_dispatch_prefix.restype = ctypes.c_int
        self.lib.monoid_batch_engine_dispatch_prefix.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # raw_concat
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.c_int,                     # B
            ctypes.c_int,                     # total_chars
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        rc = self.lib.monoid_batch_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, md: MonoidData, dm: DFAMatrices,
                      max_total_chars: int = 1 << 22,
                      max_batch: int = 1 << 18) -> MonoidBatchEngine:
        return MonoidBatchEngine(self.lib, md, dm,
                                max_total_chars, max_batch)
```

- [ ] **Step 2: Smoke test the bridge**

Run: `python -c "from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator; sim = MonoidBatchGPUSimulator(); print('OK')"`
Expected: Prints "OK" (or fails with library not found if not built yet — verify library exists first)

- [ ] **Step 3: Commit**

```bash
git add src/gpu_bridge_monoid_batch.py
git commit -m "feat: Python bridge for monoid batch GPU engine"
```

---

### Task 4: GPU Correctness Tests (Python)

**Files:**
- Create: `tests/test_monoid_batch_gpu.py`

Cross-validates the monoid batch GPU engine against `simulate_monoid` (CPU) on multiple regex patterns.

- [ ] **Step 1: Create tests/test_monoid_batch_gpu.py**

```python
"""
GPU tests for the monoid batch engine — cross-validate against simulate_monoid.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
import numpy as np

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential
from src.generate_data import PATTERNS
from src.monoid import compute_monoid, simulate_monoid


def _make_dm(pattern_name):
    pat = PATTERNS[pattern_name]
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    return dm, dfa


def _get_alphabet(pattern_name):
    if pattern_name in ("abb", "even_a", "ab_star"):
        return "ab"
    if pattern_name == "binary_div3":
        return "01"
    if pattern_name == "hex_number":
        return "0123456789abcdefx"
    if pattern_name == "identifier":
        return "abcdefghijklmnopqrstuvwxyz0123456789"
    return "abcdefgh"


def _monoid_batch_gpu_available():
    try:
        from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator
        MonoidBatchGPUSimulator()
        return True
    except Exception:
        return False


skip_no_gpu = pytest.mark.skipif(
    not _monoid_batch_gpu_available(),
    reason="monoid batch GPU engine not available"
)


@pytest.fixture(scope="module")
def simulator():
    from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator
    return MonoidBatchGPUSimulator()


GPU_PATTERNS = ["abb", "binary_div3", "even_a", "ab_star",
                "hex_number", "identifier"]


@skip_no_gpu
class TestMonoidBatchGPU:

    def test_batch_cross_validate(self, simulator):
        for pattern_name in GPU_PATTERNS:
            dm, dfa = _make_dm(pattern_name)
            md = compute_monoid(dm)
            assert md is not None
            assert md.size <= 255, f"M={md.size} too large for uint8"

            engine = simulator.create_engine(md, dm)
            alphabet = _get_alphabet(pattern_name)
            rng = random.Random(hash(pattern_name) & 0xFFFFFFFF)

            strings = [""]
            for _ in range(499):
                length = rng.randint(0, 200)
                strings.append("".join(rng.choice(alphabet) for _ in range(length)))

            gpu_results = engine.simulate_batch(strings)
            cpu_results = [simulate_monoid(md, dm, s) for s in strings]

            mismatches = [
                (s, cpu, gpu)
                for s, cpu, gpu in zip(strings, cpu_results, gpu_results)
                if cpu != gpu
            ]
            assert not mismatches, (
                f"GPU vs CPU mismatches for '{pattern_name}' "
                f"({len(mismatches)}/500):\n" +
                "\n".join(f"  '{s[:40]}' exp={e} got={g}"
                          for s, e, g in mismatches[:10])
            )
            engine.destroy()

    def test_empty_batch(self, simulator):
        dm, _ = _make_dm("abb")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm)
        assert engine.simulate_batch([]) == []
        engine.destroy()

    def test_all_empty_strings(self, simulator):
        dm, dfa = _make_dm("even_a")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm)
        results = engine.simulate_batch(["", "", ""])
        expected = simulate_sequential(dfa, "")
        assert all(r == expected for r in results)
        engine.destroy()

    def test_timed(self, simulator):
        dm, _ = _make_dm("abb")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm)
        rng = random.Random(99)
        strings = ["".join(rng.choice("ab") for _ in range(100)) for _ in range(1000)]
        results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
        assert len(results) == 1000
        assert kern_ms > 0
        assert total_ms >= kern_ms
        engine.destroy()

    def test_variable_lengths(self, simulator):
        dm, dfa = _make_dm("abb")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm)
        rng = random.Random(77)

        strings = []
        for i in range(200):
            length = rng.randint(0, 500)
            strings.append("".join(rng.choice("ab") for _ in range(length)))

        gpu_results = engine.simulate_batch(strings)
        cpu_results = [simulate_sequential(dfa, s) for s in strings]
        assert gpu_results == cpu_results
        engine.destroy()
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_monoid_batch_gpu.py -v`
Expected: All tests PASS (5 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_monoid_batch_gpu.py
git commit -m "test: monoid batch GPU cross-validation against CPU"
```

---

### Task 5: Parallel Prefix Kernel

**Files:**
- Modify: `cuda/monoid_batch.cu` (replace prefix kernel stub)
- Modify: `tests/test_monoid_batch_gpu.py` (add long-string tests)

Implements the parallel prefix tree-reduce kernel for few very long strings (B ≤ 128, L > 100K). One block per string, 256 threads per block. Phase 1: thread-local sequential reduce. Phase 2: tree reduce in shared memory using M×M compose table. Phase 3: accept check.

- [ ] **Step 1: Replace the monoid_prefix_kernel stub in cuda/monoid_batch.cu**

Replace the placeholder `monoid_prefix_kernel` with:

```cuda
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

    // Shared memory layout:
    //   compose_sh:  M * sigma_ext  (char compose table)
    //   charmap_sh:  256            (raw byte -> char index)
    //   mcompose_sh: M * M          (monoid compose for tree reduce)
    //   accept_sh:   M              (accept table)
    //   reduce_sh:   MP_BLOCK_SIZE  (per-thread monoid elements)
    extern __shared__ uint8_t smem[];
    uint8_t *compose_sh  = smem;
    uint8_t *charmap_sh  = compose_sh + M * sigma_ext;
    uint8_t *mcompose_sh = charmap_sh + 256;
    uint8_t *accept_sh   = mcompose_sh + M * M;
    uint8_t *reduce_sh   = accept_sh + M;

    // Cooperative load of tables
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

    // Phase 2: tree reduce (compose newer ∘ older)
    // Thread i holds the monoid for chunk i. Thread i+stride has later chars.
    for (int stride = MP_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            uint8_t older = reduce_sh[threadIdx.x];
            uint8_t newer = reduce_sh[threadIdx.x + stride];
            reduce_sh[threadIdx.x] = mcompose_sh[newer * M + older];
        }
        __syncthreads();
    }

    // Phase 3: thread 0 writes result
    if (threadIdx.x == 0) {
        results[sid] = (int)accept_sh[reduce_sh[0]];
    }
}
```

- [ ] **Step 2: Add a standalone prefix test to the #else block in cuda/monoid_batch.cu**

Add before `bench_throughput()`:

```cuda
void test_prefix_correctness() {
    printf("test_prefix_correctness... ");

    uint8_t char_compose[2 * 3];
    uint8_t raw_char_map[256];
    uint8_t accept[2];
    uint8_t monoid_compose[4];
    int M, sigma_ext;
    uint8_t identity;
    build_even_a_tables(char_compose, raw_char_map, accept, monoid_compose,
                        &M, &sigma_ext, &identity);

    // Single long string: 100000 characters
    int B = 1;
    int L = 100000;
    int total_chars = L;

    uint8_t *raw_concat = new uint8_t[total_chars];
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
                total_chars + 1, B + 1);

    int gpu_batch_result, gpu_prefix_result;
    float kern_ms, total_ms;

    engine.dispatch_batch(raw_concat, offsets, &gpu_batch_result,
                          B, total_chars, &kern_ms, &total_ms);
    engine.dispatch_prefix(raw_concat, offsets, &gpu_prefix_result,
                           B, total_chars, &kern_ms, &total_ms);

    // CPU reference
    int cpu_result;
    cpu_monoid_batch(raw_concat, offsets, char_compose, raw_char_map, accept,
                     B, M, sigma_ext, identity, &cpu_result);

    bool ok = (gpu_batch_result == cpu_result) && (gpu_prefix_result == cpu_result);
    TEST_ASSERT(ok, "prefix correctness");
    if (ok) {
        printf("PASS (L=%d, count_a=%d, result=%d, kern=%.3fms)\n",
               L, count_a, cpu_result, kern_ms);
    } else {
        printf("FAIL (cpu=%d, batch=%d, prefix=%d)\n",
               cpu_result, gpu_batch_result, gpu_prefix_result);
    }

    delete[] raw_concat;
    engine.destroy();
}
```

Update `main()` to call `test_prefix_correctness()` and update the test count.

- [ ] **Step 3: Build and run**

Run: `make build/monoid_batch && ./build/monoid_batch`
Expected: 3/3 tests passed

- [ ] **Step 4: Add Python tests for long strings**

Add to `tests/test_monoid_batch_gpu.py` in `TestMonoidBatchGPU`:

```python
    def test_long_string_prefix(self, simulator):
        """Test parallel prefix dispatch for a single long string."""
        dm, dfa = _make_dm("even_a")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm, max_total_chars=2_000_000)
        rng = random.Random(42)

        for L in [200_000, 1_000_000]:
            s = "".join(rng.choice("ab") for _ in range(L))
            gpu_result = engine.simulate_batch([s])
            cpu_result = [simulate_monoid(md, dm, s)]
            assert gpu_result == cpu_result, (
                f"L={L}: gpu={gpu_result}, cpu={cpu_result}"
            )
        engine.destroy()

    def test_few_long_strings_prefix(self, simulator):
        """B=4 strings of L=500K each — should use prefix dispatch."""
        dm, dfa = _make_dm("abb")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm, max_total_chars=4_000_000)
        rng = random.Random(88)

        strings = ["".join(rng.choice("ab") for _ in range(500_000))
                    for _ in range(4)]
        gpu_results = engine.simulate_batch(strings)
        cpu_results = [simulate_monoid(md, dm, s) for s in strings]
        assert gpu_results == cpu_results
        engine.destroy()
```

- [ ] **Step 5: Run Python tests**

Run: `python -m pytest tests/test_monoid_batch_gpu.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add cuda/monoid_batch.cu tests/test_monoid_batch_gpu.py
git commit -m "feat: parallel prefix monoid reduce for long strings"
```

---

### Task 6: OptimizedEngine Integration

**Files:**
- Modify: `src/optimized_engine.py`
- Modify: `tests/test_optimized_engine.py`

Adds `"monoid_batch+gpu"` config option and updates auto-selection to prefer the monoid batch kernel for small-monoid DFAs (M ≤ 255).

- [ ] **Step 1: Add setup and dispatch methods to OptimizedEngine**

Add to `src/optimized_engine.py`:

After the existing `_setup_kgram_gpu` method (around line 219):

```python
    def _setup_monoid_batch_gpu(self):
        if self._md is None:
            raise RuntimeError("Monoid computation failed; cannot use monoid batch GPU")
        if self._md.size > 255:
            raise RuntimeError(f"Monoid size {self._md.size} > 255; use monoid+gpu instead")
        from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator
        sim = MonoidBatchGPUSimulator()
        self._monoid_batch_gpu = sim.create_engine(self._md, self._dm)
        self._scan_backend = 'monoid_batch+gpu'
        self._selection_reason = (
            f'GPU monoid batch (M={self._md.size}, '
            f'sigma={len(self._dm.alphabet)})'
        )
```

Add `self._monoid_batch_gpu = None` to `__init__` (around line 69, after `self._kgram_gpu = None`).

Add the new config case in `__init__` (around line 94, before the `else` clause):

```python
        elif config == "monoid_batch+gpu":
            self._force_monoid()
            self._setup_monoid_batch_gpu()
```

Update the error message in the `else` clause to include `'monoid_batch+gpu'`.

- [ ] **Step 2: Update dispatch methods**

In `_match_one` (around line 253), add before the `self._gpu_engine` check:

```python
        if self._monoid_batch_gpu is not None:
            return self._monoid_batch_gpu.simulate_batch([s])[0]
```

In `match_batch` (around line 266), add before the `self._gpu_engine` check:

```python
        if self._monoid_batch_gpu is not None:
            return self._monoid_batch_gpu.simulate_batch(strings)
```

In `match_batch_timed` (around line 276), add before the `self._gpu_engine` check:

```python
        if self._monoid_batch_gpu is not None:
            results, kern_ms, total_ms = self._monoid_batch_gpu.simulate_batch_timed(strings)
            return results, {'kernel_ms': kern_ms, 'total_ms': total_ms}
```

- [ ] **Step 3: Add tests**

Add to `tests/test_optimized_engine.py`:

```python
def _monoid_batch_gpu_available():
    try:
        from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator
        MonoidBatchGPUSimulator()
        return True
    except Exception:
        return False

skip_no_monoid_batch_gpu = pytest.mark.skipif(
    not _monoid_batch_gpu_available(),
    reason="monoid batch GPU not available"
)


@skip_no_monoid_batch_gpu
class TestMonoidBatchGPUConfig:

    def test_monoid_batch_gpu_config(self):
        engine = OptimizedEngine("(a|b)*abb", config="monoid_batch+gpu")
        info = engine.config_info
        assert info["scan_backend"] == "monoid_batch+gpu"

    def test_monoid_batch_gpu_correctness(self):
        engine = OptimizedEngine("(a|b)*abb", config="monoid_batch+gpu")
        baseline = OptimizedEngine("(a|b)*abb", config="baseline")
        rng = random.Random(42)
        strings = ["".join(rng.choice("ab") for _ in range(rng.randint(0, 200)))
                    for _ in range(100)]
        gpu_results = engine.match_batch(strings)
        cpu_results = baseline.match_batch(strings)
        assert gpu_results == cpu_results

    def test_monoid_batch_gpu_timed(self):
        engine = OptimizedEngine("(a|b)*abb", config="monoid_batch+gpu")
        rng = random.Random(99)
        strings = ["".join(rng.choice("ab") for _ in range(100)) for _ in range(500)]
        results, timing = engine.match_batch_timed(strings)
        assert len(results) == 500
        assert 'kernel_ms' in timing
        assert timing['kernel_ms'] > 0
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_optimized_engine.py -v`
Expected: All existing + 3 new tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimized_engine.py tests/test_optimized_engine.py
git commit -m "feat: monoid_batch+gpu config in OptimizedEngine"
```

---

### Task 7: Benchmark

**Files:**
- Modify: `cuda/monoid_batch.cu` (already has bench_throughput in standalone)

This task runs the full benchmark to measure throughput and validate performance targets from the spec.

- [ ] **Step 1: Build and run the standalone benchmark**

Run: `make build/monoid_batch && ./build/monoid_batch`
Expected: Throughput table printed. Record results for spec comparison.

- [ ] **Step 2: Run Python end-to-end benchmark**

Run an inline benchmark from Python:

```bash
python -c "
from src.optimized_engine import OptimizedEngine
import random, time

engine = OptimizedEngine('(b*ab*ab*)*b*', config='monoid_batch+gpu')
print(engine.config_info)

rng = random.Random(42)
for B, L in [(1024, 128), (4096, 512), (65536, 128), (262144, 128), (262144, 2048)]:
    strings = [''.join(rng.choice('ab') for _ in range(L)) for _ in range(B)]
    results, timing = engine.match_batch_timed(strings)
    total_chars = B * L
    kern_ms = timing['kernel_ms']
    gcs = total_chars / (kern_ms * 1e6)
    print(f'B={B:>7d}  L={L:>5d}  kern={kern_ms:.3f}ms  {gcs:.1f} Gc/s')
"
```

Expected: Throughput numbers significantly above TC V3's 116 Gc/s.

- [ ] **Step 3: Commit benchmark results**

If standalone tests pass and benchmark numbers are collected, no code changes needed. The benchmark output documents performance.

```bash
git add -u
git commit -m "bench: monoid batch pipeline throughput measurements"
```
