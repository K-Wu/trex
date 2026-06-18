# FP16 Tensor Core DFA Evolution Engine

## Problem

The existing tensor core kernel (`batched_evolution.cu`) achieves only 50 Gc/s on
H200 — 13× slower than the scalar monoid_batch kernel (666 Gc/s). Profiling reveals
TC utilization of 2-3%: the kernel spends 97% of its time on **INT32→INT8 data
format conversion**, not on tensor core compute.

The root cause: INT8 MMA produces INT32 accumulators. To chain MMAs (feed result of
position t into position t+1), the kernel must store INT32 to shared memory, threshold
each element (`v > 0 ? 1 : 0`), convert to INT8, store back, and reload as an INT8
fragment. This cycle takes ~180 cycles per position; the MMA itself takes ~5 cycles.

## Key Insight

**Use FP16 MMA with FP16 accumulator.** For DFA transitions (permutation matrices
acting on one-hot state vectors), the MMA output is already exactly {0.0, 1.0} —
no threshold or conversion is needed. The FP16 result feeds directly into the next MMA.

Mathematical proof: a DFA transition matrix M is a permutation matrix (entries in
{0, 1}, exactly one 1 per row and column). A DFA state vector s is one-hot (entries
in {0, 1}, exactly one 1). The product (M × s)[i] = M[i][s_active], which is 0 or 1.
The output is again one-hot. This invariant holds for all N, all L, all inputs. FP16
represents 0.0 and 1.0 exactly, and integer sums up to 2048 are exact in FP16's
11-bit mantissa — no rounding occurs.

Secondary advantage: FP16's MMA instruction shape is `m16n8k16`, where K=16 matches
our DFA state count N=16 naturally. INT8's `m16n8k32` forces zero-padding K from 16
to 32, wasting 50% of TC throughput.

## Architecture

### Kernel: FP16 Batched State-Vector Evolution

File: `cuda/fp16_evolution.cu`

Same algorithm as `batched_evolution.cu`, but with FP16 data path throughout:

```
              INT8 (current)                    FP16 (proposed)
              ──────────────                    ───────────────
Input:        int8_t {0,1}                      half {0.0, 1.0}
MMA:          int8 × int8 → int32               half × half → half
Threshold:    (v > 0 ? 1 : 0) — REQUIRED        NOT NEEDED (already {0,1})
Convert:      int32 → int8 — REQUIRED            NOT NEEDED (stays half)
Store:        1024B int32 per tile               512B half per tile
```

### Inner Loop (N=16, sigma=2)

```cuda
// FP16 fragments
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_T0, frag_T1;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_S;
wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc0, frag_acc1;

// T matrices loaded once (constant across all positions)
wmma::load_matrix_sync(frag_T0, T0_sh, 16);
wmma::load_matrix_sync(frag_T1, T1_sh, 16);

for (int t = 0; t < L; t++) {
    wmma::load_matrix_sync(frag_S, S_sh, 16);

    // Two MMAs: T0 × S and T1 × S (FP16 → FP16)
    wmma::fill_fragment(frag_acc0, __float2half(0.0f));
    wmma::mma_sync(frag_acc0, frag_T0, frag_S, frag_acc0);
    wmma::fill_fragment(frag_acc1, __float2half(0.0f));
    wmma::mma_sync(frag_acc1, frag_T1, frag_S, frag_acc1);

    // Store both accumulators to smem (row-major, half)
    wmma::store_matrix_sync(acc0_sh, frag_acc0, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(acc1_sh, frag_acc1, 16, wmma::mem_row_major);
    __syncwarp();

    // Per-column select — NO threshold, NO conversion
    for (int e = lane; e < 256; e += 32) {
        int col = e / 16;
        int row = e % 16;
        uint8_t ch = input[t * B_padded + warp_col_start + col];
        if (ch < 2) {
            S_sh[col * 16 + row] = (ch == 0)
                ? acc0_sh[row * 16 + col]
                : acc1_sh[row * 16 + col];
        }
    }
    __syncwarp();
}
```

### Kernel Variants

Two variants, in order of implementation:

**FP16-V1 (smem path):** Store acc to shared memory as half, select per-column, write
to S_sh. Simple, correct, isolates the FP16 improvement.

**FP16-V2 (register path):** Access `frag_acc.x[i]` directly in registers (like the
existing INT8 V2). Requires probing the FP16 accumulator fragment layout — same
approach as `probe_frag_layout.cu`. Eliminates the acc store/load round-trip.

The V2 register-level select for INT8:
```cuda
// INT8 V2: select + threshold + cast = 3 operations per element
int32_t v = (ch == 0) ? frag_acc0.x[i] : frag_acc1.x[i];
S_sh[pos] = (int8_t)(v > 0 ? 1 : 0);
```

Becomes for FP16 V2:
```cuda
// FP16 V2: select only = 1 operation per element
S_sh[pos] = (ch == 0) ? frag_acc0.x[i] : frag_acc1.x[i];
```

### Threading Model

Identical to `batched_evolution.cu`:
- 1 warp (32 threads) per 16-string tile
- 4 warps per block → 64 strings per block
- Grid: `ceil(B / 64)` blocks

### Shared Memory

Per-block layout for N=16, sigma=2:

```
FP16-V1:
  T0_sh[16×16]      = 512B  (half, transition matrix char 0)
  T1_sh[16×16]      = 512B  (half, transition matrix char 1)
  S_sh[4×16×16]     = 2048B (half, state tiles for 4 warps)
  acc0_sh[4×16×16]  = 2048B (half, accumulator buffer 0)
  acc1_sh[4×16×16]  = 2048B (half, accumulator buffer 1)
  Total:              7168B

FP16-V2 (register acc):
  T0_sh[16×16]      = 512B
  T1_sh[16×16]      = 512B
  S_sh[4×16×16]     = 2048B
  Total:              3072B
```

Compare INT8 V1: 9728B, INT8 V2: 1536B.

FP16 uses 2× the bytes per element (half vs int8), but eliminates the INT32 acc
buffers (4× per element). Net: FP16-V2 uses 2× the smem of INT8 V2, but the
dominant factor is occupancy — both allow high warp counts.

## Multi-N Support (N=32, N=64)

For N > 16, tile the matmul into 16×16 blocks.

### Tiling Scheme

T[N×N] × S[N×16] tiled as:

```
For each output row group r = 0..(N/16-1):
    acc = 0
    for k = 0..(N/16-1):
        acc = MMA(T[r][k], S[k], acc)    // accumulate K-tiles
```

MMA count per transition: (N/16)² instructions
MMA count per position (sigma=2): 2 × (N/16)²

| N  | Tiles per matmul | wmma calls per position (σ=2) | T matrix smem (σ=2) |
|----|-----------------|-------------------------------|---------------------|
| 16 | 1               | 2                             | 1024B               |
| 32 | 4               | 8                             | 4096B               |
| 48 | 9               | 18                            | 9216B               |
| 64 | 16              | 32                            | 16384B              |

All fit comfortably within the 228KB smem budget per SM.

### Template on N

The kernel is templated on N so tile loops unroll at compile time:

```cuda
template <int N>
__global__ void fp16_evolution_kernel(...) {
    constexpr int N_TILES = N / 16;
    // ...
    for (int t = 0; t < L; t++) {
        for (int sigma_idx = 0; sigma_idx < sigma; sigma_idx++) {
            for (int r = 0; r < N_TILES; r++) {
                wmma::fill_fragment(acc, __float2half(0.0f));
                for (int k = 0; k < N_TILES; k++) {
                    wmma::mma_sync(acc, frag_T[sigma_idx][r][k], frag_S[k], acc);
                }
                // store acc row group r
            }
        }
        // select per column, write to S_sh
    }
}
```

Supported N values: 16, 32, 48, 64 (compile-time instantiated). DFAs with N not a
multiple of 16 are padded to the next multiple (e.g., N=20 → N_padded=32); extra
rows/cols in T are identity, extra elements in S are zero.

### State Tiling

State S[N×16] has N/16 row groups. Each row group is a separate `matrix_b` fragment.
The select loop after MMA operates on all row groups independently.

## Sigma-Branching Strategy

### Sigma=2 (Binary Alphabet)

Primary target. Compute T[0]×S and T[1]×S, select per column. Identical to the
existing binary kernel architecture.

### Sigma=3-4 (Small Alphabets)

Same approach, linear in sigma. For sigma=4:

```cuda
// 4 MMAs per position per tile (N=16)
for (int c = 0; c < 4; c++) {
    wmma::fill_fragment(frag_acc[c], __float2half(0.0f));
    wmma::mma_sync(frag_acc[c], frag_T[c], frag_S, frag_acc[c]);
}
// 4-way select per column based on actual character
```

### Sigma > 4

Deferred to follow-up. Use ballot-based grouping from the existing general kernel
in `batched_evolution.cu`. The FP16 advantage (no conversion) applies equally to
the general kernel.

### Initial Scope

Sigma=2 only for the first implementation. This covers the majority of practical
regex patterns and allows clean benchmarking against the existing binary INT8 kernel.

## Input Layout

Position-contiguous layout (same as `batched_evolution.cu`):

```
input[t * B_padded + string_id] = character index at position t for string string_id
```

The Python bridge transposes from string-contiguous (natural Python layout) to
position-contiguous (optimal for per-column character lookup in the inner loop).

Strings shorter than L_max are padded with an identity character (>= sigma) that
leaves the state unchanged (the select skips identity characters).

## C API

```c
extern "C" {
    int fp16_engine_device_check(void);

    int fp16_engine_init(
        const float *T_matrices,    // [sigma][N][N] row-major, float (converted to half on GPU)
        const float *accept_mask,   // [N] float
        const float *start_vec,     // [N] float (one-hot)
        int N,                      // number of DFA states (16, 32, 48, or 64)
        int sigma,                  // alphabet size
        int max_B,                  // max batch size
        int max_L                   // max string length
    );

    int fp16_engine_dispatch(
        const uint8_t *raw_concat,  // [total_chars] concatenated raw string bytes
        const int     *offsets,     // [B+1] string boundary offsets
        int           *results,    // [B] output: 1=accept, 0=reject
        int B,                     // number of strings
        int total_chars,           // sum of string lengths
        float *kernel_ms,          // output: kernel time in ms
        float *total_ms            // output: total dispatch time in ms
    );

    void fp16_engine_destroy(void);
}
```

The API accepts float inputs (natural from Python/numpy) and converts to half
internally during init. This avoids requiring FP16 support in the Python bridge.

The dispatch accepts the same raw_concat + offsets format as `prefix_compose` and
`monoid_batch`, converting to position-contiguous layout internally.

## Python Bridge

File: `src/gpu_bridge_fp16_evolution.py`

Follows the same pattern as `gpu_bridge_prefix_compose.py`:

```python
class FP16EvolutionEngine:
    def __init__(self, lib, dm: DFAMatrices, max_total_chars, max_batch):
        # Build T_matrices[sigma][N][N] from dm.matrices
        # Build accept_mask[N], start_vec[N]
        # Call fp16_engine_init(...)

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        # Prepare raw_concat, offsets
        # Call fp16_engine_dispatch(...)
        # Return list of bools

    def simulate_batch_timed(self, strings) -> tuple:
        # Same but returns (results, kernel_ms, total_ms)

    def destroy(self):
        # Call fp16_engine_destroy()

class FP16EvolutionGPUSimulator:
    def create_engine(self, dm, max_total_chars, max_batch) -> FP16EvolutionEngine:
        ...
```

## OptimizedEngine Integration

### New config: `"fp16_tc+gpu"`

```python
elif config == "fp16_tc+gpu":
    self._force_baseline()
    self._setup_fp16_tc_gpu()
```

### Updated auto-selection

```python
def _auto_select(self):
    self._build_dfa()
    n_states = self._dfa.n_states

    if n_states > self._dfa_state_cap:
        # NFA fallback (unchanged)
        ...
        return

    # Tier 1: Monoid for small DFAs with small monoid
    md = compute_monoid(self._dm, max_size=self._monoid_cap)
    if md is not None and md.size <= 255:
        try:
            self._setup_monoid_batch_gpu()
            return
        except Exception:
            pass

    # Tier 2: FP16 TC for N ≤ 64 (replaces prefix+gpu in auto-selection)
    if n_states <= 64:
        try:
            self._setup_fp16_tc_gpu()
            return
        except Exception:
            pass

    # Tier 3: Prefix compose fallback
    try:
        self._setup_prefix_gpu()
        return
    except Exception:
        pass

    # Tier 4: CPU fallback
    ...
```

The key change: FP16 TC takes the Tier 2 slot for N ≤ 64, replacing prefix_compose
in auto-selection. Prefix compose remains available as an explicit config.

For N=16 with small monoid (M ≤ 255), monoid_batch remains Tier 1 — its O(1) table
lookup is hard to beat. FP16 TC serves the cases where monoid is infeasible
(M > 255 or N > 16).

## Testing

### Built-in CUDA tests (in fp16_evolution.cu)

1. `test_basic_correctness`: B=8, L=16, sigma=2, regex `(a|b)*abb`. Verify against
   known accept/reject for specific strings.

2. `test_large_random`: B=4096, L=256, sigma=2. Generate random strings, cross-validate
   against sequential DFA simulation (built into the test, same as batched_evolution).

3. `test_fp16_invariant`: B=64, L=100000. After processing, read back all state vectors
   and verify every entry is exactly 0.0 or 1.0. Validates the no-threshold invariant
   over 100K positions.

4. `test_n32`: Construct a 32-state DFA, B=1024, L=128. Cross-validate against
   sequential simulation. Exercises the multi-tile path.

### Python tests (tests/test_fp16_evolution.py)

1. Cross-validate against monoid_batch and prefix_compose for multiple regexes
2. Edge cases: empty batch, B=1, empty strings, single-character strings
3. Large-scale: B=65536, L=512, verify match count consistency across backends

### FP16 precision validation

No runtime test needed — the invariant is mathematical (proven above). But
test_fp16_invariant provides empirical confirmation.

## Benchmark Plan

### Built-in benchmark (in fp16_evolution.cu)

Configs matching existing benchmarks for direct comparison:

```
B × L: {1024, 4096, 16384, 65536, 262144} × {128, 512, 2048}
N: 16 (primary), 32, 64 (secondary)
sigma: 2
```

Report per config: kernel_ms, Gc/s, estimated TC utilization.

### TC utilization estimate

```
mma_count = ceil(B/16) × L × sigma × (N/16)²
mma_peak_per_sec = n_sms × TC_units_per_sm × clock_hz / cycles_per_mma
                 = 132 × 4 × 1.785e9 / 4
                 = 235.6 billion MMA/s
tc_util = mma_count / (kernel_seconds × mma_peak_per_sec)
```

### Success criteria

| N  | Target Gc/s | Comparison         | TC util target |
|----|-------------|--------------------|---------------|
| 16 | 400+        | ≥0.6× monoid (666) | ≥15%          |
| 32 | 150+        | ≥1.5× prefix (~100)| ≥10%          |
| 64 | 40+         | ≥1.5× prefix (~30) | ≥5%           |

Stretch goals (with V2 register path):
| 16 | 600+        | ~1× monoid         | ≥25%          |
| 32 | 200+        | ≥2× prefix          | ≥15%          |

## Performance Analysis

### Why FP16 should be 5-10× faster than INT8

The existing INT8 V2 inner loop per position per tile:
- `wmma::load_matrix_sync` (S from smem): ~4 cycles
- 2× `wmma::mma_sync`: ~8 cycles (2 MMA, pipelined)
- 4× `__shfl_sync` (broadcast characters): ~4 cycles
- 8× REGSEL (select + threshold + INT32→INT8 + smem write): ~40-60 cycles
  - Each: select (1 op) + threshold (1 compare + 1 select) + cast (1 op) + store (1 op)
- `__syncwarp`: ~4 cycles
- **Total: ~60-80 cycles** (measured: ~180 cycles including pipeline stalls)

Proposed FP16-V1 inner loop per position per tile:
- `wmma::load_matrix_sync` (S from smem, half): ~4 cycles
- 2× `wmma::mma_sync` (FP16→FP16): ~8 cycles
- 2× `wmma::store_matrix_sync` (acc to smem, half): ~8 cycles
- 8× select + write (select only, no threshold/cast): ~16 cycles
- `__syncwarp`: ~4 cycles
- **Total: ~40 cycles**

Proposed FP16-V2 inner loop (register acc):
- `wmma::load_matrix_sync`: ~4 cycles
- 2× `wmma::mma_sync`: ~8 cycles
- 4× `__shfl_sync`: ~4 cycles
- 8× select + smem write (1 op + 1 store): ~16 cycles
- `__syncwarp`: ~4 cycles
- **Total: ~36 cycles**

Expected speedup over INT8 V2:
- FP16-V1: 180/40 ≈ 4.5× → 50 × 4.5 = **225 Gc/s** (conservative)
- FP16-V2: 180/36 ≈ 5× → 50 × 5 = **250 Gc/s** (conservative)

With higher occupancy (smaller smem footprint → more warps → better latency hiding):
- Optimistic: 8-10× → **400-500 Gc/s**
- Stretch: **600-800 Gc/s** with careful pipelining

### Scaling with N

For N=32 (4 MMAs per transition vs 1 for N=16):
- MMA fraction of inner loop increases (good — TC does more useful work)
- Overhead per position: ~40 + 4×8 = ~72 cycles (V1 with 4 MMA pairs)
- MMA time: 8 × 4 = 32 cycles (4 TC units)
- Overhead-limited, not TC-limited
- Estimate: 150-250 Gc/s

For N=64 (16 MMAs per transition):
- MMA becomes dominant: 32 × 4 = 128 cycles
- Overhead: ~100 cycles
- Total: ~230 cycles per position
- Estimate: 40-80 Gc/s
