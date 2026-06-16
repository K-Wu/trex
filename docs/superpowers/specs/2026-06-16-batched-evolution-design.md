# Batched State-Vector Evolution Design Spec

**Date:** 2026-06-16
**Status:** Draft
**Builds on:** `docs/REFRAMING.md`, v4 parallel engine, monoid scan engine, `OptimizedEngine`

## Goal

Replace the matrix-composition prefix scan (0.53% tensor core utilization) with batched state-vector evolution via character-grouped GEMMs. The key shape change: N×N × N×N (tiny, overhead-dominated) becomes N×N × N×(B/|Σ|) (large, bandwidth- or compute-limited). Implement three configurations — batched evolution for throughput, monoid scan for latency (already exists), and pattern packing for multi-regex — with auto-dispatch.

---

## 1. Architecture Overview

### 1.1 Engine Landscape

| Engine | GEMM Shape | Use Case | Bottleneck |
|--------|-----------|----------|------------|
| **BatchedEvolution** (A) | N×N × N×(B/\|Σ\|) | Many strings, one regex | Memory BW |
| **MonoidScan** (B, exists) | None (table lookup) | Single/few strings, low latency | Memory latency |
| **PackedEvolution** (C) | (NP)×(NP) × (NP)×(B/\|Σ\|) | Many strings, many regexes | Compute |
| v4 PrefixScan (legacy) | N×N × N×N | Fallback | MMA pipeline stalls |

### 1.2 Auto-Dispatch Rules

In `OptimizedEngine.match_batch(strings)`:

```
if B == 0:            return []
if B <= 16:           use monoid scan (Config B, already implemented)
if B >= 256:          use batched evolution GPU (Config A)
if 16 < B < 256:     use monoid scan (amortization favors avoiding GPU launch overhead)
```

In the new `PackedEngine.match_batch(strings)`:
```
Always use packed evolution GPU (Config C) — the caller opted into multi-pattern mode.
```

These thresholds are initial estimates; final values will be calibrated by benchmarking.

### 1.3 H200 Throughput Projections

**Config A** (B=65536, |Σ|=2, N=16, L=any):
- Per position: read/write N×B = 1 MB state (×2 for read+write) + B chars = 2.06 MB
- At 4,917 GB/s HBM: 0.42 μs per position
- Input throughput: B / 0.42 μs = **156 Gchar/s**
- vs v4 prefix scan: **60× faster**
- vs monoid R1 (93 Gchar/s): **1.7× faster**, processing 65K strings in parallel

**Config C** (P=16 patterns, NP=256, B=4096, |Σ|=2):
- GEMM: 256×256 × 256×2048. Arithmetic intensity = 2×256²/(2×256+1) ≈ 256 ops/byte
- Deep in compute territory — tensor cores are the bottleneck, finally saturated
- Estimated: near-peak INT8 throughput (approaching 3,958 TOPS)

---

## 2. Config A — Batched State-Vector Evolution

### 2.1 Core Algorithm

```
Input:
    T[c] ∈ int8[N][N]     for each c ∈ Σ     (transition matrices, from DFAMatrices)
    strings ∈ uint8[B][L]                      (B strings, each length L)
    start_state ∈ int                           (DFA start state index)
    accept_mask ∈ bool[N]                       (which states are accepting)

State matrix:
    S ∈ int8[N][B]         (column j = state vector of string j)

Initialize:
    S = 0
    S[start_state, :] = 1                      (all strings start in start_state)

For t = 0, 1, ..., L-1:
    For each character c ∈ Σ:
        group_c = { j : strings[j][t] == c }
        S[:, group_c] = T[c] × S[:, group_c]   (N×N × N×|group_c| GEMM)
    S = min(S, 1)                               (Boolean threshold, prevents int8 overflow)

Result:
    accepts[j] = any(S[k, j] > 0 for k in accept_states)
```

### 2.2 Equal-Length Requirement and Padding

All strings in a batch must have the same length L. The Python bridge handles variable-length input by:

1. Sorting strings by length
2. Grouping into length-buckets (e.g., bucket widths of 64 or 128)
3. Padding shorter strings within each bucket to the bucket ceiling with a "freeze" character
4. Running one batched evolution kernel per bucket
5. Merging results

The "freeze" character maps to the identity matrix (no state change). If no identity character exists in the alphabet, we add one: a virtual character σ_id with T[σ_id] = I_N. This adds one row to the transition matrix stack but costs nothing at runtime (identity GEMM is a no-op in exact arithmetic, and the group for σ_id contains only padding positions).

### 2.3 Character Grouping — Binary Alphabet (Masked GEMM)

For |Σ| = 2, at each position t, each string has character 0 or 1. Instead of gathering/scattering columns, we apply both transition matrices to all columns and use a mask to select:

```
mask[j] = (input[j][t] == 0) ? 1 : 0       // binary mask, shape [1, B]

S_new = T[0] × (S ⊙ mask) + T[1] × (S ⊙ (1 - mask))
```

Where ⊙ is element-wise broadcast multiplication (each column of S multiplied by the corresponding mask bit).

**Optimized kernel implementation:** Rather than two full GEMMs + masking, the kernel can process each column by selecting T[0] or T[1] based on the mask bit. Since the WMMA fragment loading operates on 16-column tiles, each 16-column tile loads either T[0] or T[1] based on the majority character in that tile — or more precisely, the kernel loads both T[0] and T[1] into shared memory and, for each column within a tile, selects the correct matrix row during fragment construction.

In practice, the simplest correct implementation:
1. At position t, read `input[0..B-1][t]` to build a per-column character array
2. For each WMMA tile of 16 columns: check if all 16 columns share the same character
   - If yes (common for sorted or correlated inputs): single MMA with that character's matrix
   - If mixed: two MMAs (one per character), mask and add

**Why masked GEMM for binary:** For |Σ|=2, each GEMM operates on B/2 columns on average. The masked approach avoids the scatter-gather overhead (column reindexing) at the cost of potentially computing both GEMMs and masking. Since the two GEMMs together touch all B columns exactly once, the total work is identical — the difference is avoiding the gather/scatter memory traffic.

### 2.4 Character Grouping — Large Alphabet (Scatter-Gather)

For |Σ| > 2 (e.g., byte alphabet with |Σ|=256), masked GEMM wastes (|Σ|-1)/|Σ| of compute. Instead:

**Precompute phase (CPU, once per batch):**
```
For t = 0 .. L-1:
    For c = 0 .. |Σ|-1:
        group_indices[t][c] = list of j where input[j][t] == c
        group_sizes[t][c] = len(group_indices[t][c])
```

Upload `group_indices` as a flat array with offset pointers. Total memory: O(B × L) uint32 indices.

**Runtime phase (GPU, per position t):**
```
For each character c with group_sizes[t][c] > 0:
    Gather: S_gathered[:, 0..g-1] = S[:, group_indices[t][c]]       (g = group size)
    GEMM:   S_gathered = T[c] × S_gathered                          (N×N × N×g)
    Scatter: S[:, group_indices[t][c]] = S_gathered[:, 0..g-1]
```

The gather/scatter adds memory traffic but ensures each GEMM is dense with no wasted compute. For |Σ|=256 and B=65536, each GEMM averages N×N × N×256 — still much larger than the prefix scan's N×N × N×N.

**Optimization:** Coalesce the gather/scatter by pre-sorting the group indices. Since the same indices are reused L times, this sorting cost is amortized.

### 2.5 Data Layout

```
Transition matrices:   int8[|Σ|][N][N]     column-major per matrix, contiguous across chars
                       Total: |Σ| × N² bytes (e.g., 2 × 256 = 512 bytes for binary, N=16)
                       → fits in shared memory trivially

Input strings:         uint8[L][B]          position-contiguous (column-major over strings)
                       Reading all B characters at position t is one coalesced load
                       Total: B × L bytes

State matrix:          int8[N][B]           column-major, padded to 16-column alignment
                       Total: N × B_padded bytes (e.g., 16 × 65536 = 1 MB)

Group indices:         uint32[L][|Σ|][max_group_size]   (scatter-gather mode only)
  (or flattened)       with offset array uint32[L][|Σ|+1]
                       Total: O(B × L × 4) bytes
```

### 2.6 Kernel Design

```c
// Persistent kernel — one launch for the entire L-step evolution
__global__ void batched_evolution_kernel(
    const int8_t  *T_all,        // [|Σ|, N, N]  transition matrices
    const uint8_t *input,        // [L, B]        input strings (position-contiguous)
    int8_t        *S,            // [N, B]        state matrix (in-place)
    const int8_t  *accept_mask,  // [N]           accept states
    int           *results,      // [B]           output: 1=accept, 0=reject
    int B, int L, int N, int sigma,
    // scatter-gather mode only:
    const uint32_t *group_indices,  // flattened column indices
    const uint32_t *group_offsets   // [L, |Σ|+1] offsets into group_indices
) {
    extern __shared__ int8_t smem[];
    // Load all transition matrices into shared memory
    // smem layout: T[0..sigma-1], each N×N

    // Initialize S: S[start, j] = 1, rest = 0
    // (can be done as a separate kernel or here)

    for (int t = 0; t < L; t++) {
        if (sigma == 2) {
            // Binary path: masked GEMM
            // Load input[t][0..B-1] as mask bits
            // For each 16-column WMMA tile:
            //   select T[0] or T[1] per column
            //   wmma::mma_sync
        } else {
            // General path: scatter-gather per character
            for (int c = 0; c < sigma; c++) {
                uint32_t off = group_offsets[t * (sigma+1) + c];
                uint32_t g   = group_offsets[t * (sigma+1) + c + 1] - off;
                if (g == 0) continue;
                // Gather S columns by group_indices[off..off+g-1]
                // GEMM: T[c] × S_gathered (N×N × N×g)
                // Scatter back
            }
        }
        // Boolean threshold: clamp S to {0,1}
        __syncthreads();
    }

    // Fused acceptance check
    // For each column j, check if any accept state is active
    // results[j] = (S[accept_states, j] > 0) ? 1 : 0
}
```

### 2.7 Boolean Threshold

After each position's GEMM, clamp all values in S to {0, 1}: `S[i][j] = min(S[i][j], 1)`.

For DFA transition matrices (permutation matrices with exactly one 1 per column), the product T × S already produces {0, 1} values — the threshold is a no-op. For NFA matrices (multiple 1s per column), values can accumulate past 1 when multiple paths converge; the threshold prevents int8 overflow across L steps.

The threshold is a simple element-wise `min(val, 1)` over the N×B state matrix — negligible cost compared to the GEMM.

---

## 3. Config C — Pattern Packing

### 3.1 Block-Diagonal Construction

Given P compiled DFA patterns with state counts N₁, N₂, ..., N_P and alphabets Σ₁, Σ₂, ..., Σ_P:

1. **Alphabet union:** Σ = Σ₁ ∪ Σ₂ ∪ ... ∪ Σ_P
2. **Padding:** Each Nᵢ padded to next multiple of 16 → Nᵢ'. NP = Σ Nᵢ'.
3. **Block-diagonal matrix:** For each c ∈ Σ:
   ```
   T_packed[c] = diag(T₁'[c], T₂'[c], ..., T_P'[c])    size NP × NP
   ```
   where Tᵢ'[c] = Tᵢ[c] if c ∈ Σᵢ, else I_{Nᵢ'} (identity — character doesn't affect this pattern).

4. **State matrix:** `int8[NP][B]` — column j holds the concatenated state vectors for all P patterns. Pattern p's state occupies rows `offset_p` to `offset_p + Nᵢ' - 1`.

5. **Start state:** For each pattern p and string j: `S[offset_p + start_p, j] = 1`.

6. **Accept check:** For each pattern p and string j: `accepts[p][j] = any(S[offset_p + k, j] > 0 for k in accept_states_p)`.

### 3.2 Why This Works

For P=16 patterns with N=16: NP = 256. The GEMM per position is 256×256 × 256×(B/|Σ|). For B=4096, |Σ|=2: GEMM is 256×256×2048. This is a textbook tensor-core-friendly shape with:
- Arithmetic intensity: 2×256²/(2×256+1) ≈ 256 ops/byte — deep compute territory
- Tensor core utilization: near-peak (256×256 tiles fully occupy the MMA pipeline)

The off-diagonal blocks of T_packed are zero (patterns don't interact). A dense GEMM wastes compute on these blocks but still achieves high utilization because the tile dimensions are large. An optimized kernel can skip zero blocks to reclaim the wasted factor, but this is an optimization, not a correctness requirement.

### 3.3 API

```python
class PackedEngine:
    def __init__(self, regexes: list[str]):
        """Compile multiple regex patterns into a packed block-diagonal DFA."""
        # For each regex: compile DFA, build matrices
        # Construct block-diagonal T_packed and metadata (offsets, accept masks)

    def match_batch(self, strings: list[str]) -> list[list[bool]]:
        """Match all strings against all patterns.
        Returns results[pattern_idx][string_idx]."""

    def match_batch_timed(self, strings: list[str]) -> tuple[list[list[bool]], dict]:
        """Like match_batch but with timing breakdown."""

    @property
    def config_info(self) -> dict:
        """Returns metadata: n_patterns, state counts, NP, etc."""
```

---

## 4. Python Bridge

### 4.1 BatchedEvolutionEngine (Config A)

```python
class BatchedEvolutionEngine:
    def __init__(self, lib, dm: DFAMatrices, max_B: int, max_L: int):
        """Upload transition matrices, allocate state matrix and group buffers."""

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        """
        1. Group strings by length into buckets
        2. For each bucket: pad to bucket ceiling, transpose to [L][B] layout
        3. For scatter-gather mode: precompute group indices on CPU
        4. Upload input + group indices
        5. Launch kernel
        6. Read back results
        7. Merge results across buckets
        """

    def simulate_batch_timed(self, strings: list[str]) -> tuple[list[bool], float, float]:
        """Returns (results, kernel_ms, total_ms)."""
```

### 4.2 Length Bucketing

To minimize padding waste, group strings by similar length:

```
bucket_width = 64  (or 128, tunable)
For each string:
    bucket = ceil(len(string) / bucket_width) * bucket_width
    assign to bucket group
For each bucket group with B_bucket strings:
    pad all strings to bucket length
    run batched evolution
```

Worst-case padding overhead: bucket_width - 1 extra positions per string = (bucket_width - 1) / avg_length fraction of wasted work. For bucket_width=64, avg_length=512: 12% overhead.

### 4.3 Char-Mapping Optimization

The current Python char-mapping (88 ms/Mchar) is the bottleneck. For Config A, we need to transpose B×L characters to L×B layout and map to alphabet indices. This should be done in C via a ctypes helper or entirely on GPU:

```c
// CPU-side helper compiled into the .so
void prepare_batch(
    const char *strings,     // concatenated string bytes
    const int  *lengths,     // per-string lengths
    uint8_t    *output,      // [L][B] transposed, alphabet-indexed
    int B, int L, int sigma,
    const int  *char_to_idx  // [256] char → alphabet index mapping
);
```

This converts the O(B×L) Python loop into a single C function call.

---

## 5. CUDA Implementation

### 5.1 File Structure

```
cuda/batched_evolution.cu     — Config A kernel + library API
cuda/packed_evolution.cu      — Config C kernel (reuses A's inner loop)
src/gpu_bridge_batched.py     — Python ctypes bridge for Config A
src/gpu_bridge_packed.py      — Python ctypes bridge for Config C
src/packed_engine.py          — PackedEngine Python API
```

### 5.2 Library C API

```c
// Config A
int batched_engine_init(int N, int sigma,
                        const int8_t *trans_matrices,  // [sigma, N, N]
                        const int8_t *accept_mask,     // [N]
                        int start_state,
                        int max_B, int max_L);
void batched_engine_destroy();
int batched_engine_dispatch(
    const uint8_t *input,       // [L, B] position-contiguous
    int B, int L,
    int *results,               // [B] output
    float *kernel_ms,
    float *total_ms
);

// Config C
int packed_engine_init(int NP, int sigma, int n_patterns,
                       const int8_t *trans_packed,     // [sigma, NP, NP]
                       const int8_t *accept_masks,     // [n_patterns, NP]
                       const int *start_states,        // [n_patterns]
                       const int *pattern_offsets,      // [n_patterns] row offsets
                       int max_B, int max_L);
void packed_engine_destroy();
int packed_engine_dispatch(
    const uint8_t *input,       // [L, B]
    int B, int L,
    int *results,               // [n_patterns * B] output (row-major)
    float *kernel_ms,
    float *total_ms
);
```

---

## 6. Integration into OptimizedEngine

### 6.1 New Configs

```python
config="batched"       # Config A CPU simulation (for testing)
config="batched+gpu"   # Config A on GPU
config="packed+gpu"    # Config C on GPU (via PackedEngine)
```

### 6.2 Auto-Selection Update

When `match_batch` is called:

```python
def match_batch(self, strings):
    B = len(strings)
    if self._gpu_engine is not None:         # monoid+gpu already set
        return self._gpu_engine.simulate_batch(strings)
    if B >= 256 and self._batched_gpu is not None:
        return self._batched_gpu.simulate_batch(strings)
    # ... existing dispatch (monoid, kgram, sequential)
```

The auto-selector in `__init__` will attempt to initialize the batched GPU engine alongside the monoid engine. At match time, dispatch based on B.

---

## 7. Test Plan

### Correctness Tests

**T1: Batched evolution CPU correctness**
- Cross-validate `simulate_batched_cpu(dm, strings)` against `simulate_sequential` for patterns: abb, even_a, binary_div3, ab_star, hex_number, identifier
- B in {1, 16, 256, 65536}, L in {1, 32, 128, 1024}
- Random strings seeded for reproducibility

**T2: Batched evolution GPU correctness**
- Cross-validate GPU kernel results against CPU batched evolution
- Same parameter grid as T1

**T3: Character grouping modes**
- For binary alphabet: verify masked GEMM mode matches sequential
- For byte alphabet (hex_number, identifier): verify scatter-gather mode matches sequential

**T4: Variable-length padding**
- Batch of strings with lengths {10, 50, 100, 500}
- Verify padded results match individual per-string results

**T5: Pattern packing correctness**
- Pack 2, 4, 8, 16 patterns together
- Cross-validate each pattern's results against individual `OptimizedEngine` runs

**T6: NFA compatibility**
- Run batched evolution with NFA matrices (non-permutation)
- Verify Boolean threshold prevents overflow
- Cross-validate against `simulate_nfa`

### Performance Benchmarks

**P1: Throughput scaling with B**
- Config A, pattern=abb, L=512
- B in {64, 256, 1024, 4096, 16384, 65536}
- Measure Gchar/s, compare to monoid R1 and v4 prefix scan

**P2: Throughput scaling with L**
- Config A, pattern=abb, B=65536
- L in {32, 128, 512, 2048, 8192}
- Measure Gchar/s

**P3: Effective FLOP/s**
- Estimate tensor core utilization from kernel time and FLOP count
- Compare to 3,958 INT8 TOPS peak

**P4: Multi-pattern scaling (Config C)**
- P in {1, 2, 4, 8, 16}, B=4096, L=512
- Measure Gchar/s per pattern and total

**P5: End-to-end including preprocessing**
- Full pipeline: regex compile → matrix build → GPU init → batch prep → kernel → readback
- Compare to monoid pipeline and v4 pipeline
- Identify bottleneck at each B

---

## 8. Implementation Priority

1. **Config A binary-alphabet kernel** — highest impact, simplest (masked GEMM, |Σ|=2). Proves the concept and delivers the throughput win.
2. **Config A scatter-gather kernel** — generalizes to arbitrary alphabets.
3. **Config A Python bridge + integration** — wire into OptimizedEngine with auto-dispatch.
4. **Config C pattern packing** — builds directly on Config A's kernel with larger matrices.
5. **Benchmarks and profiling** — validate throughput projections.
6. **C-accelerated batch prep** — eliminate the Python char-mapping bottleneck.
