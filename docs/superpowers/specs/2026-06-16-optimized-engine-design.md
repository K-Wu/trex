# Optimized DFA/NFA Engine Design Spec

**Date:** 2026-06-16
**Status:** Draft
**Builds on:** `docs/OPTIMIZATIONS.md`, v4 parallel engine (`cuda/parallel_dfa_engine.cu`)

## Goal

Implement four composable optimizations that compound to eliminate the fundamental inefficiency identified in OPTIMIZATIONS.md: feeding 93.75%-zero permutation matrices through O(N³) tensor-core matmul. The optimizations target different axes and auto-select based on measured regex/input characteristics.

---

## 1. Composability Model

### Axes of Optimization

| Axis | Options | What It Reduces |
|------|---------|-----------------|
| **Representation** | DFA (small N) vs. NFA (no blowup) | Construction cost / state count |
| **Encoding** | int8 (current) vs. binary (b1) | Memory per matrix: 8x reduction |
| **Scan backend** | Matrix scan (MMA) vs. Monoid scan (table lookup) | Per-step cost: O(N³) → O(1) |
| **Input compression** | Raw chars vs. k-gram | Scan length: L → L/k |

### Composition Rules

**Stack freely (orthogonal):**
- k-gram + monoid → O(L/k) lookups + O(log(L/k)) integer scan
- k-gram + matrix scan → matrix scan over L/k matrices instead of L
- Binary encoding + matrix scan → 8x bandwidth improvement
- Any combination of the above

**Alternatives (pick one per axis):**
- DFA path XOR NFA path (different construction)
- Monoid scan XOR matrix scan (monoid replaces matrix math entirely)

### Auto-Selection Heuristic

Decided at regex compile time:

```
1. Try DFA construction (cap: N ≤ 64 states, timeout: 1s)
   → Success: DFA path
   → Failure: NFA path

2. If DFA path:
   a. Compute transition monoid (cap: M ≤ 65,536 elements)
   b. If M fits → monoid scan backend
   c. Else → binary matrix scan backend

3. If NFA path:
   → Always binary matrix scan backend
   → (Monoid too large for NFA typically)

4. k-gram (always applied):
   k = max k such that |Σ|^k ≤ 65,536
   → Binary alphabet (|Σ|=2): k=16
   → Small alphabet (|Σ|≤16): k=4
   → Byte alphabet (|Σ|=256): k=2

5. Encoding:
   → Monoid scan: integer encoding (1-2 bytes per element)
   → Matrix scan: binary encoding (32 bytes per 16×16 matrix)
```

---

## 2. Transition Monoid

### Precompute Phase (one-time per regex)

1. Initialize `M = { T[c] : c ∈ Σ }` — one permutation matrix per character
2. BFS closure: for each pair `(A, B) ∈ M × M`, compute `P = A × B`
   - If P is new (not in M): add to M, enqueue for further products
   - Repeat until no new elements found
3. Build composition table: `compose[i][j] = index_of(element_i ∘ element_j)` — an M×M array of uint16
4. Build char-to-monoid mapping: `char_to_monoid[c] = index_of(T[c])`
5. Build accept table: for each monoid element m, precompute `accept[m] = (m × start_vec) ∩ accept_states ≠ ∅`

### Online Phase

```
for each char c in input:
    monoid_idx = char_to_monoid[c]     // array lookup
prefix_scan(monoid_indices, compose)   // integer scan, O(1) per composition
result = accept[final_monoid_index]    // single lookup
```

### Complexity Guard

If monoid exceeds 65,536 elements during BFS, abort and fall back to matrix scan. Typical sizes for small DFAs:
- `(a|b)*abb` (4 states): ~14 elements
- `[0-9]+\.[0-9]+` (identifier-like): ~50-200 elements
- Worst case N^N: only tractable for N ≤ 5

### CUDA Kernel: Integer Prefix Scan

The monoid scan kernel is a standard parallel prefix scan over uint16 values with a custom binary operator (table lookup). No tensor cores needed.

**R1 variant (many short strings):** One warp per string, sequential scan with table lookup.
**R3 variant (long strings):** Decoupled look-back with table-lookup composition instead of matmul.

Both reuse the v4 dispatch infrastructure, replacing the matmul calls with `compose[a][b]` lookups from shared memory.

### Implementation Files

- `src/monoid.py`: `compute_monoid(dm: DFAMatrices) → MonoidData` — BFS closure, composition table, accept table
- `cuda/monoid_scan.cu`: Integer prefix scan kernels (R1 + R3 variants)
- `src/gpu_bridge_monoid.py`: Python ctypes bridge for monoid scan engine

---

## 3. k-Gram Precomputation

### Concept

Precompute the composed result for every possible k-character substring. The input string is chunked into L/k k-grams, each resolved by a single table lookup. Reduces scan length by factor k.

### Two Modes

**Mode A — k-gram → monoid index (when monoid scan is active):**
```
For each k-gram w = (c_1, c_2, ..., c_k):
    kgram_monoid[w] = compose(compose(...compose(char_to_monoid[c_1], char_to_monoid[c_2])...), char_to_monoid[c_k])
```
This is k-1 table lookups per entry, done entirely in Python during precomputation. No matrix math.

**Mode B — k-gram → matrix (when matrix scan is active):**
```
For each k-gram w = (c_1, c_2, ..., c_k):
    kgram_matrix[w] = T[c_k] × ... × T[c_1]
```
Precomputed via recursive doubling:
- Round 0: 1-gram table = per-character matrices (|Σ| entries)
- Round 1: 2-gram table = all pairs (|Σ|² entries, each one matmul)
- Round r: 2^r-gram table from 2^(r-1)-gram table
- Each round is a batch of independent matmuls → batch GEMM on tensor cores

Total rounds: log₂(k). Total matmuls: |Σ|^k × log₂(k).

### k Selection

| Alphabet Size |Σ|| Max k (|Σ|^k ≤ 65536) | Table Memory (binary) | Table Memory (int8) |
|---|---|---|---|
| 2 | 16 | 65,536 × 32 B = 2 MB | 65,536 × 256 B = 16 MB |
| 4 | 8 | 65,536 × 32 B = 2 MB | 65,536 × 256 B = 16 MB |
| 16 | 4 | 65,536 × 32 B = 2 MB | 65,536 × 256 B = 16 MB |
| 256 | 2 | 65,536 × 32 B = 2 MB | 65,536 × 256 B = 16 MB |

With monoid scan, tables are just uint16 indices: 65,536 × 2 B = 128 KB — trivially fits.

### Tail Handling

If `L % k ≠ 0`, the last partial block (1 to k-1 characters) is handled by:
- Monoid mode: sequential compose of individual char monoid indices
- Matrix mode: sequential matmul of individual char matrices

This happens once per string — negligible cost.

### Online Data Flow

```
Input string (L chars)
  → chunk into L/k k-grams + tail
  → table lookup: k-gram → monoid index (or matrix)
  → prefix scan over L/k elements
  → compose tail if present
  → check acceptance
```

### Implementation Files

- `src/kgram.py`: `precompute_kgrams(dm, k, monoid=None) → KGramTable`
- k-gram index computation for monoid mode
- Recursive doubling batch GEMM for matrix mode
- CUDA kernel integration: k-gram lookup is a preprocessing step before the existing scan kernels

---

## 4. Binary Encoding (WMMA precision::b1)

### Matrix Representation

Each 16×16 Boolean matrix stored as 16 × uint16_t (one uint16 per row, each bit = one column). Total: 32 bytes per matrix.

```c
// Packing: matrix[row][col] → (packed[row] >> col) & 1
// Row-major bit packing for WMMA compatibility
```

### WMMA b1 Kernel

```c
wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::precision::b1, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::precision::b1, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 8, 8, 128, int32_t> c_frag;

// MMA computes: C[i][j] = popcount(A_row_i AND B_col_j)
// Threshold: result = (C > 0) ? 1 : 0
```

Note: WMMA b1 shape is 8×8×128 on SM ≥ 7.5. For 16×16 matrices, decompose into 2×2 tiles of 8×8 with 128-bit (16-byte) K dimension. Since our matrices are 16×16 = 256 bits = 32 bytes, the K dimension maps to 2 WMMA tiles along K.

Alternative: Use PTX `mma.sync` directly for 16×16 b1 if available on SM 9.0 (Hopper). Check CUDA documentation for exact supported shapes.

If b1 WMMA proves too constrained on shape, fall back to: store as int8 but use the binary structure to skip zero elements via bitmask — still saves bandwidth even without b1 MMA.

### Python Side

`BinaryDFAMatrices` class:
- Converts DFAMatrices to packed uint16 format
- Stores `packed_stack`: shape `(|Σ|, 16)` of uint16
- Conversion methods: `to_binary()`, `from_binary()`

### Implementation Files

- `src/binary_matrices.py`: `BinaryDFAMatrices` class, packing/unpacking utilities
- `cuda/binary_scan.cu`: WMMA b1 scan kernels (or fallback int8-with-bitmask)
- Integration into existing R1/R3 dispatch as an alternative matmul function

---

## 5. NFA Path

### Construction

Expose NFA transition matrices directly from `regex_to_dfa.py`:

```python
def compile_nfa_matrices(regex: str) -> NFAMatrices:
    nfa = parse_and_build_nfa(regex)
    # NFA transition matrix: T[c][i][j] = 1 iff state j can reach state i on char c
    # Multiple 1s per column (nondeterminism)
    # State count n = O(|regex|)
```

### Multi-Tile MMA

For NFA state count n > 16, the n×n matrix is decomposed into `ceil(n/16)²` tiles of 16×16. Block matrix multiplication:

```
For tiles (I, J) of result C = A × B:
    C_tile[I][J] = Σ_K  A_tile[I][K] × B_tile[K][J]
```

Each tile product is one WMMA MMA call. Total MMA calls per matrix multiply: `ceil(n/16)³`.

### Boolean Threshold

After each matmul step in the prefix scan, apply threshold: `result[i][j] = min(result[i][j], 1)`. This prevents int8 overflow from accumulating multiple 1s. With binary encoding, this is implicit (popcount > 0 → 1).

### When to Use

- DFA construction exceeds N=64 states or times out
- Pattern uses features that cause exponential DFA blowup (e.g., `.*a.{n}b` for large n)
- User explicitly requests NFA mode

### Implementation Files

- `src/nfa_matrices.py`: `NFAMatrices` class — NFA construction, multi-tile matrix layout
- `cuda/nfa_scan.cu`: Multi-tile binary scan kernel
- `src/gpu_bridge_nfa.py`: Python ctypes bridge

---

## 6. Unified API

### `OptimizedEngine` Class

```python
class OptimizedEngine:
    def __init__(self, regex: str, config: str | None = None):
        """
        config=None: auto-select based on regex analysis
        config="monoid": force monoid scan
        config="monoid+kgram": force monoid + k-gram
        config="binary": force binary matrix scan
        config="nfa": force NFA path
        config="nfa+kgram": force NFA + k-gram
        config="baseline": force v4 int8 matrix scan (no optimizations)
        """

    def match_batch(self, strings: list[str]) -> list[bool]:
        """Match a batch of strings against the compiled regex."""

    def match_batch_timed(self, strings: list[str]) -> tuple[list[bool], dict]:
        """Like match_batch but returns timing breakdown."""

    @property
    def config_info(self) -> dict:
        """Returns which optimizations are active and why."""
```

### Config Info Example

```python
{
    "representation": "dfa",
    "dfa_states": 5,
    "monoid_size": 14,
    "scan_backend": "monoid",
    "kgram_k": 8,
    "encoding": "integer",  # monoid uses integer, not matrix
    "alphabet_size": 2,
    "selection_reason": "DFA succeeded (5 states), monoid small (14 elements), binary alphabet allows k=8"
}
```

### Implementation File

- `src/optimized_engine.py`: `OptimizedEngine` class with auto-selection logic

---

## 7. Test Plan

### Correctness Tests (per optimization)

**T1: Monoid correctness**
- Verify monoid closure: for all pairs in M, product is in M
- Verify composition table: `compose[i][j]` matches actual matrix product
- Cross-validate monoid scan against matrix scan for all 6 test patterns
- Test monoid size guard: patterns with large monoid correctly fall back

**T2: k-gram correctness**
- Verify k-gram tables match sequential composition of k individual matrices
- Test all k values: k=2,4,8,16 for binary alphabet; k=2 for byte alphabet
- Test tail handling: L%k = 0, 1, k-1
- Cross-validate k-gram scan against non-k-gram scan

**T3: Binary encoding correctness**
- Verify binary packing/unpacking roundtrips
- Cross-validate binary MMA against int8 MMA for all test patterns
- Test with DFA matrices (permutation) and NFA matrices (multiple 1s)

**T4: NFA path correctness**
- Cross-validate NFA path against DFA path for patterns where both work
- Test patterns with known DFA blowup (verify NFA handles them)
- Test multi-tile MMA for NFA state count > 16
- Verify Boolean threshold prevents overflow

**T5: Auto-selection correctness**
- Verify correct backend selection for small DFA, large DFA, NFA-triggering patterns
- Verify all configs produce identical results on same inputs

**T6: Composition correctness**
- Monoid + k-gram: cross-validate against monoid-only and k-gram-only
- k-gram + binary: cross-validate against int8 k-gram

### Performance Benchmarks

**P1: Monoid scan vs. matrix scan** — varying L (64 to 1M), B (1 to 100K), all test patterns
**P2: k-gram speedup** — varying k, comparing scan throughput with and without k-gram
**P3: Binary vs. int8** — memory bandwidth measurement, scan throughput
**P4: NFA vs. DFA** — construction time, scan throughput, pattern size scaling (10 to 1000 NFA states)
**P5: Auto-selected vs. v4 baseline** — end-to-end comparison across all test patterns and input sizes
**P6: Compound optimization** — monoid+kgram vs. monoid-only vs. kgram-only vs. baseline

---

## 8. Implementation Priority

Ordered by impact-to-effort ratio:

1. **Transition monoid** — highest standalone impact, pure Python precompute + simple CUDA integer scan kernel. For the `(a|b)*abb` DFA (14-element monoid), eliminates all matrix math from the hot path.

2. **k-gram precomputation** — compounds with monoid (k-gram → monoid index, zero matrix math). Straightforward table construction. Reduces scan length by 8-16x for binary alphabet.

3. **Binary encoding** — requires new WMMA b1 kernel. 8x memory reduction. Most impactful for matrix scan backend (not needed for monoid scan which is already O(1) per step). Critical for NFA path.

4. **NFA path** — requires NFA matrix export, multi-tile MMA, Boolean threshold. Enables patterns DFA can't handle. Depends on binary encoding for practical performance.

5. **Auto-selection + unified API** — ties everything together. Depends on all above being independently testable first.
