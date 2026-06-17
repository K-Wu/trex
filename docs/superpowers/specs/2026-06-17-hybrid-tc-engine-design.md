# Hybrid TC Engine — K-gram Fusion & Backend Auto-Selection

## Problem

Tensor core utilization for 16×16 DFA state evolution is <1%. The arithmetic
intensity (AI = 2N/3 ≈ 10.7 ops/byte for N=16) is 75× below the H200 balance
point (805 ops/byte). Monoid table-lookup achieves 93 Gc/s vs 28 Gc/s for TC —
TC cannot win for N≤16 because it replaces an O(1) lookup with an O(N³) MMA.

For N>16, monoid's state space M≤N! becomes impractical, and TC is the only
viable approach. K-gram fusion (precomputing product matrices for k-character
sequences) increases effective AI by k×, making TC meaningful for medium-N DFAs.

## Architecture

Three-tier hybrid with auto-selection:

```
OptimizedEngine.match_batch(strings)
│
├── N ≤ 16, monoid fits ──→ Monoid R1 GPU kernel
│   93 Gc/s. O(1) table lookup per char. No matmul.
│
├── N > 16, σ^k ≤ budget ──→ K-gram TC kernel (NEW)
│   Precompute σ^k product matrices. One MMA per k chars.
│   AI = k × 16 ops/byte. σ=2, k=16 → AI=256 (~32% TC util).
│
└── N > 16, σ^k too large ──→ TC state evolution (existing)
    One MMA per char. AI=16 ops/byte. ~28 Gc/s.
```

### Selection boundaries

| N     | σ   | k (auto) | Table size | Backend          |
|-------|-----|----------|------------|------------------|
| ≤16   | any | —        | —          | Monoid R1        |
| 17-64 | 2   | 16       | 16 MB      | K-gram TC        |
| 17-64 | 4   | 8        | 16 MB      | K-gram TC        |
| 17-64 | 26  | 3        | 4.5 MB     | K-gram TC        |
| 17-64 | 256 | 2        | 16 MB      | K-gram TC        |
| >64   | 2   | 10       | 256 KB     | K-gram TC        |
| >64   | 256 | —        | too large  | TC state evol.   |

### Auto-k selection

```
auto_k_for_gpu(sigma, n_states, max_table_bytes=48_000_000):
    matrix_bytes = n_states * n_states   # int8
    k = 1
    while sigma^(k+1) * matrix_bytes <= max_table_bytes:
        k += 1
    return k
```

The 48 MB budget targets H200 L2 cache. The k-gram table should fit in L2 for
good throughput — global memory random access would be too slow.

## K-gram TC CUDA Kernel

### File: `cuda/kgram_evolution.cu`

### The batching incompatibility

WMMA computes `C[16×16] = A[16×16] × B[16×16]`. Matrix A (the transition
matrix T) is shared across all 16 columns of B (16 strings' state vectors).
In the existing per-character kernel, all 16 strings at position `pos` use the
same character mapping: `T[c]` where `c = input[pos]`... **NO** — different
strings have different characters at position `pos`. The existing kernel handles
this via the `select-by-threshold` trick:

```
For sigma=2 (binary alphabet):
  acc0 = T0 × S    (what-if char=0)
  acc1 = T1 × S    (what-if char=1)
  result = select(acc0, acc1, char)  // per-element select by actual char
```

This works because with σ MMA calls, we compute ALL possible transitions and
select per-column. For k-gram with σ=2 and k=16, we'd need σ^k = 65536 MMAs
to compute all possibilities and select — **worse than the original**.

### Correct k-gram TC design: single-string mode

K-gram TC is viable only in **single-string-per-warp** mode: one warp processes
one string, using 1 column of the 16×16 tile. The other 15 columns are wasted.

```c
// Each warp processes ONE string
// Grid: (num_strings, 1)
// 32 threads per warp, 4 warps per block → 128 threads/block

int string_id = blockIdx.x * WARPS_PER_BLOCK + warpId;

for (int pos = 0; pos < L; pos += k) {
    // Compute k-gram index (only lane 0 computes, broadcasts)
    uint32_t idx = 0;
    for (int i = 0; i < k; i++)
        idx = idx * sigma + input[(pos + i) * B_padded + string_id];

    // Load T_kgram[idx] into fragment A
    wmma::load_matrix_sync(frag_T, &T_kgram[idx * N * N], N);

    // MMA: T_kgram × S → acc (only column 0 has the state vector)
    wmma::fill_fragment(acc, 0);
    wmma::mma_sync(acc, frag_T, frag_S, acc);

    // Store, reload
    wmma::store_matrix_sync(smem, acc, N, wmma::mem_row_major);
    wmma::load_matrix_sync(frag_S, smem, N);
}
```

**Throughput estimate (σ=2, k=16, N=16):**
- Per string: L/k MMA calls = L/16 MMAs
- Each MMA: ~4 cycles compute + ~30 cycles L2 load T_kgram = ~34 cycles
- Total per string: L/16 × 34 = 2.125L cycles
- At 1.98 GHz: 1.98e9 / 2.125 = 932 Mchar/s per warp
- H200 has 132 SMs × 4 warps/SM = 528 warps
- Peak: 528 × 932 Mchar/s = **492 Gc/s** (theoretical upper bound)
- Realistic (memory/occupancy losses): ~50-100 Gc/s

**vs existing batched kernel (σ=2, k=1):**
- Per warp: processes 16 strings simultaneously, 2 MMAs/char
- Per string: L × 2/16 = L/8 MMA-equivalents
- K-gram at k=16: L/16 MMAs per string → 50% fewer operations

**When k-gram wins over batched:**
- k-gram: 1 MMA per k chars, but only 1 string per warp
- batched: σ MMAs per char, but 16 strings per warp
- k-gram MMA count per string: L/k
- batched MMA count per string: L×σ/16
- k-gram wins when L/k < L×σ/16 → k > 16/σ
- For σ=2: k > 8. At k=16 → 2× fewer MMAs per string.
- For σ=4: k > 4. At k=8 → 2× fewer.
- For σ=256: k > 0.0625. Always wins (k≥1).

### Multi-tile extension (N>16)

For N=32 (2×2 tiles):
- Each MMA step: (N/16)² = 4 MMAs for the full T×S multiply
- T_kgram entry: 32×32 = 1024 bytes
- K-gram table at σ=2, k=16: 65536 × 1024 = 64 MB (too large for L2)
- Auto-reduce k: at k=14, 16384 × 1024 = 16 MB (fits)

For N=64 (4×4 tiles):
- Each MMA step: 16 MMAs
- T_kgram entry: 4096 bytes
- K-gram at σ=2, k=13: 8192 × 4096 = 32 MB

### Tail handling

When `L % k != 0`, the last `L % k` characters use per-character MMA with the
base T matrices `T_base[σ][N][N]` (always tiny, fits in shared memory):

```c
for (int pos = L - (L % k); pos < L; pos++) {
    uint8_t c = input[pos * B_padded + string_id];
    wmma::load_matrix_sync(frag_T, &T_base[c * N * N], N);
    wmma::fill_fragment(acc, 0);
    wmma::mma_sync(acc, frag_T, frag_S, acc);
    // store/reload
}
```

## C API

```c
// cuda/kgram_evolution.cu

void kgram_engine_init(
    const int8_t *T_kgram,      // [n_entries, N, N] precomputed products
    const int8_t *T_base,       // [sigma, N, N] per-char (for tail)
    const int8_t *accept_mask,  // [N] accept indicator
    const int8_t *start_vec,    // [N] start state
    int N, int sigma, int k,
    int n_entries,              // σ^k
    int max_B, int max_L
);

float kgram_engine_dispatch(
    const uint8_t *input,       // [L, B_padded] position-contiguous
    int B, int B_padded, int L,
    int *results                // [B] output accept/reject
);

void kgram_engine_destroy(void);
```

## Python Bridge

### File: `src/gpu_bridge_kgram.py`

```python
class KGramGPUSimulator:
    def create_engine(self, dm: DFAMatrices, k: int):
        # 1. Precompute k-gram matrices using kgram.precompute_kgrams(dm, k)
        #    (Mode B — matrix composition, no monoid needed)
        # 2. Flatten table: [σ^k, N, N] int8 contiguous array
        # 3. Extract T_base: [σ, N, N] from dm.matrices
        # 4. Build accept_mask, start_vec from dm
        # 5. Load libkgram_evolution.so, call kgram_engine_init
        return KGramGPUEngine(...)

class KGramGPUEngine:
    def simulate_batch(self, strings: list[str]) -> list[bool]:
        # 1. Map chars → indices, pad, layout position-contiguous
        # 2. Call kgram_engine_dispatch
        # 3. Return bool results

    def simulate_batch_timed(self, strings) -> tuple:
        # Same but returns (results, kernel_ms, total_ms)
```

## OptimizedEngine Integration

### New config: `"kgram+gpu"`

```python
# In OptimizedEngine.__init__:
elif config == "kgram+gpu":
    self._force_baseline()  # builds DFA + DFAMatrices
    self._setup_kgram_gpu()

def _setup_kgram_gpu(self):
    from src.gpu_bridge_kgram import KGramGPUSimulator
    sigma = len(self._dfa.alphabet)
    k = auto_k_for_gpu(sigma, self._dfa.n_states)
    sim = KGramGPUSimulator()
    self._kgram_gpu = sim.create_engine(self._dm, k)
    self._kgram_k = k
    self._scan_backend = 'kgram+gpu'
    self._selection_reason = f'GPU k-gram TC (k={k}, table={sigma**k} entries)'
```

### Updated auto-selection

```python
def _auto_select(self):
    self._build_dfa()
    n = self._dfa.n_states
    sigma = len(self._dfa.alphabet)

    if n > self._dfa_state_cap:
        # NFA fallback (unchanged)
        ...
        return

    # Tier 1: Monoid for small DFAs
    if n <= 16:
        md = compute_monoid(self._dm, max_size=self._monoid_cap)
        if md is not None:
            # existing monoid+kgram path
            ...
            return

    # Tier 2: K-gram TC for medium DFAs
    k = auto_k_for_gpu(sigma, n)
    if k >= 2:
        self._setup_kgram_gpu()
        return

    # Tier 3: Direct TC state evolution
    self._setup_batched_gpu()
```

## TC Utilization Analysis

### Arithmetic intensity derivation

For a single N×N WMMA MMA (N=16):
- Compute: 2 × 16³ = 8,192 INT8 ops
- Data: 2 × 256 bytes (load A + B) = 512 bytes
- AI = 16 ops/byte

H200 SXM balance point:
- 3,958 INT8 TOPS / 4.915 TB/s = 805 ops/byte

Ratio: 16/805 = **2.0% maximum TC utilization** for N=16 per-character evolution.

### K-gram multiplier

Processing k chars per MMA:
- Compute: 8,192 ops (same)
- Effective work: k × 8,192 ops (k chars' worth of transitions)
- Data: still 512 bytes per MMA
- Effective AI = k × 16 ops/byte

| k  | Effective AI | TC util estimate | Table (σ=2) |
|----|-------------|-----------------|-------------|
| 1  | 16          | 2.0%            | 512 B       |
| 4  | 64          | 8.0%            | 4 KB        |
| 8  | 128         | 15.9%           | 64 KB       |
| 16 | 256         | 31.8%           | 16 MB       |
| 20 | 320         | 39.8%           | 256 MB      |

### Why monoid wins for N≤16

Monoid replaces the entire matmul with O(1) table lookup:
- Compose table: M² entries, M ≤ ~200 for typical 16-state DFAs
- Table fits in L1/registers (~40 KB)
- 1-2 cycles per character vs ~34 cycles per k-gram MMA step
- Measured: 93 Gc/s (monoid R1) vs 28 Gc/s (TC sparse)

For N>16, monoid state space M ≤ N! grows explosively. At N=17, the worst-case
monoid has 17! ≈ 3.6×10¹⁴ elements — infeasible to enumerate. TC becomes the
only viable approach, and k-gram fusion makes it efficient.
