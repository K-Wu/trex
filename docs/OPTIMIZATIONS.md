# Optimization Opportunities: Beyond Naive Int8 Matrix-Product DFA Simulation

## The Core Inefficiency

A DFA transition matrix is a permutation matrix: exactly one 1 per column, everything else zero. That is N bits of information stored in N² bytes of int8, and O(N³) compute per composition for O(N) actual work. Tensor cores are designed for large dense GEMMs; feeding them 16×16 tiles that are 93.75% zeros is a fundamental mismatch.

Three independent improvements exist. Each changes the complexity along a different axis, and they compound.

---

## 1. Binary (1-Bit) Tensor Cores

### Problem

All matrix entries are ∈ {0, 1}. Storing them as int8 wastes 7 bits per element and 8× memory bandwidth.

### Solution

NVIDIA WMMA supports `precision::b1` on SM ≥ 7.5 (Turing+). A 16×16 Boolean matrix becomes 256 bits = 32 bytes, down from 256 bytes in int8. The MMA operation is bitwise-AND plus popcount:

```
C[i][j] = popcount(A_row_i AND B_col_j)
```

For Boolean matrix multiplication (OR-AND semiring), threshold the result:

```
result[i][j] = 1  if  C[i][j] > 0
               0  otherwise
```

This is exact for both DFA matrices (permutation — popcount always 0 or 1) and NFA matrices (Boolean reachability — popcount can be > 1, thresholded to 1).

### Impact

| Metric | Int8 | Binary | Improvement |
|--------|------|--------|-------------|
| Memory per matrix | 256 B | 32 B | 8× |
| Matrices fitting in 48 KB shared memory | 192 | 1,536 | 8× |
| HBM bandwidth per matrix load | 256 B | 32 B | 8× |
| MMA throughput (A100) | 624 TOPS | higher effective TOPS | architecture-dependent |

For large L where the prefix scan is memory-bandwidth-bound, the 8× data reduction translates directly to ~8× throughput improvement.

### Applicability to NFA Path

Binary representation is natural for NFA simulation. NFA transition matrices have multiple 1s per column (nondeterminism). The Boolean matmul computes reachable states correctly:

```
active_states_{t+1} = threshold(T[c_t] × active_states_t)
```

No DFA construction needed. NFA state count is always O(|pattern|), avoiding exponential blowup. The binary encoding is the native representation for Boolean reachability.

---

## 2. k-Gram Precomputation (Four Russians Method)

### Problem

The prefix scan processes L matrices, one per input character. Scan depth is O(log L). For a 1 MB input, that is 20 scan steps even with perfect parallelism.

### Solution

Precompute the composed transition matrix for every possible k-character substring:

```
T_block[w] = T[w_k] × T[w_{k-1}] × ... × T[w_1]
for all |Σ|^k strings w of length k
```

The input string is then chunked into L/k k-grams, each mapped to its precomputed matrix via table lookup. The prefix scan operates over L/k matrices instead of L.

The precomputation is |Σ|^k independent chains of k-1 matrix multiplications — an embarrassingly parallel workload, ideal for GPU batch GEMM. It can also be computed recursively:

```
T_block[w₁w₂] = T_block[w₂] × T_block[w₁]
```

Building 2-gram tables from 1-gram tables, 4-gram from 2-gram, and so on via log₂(k) rounds of batch GEMM.

### Concrete Numbers

Binary alphabet (|Σ| = 2):

| k | Tables | Precompute matmuls | Precompute memory (binary) | Scan depth reduction |
|---|--------|--------------------|----------------------------|----------------------|
| 8 | 256 | 1,792 | 8 KB | 8× |
| 12 | 4,096 | 45,056 | 128 KB | 12× |
| 16 | 65,536 | 983,040 | 2 MB | 16× |

Byte alphabet (|Σ| = 256):

| k | Tables | Precompute memory (binary) | Precompute memory (int8) | Scan depth reduction |
|---|--------|----------------------------|--------------------------|----------------------|
| 1 | 256 | 8 KB | 64 KB | 1× (baseline) |
| 2 | 65,536 | 2 MB | 16 MB | 2× |
| 3 | 16.7M | 512 MB | 4 GB | 3× (memory-limited) |

For byte input, k = 2 is the practical sweet spot: 65,536 matrices fit in GPU memory, the precomputation is one large batch GEMM (exactly what tensor cores excel at), and scan depth halves.

For binary input, k = 8–16 is practical and gives an order-of-magnitude scan depth reduction.

### Impact on Overall Complexity

| | Without k-gram | With k-gram |
|--|----------------|-------------|
| Scan length | L | L/k |
| Scan depth | O(log L) | O(log(L/k)) |
| Precompute work | — | O(\|Σ\|^k · (k-1) · N³) |
| Precompute depth | — | O(log k) rounds of batch GEMM |
| Total online work | O(L · N³) | O(L/k · N³) |

The precompute cost is paid once per regex. The online cost drops by factor k per string.

---

## 3. Transition Monoid Collapse

### Problem

The prefix scan composes matrices via O(N³) matmul at each step. But the set of all possible products of the DFA's transition matrices is finite and usually small.

### Background

The transition monoid (or syntactic monoid) of a DFA is the set of all distinct products of its per-character transition matrices, closed under composition. For N states:

- Worst case: N^N elements (all functions from N states to N states)
- Typical small regex: 10–50 elements
- Regular languages always have finite syntactic monoids (Myhill-Nerode theorem)

### Solution

Precompute the monoid offline:

```
1. Initialize M = { T[c] : c ∈ Σ }
2. For each pair (A, B) ∈ M × M:
       P = A × B
       if P ∉ M: add P to M
3. Repeat until M is closed under composition
4. Build lookup table: compose[i][j] = monoid index of (element_i ∘ element_j)
```

Now each transition matrix is a small integer (its monoid index), and composition is a table lookup in O(1).

### Impact

The online prefix scan becomes a scan over integers with table-lookup composition:

```
map each input char → monoid index  (array lookup, O(1) per char)
prefix scan with compose[i][j]      (table lookup per step, O(1) per composition)
check final monoid element for acceptance
```

| | Matrix prefix scan | Monoid prefix scan |
|--|--------------------|--------------------|
| Per-step cost | O(N³) matmul | O(1) table lookup |
| Memory per element | N² bytes | 1–2 bytes (index) |
| Scan depth | O(log L) | O(log L) |
| Precompute | — | monoid closure (offline, one-time) |

Tensor cores are used only during monoid construction (many independent matmuls to discover all products). The online scan is trivially fast — no matrix arithmetic at all.

### Example

The DFA for `(a|b)*abb` has 4 states and its transition monoid has ~14 elements. The composition table is 14 × 14 = 196 bytes. The entire online computation reduces to a prefix scan over a 196-byte table.

### Limitations

- Monoid size is hard to predict before computation. Worst case N^N is intractable for N > 5.
- Monoid construction itself can be expensive (many matmuls to reach closure).
- Most useful for small DFAs applied to very long or very many input strings, where the one-time precompute amortizes.
- Does not generalize to NFA path (NFA transition monoid can be exponentially larger).

---

## 4. Hierarchical Two-Level Scan

### Problem

The Hillis-Steele scan does O(log L) rounds of global-memory reads and writes. Each round reads/writes L matrices (or L/k with k-gram), touching HBM at ~2 TB/s.

### Solution

Exploit the GPU memory hierarchy with a two-level scan:

```
Level 1 — Intra-block (shared memory, ~19 TB/s):
    Each thread block loads a chunk of B consecutive matrices into shared memory.
    Compute local prefix products entirely in SMEM.
    Output: one "summary" matrix per block (product of all B matrices in the chunk).

Level 2 — Inter-block (global memory, ~2 TB/s):
    Prefix scan over the L/(k·B) summary matrices in HBM.

Level 1 again — Distribute:
    Each block multiplies its local prefixes by the inter-block prefix from Level 2.
```

This is a standard Brent-Kung two-level pattern, but the key numbers are:

| Encoding | Matrix size | Shared memory (48 KB) fits | Chunk size B |
|----------|-------------|---------------------------|--------------|
| Int8 | 256 B | 192 matrices | ~192 |
| Binary | 32 B | 1,536 matrices | ~1,536 |

With binary encoding, each thread block handles ~1,536 characters entirely in shared memory at ~10× the bandwidth of HBM. Global memory traffic drops to L/1536 matrices per scan round.

### Combined with k-gram

If k-gram precomputation reduces input to L/k matrices, and each block handles B of those in shared memory, global traffic is L/(k·B) matrices per round. For k = 8, binary encoding, B = 1,536:

```
1M character input → 125K k-grams → 82 blocks → ~7 rounds of inter-block scan
Global memory traffic: 82 × 32 B × 7 rounds ≈ 18 KB total (negligible)
```

The scan becomes almost entirely shared-memory-resident.

---

## 5. NFA Path (Eliminating DFA Construction)

### Problem

Subset construction (NFA → DFA) can blow up exponentially. An NFA with n states can produce a DFA with up to 2^n states. For multi-pattern matching (Aho-Corasick + union of regex patterns), n can be thousands, making DFA construction intractable.

### Solution

Skip DFA construction entirely. Simulate the NFA directly via Boolean matrix products:

```
NFA transition matrix T[c] ∈ {0,1}^{n×n}:
    T[c][i][j] = 1  iff  NFA state j transitions to state i on character c

Active states: s ∈ {0,1}^n  (Boolean vector, multiple states active simultaneously)

Advance: s_{t+1} = threshold( T[c_t] × s_t )
         where threshold(v) = min(v, 1) element-wise
```

The prefix scan works identically — matrix products are associative regardless of whether matrices are permutation (DFA) or general Boolean (NFA). The only addition is the per-step Boolean threshold.

### Comparison

| | DFA path | NFA path |
|--|----------|----------|
| Construction | Subset construction (up to 2^n blowup) | Thompson's construction (O(\|pattern\|), trivial) |
| State count | Minimized, often small | Linear in \|pattern\|, larger than DFA |
| Matrix structure | Permutation (one 1 per column) | General Boolean (multiple 1s per column) |
| Arithmetic | Exact in int8 (no overflow) | Needs threshold after each product |
| Matrix size | Usually ≤ 16 for simple regex | Could be 32, 64, or larger |
| Multi-tile needed | Rarely | More often (n > 16) |
| Binary encoding | Natural (values ∈ {0,1}) | Natural (values ∈ {0,1}) |

For n > 16, multi-tile MMA is needed: decompose the n×n matrix into ceil(n/16)² tiles of 16×16 and compute block matrix multiplication. Each tile is one tensor-core MMA.

### NFA + Binary + k-gram

The most GPU-friendly combination for large patterns:

```
Regex → NFA (Thompson's, O(|pattern|))
      → Binary NFA matrices (n×n bits, 32 bytes for n=16)
      → k-gram precompute (batch GEMM on tensor cores)
      → Prefix scan (binary MMA on tensor cores)
      → Threshold and check accept
```

No DFA construction, no exponential blowup, 8× memory reduction from binary, k× scan depth reduction from k-grams.

---

## 6. Batch Packing (Filling the Tensor Core Tile)

### Problem

A single 16×16×16 MMA completes in one cycle, but the tensor core warp-level unit is capable of much larger operands. A single 16×16 multiply underutilizes the hardware.

### Solution

Pack multiple independent matrix multiplications to better fill the hardware:

**Option A — Multiple strings:** For B strings at the same position, pack their state vectors as columns:

```
States ∈ {0,1}^{N×B}  (one column per string)
Advance: States' = T[c] × States  (N×N × N×B, proper-sized GEMM for B ≥ 16)
```

This is the "batched sequential" mode. O(L) depth but massive data parallelism. Best for many short strings.

**Option B — Multiple patterns:** For P patterns each with N_i ≤ 16 states, build a block-diagonal matrix:

```
T_combined = diag(T₁[c], T₂[c], ..., T_P[c])
```

If P patterns with 16 states each → 16P × 16P block-diagonal matrix. Composed via block-diagonal matmul (P independent 16×16 matmuls). Can be packed into larger MMA operations.

**Option C — k-gram precomputation as batch GEMM:** The k-gram precomputation produces |Σ|^k independent matrix multiplications. For |Σ|=256, k=2, that is 65,536 independent 16×16 matmuls — a single call to batched GEMM fills the GPU completely.

---

## 7. Transition Monoid + k-Gram Hybrid

### Insight

The transition monoid and k-gram precomputation solve the same subproblem from different angles:

- k-gram: precompute products for fixed-length substrings (|Σ|^k entries)
- Monoid: precompute all reachable products (M entries, M ≤ N^N)

They can be combined. First compute the monoid (all reachable products). Then, for k-gram precomputation, each k-gram's product is guaranteed to be a monoid element. The k-gram table maps k-grams to monoid indices, and the online scan uses the monoid's O(1) composition table.

```
Offline:
    1. Compute transition monoid M (tensor-core batch GEMM for matrix products)
    2. Build k-gram → monoid index lookup table
    3. Build compose[i][j] table

Online:
    1. Chunk input into k-grams
    2. Map each k-gram to monoid index (array lookup)
    3. Prefix scan with compose table (integer scan, O(1) per step)
    4. Look up final monoid element's accept status
```

This is the fastest possible online phase: O(L/k) table lookups plus an O(log(L/k))-depth integer prefix scan. No matrix arithmetic during matching.

---

## Summary: Compound Improvements

| Optimization | Reduces | Factor | Tensor cores used for |
|---|---|---|---|
| Binary encoding | Memory per matrix | 8× | Online scan (binary MMA) |
| k-gram precomputation | Scan length | k× | Precompute (batch GEMM) |
| Transition monoid | Per-step cost | N³ → O(1) | Monoid construction only |
| Hierarchical scan | HBM traffic | ~B× (chunk size) | Both levels |
| NFA path | Construction cost | eliminates 2^n blowup | Online scan |
| Batch packing | MMA utilization | up to B× | All phases |

### Recommended Configurations

**Config A — Maximum throughput, small patterns (N ≤ 16):**

```
DFA + monoid precompute + k-gram tables + scalar prefix scan
Online: O(L/k) lookups + O(log(L/k)) integer scan
Tensor cores: precompute phase only
```

**Config B — General patterns, arbitrary size:**

```
NFA + binary matrices + k-gram precompute + hierarchical binary-MMA scan
Online: O(L/k · log(L/k)) binary MMAs in O(log(L/k)) depth
Tensor cores: precompute + online scan
```

**Config C — Multi-pattern matching (IDS/DPI):**

```
NFA union + block-diagonal binary matrices + batch packing + hierarchical scan
Tensor cores: online scan with high utilization from pattern packing
```

---

## Implementation Priority

Ordered by impact-to-effort ratio:

1. **k-gram precomputation** — highest standalone impact, straightforward batch GEMM, benefits all configurations
2. **Binary encoding** — 8× memory improvement, requires WMMA b1 kernel variant
3. **Hierarchical scan** — significant for large L, standard two-level Brent-Kung, well-understood
4. **NFA path** — eliminates DFA construction, needs Boolean threshold in scan, enables large patterns
5. **Transition monoid** — maximum online speed, but monoid computation can be expensive and does not generalize to NFA
6. **Batch packing** — improves utilization, most impactful for multi-string and multi-pattern workloads

---

## Code Changes Required

| Component | File | Change |
|---|---|---|
| Binary matrix class | `simulation.py` | `BinaryDFAMatrices`: store as packed `uint16` arrays, binary matmul via numpy bitwise ops |
| k-gram precomputation | `simulation.py` | `precompute_kgrams(matrices, k)` → lookup table of composed matrices |
| Transition monoid | `simulation.py` | `compute_monoid(matrices)` → monoid elements + composition table |
| Binary MMA kernel | `tensor_core_dfa_scan.cu` | WMMA `precision::b1` variant of scan kernels |
| Batch GEMM kernel | `tensor_core_dfa_scan.cu` | Kernel for k-gram precomputation (|Σ|^k independent matmuls) |
| Hierarchical scan | `tensor_core_dfa_scan.cu` | Two-level Brent-Kung: shared-memory intra-block + global inter-block |
| NFA matrix builder | `regex_to_dfa.py` | Export NFA transition matrices directly (skip subset construction) |
| NFA simulation backend | `simulation.py` | Prefix scan with per-step Boolean threshold |
