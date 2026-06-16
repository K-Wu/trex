# Architectural Reframing: From Matrix Prefix Scan to Batched State Evolution

## Abstract

The naive approach to tensor-core-accelerated regex matching — parallel prefix scan over N×N DFA transition matrices using int8 WMMA — achieves poor hardware utilization because the matrices are fundamentally small (N ≤ 16). Each 16×16 MMA instruction completes in ~1 cycle of compute but incurs ~100 cycles of fragment load/store and synchronization overhead. No optimization to the prefix scan (k-gram precomputation, binary encoding, hierarchical decomposition) changes this because they all still produce 16×16 matmuls.

This document proposes a fundamental reframing: replace the matrix-composition prefix scan with batched state-vector evolution via character-grouped GEMMs. The key operation changes from N×N × N×N (tiny, low utilization) to N×N × N×B (large, high utilization), where B is the number of strings processed simultaneously. This trades O(log L) depth for O(L) depth but transforms each step into a properly-sized GEMM that saturates the tensor core pipeline.

We analyze the throughput implications, describe three concrete configurations (batched evolution, monoid scalar scan, pattern packing), and show that the batched approach can reach HBM bandwidth limits (~149 GB/s on A100) for realistic workloads.

---

## 1. The Utilization Problem

### 1.1 Why 16×16 MMA Is Wasteful

A tensor core MMA instruction performs a 16×16×16 fused multiply-accumulate in a single cycle. However, each invocation requires:

- Loading WMMA fragments from shared/global memory into registers (~20–40 cycles)
- The MMA instruction itself (~1 cycle)
- Storing the accumulator fragment back to memory (~20–40 cycles)
- Warp synchronization barriers (~5–10 cycles)

For a single 16×16×16 int8 MMA, the arithmetic intensity is 8,192 int8 ops in ~1 cycle of compute wrapped in ~100 cycles of overhead. Effective utilization: roughly 1%.

cuBLAS achieves high utilization by tiling large GEMMs (M, N, K ≥ 128) so that the tensor core pipeline is continuously fed without stalls. For a 4096×4096×4096 GEMM, the ratio inverts: ~99% compute, ~1% overhead.

### 1.2 Why the Prefix Scan Cannot Fix This

The parallel prefix scan over L transition matrices has O(log L) depth. At step s, it performs (L − 2^s) independent N×N × N×N matmuls. For L = 1M, the first step has 500K independent 16×16 matmuls — the GPU is fully occupied, but each individual MMA instruction is still a 16×16 tile with 1% utilization.

The batch of 500K independent matmuls could theoretically be dispatched as a cuBLAS strided batched GEMM, which pipelines better than hand-written WMMA. But each matmul is still 16×16, and cuBLAS cannot fuse independent small matmuls into a single large one because the operands differ.

Every optimization in the current framework inherits this problem:

| Optimization | Effect on matrix size | Utilization improvement |
|---|---|---|
| k-gram precomputation | Same N×N, fewer of them | None per-matmul |
| Binary (1-bit) encoding | Same N×N, 8× less memory | None per-matmul (memory-bound improvement only) |
| Hierarchical scan | Same N×N, better locality | None per-matmul |
| Transition monoid | Eliminates matmuls entirely | N/A (no tensor cores in online phase) |

The matrices are N×N because N is the DFA state count. No algebraic transformation changes this: the prefix scan fundamentally composes N×N objects.

### 1.3 The Wrong Question and the Right Question

Wrong question: *How do we prefix-scan N×N matrices faster on tensor cores?*

Right question: *How do we reshape the workload so each tensor-core operation involves large matrices?*

---

## 2. The Reframing: Batched State-Vector Evolution

### 2.1 From Matrix Composition to State Evolution

The current pipeline separates matrix composition from state-vector application:

```
Step 1 (prefix scan):   P = T[c_{L-1}] × T[c_{L-2}] × ... × T[c_0]     → N×N product
Step 2 (apply):          s_final = P × s_0                                → N×1 result
```

Step 1 requires O(L) matmuls of size N×N × N×N. Step 2 is a single matvec.

The reframing fuses these steps and processes B strings simultaneously:

```
State matrix:   S ∈ {0,1}^{N × B}       (column j = state vector of string j)
Initialize:     S[:, j] = e_{start}      (one-hot start state for all strings)

For t = 0, 1, ..., L-1:
    S ← T[c_t] × S                       (N×N × N×B  GEMM)
```

Each step is a matrix-matrix multiply where the right operand has B columns. For B ≥ 256, this is a properly-sized GEMM that achieves high tensor core utilization.

### 2.2 Handling Per-String Character Variation

Different strings have different characters at position t, so T[c_t] varies across the B columns. The solution is character-grouped GEMMs:

```
For each position t = 0, 1, ..., L-1:
    For each character c ∈ Σ:
        group_c ← { j : string_j[t] = c }
        S[:, group_c] ← T[c] × S[:, group_c]          ← one GEMM per character
```

At each position, this performs |Σ| GEMMs. The j-th GEMM operates on the subset of strings whose character at position t equals c. The expected group size is B / |Σ|.

For B = 10,000 strings over |Σ| = 2 (binary alphabet): each GEMM is 16×16 × 16×5,000. The output matrix is 16 × 5,000 — a 10,000× increase in useful work per GEMM compared to the prefix-scan approach.

For B = 65,536 strings over |Σ| = 256 (byte alphabet): each GEMM is 16×16 × 16×256. Still 256× more work per GEMM than the prefix scan.

### 2.3 Complexity Comparison

| | Prefix scan | Batched state evolution |
|---|---|---|
| Depth | O(log L) | O(L) |
| Work per step | N³ (N×N × N×N matmul) | N² · B/\|Σ\| (N×N × N×(B/\|Σ\|) GEMM, ×\|Σ\| groups) |
| GEMM dimensions | N × N × N | N × N × (B/\|Σ\|) |
| Total work | O(L · N³ · log L) | O(L · N² · B) |
| Tensor core utilization | ~1% (16×16 tiles) | High (large B dimension) |
| Strings per invocation | 1 (apply to state vec after scan) | B (all simultaneous) |

The batched approach does more total arithmetic (factor B/N), but each FLOP maps to actual tensor core utilization. The effective throughput is higher because the hardware is no longer stalled on overhead.

### 2.4 Implementation: Grouped Gather-Advance Kernel

The inner loop can be implemented as a single persistent CUDA kernel:

```
__global__ void grouped_advance_kernel(
    const int8_t *trans_matrices,    // [|Σ|, N, N]  transition matrices
    int8_t *states,                  // [N, B]        state matrix (modified in place)
    const uint8_t *input_strings,    // [B, L]        input strings (row-major)
    int L, int B, int N, int sigma
) {
    // Each warp group handles one character class per position
    for (int t = 0; t < L; t++) {
        for (int c = 0; c < sigma; c++) {
            // Identify columns where input_strings[j][t] == c
            // Perform GEMM: T[c] × S[:, group_c]
            // This maps to cuBLAS or hand-tuned WMMA with large N×B tiles
        }
        __syncthreads();  // all characters processed before next position
    }
}
```

In practice, the character grouping and GEMM dispatch would use cuBLAS `cublasGemmStridedBatched` or a custom kernel that:

1. Loads T[c] into shared memory (only |Σ| matrices, each N×N — fits trivially)
2. For each position t, scatters column indices by character
3. Performs the grouped GEMM using tensor cores with the large B dimension

The shared-memory footprint is small: |Σ| × N² bytes for transition matrices (e.g., 2 × 256 = 512 bytes for binary alphabet, 16-state DFA) plus N × B bytes for the state tile currently in flight.

---

## 3. Throughput Analysis

### 3.1 Roofline Model

On A100 (624 int8 TOPS, 2 TB/s HBM2e bandwidth):

**Compute bound:**
Per position: |Σ| GEMMs, each N×N × N×(B/|Σ|) = 2N² · B/|Σ| int8 ops per GEMM, times |Σ| = 2N²B total ops.
For N = 16, B = 65536: 2 × 256 × 65536 = 33.6M ops per position.
Time per position: 33.6M / 624T = 0.054 μs.

**Memory bound:**
Per position: read B bytes of input characters + read/write N × B bytes of state matrix = B(1 + 2N) bytes.
For N = 16, B = 65536: 65536 × 33 = 2.1 MB per position.
Time per position: 2.1 MB / 2 TB/s = 1.05 μs.

The kernel is **memory-bandwidth-limited** — the tensor cores complete in 0.054 μs but must wait 1.05 μs for data. This means the tensor cores are working and the bottleneck has shifted from "too little compute per MMA" to "memory bandwidth", which is the correct regime for a well-optimized GPU kernel.

### 3.2 End-to-End Throughput Projections

| Workload | B | L | |Σ| | N | Compute (μs) | Memory (μs) | Bottleneck | Throughput |
|---|---|---|---|---|---|---|---|---|
| Binary regex, 64K strings | 65,536 | 1M | 2 | 16 | 0.054M | 1.05M | Memory | 62 GB/s |
| Byte regex, 64K strings | 65,536 | 1M | 256 | 16 | 0.054M | 1.05M | Memory | 62 GB/s |
| Binary regex, 4K strings | 4,096 | 1M | 2 | 16 | 0.0034M | 0.066M | Memory | 62 GB/s |
| Small batch, short strings | 256 | 1K | 2 | 16 | 3.4 | 66 | Memory | 3.9 GB/s |

The throughput is approximately constant at B × L / memory_time ≈ HBM_bandwidth / (1 + 2N), regardless of batch size, because the bottleneck is reading/writing the state matrix. For N = 16: ~2 TB/s / 33 ≈ 62 GB/s of input throughput.

### 3.3 Comparison to Prefix Scan

The prefix-scan approach on the same hardware:

Per scan step: ~L/2 independent 16×16 matmuls. Each matmul takes ~100 cycles overhead for ~1 cycle compute. Effective throughput: ~1% of peak.

```
Prefix scan:           624 TOPS × 1% utilization = 6.24 TOPS effective
Batched state evo:     624 TOPS, memory-limited at ~62 GB/s input throughput
```

For 64K strings of 1M characters (64 GB input):

```
Prefix scan:           would need to scan each string separately (or batch the scan)
                       64K × 20 scan steps × ~100 μs per step ≈ 128 seconds
Batched state evo:     64 GB / 62 GB/s ≈ 1.03 seconds
```

The batched approach is roughly two orders of magnitude faster for throughput workloads, because it solves the utilization problem at the root.

---

## 4. Three Configurations

### 4.1 Configuration A: Batched State Evolution (Throughput)

**Use case:** Match many strings against one regex (grep, IDS, log scanning, database LIKE).

```
Offline:   regex → DFA → transition matrices T[c] (CPU, microseconds)
Online:    grouped character-advance GEMMs for B strings simultaneously

Depth:     O(L)
GEMM size: N × N × B/|Σ|
Utilization: high (memory-bandwidth-limited)
```

Tensor cores are used in the online phase, doing large GEMMs. The O(L) depth is acceptable because the wall-clock time is dominated by memory bandwidth, not arithmetic depth.

### 4.2 Configuration B: Monoid + Scalar Scan (Latency)

**Use case:** Match one string against one regex with minimum latency.

```
Offline:   regex → DFA → transition monoid M (tensor-core batch GEMM for closure)
           build k-gram → monoid index lookup table
           build compose[i][j] multiplication table

Online:    chunk input into k-grams → lookup monoid indices → scalar prefix scan
           O(1) per composition via table lookup

Depth:     O(log(L/k))
GEMM size: N/A online (tensor cores used only in offline monoid construction)
Utilization: N/A online (trivial scalar ops)
```

Tensor cores are used once during preprocessing to compute the monoid (thousands of independent matmuls — a proper batch GEMM). The online phase is a scalar scan with table lookup, running at memory bandwidth on CUDA cores.

### 4.3 Configuration C: Pattern Packing + Batched Evolution (Maximum)

**Use case:** Match many strings against many regex patterns simultaneously (multi-rule IDS, complex log parsing).

```
Offline:   P regex patterns → P DFAs → block-diagonal transition matrices
           T_combined[c] = diag(T₁[c], T₂[c], ..., T_P[c])    size (NP) × (NP)

Online:    batched state evolution with B strings
           GEMM size: (NP) × (NP) × (NP) × B/|Σ|

Depth:     O(L)
GEMM size: (NP) × (NP) × B/|Σ|
Utilization: maximum (both matrix dimensions and batch dimension are large)
```

For P = 16 patterns with N = 16 states each: NP = 256. The GEMM is 256 × 256 × (B/|Σ|). For B = 4096, |Σ| = 256: GEMM is 256 × 256 × 16. For |Σ| = 2: GEMM is 256 × 256 × 2048. These are textbook tensor-core-friendly dimensions.

The block-diagonal structure means the off-diagonal blocks of the product are zero (patterns don't interact). An optimized kernel can skip these blocks, but even a dense cuBLAS GEMM achieves high utilization at 256×256 tile size. The extra zero-block computation is wasted but is a small constant factor.

---

## 5. Implementation Roadmap

### 5.1 Data Layout

```
Transition matrices:   T[c] ∈ int8[N][N]          column-major for WMMA compatibility
                       stored contiguously: int8[|Σ|][N][N]

Input strings:         int8[B][L]                  row-major (string-contiguous)
                       or int8[L][B]               column-major (position-contiguous)

State matrix:          int8[N][B]                  column-major (one column per string)
                       padded to 16-column alignment
```

Position-contiguous layout (int8[L][B]) is preferred: at each position t, reading all B characters is a contiguous memory access, enabling coalesced loads for the character grouping step.

### 5.2 Character Grouping Strategy

Two approaches to the character-grouped GEMM dispatch:

**Approach A — Scatter-gather:**
At each position t, sort/partition the B column indices by character value. Build a gather index for each character class. Execute one GEMM per class using indirect indexing.

Cost: O(B) work per position for the partition. Amortizable by precomputing the character-group indices for all positions offline (O(B × L) memory, but only O(B) accessed per step).

**Approach B — Masked GEMM:**
Instead of physically gathering columns, use a mask to zero out columns that don't belong to the current character class. Execute one GEMM for each character, but only the masked columns contribute.

Cost: no scatter/gather overhead, but wasted compute on zeroed columns. For |Σ| = 2, 50% waste (each GEMM processes B columns but half are zeroed). For |Σ| = 256, 99.6% waste — unacceptable.

**Approach C — Pre-sorted batch (recommended for small |Σ|):**
If all strings are known upfront, pre-sort them so that at each position t, strings with the same character are contiguous. This is possible only if the sort order is consistent across positions, which it generally isn't. However, for binary alphabet, a simple bit at position t partitions the batch into two contiguous halves — a radix-sort-like decomposition.

**Recommendation:** Approach A (scatter-gather) for general alphabets. Precompute character-group indices offline for the entire input (one pass over the input, O(B × L) memory for the index arrays). At runtime, each GEMM uses the precomputed column indices for contiguous memory access.

### 5.3 Kernel Structure

```
Phase 1 — Precompute (CPU, once):
    Compile regex → DFA → transition matrices
    Precompute character-group indices for input batch

Phase 2 — Upload (H2D, once):
    Copy transition matrices to device constant memory (tiny: |Σ| × N × N bytes)
    Copy input strings to device global memory (B × L bytes)
    Copy character-group indices to device global memory
    Allocate state matrix on device (N × B bytes)

Phase 3 — Execute (GPU):
    Initialize state matrix: S[:, j] = one-hot start state
    For t = 0, 1, ..., L-1:
        For each character c:
            columns = group_indices[t][c]
            S[:, columns] = T[c] × S[:, columns]     // cuBLAS GEMM or custom kernel
    
Phase 4 — Readback (D2H):
    Copy state matrix to host
    Check S[accept_states, j] for each string j
```

### 5.4 Optimization Opportunities Within This Framework

**Shared-memory caching of T[c]:** All |Σ| transition matrices fit in shared memory (|Σ| × N² bytes). Load once per thread block, reuse across all positions.

**Double buffering:** While processing position t, prefetch the character-group indices for position t+1.

**Fused acceptance check:** Instead of reading back the full state matrix, fuse a reduction kernel that checks the accept mask and outputs a boolean vector of length B.

**Variable-length strings:** Strings shorter than L can be handled by masking: once a string terminates, its column is frozen (not updated). The GEMM still operates on the full batch, but terminated strings contribute zero useful work. For batches with similar string lengths, the waste is small.

**Multi-tile for large N:** If the DFA/NFA has N > 16 states, the N×N transition matrix spans multiple 16×16 tiles. The GEMM is still well-formed — cuBLAS handles multi-tile automatically. Utilization improves further because each GEMM is even larger.

---

## 6. When Each Approach Wins

### 6.1 Decision Matrix

| Scenario | Recommended approach | Why |
|---|---|---|
| Many strings, one pattern | Batched state evolution (§4.1) | Large B dimension fills tensor cores |
| One string, latency-critical | Monoid + scalar scan (§4.2) | Eliminates matrix arithmetic from online phase entirely |
| Many strings, many patterns | Pattern packing + batched evolution (§4.3) | Both N and B dimensions are large |
| One string, one pattern, throughput irrelevant | CPU sequential or prefix scan | GPU overhead not justified |
| Streaming input (strings arrive online) | Batched evolution with rolling batch | Accumulate strings into batches, flush when batch is full |

### 6.2 Crossover Points

**Batched evolution vs. CPU sequential (single-core):**
CPU processes one string at ~1 byte/ns (sequential DFA lookup). For B strings of length L: CPU time ≈ B × L ns.
GPU batched evolution: ~L × 1.05 μs (memory-limited, independent of B for large B).
Crossover: B × L ns > L × 1.05 μs → B > 1,050. For batches larger than ~1,000 strings, the GPU wins.

**Batched evolution vs. prefix scan on GPU:**
Prefix scan: O(log L) depth, ~1% utilization, dominated by per-MMA overhead.
Batched evolution: O(L) depth, ~100% utilization (memory-limited).
For the total time to process B strings:
- Prefix scan: B × log₂(L) × overhead_per_mma ≈ B × 20 × 100 cycles at 1.4 GHz ≈ B × 1.43 μs (for L = 1M)
- Batched: L × 1.05 μs = 1.05s (for L = 1M, B-independent)
Crossover: B × 1.43 μs > 1.05s → B > 734K. For fewer than ~700K strings, the prefix scan (run B times) is faster.

This crossover is high, meaning for moderate batch sizes the per-string prefix scan has lower total latency. But the prefix-scan number assumes optimistic 100-cycle overhead — in practice, with kernel launch overhead and memory stalls, the crossover is much lower (~1K–10K strings).

---

## 7. Relationship to Prior Work

### 7.1 State Space Models (Mamba, S4)

The DFA recurrence h_t = T[c_t] × h_{t-1} is structurally identical to a linear state space model with input-dependent transition matrices. The Mamba architecture faces the same sequential bottleneck and solves it with a parallel associative scan. However, Mamba's key optimization (diagonal state matrices, enabling element-wise parallel scan) does not apply here because DFA transition matrices are full rank.

The batched state-evolution approach is analogous to how Mamba processes multiple sequences in a batch: the sequential recurrence is unavoidable per-sequence, but batching across sequences provides the parallelism that fills the hardware.

### 7.2 GPU Automata Processing (ngAP, AsyncAP)

The state-of-the-art GPU automata engines (ngAP, ASPLOS 2024; AsyncAP, SIGMETRICS 2023) process automata using CUDA-core worklist algorithms with input-level parallelism (multiple input symbols processed concurrently). They achieve high throughput by parallelizing across input positions and active states.

The batched state-evolution approach is complementary: instead of parallelizing across positions within one string, it parallelizes across strings. For workloads with many independent strings (network packet inspection, log scanning), the batch dimension provides more parallelism than the position dimension.

### 7.3 Boolean SpMV / SpMM Approaches (TACO 2025)

The TACO 2025 "AutomataBLAS" approach formulates NFA simulation as sparse matrix–vector multiplication on CUDA cores. The batched state-evolution approach can be viewed as a dense variant: instead of SpMV (sparse T × dense vector s), it performs dense MM (dense T × dense matrix S). For small N (≤ 16), the transition matrices are small enough that dense multiplication is competitive with sparse, and the dense formulation maps directly to tensor cores.

---

## 8. Summary

The core insight is a change in what is multiplied, not how:

| | Prefix scan | Batched state evolution |
|---|---|---|
| Operation | N×N × N×N | N×N × N×B |
| What grows | Length L (scan depth) | Batch B (GEMM width) |
| Tensor core regime | Overhead-dominated (tiny tiles) | Bandwidth-dominated (large GEMM) |
| Utilization | ~1% | Limited by HBM bandwidth |
| Depth | O(log L) | O(L) |
| Best for | Single-string latency (but use monoid instead) | Multi-string throughput |

The prefix scan is the theoretically elegant solution (O(log L) depth), but it is architecturally mismatched to tensor cores because it produces irreducibly small matrix operands. The batched state-evolution formulation abandons logarithmic depth in exchange for large GEMM operands that properly utilize the hardware. For throughput workloads — which dominate real-world regex matching applications — this is the correct trade.

For latency workloads, the monoid + scalar scan approach is superior to both: it precomputes all matrix products offline (using tensor-core batch GEMM for the closure computation) and reduces the online phase to a trivial integer scan with O(1) composition via table lookup.

The optimal system implements all three and dispatches based on workload characteristics: batched evolution for throughput, monoid scan for latency, pattern packing when matching multiple regex simultaneously.
