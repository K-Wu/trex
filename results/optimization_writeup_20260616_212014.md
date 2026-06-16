# TERX Optimization Report

**Date:** 2026-06-16
**Hardware:** NVIDIA H200 NVL (SM 9.0, 132 SMs, 4,917 GB/s peak BW, 3,958 INT8 TOPS)
**Pattern set:** Binary-alphabet DFAs with 2–5 states

---

## 1. The Problem with Tensor-Core DFA Simulation

The v4 baseline encodes DFA transitions as 16×16 int8 permutation matrices and uses WMMA `mma_sync` to compose them via parallel prefix scan. The appeal is clear: tensor cores deliver massive throughput on matrix operations.

The reality: **0.53% tensor core utilization.**

Each step in the prefix scan depends on the previous result: `acc = T[c_i] × acc`. The MMA pipeline stalls after every `mma_sync` waiting for the accumulator. No amount of occupancy (we achieve 100%) or warp interleaving can break this serial dependency chain. A batch GEMM of independent matmuls would saturate the tensor cores; a sequential scan fundamentally cannot.

The 16×16 matrices are also 93.75% zeros (permutation matrices have exactly one 1 per column), so 15/16 of every MMA's multiply-adds are wasted on zero × zero = zero.

| Metric | V4 MMA R1 (best) | H200 INT8 Peak | Utilization |
|--------|-----------------|----------------|-------------|
| Compute | 21 TFLOP/s | 3,958 TOPS | 0.53% |
| Bandwidth | 10.6 GB/s | 4,917 GB/s | 0.22% |

This is the starting point. Every optimization below is measured against this baseline.

---

## 2. The Optimizations

### 2.1 Transition Monoid

**Concept:** The set of all products of DFA transition matrices is finite (the transition monoid). For a DFA with N states, the monoid has at most N^N elements — but in practice far fewer: 2–14 elements for our patterns. Precompute all products, assign each an integer index, and build a composition table: `compose[i][j] = index of (element_i ∘ element_j)`.

At scan time, replace O(N³) matmul with O(1) table lookup.

**Preprocessing cost:**

| Pattern | DFA States | Monoid Size | Monoid Compute Time |
|---------|-----------|-------------|-------------------|
| abb | 4 | 7 | 0.55 ms |
| even_a | 2 | 2 | 0.06 ms |
| binary_div3 | 3 | 6 | 0.42 ms |
| ab_star | 3 | 6 | 0.43 ms |
| hex_number | 5 | 8 | 0.76 ms |
| identifier | 3 | 3 | 0.13 ms |

The monoid BFS closure is cheap: under 1 ms for all tested patterns. Cost scales with N^N worst-case but is bounded by the `max_size=65536` cap.

**GPU kernel throughput:**

| Kernel | Config | Gchar/s | vs V4 MMA |
|--------|--------|---------|-----------|
| Monoid R1 | B=100K, L=128 | 93.0 | **38x** |
| Monoid R1 | B=10K, L=512 | 81.2 | **31x** |
| Monoid R3 | L=16M | 18.7 | **20x** |
| V4 MMA R1 | B=100K, L=128 | 2.4 | 1x |
| V4 MMA R3 | L=16M | 0.95 | 1x |

**Why it works:** Replaces a compute-bound kernel (sequential dependent MMA at 0.5% utilization) with a memory-bound kernel (integer table lookup). The compose table for M=7 is 98 bytes — fits in L1 cache. Each character requires one `uint16_t` load + one shared memory lookup, versus one 16×16 matmul.

**Resource comparison:**

| Resource | Monoid | V4 MMA |
|----------|--------|--------|
| Registers/thread | 25 | 32 |
| Shared mem/block | 8 B | 7,168 B |
| Input per char | 2 B (uint16) | 4 B (int) |
| Work per char | 1 table lookup | 8,192 FLOPs |

**Inflection point:** The monoid approach degrades when M grows large. The compose table is M² × 2 bytes in shared memory, capped at 48 KB → M ≤ 155. For M > ~1000, the table no longer fits in L1 cache and lookup latency increases. The `max_size=65536` cap ensures we fall back to matrix scan before this becomes pathological.

**Verdict:** Pure win for small DFAs (N ≤ ~10, M ≤ ~200). Essentially all "interesting" regex patterns for binary/small alphabets fall in this range.

---

### 2.2 k-Gram Precomputation

**Concept:** Precompute the composed monoid element (or matrix) for every possible k-character substring. At scan time, chunk the input into L/k blocks and look up each block in O(1). Reduces scan length by factor k.

**k selection:** `k = max k such that |Σ|^k ≤ 65536`.

| Alphabet Size | k | Table Entries | Table Size (monoid) |
|--------------|---|---------------|-------------------|
| 2 (binary) | 16 | 65,536 | 128 KB |
| 4 | 8 | 65,536 | 128 KB |
| 16 | 4 | 65,536 | 128 KB |
| 36 (identifier) | 3 | 46,656 | 91 KB |
| 256 (byte) | 2 | 65,536 | 128 KB |

**Preprocessing cost — the critical tradeoff:**

| k | Table Entries | Precompute Time | Scan Reduction |
|---|--------------|-----------------|----------------|
| 2 | 4 | 0.01 ms | 2x |
| 4 | 16 | 0.01 ms | 4x |
| 8 | 256 | 0.23 ms | 8x |
| 12 | 4,096 | 5.5 ms | 12x |
| 16 | 65,536 | **115 ms** | 16x |

k=16 for binary alphabet takes **115 ms** — this is the single most expensive preprocessing step and dominates the entire monoid pipeline (115 ms out of ~120 ms total CPU precompute).

**CPU-side throughput impact:**

| Backend | Mchar/s (CPU, Python) |
|---------|----------------------|
| Sequential DFA | 20.8 |
| Monoid (per-char) | 12.0 |
| Monoid + k-gram (k=16) | 22.2 |

Per-character monoid is *slower* than sequential in Python because each `compose_table[i, j]` numpy indexing call has Python overhead that exceeds a simple dict lookup. The k-gram precomputation amortizes this: 16 characters resolved per lookup.

**GPU-side impact:** On GPU, the monoid kernel already processes raw monoid indices without k-gram — the k-gram table is consumed during the Python `_prepare_batch` char-mapping step. k-gram reduces the number of elements sent to the GPU kernel, which helps when transfer time dominates, but the kernel is already so fast that this rarely matters.

**Inflection point:** k=16 precompute (115 ms) is amortized over all queries to the compiled engine. Break-even versus k=8 (0.23 ms precompute, 8x scan reduction instead of 16x):

- For GPU path: kernel time is negligible vs preprocessing, so k=8 is almost always better
- For CPU path: k=16 needs ~5.2M characters processed at 22 Mchar/s to recoup the 115 ms investment over k=8's 21 Mchar/s

**Verdict:** k-gram compounds freely with monoid. For CPU simulation, k=16 provides measurable speedup. For GPU simulation, the benefit is marginal because kernel time is already dominated by preprocessing. **k=8 is the sweet spot** for binary alphabet when amortization is uncertain.

---

### 2.3 NFA Path

**Concept:** Skip DFA construction entirely. Build transition matrices directly from the NFA, where N = O(|pattern|). The matrices are Boolean (multiple 1s per column, representing nondeterminism). Simulation uses matrix-vector products with Boolean thresholding: `state = min(T[c] @ state, 1)`.

**When it matters:** DFA construction can blow up exponentially. A pattern like `.*a.{20}b` produces an NFA with ~25 states but a DFA with potentially millions. The NFA path caps state count at O(|pattern|) regardless.

**Preprocessing cost:**

| Pattern | NFA States (raw) | NFA States (padded) | NFA Build Time |
|---------|-----------------|--------------------|----|
| abb | 11 | 16 | 0.02 ms |
| even_a | 6 | 16 | 0.03 ms |
| hex_number | 88 | 96 | 0.14 ms |
| identifier | 144 | 144 | 0.86 ms |

NFA matrix construction is fast: no subset construction, no minimization. For patterns where DFA blows up, the NFA path goes from "impossible" to sub-millisecond.

**Tradeoff vs DFA:**

| Aspect | DFA Path | NFA Path |
|--------|----------|----------|
| Construction | Exponential worst-case | O(\|pattern\|) always |
| Matrix size | N×N (N = DFA states, small) | n×n (n = NFA states, larger) |
| Matrix density | Permutation (1 per column) | Boolean (multiple 1s) |
| Monoid applicable? | Yes (small M) | Rarely (M ≈ n^n, too large) |
| Scan cost | O(N³) per char (MMA) | O(n³) per char (MMA) |

**Inflection point:** The NFA path is slower per-character (larger matrices) but makes infeasible patterns tractable. The auto-selector triggers NFA when DFA states exceed 64.

**Verdict:** Not a performance optimization — it's a **feasibility** optimization. It extends the regex space TERX can handle from "small DFAs" to "arbitrary patterns."

---

## 3. End-to-End Pipeline Costs

### 3.1 Full Preprocessing Breakdown

For the `abb` pattern (representative binary-alphabet DFA):

| Step | Time (ms) | Cumulative |
|------|-----------|------------|
| regex → DFA | 0.04 | 0.04 |
| DFA → matrices | 0.01 | 0.05 |
| Monoid BFS | 0.55 | 0.60 |
| k-gram (k=16) | **115** | 116 |
| GPU monoid engine init | **0.5–156** ¹ | 116–272 |
| **Total** | | **116–272 ms** |

¹ First call to `create_engine` in a process includes CUDA context initialization (~155 ms). Subsequent calls: ~0.5 ms.

For comparison, v4 baseline:

| Step | Time (ms) |
|------|-----------|
| regex → DFA | 0.04 |
| DFA → matrices | 0.01 |
| GPU v4 engine init | **0.5–149** ¹ |
| **Total** | **0.5–149 ms** |

¹ Same CUDA context init penalty on first call.

**The k-gram k=16 table is the bottleneck.** At 115 ms, it's 200x more expensive than the monoid computation itself. With k=8 (0.23 ms), total preprocessing drops to ~1 ms (excluding first-time CUDA init).

### 3.2 Python char-mapping overhead

Before the GPU kernel runs, Python must map each character to its monoid index via `md.char_to_monoid`. This is a Python dict lookup per character:

| Chars | Mapping Time | Rate |
|-------|-------------|------|
| 1M | 88 ms | 11.4 Mchar/s |

This is a significant bottleneck: mapping 10M characters takes **880 ms** in Python, while the GPU kernel processes them in **1.2 ms**. The char-mapping step is **700x slower** than the kernel.

For production use, this mapping should be pushed to C/Cython or done on-GPU.

### 3.3 Amortization

Preprocessing is one-time per compiled regex. Over repeated queries:

| Queries × 10M chars | Monoid Total | V4 Total | Winner |
|---------------------|-------------|----------|--------|
| 1 × 10M | 122 ms | 156 ms | Monoid |
| 10 × 10M | 133 ms | 213 ms | Monoid |
| 100 × 10M | 243 ms | 787 ms | Monoid |
| 1000 × 10M | 1,349 ms | 6,527 ms | Monoid |

Monoid wins from the very first query because even with the k=16 preprocessing overhead, the kernel is 14.5x faster. The crossover point doesn't exist — monoid dominates at all batch counts.

With k=8 instead of k=16, preprocessing drops from ~120 ms to ~1 ms, and the kernel speedup is identical (k-gram only affects CPU-side batching, not the GPU kernel). This makes monoid strictly dominant.

---

## 4. Composability Matrix

| | Monoid | k-gram | NFA | Binary Encoding |
|---|---|---|---|---|
| **Monoid** | — | ✅ Stacks (k-gram → monoid index) | ❌ Alt (monoid too large for NFA) | ❌ N/A (no matrices) |
| **k-gram** | ✅ | — | ✅ Stacks (k-gram → matrix) | ✅ Stacks |
| **NFA** | ❌ | ✅ | — | ✅ Stacks (binary NFA matrices) |
| **Binary** | ❌ | ✅ | ✅ | — |

**Practical configurations:**

| Config | When | Preprocessing | Kernel Throughput |
|--------|------|---------------|-------------------|
| DFA + Monoid | Small DFA, small monoid | ~1 ms | 93 Gchar/s |
| DFA + Monoid + k-gram(k=8) | Same, want CPU speedup | ~1 ms | 93 Gchar/s |
| DFA + Monoid + k-gram(k=16) | Same, heavy CPU use | ~120 ms | 93 Gchar/s |
| DFA + Matrix scan (v4) | Large monoid | ~0.5 ms | 2.6 Gchar/s |
| NFA + Matrix scan | DFA blows up | ~1 ms | Depends on n |

---

## 5. Summary of Findings

**What works:**
1. **Transition monoid** is the highest-impact optimization. 20–38x GPU kernel speedup by replacing serial-dependent MMA (0.5% tensor utilization) with O(1) table lookup. Preprocessing is sub-millisecond.
2. **NFA path** makes previously infeasible patterns tractable. Not faster per-character, but enables the system to handle arbitrary regex.
3. **Auto-selection** (DFA cap → monoid cap → fallback) correctly routes patterns to the best backend with zero user configuration.

**What's marginal:**
1. **k-gram with k=16** costs 115 ms to precompute for a ~10% CPU throughput gain. k=8 gives 8x scan reduction for 0.23 ms precompute — better ROI in almost all scenarios.
2. **k-gram on GPU path** provides negligible benefit since the char-mapping Python overhead (88 ms/Mchar) dwarfs the kernel time.

**What limits further gains:**
1. **Python char-mapping** at 11 Mchar/s is the actual bottleneck for GPU-accelerated workloads. The GPU kernel is 700x faster than the Python preprocessing that feeds it.
2. **Monoid R1 thread utilization**: only lane 0 of each warp is active (96.9% of threads idle). A warp-cooperative scan could improve throughput further.
3. **Monoid R3 look-back**: sequential tile chaining limits scaling. Hierarchical or tree-based reduction could improve long-string throughput.

**The fundamental insight:** For small DFAs over small alphabets, the transition monoid collapses the per-character cost from O(N³) compute to O(1) memory access. This transforms the problem from one that fundamentally cannot use tensor cores (serial dependency chain → 0.5% utilization) to one that doesn't need them (simple integer scan → memory-bandwidth limited). The H200's 3,958 INT8 TOPS are irrelevant; its memory subsystem matters.
