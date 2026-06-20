# TERX Kernel Variants — Problem Framing, Performance, and Selection Guide

## The Core Problem

Every kernel in this repo solves the same problem: **run a finite-state automaton over batches of strings on GPU**. Given an automaton with N states, alphabet size sigma, B strings each of length L, compute which strings are accepted.

The mathematical core is a sequential chain of matrix-vector multiplies:

```
s_{t+1} = T[input[t]] × s_t
```

where `s` is the state vector (N elements) and `T[c]` is the N×N transition matrix for character `c`. This chain is **inherently serial per string** — position t+1 depends on position t. The entire design space is about how to represent T, how to compute the multiply, and how to exploit parallelism despite the serial dependency.

The kernels split along three axes:

| Axis | Options |
|------|---------|
| **State type** | Boolean {0,1} (NFA/DFA) vs. real-valued FP16 (probabilistic/HMM) |
| **Compute unit** | Tensor cores (WMMA/wgmma) vs. CUDA INT cores (bitwise) vs. smem lookup tables |
| **Parallelism model** | Batch across strings vs. parallel prefix over positions vs. serial chain |

---

## Part 1: DFA Kernels (N ≤ 16)

These kernels target deterministic finite automata with small state spaces (N ≤ 16), where each state vector is one-hot and fits in a single WMMA tile.

### `tensor_core_dfa_scan.cu` — Parallel Prefix via Tensor Cores

**Framing.** Treats DFA simulation as a parallel prefix scan over position indices. Each position maps to a 16×16 transition matrix. The scan composes matrices via WMMA int8 MMA using Blelloch's work-efficient algorithm: O(L) work, O(log L) depth.

**When it matters.** Useful when L is large and B is small (few very long strings). The prefix scan extracts parallelism over positions rather than across strings. For B=1, L=10M, this gives O(L/P) time instead of O(L) serial.

**Hardware.** WMMA 16×16 int8, 8 warps/block. Shared memory holds per-warp B transpose (256 B) and int32 accumulator (1 KB). Two modes: "Basic" (malloc per call) and "Optimized" (persistent GPUContext with pre-allocated memory).

**Performance.** ~42 TFLOPS. Not competitive with monoid-based approaches for small N because the O(N³) matmul dominates over O(N) table lookup.

### `parallel_dfa_engine.cu` — Adaptive Multi-String DFA

**Framing.** Production multi-string DFA engine with adaptive dispatch. Supports variable-length strings (CSR format) and multi-DFA single-pass fusion.

**Architecture.** Two execution regimes:
- **R1 (short strings, L ≤ 1024):** Warp-per-string sequential matmul chain. Each warp processes one string serially, relies on batch width for occupancy.
- **R3 (long strings, L > 1024):** Decoupled look-back persistent scan. Each block processes a tile of positions, communicates partial results via global memory status flags.

Uses in-place right-multiply (`C = A × C`) to avoid temp allocations. Includes stream compaction for cascading pipelines (multi-stage regex).

### `monoid_scan.cu` — Table-Lookup DFA via Monoid Composition

**Framing.** Exploits the algebraic structure: the set of all reachable DFA state transformations forms a finite monoid (typically M ≤ 256 elements). Precomputes a composition table `compose[a][b]` on the host. Each character maps to a monoid element via `charmap[c]`, and the DFA simulation becomes sequential table lookups: `acc = compose[charmap[input[t]]][acc]`.

**Key insight.** Replaces O(N³) matrix multiply with O(1) shared memory lookup. For N=16, this is 16³ = 4096 FMA vs. 2 smem reads. The monoid approach wins by 2000×+ per operation.

**Architecture.**
- **R1:** Warp-per-string sequential scan. Lane 0 does all work; other lanes exist for occupancy.
- **R3:** Decoupled look-back for long strings.
- Composition table in smem: M × M × 2 bytes, loaded cooperatively.

### `monoid_batch.cu` — High-Throughput Batch Monoid

**Framing.** The production throughput kernel for DFA batches. Thread-per-string: each thread runs the monoid composition loop independently. Inner loop is 2 smem reads + 1 register update per character.

**Optimizations.**
- **Fused table:** Merges charmap + compose into a single (M × 256) table, reducing smem reads from 2 to 1 per character.
- **K-gram amortization:** Precomputes composition for all σ^K character patterns (e.g., 2^4 = 16 patterns for binary alphabet, K=4). Processes K characters per smem lookup, reducing memory accesses K×.
- **L2 prefetch:** A small kernel pre-reads input data at 128-byte stride to warm L2 after H2D DMA transfer. Fixes a 30× cold-cache penalty.

**Performance.** Target 1500–2500 Gc/s, bottlenecked by HBM bandwidth (~4.8 TB/s). This is the fastest DFA kernel in the repo for small monoids.

### `prefix_compose.cu` — Fused DFA Simulation Engine

**Framing.** Function-map composition approach: each transition is a map f: {0..N-1} → {0..N-1} stored as N bytes. Composition is O(N) smem gathers instead of O(N³) matmul. Three-component architecture:

1. **L2 prefetch kernel** — warms L2 after H2D DMA.
2. **Thread kernel (high-B)** — one thread per string, 1 smem lookup per character.
3. **Warp kernel (low-B)** — one warp per string, 16 lanes track the full map.

Auto-dispatches based on batch size threshold.

### `kgram_evolution.cu` — K-gram Tensor Core Evolution

**Framing.** Bridges the TC and monoid worlds. Precomputes product matrices for all σ^K k-grams (e.g., 16 matrices for K=4 binary). Each WMMA call applies a pre-composed matrix covering K characters, increasing arithmetic intensity K× compared to per-character evolution.

**Architecture.** Warp-per-string: each warp processes 1 string with one 16×16 WMMA tile. Uses cp.async for global→shared async copy. V2 batches 2 strings per warp by packing into the N=16 column dimension.

**Niche.** Useful when the state space is exactly N=16 and you want TC acceleration without the monoid precomputation step.

---

## Part 2: NFA Kernels (N = 64)

These kernels target non-deterministic automata with 64 states. The state vector is no longer one-hot — multiple states can be active simultaneously. The 64×64 matrix multiply requires tiling across multiple WMMA/wgmma calls.

Source: `cuda/nfa_tc_evolution.cu` and `cuda/probe_wgmma_layout.cu`

### V2 — TC Baseline (Shared Memory T Reload)

**Framing.** 4×4 tiling of 16×16 WMMA for 64×64 matmul. Each position: 32 MMA calls (4 row-tiles × 4 K-tiles × 2 characters). Both T and S live in shared memory; T fragments are reloaded from smem for every MMA.

**Why it's slow.** 32 `load_matrix_sync` calls per position for T fragments, each stalling until data arrives. The smem reload latency dominates.

**Performance.** 2.6 Gc/s, 42 TFLOPS. 8 blocks/SM (~40 regs/thread).

**Verdict.** Never use. V3 is strictly better.

### V3 — TC with Full T-Caching

**Framing.** Same as V2 but loads all 32 T fragments into registers before the L-loop. The per-position inner loop eliminates all T smem reads. Only S tiles traverse smem.

**Per-position serial chain:**
1. Load 4 S tiles from smem
2. 32 MMA (from register-cached T)
3. REGSEL: threshold accumulator to {0,1}, write S back to smem
4. `__syncwarp()`

**Trade-off.** ~170 registers/thread → only 2 blocks/SM. The register cache eliminates 32 smem loads per position but sacrifices occupancy. Net: 4.5× faster than V2 — the T-cache wins decisively.

**Performance.** 12.5 Gc/s, 205 TFLOPS, ~21% TC utilization.

**Verdict.** Best TC kernel for Boolean NFA. Use when sharing infrastructure with the probabilistic path (V5).

### V3.5 — Partial T-Caching (Failed Experiment)

**Hypothesis.** Cache only 2 of 4 T row-tiles in registers; reload the other 2 from smem. Lower register pressure → 4 blocks/SM → more warps → better TC utilization.

**Result.** 3× slower than V3.

**Why it failed.** The extra smem loads fall on the serial critical path. Each position now has 8 additional `load_matrix_sync` calls that cannot overlap with MMA. The bottleneck is the serial dependency chain, not insufficient warp-level parallelism. **Adding warps does not help when the critical path itself gets slower.**

**Verdict.** Never use. Exists as empirical proof that occupancy optimization is counterproductive for serial state evolution.

### V4 — Bit-Parallel Chunked (CUDA INT Cores)

**Framing.** Completely different compute model. State is a `uint64_t` bitmask. Boolean matmul becomes bitwise AND + test + OR over 64-bit words. One thread per string, fully independent.

**K-gram composition:** Host precomputes 2^K = 16 composed matrices (one per 4-bit character pattern). Each matrix is 64 × uint64_t = 512 bytes. All 16 fit in 8 KB smem. Per 4-character chunk: load 4 chars, compute pattern index, lookup composed matrix, apply 64 AND/OR ops.

**Why it dominates Boolean NFA.** Three compounding factors:
1. **8× Boolean throughput:** 128 CUDA INT cores × 64 Boolean ops per uint64_t = 8192 Boolean/cycle/SM, vs. 4 TC units × 4096/16 cycles = 1024 Boolean/cycle/SM.
2. **No serial overhead:** No fragment load/store, no smem S tiles, no `__syncwarp()`. Just load → lookup → bitwise → store.
3. **25× occupancy:** ~11 regs/thread → 16+ blocks/SM vs. V3's 2 blocks/SM.

**Performance.** 185 Gc/s, 3030 eq.TFLOPS. **16× faster than V3.**

**Verdict.** Always use for Boolean NFA. The ISA-level throughput advantage is fundamental and cannot be overcome by TC optimization.

### V5 — TC Probabilistic NFA (Real-Valued)

**Framing.** Identical to V3 architecture but without Boolean threshold. State carries real-valued FP16 probabilities. This is the HMM forward algorithm.

**Why TC wins here.** With real-valued matrices, there is no bitwise shortcut. The computation is genuine FP16 matrix-vector multiply. TC hardware delivers 989 TFLOPS vs. 134 TFLOPS for CUDA FP16 cores — a 7.4× hardware advantage. Measured: V5 at 178 TFLOPS vs. V6 at 11 TFLOPS (16× gap; V6 is intentionally naive).

**Performance.** 10.9 Gc/s, 178 TFLOPS. 2 blocks/SM.

**Verdict.** Always use for probabilistic/real-valued NFA.

### V6 — CUDA-Core FP16 Baseline

**Framing.** Naive scalar FP16 matmul: each thread handles one string, state is `half[64]` in registers, explicit `__hfma` loops. 4096 FMA per position per string.

**Performance.** 0.7 Gc/s, 11 TFLOPS.

**Verdict.** Reference implementation for correctness validation of V5. Not for production.

### wgmma RS Register Chain (Hopper, Experimental)

**Framing.** Exploits Hopper's wgmma instruction to eliminate V3's smem round-trip entirely. Key discovery: the wgmma RS form (register A, shared B) has identical accumulator and A-input register layouts — the accumulator feeds directly as A input for the next call. Zero-cost register chain.

**Reformulation.** `S' = T × S` becomes `S'^T = S^T × T^T` (right-multiply). State S^T lives in registers throughout; only T^T touches shared memory.

**Architecture.** 32 pipelined async MMA per position (4 K-tiles × 4 N-tiles × 2 characters). 128 threads (1 warpgroup) process 1 string. ~56 registers/thread.

**Performance.** 82.7 cy/pos = 46 ns/pos. **2.6× faster than V3** (218 cy). TC pipeline utilization: ~82%.

**Trade-off.** Processes 1 string per warpgroup (vs. V3's 64 strings per block). Wins on latency, loses on throughput. Best for latency-critical workloads or as the inner loop of a prefix scan.

---

## Part 3: DFA Engine Variants (N = 16, `batched_evolution.cu`)

These are the production DFA kernels for N=16 with the batch-parallel model (64 strings per block).

### V1 — Store-to-Smem Select

After MMA, `store_matrix_sync` writes accumulator to smem, then selects between T0 and T1 results in smem, then reloads as matrix_b. Full smem round-trip per position.

### V2 — Register-Level Select (REGSEL)

Uses the probed accumulator layout formula to select between T0/T1 results directly in registers, avoiding the smem round-trip for the select step. Still writes S back to smem for the next iteration's matrix_b load.

**Performance.** 112 Gc/s, 32 regs/thread.

### V3 — Relaxed Launch Bounds

V2's inner loop with relaxed `__launch_bounds__`, allowing the compiler more register freedom. Small occupancy trade for slightly better instruction scheduling.

**Performance.** 116 Gc/s, 40 regs/thread. **Fastest DFA batch variant.**

### V4 — Register Pipeline (acc→matb Shuffle)

Attempts to eliminate the smem S round-trip entirely by shuffling accumulator registers directly into matrix_b register positions. No S_sh reads in the inner loop.

**Performance.** 80 Gc/s. **Slower than V2/V3** because the shuffle overhead exceeds the smem load cost it replaces. The WMMA fragment layouts for accumulator and matrix_b differ, requiring expensive register permutations.

### V5 — V4 + Relaxed Launch Bounds

V4 body with relaxed bounds. Same outcome: register pipeline doesn't pay off.

**Performance.** 82 Gc/s.

---

## Part 4: Infrastructure Kernels

### `probe_frag_layout.cu` — WMMA Fragment Layout Probing

Empirical probes that determine how WMMA stores data in registers. Prints the mapping from (thread, element) to (row, col) for matrix_a, matrix_b, and accumulator fragments. Results feed the REGSEL optimization in V2/V3.

### `probe_wgmma_layout.cu` — Hopper wgmma Layout Verification

Comprehensive probe suite for wgmma m64n16k16 on H200. Verified:
- Accumulator layout formula (1024/1024 elements correct)
- A register layout = accumulator layout (enables zero-cost chain)
- Multi-step chain correctness
- Pipelined throughput benchmarks

### `profile_kernels.cu` — Profiling Harness

Unified benchmark harness measuring all 4 kernel types (monoid R1/R3, MMA R1/R3) with detailed per-SM metrics.

---

## Selection Heuristic

```
Is the automaton Boolean (NFA/DFA with {0,1} states)?
├── Yes
│   ├── N ≤ 16?
│   │   ├── Yes: How many states in the transition monoid?
│   │   │   ├── Small (M ≤ 256): monoid_batch.cu (thread-per-string, K-gram fused table)
│   │   │   │   → 1500-2500 Gc/s, HBM-bandwidth bound
│   │   │   └── Large / not precomputed: batched_evolution.cu V3 (WMMA, 116 Gc/s)
│   │   └── No (17 ≤ N ≤ 64): nfa_tc_evolution.cu V4 (bit-parallel, 185 Gc/s)
│   └── String count vs. length?
│       ├── Many short strings (B >> L): batch kernels above
│       └── Few long strings (L >> B): prefix_compose.cu (warp kernel) or
│           monoid_scan.cu R3 (decoupled look-back)
│
└── No (real-valued / probabilistic)
    └── N ≤ 64: nfa_tc_evolution.cu V5 (TC, 178 TFLOPS)
        └── Latency-critical, single string: wgmma RS chain (82 cy/pos)
```

### Key Principles

1. **Boolean NFA: avoid tensor cores.** CUDA INT cores process 64 Boolean values per instruction; TC processes 4096 but takes 16 pipeline cycles on 4 units. Net 8× throughput advantage for bitwise ops. No amount of TC optimization overcomes this ISA-level gap. (V4 vs V3 = 16× measured.)

2. **Real-valued NFA: use tensor cores.** No bitwise shortcut exists. TC delivers 989 TFLOPS vs 134 TFLOPS for CUDA FP16 cores. (V5 vs V6 = 16× measured.)

3. **Small monoids beat matrix multiply.** For DFA with N ≤ 16 and precomputable monoid (M ≤ 256), a single smem table lookup replaces 4096 FMA operations. The monoid kernels are HBM-bandwidth-bound, not compute-bound.

4. **Occupancy does not help serial chains.** V3.5 proved that doubling warps (via partial T-caching) made things 3× worse. The bottleneck is the per-position serial dependency chain, not the scheduler's ability to hide latency. Optimize the chain length, not the occupancy.

5. **Register caching beats smem caching.** V3 vs V2 (5× for NFA), V2 vs V1 (7× for DFA): keeping transition matrices in registers eliminates smem load latency from the critical path. The trade-off is lower occupancy (more registers per thread), but for serial chains, shorter critical path always wins.

6. **K-gram amortization is universally helpful.** Both monoid_batch and V4 use it: precompute composed operations for K-character patterns, process K chars per lookup/matmul. Reduces per-character overhead by K×.

7. **L2 prefetch after H2D DMA.** GPU DMA bypasses L2 cache, causing 30× cold-cache penalty on first access. A tiny prefetch kernel (one byte per 128-byte cache line) fixes this. Used in monoid_batch and prefix_compose.

8. **Latency vs throughput is a real trade-off.** wgmma RS processes 1 string per warpgroup (128 threads) at 82 cy/pos. V3 processes 64 strings per block (128 threads) at 218 cy/pos. wgmma wins 2.6× on latency; V3 wins ~3× on throughput. Choose based on workload.
