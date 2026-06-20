# NFA Kernel Variants — Architecture, Performance, and When to Use Each

All kernels evolve NFA state vectors through transition matrices on GPU.
N=64 states, σ=2 (binary alphabet), batched across B strings of length L.
Benchmarked on NVIDIA H200 NVL (SM 9.0, 132 SMs, 989 FP16 TC TFLOPS, 134 FP16 CUDA TFLOPS).

Source: `cuda/nfa_tc_evolution.cu`

---

## Kernel Summary

| Variant | Compute Unit | Problem Domain | Peak Gc/s | Peak TFLOPS | Occupancy |
|---------|-------------|----------------|-----------|-------------|-----------|
| V2 | Tensor Core | Boolean NFA | 2.6 | 42 | 8 blocks/SM |
| V3 | Tensor Core | Boolean NFA | 12.5 | 205 | 2 blocks/SM |
| V3.5 | Tensor Core | Boolean NFA | 4.2 | 69 | 4 blocks/SM |
| V4 | CUDA INT | Boolean NFA | 189 | 3086 eq. | ~16 blocks/SM |
| V5 | Tensor Core | Probabilistic NFA | 10.9 | 178 | 2 blocks/SM |
| V6 | CUDA FP16 | Probabilistic NFA | 0.7 | 11 | 8 blocks/SM |

---

## V2 — TC Baseline (smem T reload)

**Architecture.** 4 warps/block, 64 strings/block (16 per warp). Loads T0
and T1 transition matrices into shared memory once. Each position: loads 4 S
tiles from smem, reloads 32 T fragments from smem, runs 32 MMA (4 row-tiles ×
4 k-tiles × 2 chars), then REGSEL writes back to smem with Boolean threshold.

**Launch bounds.** `__launch_bounds__(128, 8)` — targets 8 blocks/SM.

**Shared memory.** T0(8 KB) + T1(8 KB) + S(8 KB) = 24 KB.

**Bottleneck.** 32 smem T-fragment loads per position dominate the serial
chain. Each `load_matrix_sync` stalls until data arrives; there's no
opportunity to overlap loads with MMA because the MMA depends on the T fragment
being loaded.

**Performance (B=262K, L=2048).** 2.6 Gc/s, 42 TFLOPS, 210 ms.

**When to use.** Never. V3 is strictly better — same smem footprint, same
correctness, 5× higher throughput.

---

## V3 — TC with Full T-Caching

**Architecture.** Same threading model as V2. Key change: loads all 32 T
fragments (16 for T0, 16 for T1) into registers before the L-loop. The per-
position inner loop eliminates all T smem reads — only S tiles are loaded from
smem, and only S is written back after REGSEL.

**Launch bounds.** `__launch_bounds__(128, 2)` — register pressure (~170
regs/thread) limits to 2 blocks/SM (8 warps).

**Shared memory.** Same 24 KB. T stays resident in smem but is only read once
during the initial fragment load.

**Per-position serial chain:**
1. Load 4 S tiles from smem (4 × `load_matrix_sync`)
2. 32 MMA ops from register-cached T fragments
3. REGSEL: read 8 accumulator elements, shuffle chars, threshold to {0,1},
   write 8 elements to S in smem
4. `__syncwarp()`

**Bottleneck.** The serial dependency chain between positions. State at
position t+1 depends on position t's output, forcing the MMA → REGSEL → smem
store → sync → smem load → MMA pipeline to execute sequentially. With only
8 warps/SM, the warp scheduler has limited opportunity to interleave
independent work while a warp stalls on MMA pipeline latency.

TC utilization: 8 warps issuing 32 MMA into a 4-unit × 16-deep pipeline
= 8/64 = 12.5% theoretical. Measured ~21%, because consecutive MMA within a
warp partially pipeline (dependent MMA issues after data is ready, not after
full pipeline drain).

**Performance (B=262K, L=2048).** 12.5 Gc/s, 205 TFLOPS, 43 ms.

**When to use.** Best TC kernel for Boolean NFA. Use when you need TC
specifically (e.g., FP16 pipeline, shared engine struct with probabilistic
mode) but the problem is Boolean. 16× slower than V4 at this problem, but
shares the same engine/dispatch infrastructure and verifies correctness against
V2.

---

## V3.5 — TC with Partial T-Caching (Failed Optimization)

**Hypothesis.** Cache only 2 of 4 T row-tiles in registers; reload the other
2 from smem each position. Lower register pressure → 4 blocks/SM (16 warps)
→ better TC utilization via warp interleaving.

**Architecture.** Row-tiles 0–1: 16 T fragments cached in registers, used
directly in MMA. Row-tiles 2–3: 8 T fragments loaded from smem per position
via `load_matrix_sync`, then MMA.

**Launch bounds.** `__launch_bounds__(128, 4)` — ~103 regs/thread → 4
blocks/SM (16 warps).

**Result.** 3× slower than V3 at all batch sizes.

**Why it failed.** The extra smem loads for uncached T tiles fall on the serial
critical path. Each position now has 8 additional `load_matrix_sync` calls
that cannot overlap with MMA (the MMA depends on the T fragment just loaded).
The 2× increase in warps (16 vs 8) provides more scheduling slack, but the
serial chain is longer by the smem load latency — a net loss.

The bottleneck for sequential NFA evolution is the serial dependency chain
between positions, not insufficient warp-level parallelism. Adding warps does
not help when the critical path itself gets slower.

**Performance (B=262K, L=2048).** 3.8 Gc/s, 62 TFLOPS, 142 ms.

**When to use.** Never. Included as empirical evidence that occupancy-focused
TC optimization is counterproductive for serial state evolution.

---

## V4 — Bit-Parallel Chunked (CUDA INT Cores)

**Architecture.** Completely different from TC variants. Each thread handles
one string independently. State is a `uint64_t` bitmask (64 bits = 64 NFA
states). Boolean matmul becomes bitwise AND + test + OR over 64-bit words.

Pre-composition: host builds a lookup table of 2^K=16 pre-composed transition
matrices (one per 4-bit character pattern). Each composed matrix M maps a
4-character chunk to a single matrix multiply. Stored as 64 × uint64_t per
pattern = 512 bytes × 16 patterns = 8 KB in shared memory.

Per chunk (4 positions): load 4 characters, compute 4-bit pattern index, look
up composed matrix M[pattern], apply 64 AND-test-OR operations. Amortizes the
matmul over K=4 positions.

**Launch bounds.** `__launch_bounds__(128, 16)` — ~11 regs/thread → up to
25 blocks/SM. Massively higher occupancy than any TC variant.

**Shared memory.** M_table(8 KB) + T_bits(1 KB) = 9 KB.

**Why it dominates Boolean NFA.** CUDA INT cores deliver 8× higher Boolean
throughput than tensor cores:
- CUDA: 128 cores/SM × 64 Boolean ops per `uint64_t` instruction = 8192
  Boolean/cycle/SM
- TC: 4 units × 4096 Boolean/MMA ÷ 16 cycle pipeline = 1024 Boolean/cycle/SM

Additionally, each thread is fully independent — no warp-level coordination,
no smem S tiles, no `__syncwarp()`. The serial dependency chain is just
"load chars → lookup → 64 AND/OR → store state" with no MMA pipeline
latency, no fragment load/store overhead.

**Performance (B=262K, L=2048).** 185 Gc/s, 3030 eq.TFLOPS, 2.9 ms.

**Performance scaling:**

| B | L=128 | L=512 | L=2048 |
|---|-------|-------|--------|
| 4K | 10.4 Gc/s | 11.6 Gc/s | 11.9 Gc/s |
| 16K | 42.1 Gc/s | 45.2 Gc/s | 47.0 Gc/s |
| 65K | 91.1 Gc/s | 155.1 Gc/s | 143.1 Gc/s |
| 262K | 107.4 Gc/s | 188.4 Gc/s | 184.9 Gc/s |

Throughput scales well with batch size. At B=65K/L=128, occupancy is marginal
(~4 blocks/SM with B_padded=65536, 128 threads/block = 512 blocks total /
132 SMs ≈ 4 blocks/SM). At B=262K the GPU is fully saturated.

**When to use.** Always, for Boolean NFA at N=64 with binary alphabet. 16×
faster than the best TC kernel (V3). The only limitation: restricted to
Boolean automata — cannot handle real-valued transition matrices.

---

## V5 — TC Probabilistic NFA (Real-Valued)

**Architecture.** Identical to V3 (all T fragments cached, 2 blocks/SM) but
without the {0,1} Boolean threshold in REGSEL. The state vector carries
real-valued FP16 probabilities through the evolution. Accept check is a dot
product: `sum(s[i] * accept_weights[i]) > 0.5`.

This is the HMM forward algorithm: `s_{t+1} = T[observation_t] × s_t` where
T is a row-stochastic matrix and s is a probability distribution.

**REGSEL simplification.** Instead of:
```
S_sh[col*N + row] = __hgt(v, h_zero) ? h_one : h_zero;
```
Just:
```
S_sh[col*N + row] = (ch == 0) ? frag_acc0.x[i] : frag_acc1.x[i];
```
Fewer instructions on the critical path. Slightly simpler than V3 but same
MMA count and same serial dependency structure.

**Performance (B=262K, L=2048).** 10.9 Gc/s, 178 TFLOPS, 49 ms.

Slightly lower than V3's 205 TFLOPS. The real-valued state means more
non-zero elements flow through MMA, potentially causing more accumulator
additions and slightly different FP16 rounding behavior. The performance
delta is within expected variance for the same architecture.

**When to use.** The right kernel for real-valued automata: probabilistic
NFAs, weighted automata, HMM forward passes, soft attention over finite-state
machines. TC is the only viable compute path because no bitwise shortcut
exists for real-valued matmul. V5 at 178 TFLOPS exceeds the H200's entire
CUDA-core FP16 peak of 134 TFLOPS.

---

## V6 — CUDA-Core FP16 Baseline (Real-Valued)

**Architecture.** Each thread handles one string. State is `half[64]` in
registers. Per position: explicit scalar FP16 matrix-vector multiply using
`__hfma` intrinsics. N² = 4096 FMA operations per position per string, fully
sequential within the thread.

No tensor cores, no shared memory tiling, no warp cooperation. This is
intentionally naive to establish the CUDA-core throughput floor for comparison
with V5.

**Launch bounds.** `__launch_bounds__(128, 8)`.

**Performance (B=262K, L=2048).** 0.7 Gc/s, 11 TFLOPS, 791 ms.

**Why it's slow.** Each thread does 4096 FMA per position sequentially. The
H200 has 128 FP16 CUDA cores per SM, each doing 2 ops/cycle. At 1.98 GHz:
128 × 2 × 1.98 = 507 GFLOPS/SM. For 64×64 matmul (8192 flops): 8192/507 =
16 µs/position/SM if fully utilized. With serial state dependency preventing
cross-position overlap, actual throughput is far below peak.

A well-optimized CUDA FP16 kernel with smem tiling would improve this, but
would cap near the 134 TFLOPS theoretical peak — still below V5's 178
TFLOPS.

**When to use.** Reference implementation for correctness validation of V5.
Not competitive for production workloads.

---

## Decision Matrix

### Boolean NFA (states ∈ {0,1})

Use **V4** (bit-parallel). Always. At every batch size and string length.

The 8× Boolean throughput advantage of CUDA INT cores over tensor cores is
fundamental to the ISA. No TC optimization can overcome it because:

1. `uint64_t` bitwise AND processes 64 Boolean values in one instruction on
   one CUDA core.
2. TC MMA processes 16×16×16 = 4096 values but takes 16 cycles on one TC
   unit, and only 4 TC units exist per SM.
3. 128 CUDA cores × 64 Boolean/op = 8192 Boolean/cycle vs
   4 TC × 4096/16 = 1024 Boolean/cycle.

V3 is 16× slower. V3.5 (the occupancy optimization attempt) is 48× slower.

### Probabilistic NFA (states ∈ ℝ, FP16)

Use **V5** (TC). The roles reverse completely.

With real-valued matrices, there is no bitwise shortcut. The computation
is genuine FP16 matrix-vector multiply, and TC hardware delivers 7.4× higher
throughput than CUDA FP16 cores (989 vs 134 TFLOPS peak). Measured:
V5 at 178 TFLOPS vs V6 at 11 TFLOPS — a 16× gap (V6 is naive; gap would
narrow to ~6-7× with an optimized CUDA kernel, but TC still wins).

### Mixed / uncertain

Use V4 for the Boolean path and V5 for the real-valued path. Both share
the same `NFATCEngine` struct and dispatch infrastructure — switching is
just `engine.kernel_variant = 4` or `engine.kernel_variant = 5`.

---

## Why TC Cannot Win for Boolean NFA — The Full Argument

Three independent factors compound against tensor cores for Boolean matmul:

**1. Throughput gap (8×).** Bitwise ops on CUDA INT cores process 64 Boolean
values per instruction. TC MMA processes 4096 values per instruction but takes
16 pipeline cycles and uses one of only 4 TC units. Net: CUDA delivers 8×
more Boolean operations per cycle per SM.

**2. Serial dependency overhead.** NFA state evolution is inherently serial:
`s_{t+1} = T[c_t] × s_t`. Each position depends on the previous. The TC
serial chain per position is: load S from smem → 32 MMA → REGSEL → store S
to smem → syncwarp → repeat. The bit-parallel chain is: load chars → lookup
composed matrix → 64 AND/OR → repeat. The TC chain includes fragment
load/store overhead, syncwarp, and MMA pipeline latency that the bit-parallel
chain completely avoids.

**3. Occupancy gap.** V4 runs ~16 blocks/SM (~11 regs/thread). V3 runs 2
blocks/SM (~170 regs/thread). Higher occupancy means better latency hiding
for global memory loads (character data), further widening the throughput gap.

**V3.5 disproved the occupancy hypothesis.** If the bottleneck were
insufficient warps to hide MMA pipeline latency, partial T-caching (trading
register cache for occupancy) should have helped. Instead it made things 3×
worse, confirming that the serial dependency chain — not warp scheduling — is
the binding constraint.

---

## Full Benchmark Data

### Boolean NFA (V2, V3, V3.5, V4) at B=262K

| Variant | L=128 | L=512 | L=2048 | Unit |
|---------|-------|-------|--------|------|
| V2 (TC, smem reload) | 2.5 Gc/s | 2.5 Gc/s | 2.6 Gc/s | |
| | 42 TFLOPS | 42 TFLOPS | 42 TFLOPS | |
| V3 (TC, cached T) | 11.6 Gc/s | 11.7 Gc/s | 11.4 Gc/s | |
| | 190 TFLOPS | 191 TFLOPS | 187 TFLOPS | |
| V3.5 (TC, partial) | 4.2 Gc/s | 3.9 Gc/s | 3.8 Gc/s | |
| | 68 TFLOPS | 64 TFLOPS | 62 TFLOPS | |
| V4 (bit-parallel) | 107 Gc/s | 188 Gc/s | 185 Gc/s | |
| | 1760 eq.TF | 3086 eq.TF | 3030 eq.TF | |

### Probabilistic NFA (V5, V6) at B=262K

| Variant | L=128 | L=512 | L=2048 |
|---------|-------|-------|--------|
| V5 (TC) | 10.3 Gc/s | 10.6 Gc/s | 10.9 Gc/s |
| | 169 TFLOPS | 173 TFLOPS | 178 TFLOPS |
| V6 (CUDA) | 0.6 Gc/s | 0.6 Gc/s | 0.7 Gc/s |
| | 10 TFLOPS | 10 TFLOPS | 11 TFLOPS |

### Cross-Variant Ratios

| Comparison | Ratio | Explanation |
|-----------|-------|-------------|
| V4 / V3 (Boolean) | **16×** | Bit-parallel dominates Boolean matmul |
| V3 / V2 (Boolean) | **4.5×** | T-caching eliminates 32 smem loads/position |
| V3 / V3.5 (Boolean) | **3.0×** | Full T-cache beats partial despite lower occupancy |
| V5 / V6 (Probabilistic) | **16×** | TC dominates real-valued matmul |
| V5 / V3 (same arch, different domain) | **0.87×** | Similar — both are serial TC evolution |

---

## Resource Usage Summary

| Variant | Regs/Thread | Smem/Block | Blocks/SM | Warps/SM | Threads/SM |
|---------|-------------|------------|-----------|----------|------------|
| V2 | ~40 | 24 KB | 8 | 32 | 1024 |
| V3 | ~170 | 24 KB | 2 | 8 | 256 |
| V3.5 | ~103 | 24 KB | 4 | 16 | 512 |
| V4 | ~11 | 9 KB | 16+ | — | 2048+ |
| V5 | ~170 | 24 KB | 2 | 8 | 256 |
| V6 | ~40 | 0 | 8 | — | 1024 |
