# Register-Select V2 Kernel: Performance Report

**Date:** 2026-06-18
**Hardware:** NVIDIA H200 NVL (SM 9.0, 132 SMs, 4,917 GB/s peak BW, 3,958 INT8 TOPS)
**Pattern:** `(a|b)*abb` — 4-state DFA, N=16 (padded), |Σ|=2

---

## 1. Problem: Shared Memory Bottleneck in the Batched Kernel

The batched evolution kernel processes 16 strings per warp using tensor-core MMA. For a binary alphabet, each position requires:

1. Load S from shared memory → `load_matrix_sync`
2. Two MMAs: `acc0 = T0 × S`, `acc1 = T1 × S`
3. **Store both accumulators to shared memory** → 2 × `store_matrix_sync` (2 × 1024 bytes)
4. **Read back from shared memory** per element, select acc0 or acc1 based on input char
5. Threshold and write result to S_sh

Steps 3–4 are the bottleneck. Each position per warp moves ~3 KB through shared memory just for the select step: 2 × 1024 bytes for int32 accumulator stores, plus 256 × 4 bytes for conditional reads. This dominates the inner loop, capping V1 at **27.5 Gc/s** even at large batch sizes.

---

## 2. Solution: Register-Level Fragment Selection

The WMMA accumulator fragment layout is officially "opaque" — NVIDIA's documentation does not specify which thread holds which (row, col) element. We empirically probed the layout on H200 (SM 9.0) by writing known values into fragment elements and reading them back via `store_matrix_sync`:

```
Accumulator fragment layout (int32, 16×16×16 MMA):
  row = lane/4 + ((elem_idx >> 1) & 1) * 8
  col = (lane % 4) * 2 + (elem_idx & 1) + (elem_idx >> 2) * 8
```

Each thread holds 8 elements spanning 4 unique columns (strings) and 2 row groups. This means each thread can:

1. Read the input character for its 4 columns (via `__shfl_sync` from lanes 0–15)
2. Select directly from `frag_acc0.x[i]` or `frag_acc1.x[i]` in registers
3. Threshold and write to S_sh

This eliminates both `store_matrix_sync` calls and all reads from acc0_sh/acc1_sh.

**Shared memory reduction:** 9,728 → 1,536 bytes per block (6.3× smaller).

---

## 3. Results: V1 vs V2

Benchmark: Even-A DFA (4 states, σ=2), 20 iterations averaged, 3 warmup runs.

| B | L | V1 kern (ms) | V1 (Gc/s) | V2 kern (ms) | V2 (Gc/s) | Speedup |
|---|---|-------------|-----------|-------------|-----------|---------|
| 1,024 | 128 | 0.175 | 0.8 | 0.040 | 3.3 | 4.35× |
| 1,024 | 512 | 0.664 | 0.8 | 0.123 | 4.3 | 5.40× |
| 1,024 | 2,048 | 2.623 | 0.8 | 0.468 | 4.5 | 5.60× |
| 4,096 | 128 | 0.164 | 3.2 | 0.038 | 13.9 | 4.35× |
| 4,096 | 512 | 0.617 | 3.4 | 0.128 | 16.4 | 4.81× |
| 4,096 | 2,048 | 2.422 | 3.5 | 0.481 | 17.5 | 5.04× |
| 16,384 | 128 | 0.170 | 12.4 | 0.042 | 49.9 | 4.04× |
| 16,384 | 512 | 0.637 | 13.2 | 0.132 | 63.5 | 4.82× |
| 16,384 | 2,048 | 2.523 | 13.3 | 0.508 | 66.1 | 4.97× |
| 65,536 | 128 | 0.313 | 26.8 | 0.091 | 92.7 | 3.45× |
| 65,536 | 512 | 1.228 | 27.3 | 0.337 | 99.7 | 3.65× |
| 65,536 | 2,048 | 4.896 | 27.4 | 1.442 | 93.1 | 3.40× |
| 262,144 | 128 | 1.229 | 27.3 | 0.310 | 108.2 | 3.96× |
| 262,144 | 512 | 4.863 | 27.6 | 1.210 | 110.9 | 4.02× |
| 262,144 | 2,048 | 19.411 | 27.7 | 4.794 | 112.0 | 4.05× |

**Peak throughput: 112 Gc/s** (V2 at B=262K, L=2048) vs 27.7 Gc/s (V1). Consistent **3.4–5.6× speedup** across all configurations.

---

## 4. Analysis

### Why 4× speedup from eliminating shared memory?

The V1 inner loop per position per warp:
- 2 × `store_matrix_sync`: 2 × 256 × 4 = 2,048 bytes written to shared memory
- 256 conditional reads from shared: 256 × 4 = 1,024 bytes read
- 256 writes to S_sh: 256 bytes
- **Total shared memory traffic: ~3,328 bytes/position/warp**

The V2 inner loop:
- 4 × `__shfl_sync`: register-only, zero shared memory
- 8 conditional writes to S_sh: 8 bytes per thread (256 bytes total)
- **Total shared memory traffic: ~256 bytes/position/warp**

That's a **13× reduction** in shared memory traffic per position. The ~4× kernel speedup (rather than 13×) reflects that MMA execution and S_sh loads still consume cycles — the shared memory round-trip was the dominant but not the sole cost.

### Shared memory pressure

| Metric | V1 | V2 |
|--------|-----|-----|
| Shared memory per block | 9,728 B | 1,536 B |
| Max blocks per SM (smem-limited) | 23 | 152 |
| Actual blocks per SM (reg-limited) | 16 | 16 |

Shared memory is no longer a constraint on occupancy. Both versions are register-limited to 16 blocks/SM with `__launch_bounds__(128, 16)`.

### Scaling behavior

V2 shows stronger scaling at moderate B (1K–16K) where V1 was severely underutilized. At B=4096, L=2048: V2 achieves 17.5 Gc/s vs V1's 3.5 Gc/s — a 5× gap. This suggests V1 was bandwidth-limited on shared memory at these sizes, while V2's register-based path saturates the tensor cores earlier.

---

## 5. Comparison with Other Engines

| Engine | Peak Gc/s | Notes |
|--------|----------|-------|
| Batched V1 | 27.7 | Shared memory accumulator round-trip |
| K-gram (k=16) | 27.4 | 93.75% tile waste (1/16 columns used) |
| K-gram V2 pipelined | 30.3 | 87.5% tile waste (2/16 columns used) |
| Monoid scan | 93.0 | O(1) table lookup, limited to small DFA+alphabet |
| **Batched V2 (regsel)** | **112.0** | **Register-level select, 100% tile utilization** |

The register-select V2 batched kernel is now the fastest tensor-core kernel, exceeding even the monoid scan engine. It achieves this with full 16×16 tile utilization (16 strings per warp), register-only selection, and minimal shared memory footprint.

---

## 6. Correctness

- 21/21 built-in CUDA tests pass (including V2-vs-V1 cross-validation on 4096×256)
- 23/23 Python test suite tests pass
- V2 kernel produces bit-identical results to V1 on all tested configurations
