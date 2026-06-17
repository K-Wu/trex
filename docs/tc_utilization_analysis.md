# Tensor Core Utilization Analysis for DFA State Evolution

## Arithmetic Intensity

For a single 16x16x16 WMMA MMA (INT8):

| Metric | Value |
|--------|-------|
| Compute | 2 x 16^3 = 8,192 INT8 ops |
| Data loaded | 2 x 256 = 512 bytes (matrices A + B) |
| Arithmetic Intensity | 16 ops/byte |

H200 SXM balance point:
- INT8 compute: 3,958 TOPS
- HBM bandwidth: 4.915 TB/s
- Balance AI = 3,958 / 4.915 = **805 ops/byte**

The kernel's AI of 16 is **50x below the balance point**. At most ~2% of
tensor core capacity can be utilized for per-character 16x16 state evolution.

## K-gram Fusion

Processing k characters per MMA multiplies effective AI by k:

| k  | Eff. AI | TC util est. | Table size (sigma=2, N=16) | Table size (sigma=256, N=16) |
|----|---------|-------------|------------------------|--------------------------|
| 1  | 16      | 2.0%        | 512 B                  | 64 KB                    |
| 2  | 32      | 4.0%        | 1 KB                   | 16 MB                    |
| 4  | 64      | 8.0%        | 4 KB                   | --                       |
| 8  | 128     | 15.9%       | 64 KB                  | --                       |
| 16 | 256     | 31.8%       | 16 MB                  | --                       |
| 20 | 320     | 39.8%       | 256 MB                 | --                       |

Table size = sigma^k x N^2 bytes. Must fit in L2 cache (~48 MB on H200) for
acceptable random-access latency.

## Backend Decision Framework

### When to use each backend

| Condition | Backend | Rationale |
|-----------|---------|-----------|
| N <= 16, monoid fits | Monoid R1 | O(1) table lookup per char. 93 Gc/s measured. |
| N > 16, sigma^k fits L2 | K-gram TC | k x fewer MMAs. Best for small sigma. |
| N > 16, sigma^k too large | TC state evolution | One MMA/char. Fallback. |

### Why monoid wins for N <= 16

The monoid approach replaces O(N^3) matrix multiplication with O(1) table
lookup. For N=16, each MMA does 8,192 ops; a monoid table lookup does ~2 ops.
No amount of TC optimization can close this 4,000x gap.

Monoid is feasible when the transition monoid size M is bounded (typically
M <= 200 for regex DFAs with N <= 16). The compose table (M^2 bytes) fits in
L1/registers. Measured throughput: 93 Gc/s vs 28 Gc/s for TC sparse.

### Why monoid fails for N > 16

The worst-case monoid size is N! (all permutation matrices are reachable).
At N=17, M <= 17! ~ 3.6x10^14 -- infeasible to enumerate or store.

K-gram TC bridges this gap: TC does the O(N^2) matmul natively, and k-gram
precomputation reduces the number of matmuls by k x.

## Measured Performance

| Backend | Config | Throughput | TC Utilization |
|---------|--------|------------|----------------|
| Monoid R1 (GPU) | N=16, sigma=2 | 93 Gc/s | N/A (no TC) |
| TC sparse | N=16, sigma=2, P=1 | 28 Gc/s | <1% |
| TC batched | N=16, sigma=2 | 3.3 Gc/s | <1% |
| K-gram TC | N=16, sigma=2, k=16 | TBD | ~32% (estimated) |

## Scaling with N

For square NxN matrices, GEMM arithmetic intensity is 2N/3:

| N | AI | TC util est. | Monoid feasible? |
|---|-----|-------------|-----------------|
| 16 | 10.7 | 1.3% | Yes (M <= ~200) |
| 32 | 21.3 | 2.6% | Unlikely |
| 64 | 42.7 | 5.3% | No |
| 128 | 85.3 | 10.6% | No |
| 256 | 170.7 | 21.2% | No |
| 1024 | 682.7 | 84.8% | No |

TC utilization only approaches 100% at N ~ 1,200 -- far beyond practical DFAs.
K-gram fusion (multiplying AI by k) is the practical path to meaningful TC
utilization for medium-N DFAs.
