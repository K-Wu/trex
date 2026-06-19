# wgmma RS Register-Chain NFA Kernel Design

## Problem

V3 NFA kernel (`nfa_tc_evolution.cu`) has a serial dependency chain of ~218 cycles per
position. The bottleneck is the shared memory round-trip: after each MMA, the
accumulator must be written to smem (with threshold), then reloaded as the B fragment
for the next position. This smem round-trip dominates: MMA takes ~7 cycles but the
full chain takes ~218 cycles.

The FP16 accumulator design (`fp16_evolution.cu`) eliminates the INT32-to-INT8
conversion but still requires a smem round-trip (~130 cycles/position with WMMA).

## Key Finding: wgmma RS Register Chain

Probing on H200 (SM 9.0a, CUDA 12.8) revealed that Hopper's wgmma instruction has an
**RS form** (register A, shared B) where:

1. The A operand lives in registers (4 half2 per thread for m64n16k16)
2. The B operand comes from shared memory via a GMMA descriptor
3. The accumulator output has the **exact same register layout** as the A input

This means the accumulator feeds directly back as A for the next wgmma call — **zero
cost register chain, no shared memory round-trip for the evolving state.**

### Measured Latency

| Approach | Cycles/step | MMA/step | Threads | vs V3 |
|----------|------------|----------|---------|-------|
| wgmma RS N=64 (production) | **82** | 32 | 128 | **2.7x** |
| wgmma RS N=16 (single MMA) | 65 | 1 | 128 | 3.4x |
| FP16 WMMA + smem N=16 | 130 | 1 | 32 | 1.7x |
| V3 INT8 WMMA N=64 | 218 | 32 | 128 | 1.0x |

Benchmarks on H200 NVL, `globaltimer` measurement, fixed T in smem.

N=64 key: 32 pipelined async MMA take only 82 cycles total (2.6 cy/MMA average).
The async pipeline overlaps independent output tiles, vs V3's sync MMA at ~7 cy/MMA.

## Architecture

### Operation: S' = S x T (right-multiply chain)

wgmma RS computes C = A x B where A is in registers. We maintain the state S as the
A operand and the transition matrix T as the B operand in smem:

```
NFA evolution: S' = T x S     (left-multiply)
Transpose:     S'^T = S^T x T^T   (right-multiply)
wgmma RS:      C = A x B      where A = S^T, B = T^T
```

Precompute T^T for each character in the lookup table. The state S^T lives in registers
throughout the entire string processing — only T^T touches shared memory.

### N=64 Tiling (4x4 tiles of m64n16k16)

The NFA kernel uses N=64 states. With wgmma m64n16k16:
- A (state) is 64x16 — one K-tile of S^T (M=64 rows = 4 warps x 16)
- B (transition) is 16x16 — one tile of T^T
- C (output) is 64x16 — one output column-tile

Full 64x64 multiply requires:
- **4 K-tiles** (accumulate over K=64/16=4): 4 wgmma calls per output column-tile
- **4 N-tiles** (output columns): 4 output groups
- **Total: 16 wgmma calls per character**
- **Binary alphabet: 32 wgmma calls per position** (same as V3!)

### Pipelining Strategy

wgmma.mma_async is asynchronous. We can issue all K-tile MMA calls for one output
tile back-to-back, then commit once:

```
fence
// Output tile 0 (accumulators d0_0..d0_3)
mma d0 += A_k0 x B_k0_n0    // issue, returns immediately
mma d0 += A_k1 x B_k1_n0    // accumulate (depends on previous d0)
mma d0 += A_k2 x B_k2_n0
mma d0 += A_k3 x B_k3_n0
// Output tile 1 (independent accumulators d1_0..d1_3)
mma d1 += A_k0 x B_k0_n1
...
commit_group
wait_group 0
```

The 16 MMA for one character share a single commit/wait. Different output tiles are
independent and can overlap in the TC pipeline.

### Register Budget (per thread)

```
State (4 K-tiles x 4 regs):   16 registers   (S^T full 64-element state)
Accumulator T0 (4 N-tiles x 4): 16 registers (output for char 0)
Accumulator T1 (4 N-tiles x 4): 16 registers (output for char 1)
Overhead (desc, temp):           8 registers
Total:                          ~56 registers
```

65536 regs/SM / (128 threads x 56 regs) = ~9 blocks/SM. Reasonable occupancy.

### Per-Position Flow

```cuda
for (int pos = 0; pos < L; pos++) {
    // Load T0^T and T1^T into smem (double-buffered from previous iteration)
    // ... TMA or global load into GMMA-native layout ...
    __syncthreads();

    // Compute S^T x T0^T (char=0) and S^T x T1^T (char=1)
    // 16 + 16 = 32 wgmma RS calls, pipelined

    // Per-row select: each thread selects T0 or T1 result based on input char
    // Elements within each register pair share a row → same char → coherent select
    uint8_t ch_row0 = input[base_row + pos * stride];
    uint8_t ch_row1 = input[base_row + 8 + pos * stride];

    // Update state registers
    for (int n_tile = 0; n_tile < 4; n_tile++) {
        a_regs[n_tile] = (ch_rowX == 0) ? d0_regs[n_tile] : d1_regs[n_tile];
    }
}
```

### Shared Memory Layout

T^T matrices in GMMA-native layout (K contiguous within 8-element groups):
```
buf[cm * 128 + g * 64 + dim_local * 8 + k_local]
Descriptor: LBO=128 bytes, SBO=256 bytes
```

For N=64: each T^T tile is 16x16 = 256 halves = 512 bytes.
Full T^T: 4x4 tiles = 16 tiles = 8192 bytes.
Double-buffer: 16384 bytes = 16 KB.

### Verified Facts (from probe_wgmma_layout.cu)

1. GMMA-native smem layout: K fast, M/N slow, LBO=128, SBO=256
2. Accumulator layout: `row = w*16 + L/4 + ((e>>1)&1)*8, col = (L%4)*2 + (e&1) + (e>>2)*8`
3. f16 half2 packing confirmed (same layout as f32)
4. wgmma RS form: A in registers, B in smem (NOT the other way around!)
5. A register layout = accumulator layout (1024/1024 verified)
6. Multi-step chain verified: I x T0 x T1 = correct permutation (1024/1024)
7. Chain latency: 70 cycles/step for m64n16k16

### Comparison with V3

| | V3 (INT8 WMMA) | wgmma RS |
|---|---|---|
| Instruction | mma_sync (blocking) | mma_async (pipelined) |
| State storage | smem (round-trip per pos) | registers (zero-cost chain) |
| MMA per position | 32 | 32 |
| Threads per block | 128 (4 warps) | 128 (1 warpgroup) |
| Serial chain | ~218 cy/pos | ~70 cy/pos (N=16 measured) |
| State format | INT8 {0,1} → INT32 → threshold | FP16 {0.0, 1.0} → direct |

### Throughput vs Latency Trade-off

The wgmma RS chain processes **1 string per warpgroup** (128 threads). V3 batches
**64 strings per block** (128 threads, 4 warps x 16 strings each via WMMA's N=16
column dimension).

| Metric | V3 INT8 WMMA | wgmma RS |
|--------|-------------|----------|
| Strings per block | 64 | 1 |
| Cycles per position | 218 | 82 |
| Per-string latency | 218 cy | 82 cy |
| Per-SM throughput (est) | ~0.59 str/cy | ~0.20 str/cy |

**wgmma RS wins on latency (2.7x).** V3 wins on throughput (3x) because it batches
64 strings per block via WMMA's column dimension. The wgmma M=64 rows are consumed by
the N=64 state dimensions, leaving no room for string batching.

**Best for**: latency-critical workloads (interactive regex, real-time filtering, small
batch sizes), or as a building block for function composition (prefix scan) where the
serial chain is the critical path.

### Remaining Open Questions

1. **T^T loading**: Double-buffered T^T load (16 KB per position) needs to be hidden
   behind the MMA chain. With TC utilization high, the smem load may become the new
   bottleneck.

2. **NFA threshold**: NFA states can accumulate values > 1. Need threshold after MMA:
   `v > 0 ? 1 : 0`. Can be done in-register with `__hgt` before feeding back as A.
   Cost: ~4 instructions per register = ~64 instructions per position.

3. **Occupancy vs register pressure**: ~56 registers per thread limits to ~9 blocks/SM.

4. **Larger N-tile batching**: Using m64n32k16 or larger N to batch strings in the
   N dimension. Trades smem bandwidth (larger B) for throughput. All batched strings
   must share the same T (same character at current position).
