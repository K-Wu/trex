# Monoid Batch Pipeline: Full Performance Overhaul

## Problem

The current fastest engine (TC Batched V3) peaks at 116 Gc/s on H200. This is only 2.4% of the HBM bandwidth roofline (4,900 Gc/s at 1 byte/char). The bottleneck is the serial MMA dependency chain: each 16x16 int8 matmul takes ~40 cycles and depends on the previous position's result. Even eliminating all non-MMA overhead would only reach ~155 Gc/s.

For DFAs with small transition monoids (M <= ~1000 elements), the MMA is 4000x more expensive than necessary. A 4-state binary DFA has a 7-element monoid. The entire compose table is 14 bytes. We use a 16x16 MMA (4096 multiply-add ops) to do what a single byte lookup can do.

Additionally, the Python-side char mapping takes 88ms for 10M characters -- 700x slower than the GPU kernel itself. This dominates end-to-end latency.

## Solution

Three new components that together target 1500-2500 Gc/s for small-monoid DFAs (13-21x improvement):

1. **Monoid Batch Kernel** -- One CUDA thread per string, O(1) compose lookup per character
2. **GPU Input Pipeline** -- Char mapping and transpose on GPU, eliminating the Python bottleneck
3. **Parallel Prefix Monoid Reduce** -- O(L/threads + log threads) for few very long strings

Updated auto-selection logic routes workloads to the fastest available engine.

## Component 1: Monoid Batch Kernel

### Threading Model

- 128 threads per block (each thread processes one string)
- Target 16 blocks per SM -> 2048 strings per SM -> 270K strings across 132 SMs
- No warp synchronization, no WMMA fragments, no shared-memory state matrix
- `__launch_bounds__(128, 16)` as starting point, sweep occupancy as with V2/V3

### Shared Memory Layout (per block)

```
compose_table[M * sigma]   -- M x sigma bytes (compose(m_i, c_j) -> m_k)
char_to_monoid[256]        -- 256 bytes (raw ASCII byte -> monoid char index)
accept_table[M]            -- M bytes (1 if monoid element maps start -> accept)
Total: M*sigma + 256 + M bytes
```

For M=7, sigma=2 (Even-A DFA): 14 + 256 + 7 = 277 bytes. For M=200, sigma=36: 7456 bytes. Both trivially fit in shared memory.

### Inner Loop

```cuda
__global__ void monoid_batch_kernel(
    const uint8_t *raw_concat,     // all strings concatenated
    const int     *offsets,        // [B+1] CSR offsets into raw_concat
    const uint8_t *compose_table,  // [M * sigma] on device
    const uint8_t *char_to_monoid, // [256] raw char -> monoid char index
    const uint8_t *accept_table,   // [M] monoid element -> accept?
    int B, int M, int sigma,
    uint8_t identity_monoid,
    int *results)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= B) return;

    // Load tables into shared memory (all threads cooperate)
    extern __shared__ uint8_t smem[];
    uint8_t *compose_sh = smem;
    uint8_t *c2m_sh = compose_sh + M * sigma;
    uint8_t *accept_sh = c2m_sh + 256;
    for (int e = threadIdx.x; e < M * sigma; e += blockDim.x)
        compose_sh[e] = compose_table[e];
    for (int e = threadIdx.x; e < 256; e += blockDim.x)
        c2m_sh[e] = char_to_monoid[e];
    for (int e = threadIdx.x; e < M; e += blockDim.x)
        accept_sh[e] = accept_table[e];
    __syncthreads();

    int start = offsets[sid];
    int len = offsets[sid + 1] - start;
    uint8_t curr = identity_monoid;

    for (int t = 0; t < len; t++) {
        uint8_t ch_m = c2m_sh[raw_concat[start + t]];
        curr = compose_sh[curr * sigma + ch_m];
    }

    results[sid] = accept_sh[curr];
}
```

### Performance Model

- Per-position: 1 global read + 2 shared memory reads + 1 register update = ~4-8 instructions
- Bottleneck: HBM bandwidth (4.9 TB/s = 4900 Gc/s at 1 byte/char)
- With scheduling overhead and cache effects: **1500-2500 Gc/s** estimated
- 13-21x improvement over TC Batched V3 (116 Gc/s)

### Memory Access Pattern

Each thread reads its string sequentially from the raw concat buffer. Adjacent threads (within a warp) read from adjacent strings, which are adjacent in the concat buffer. For short strings of similar length, the access pattern is nearly coalesced. For variable-length strings, cache locality in L1/L2 handles the scatter.

No position-contiguous transpose is needed -- threads read raw string bytes directly.

### Accept Table Precomputation

For each monoid element m_i (a matrix), compute: does applying m_i to the start state vector reach any accept state?

```python
accept_table = np.zeros(M, dtype=np.uint8)
for i, matrix in enumerate(monoid.elements):
    result = matrix @ start_vec  # [N] vector
    result = (result > 0).astype(np.int8)
    if np.any(result & accept_mask):
        accept_table[i] = 1
```

This runs once at engine init. Cost: M matrix-vector multiplies of size N. For M=200, N=16: negligible.

### Identity Monoid Element

The identity element of the transition monoid is the identity matrix. Its index in the monoid element list is found during monoid construction. For characters not in the DFA alphabet (padding, unknown), map to a special "identity character" whose monoid action is the identity element:

```python
identity_idx = monoid.identity_index  # index of identity matrix in elements list
```

For strings shorter than L_max, the thread simply processes fewer characters (via the CSR offsets). No identity padding needed.

## Component 2: GPU Input Pipeline

### Current Flow (slow)

```
Python: strings -> char_to_idx dict lookup -> numpy array fill (88ms for 10M chars)
C:      batched_prepare_input() transposes to position-contiguous layout
Host:   cudaMemcpy H2D of L * B_padded bytes
GPU:    kernel
```

### Proposed Flow (fast)

```
Host:   cudaMemcpy H2D of raw concat bytes + offsets (smaller, no padding)
GPU:    gpu_prepare_input kernel (char mapping + optional transpose)
GPU:    compute kernel (monoid batch or TC batched)
```

### GPU Char Mapping Kernel

For the monoid batch path, no transpose is needed (threads read raw strings directly via CSR offsets). The char_to_monoid mapping happens inside the monoid kernel itself.

For the TC batched path (fallback), a GPU transpose kernel replaces the CPU batched_prepare_input:

```cuda
__global__ void gpu_prepare_input(
    const uint8_t *raw_concat,    // all strings concatenated
    const int     *offsets,       // [B+1] CSR
    uint8_t       *output,        // [L_max][B_padded] position-contiguous
    const uint8_t *char_map,      // [256] raw char -> DFA char index
    int B, int B_padded, int L_max,
    uint8_t identity_idx)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;  // string id
    if (sid >= B_padded) return;

    int start = (sid < B) ? offsets[sid] : 0;
    int len = (sid < B) ? (offsets[sid + 1] - start) : 0;

    for (int t = 0; t < L_max; t++) {
        uint8_t val = identity_idx;
        if (t < len) val = char_map[raw_concat[start + t]];
        output[t * B_padded + sid] = val;
    }
}
```

Grid: ceil(B_padded / 256) blocks, 256 threads. Each thread handles all positions of one string. Expected time: ~0.05ms for 10M chars.

### H2D Transfer

Transfer the raw concat buffer and offsets array to device memory. For B=262K strings of average length 100:
- Raw concat: ~26 MB
- Offsets: ~1 MB
- Total: ~27 MB vs current ~25 MB (position-contiguous with padding)

Transfer size is similar, but the GPU-side mapping eliminates the 88ms Python preprocessing.

## Component 3: Parallel Prefix Monoid Reduce

### When to Use

For workloads where B is small (1-128) and L is large (> 100K). Examples: single-document regex search, genome matching, large log file scanning. The monoid batch kernel would assign few threads and waste most of the GPU.

### Algorithm

One block per string. Block size = 256 threads.

```
Phase 1 -- Thread-local sequential reduce:
    chunk_size = ceil(L / blockDim.x)
    Each thread reduces chunk_size characters -> one monoid element
    Store per-thread results in shared memory: smem[threadIdx.x]
    __syncthreads()

Phase 2 -- Tree reduce in shared memory:
    for (stride = blockDim.x/2; stride > 0; stride >>= 1):
        if (threadIdx.x < stride):
            smem[threadIdx.x] = compose(smem[threadIdx.x], smem[threadIdx.x + stride])
        __syncthreads()

Phase 3 -- Thread 0 checks accept:
    if (threadIdx.x == 0):
        results[blockIdx.x] = accept_table[smem[0]]
```

### Performance Model

For L=10M, 256 threads per block:
- Phase 1: each thread processes ~39K chars at ~2 Gchar/s per thread = ~20us
- Phase 2: 8 tree-reduce steps, ~0.1us
- Total: ~20us per string (vs ~5ms for sequential scan of 10M chars at 2 Gc/s)

For B=1: the monoid batch kernel uses 1 thread = purely sequential. The parallel prefix uses 256 threads = 250x speedup.

### Input Format

Uses the same raw concat + CSR offsets as the monoid batch kernel. Each block reads its string's portion of the concat buffer.

## Auto-Selection Logic

### Updated Decision Tree

```python
def _select_engine(self, B, L_max):
    if self.monoid is not None and self.monoid.size <= 1024:
        if B <= 128 and L_max > 100_000:
            return "monoid_prefix"    # parallel prefix reduce
        else:
            return "monoid_batch"     # 1 thread per string
    elif self.sigma == 2 and self.N <= 16:
        return "tc_batched_v3"        # existing TC kernel
    elif self.N <= 16:
        return "tc_general"           # existing general kernel
    else:
        return "tc_multitile"         # existing multi-tile kernel
```

### Monoid Size Threshold

M <= 1024 is the cutoff. The compose table at M=1024, sigma=36 is 36 KB, which fits in shared memory. Beyond this, cache pressure degrades the lookup performance and the TC batched kernel becomes competitive.

The threshold can be tuned empirically during benchmarking. The crossover point depends on M, sigma, and the specific GPU's L1/shared memory characteristics.

### Backward Compatibility

- All existing Python APIs unchanged
- `regsel=True` parameter still selects TC V3 for the batched path
- New monoid batch path is selected automatically by default
- Existing TC kernels remain available as fallback

## New Files

- `cuda/monoid_batch.cu` -- Monoid batch kernel + parallel prefix kernel + GPU char mapping kernel + C API
- `src/gpu_bridge_monoid_batch.py` -- Python bridge to the new CUDA library (ctypes wrappers)
- Updates to `src/optimized_engine.py` -- Auto-selection routing to new engine

## Performance Targets

| Workload | Current Best | Target | Speedup |
|----------|-------------|--------|---------|
| B=262K, L=2048, M=7 | 116 Gc/s (TC V3) | 1500-2500 Gc/s | 13-21x |
| B=262K, L=128, M=7 | 112 Gc/s (TC V3) | 1500-2500 Gc/s | 13-21x |
| B=1, L=10M, M=7 | ~2 Gc/s (monoid R1) | ~500 Gc/s (prefix) | ~250x |
| End-to-end 10M chars | ~90ms (Python bottleneck) | ~1ms (GPU pipeline) | ~90x |
| B=262K, M=500 | 116 Gc/s (TC V3) | 500-1000 Gc/s | 4-9x |

## Non-Goals

- Replacing TC kernels for large-monoid or large-N DFAs (they remain the best option)
- Multi-GPU support
- Streaming / online regex matching
- wgmma / Hopper-specific async MMA (M=16 is too small for wgmma's M>=64 requirement)
