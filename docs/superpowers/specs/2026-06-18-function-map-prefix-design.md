# Function Map Parallel Prefix Engine

## Problem

Current TC-based engines (`batched_evolution.cu`, `kgram_evolution.cu`) achieve only 2–6% tensor core utilization because the DFA simulation loop is inherently serial: one 16×16 matrix multiply per character step, each depending on the previous result. TCs are idle >90% of the time.

The monoid batch kernel avoids matmul entirely (703 Gc/s via table lookups), but requires M ≤ 255. For large-monoid DFAs (M > 255) or DFAs with N > 16 states, the only current option is the serial TC kernel at 116 Gc/s.

## Core Insight

DFA transition matrices are column-stochastic binary: each column has exactly one 1 (determinism). A 16×16 matrix carries only 16 values of information — the destination state for each source state.

**Compact representation**: Each transition is a map `f: {0..N-1} → {0..N-1}`, stored as N bytes instead of N² bytes.

**Cheap composition**: `(f∘g)[j] = f[g[j]]` — N gather operations instead of N³ multiply-accumulates. For N=16: **16 ops vs 8,192 ops** per composition.

**Warp shuffle composition**: On CUDA, composing two 16-entry maps is a single `__shfl_sync` instruction (~2 cycles):
```cuda
// Lane j holds f[j] and g[j]
// Compose: result[j] = f[g[j]]
uint8_t result = (uint8_t)__shfl_sync(0xFFFF, f_val, g_val);
```
Each warp shuffle reads `f` from lane `g[j]`, which is `f[g[j]]`. The full 16-entry composition completes in one instruction.

## Architecture

Two-phase parallel prefix with a fused transition map LUT.

### Precomputation (Python, one-time)

Build a fused transition map table from the DFA:
```
tmap[raw_char * N_padded + state] = δ(state, char_mapped)   for state ∈ {0..N-1}
tmap[raw_char * N_padded + state] = state                    for unmapped chars (identity)
```
Table size: 256 × N_padded bytes (4 KB for N_padded=16). Always fits in shared memory regardless of monoid size.

### Phase 1: Block Tree Reduce (GPU Kernel 1)

Divide each string into blocks of K characters (K=32).

Each CUDA warp processes one block:
1. **Load characters**: Read K bytes from the string (global memory → registers).
2. **Map to transition maps**: For each character, each of 16 lanes reads its entry from the tmap LUT in shared memory: `map_i[j] = tmap_smem[char_i * 16 + j]`.
3. **Tree reduce**: Compose K maps into 1 via log₂(K) = 5 levels. Maps are stored in per-warp shared memory (`maps_smem[K][16]`). At each level, the warp loads two maps from smem into registers, composes via `__shfl_sync`, and writes the result back to smem. Two compositions per shuffle (one per half-warp, lanes 0–15 and 16–31). Level 0: 16 pairs → 8 shuffles. Level 1: 8 → 4. Level 2: 4 → 2. Level 3: 2 → 1. Level 4: 1 → 1. Total: 16 shuffles.
4. **Write block product**: Each of 16 lanes writes its entry to global memory: `block_products[(string_id * n_blocks + block_id) * 16 + j]`.

**Grid**: `ceil(total_blocks / WARPS_PER_BLOCK)` CUDA blocks, 128 threads (4 warps) each.

**Shared memory**: tmap LUT (256 × 16 = 4 KB, shared across all warps in the block) + workspace for tree intermediates (K × 16 bytes per warp = 512 bytes, for storing maps between tree levels).

**Composition order**: Left-to-right application. For characters c₀, c₁, ..., c_{K-1}:
- Block product = f_{K-1} ∘ ... ∘ f₁ ∘ f₀
- In the tree: level 0 composes (f₀,f₁), (f₂,f₃), etc. Each pair: `newer ∘ older`.
- Shuffle: `result = __shfl_sync(mask, newer_val, older_val)` gives `newer[older[j]]`.

### Phase 2: Block Scan + Accept (GPU Kernel 2)

Each warp processes one string by composing its block products sequentially:
1. **Load block products**: Read n_blocks × 16 bytes per string from global memory.
2. **Serial compose**: Accumulate via shuffle: `acc = __shfl_sync(mask, block_val, acc)`.
3. **Accept check**: Lane `start_state` holds the final state. Broadcast via shuffle, look up accept table.

**Grid**: `ceil(B / WARPS_PER_BLOCK)` CUDA blocks.

For n_blocks ≤ 32 (L ≤ 1024 with K=32): 15–31 compositions, ~50–100 cycles. Negligible vs Phase 1.

### Memory Layout

| Buffer | Size | Notes |
|--------|------|-------|
| tmap LUT | 256 × 16 = 4 KB | Shared memory, loaded once per block |
| Input strings | raw_concat + CSR offsets | Same format as monoid batch engine |
| Block products | B × n_blocks × 16 bytes | Global memory intermediate |
| Results | B × 4 bytes | Accept/reject per string |

For B=65536, L=512, K=32: block products = 65536 × 16 × 16 = 16.8 MB.

### Variable-Length Strings

Use the existing CSR layout (raw_concat + offsets). Each warp computes its block's character range from the string's offset and length. Characters beyond the string end are skipped (or treated as identity maps). Block count per string = ceil(string_length / K).

A block descriptor array maps global_block_id → (string_id, block_offset). Precomputed on CPU as part of dispatch.

### Composition Order Detail

For string "abcd" with K=2:
- Block 0: f_b ∘ f_a (characters a, b)
- Block 1: f_d ∘ f_c (characters c, d)
- Phase 2 compose: (f_d ∘ f_c) ∘ (f_b ∘ f_a) = f_d ∘ f_c ∘ f_b ∘ f_a
- Final state: F(start) = f_d(f_c(f_b(f_a(start))))

## Performance Model

### Throughput Estimate (H200, B=65536, L=512, K=32, N=16)

**Phase 1 (block reduce)**:
- Blocks: 65536 × 16 = 1,048,576
- Per block: K map lookups (~160 cycles) + tree reduce (16 shuffles × 10 cycles with smem staging = ~160 cycles) = ~320 cycles
- With 4224 warps: 248 blocks/warp × 320 = 79,360 cycles → **43 µs**

**Phase 2 (scan + accept)**:
- 65536 strings, 15 compositions each × ~5 cycles = ~75 cycles/string
- With 4224 warps: 16 strings/warp × 75 = 1200 cycles → **0.7 µs**

**Global memory traffic**:
- Write block products: 16.8 MB → **3.5 µs** at 4.8 TB/s
- Read block products in Phase 2: 16.8 MB → **3.5 µs**

**Total: ~51 µs → ~660 Gc/s**

With optimization (register-resident maps, coalesced loading, tuned K):
- Projected: **700–1000 Gc/s**

### Comparison

| Engine | N=16, M≤255 | N=16, M>255 | N=32 | N=64 |
|--------|-------------|-------------|------|------|
| Monoid batch (current) | 703 Gc/s | N/A | N/A | N/A |
| Serial TC (current) | 116 Gc/s | 116 Gc/s | ~30 Gc/s | ~5 Gc/s |
| **Function map prefix** | **~700–1000** | **~700–1000** | **~250–400** | **~100–200** |

Key wins:
- **5–7× over serial TC** for N=16, M>255
- **8–40× over serial TC** for N>16
- **Competitive with monoid** for N=16, M≤255 (no monoid precomputation needed)

## Integration

### New Files

- `cuda/prefix_compose.cu` — Block reduce + scan kernels, C API, built-in tests
- `src/gpu_bridge_prefix_compose.py` — Python ctypes bridge

### Modified Files

- `src/optimized_engine.py` — Add `prefix+gpu` config, update auto-selection:
  - M ≤ 255: monoid batch (unchanged)
  - M > 255: function map prefix (new)
  - N > 16: function map prefix (new, future)
- `src/simulation.py` or new utility — Extract tmap from DFAMatrices

### Precomputation

```python
def precompute_tmap(dm: DFAMatrices) -> np.ndarray:
    """Build fused transition map: tmap[char * N + state] = δ(state, char)."""
    N = dm.n_states  # padded to 16
    tmap = np.zeros(256 * N, dtype=np.uint8)
    for byte_val in range(256):
        ch_name = chr(byte_val)
        if ch_name in dm.char_to_idx:
            T = dm.matrices[ch_name]  # N×N transition matrix
            for s in range(N):
                for s2 in range(N):
                    if T[s2, s] > 0:
                        tmap[byte_val * N + s] = s2
                        break
                else:
                    tmap[byte_val * N + s] = s  # identity
        else:
            for s in range(N):
                tmap[byte_val * N + s] = s  # identity for unmapped chars
    return tmap
```

### C API

```c
int prefix_engine_init(
    const uint8_t *tmap,        // 256 × N fused transition map
    const uint8_t *accept,      // N-entry accept table
    int start_state,
    int N, int K,               // N = padded states, K = block size
    int max_total_chars,
    int max_batch
);

int prefix_engine_dispatch(
    const uint8_t *raw_concat,
    const int *offsets,
    int *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms
);

void prefix_engine_destroy();
```

## Testing

1. **Correctness**: Compare function map prefix results against sequential DFA simulation and monoid engine for all existing test regexes.
2. **Edge cases**: Empty strings, single-char strings, L < K (single block), L exactly K, padding characters.
3. **Benchmark**: Compare throughput against monoid batch and serial TC engines across (B, L, N, σ) combinations.
4. **Built-in CUDA tests**: Even-A DFA (same as other engines), run as standalone binary.

## Scope

This spec covers N ≤ 16 (single-tile maps, shuffle-based composition). N > 16 support (multi-tile maps, shared-memory-based composition) is a natural extension but out of scope for the initial implementation.
