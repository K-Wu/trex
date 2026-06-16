# TERX Parallel Multi-String & Cascading Regex Design

## Problem Statement

The current TERX kernel processes one string at a time via Blelloch prefix scan, achieving 0.4 GB/s peak throughput on H200 NVL — roughly 0.02% of the GPU's 1,979 int8 TOPS and 8% of HBM bandwidth. Three factors explain this:

1. **Serial kernel launches**: 2*log2(L)+3 separate kernel launches per string (~5-10us overhead each)
2. **Low arithmetic intensity**: OI = 5.33 ops/byte vs ridge point 412 — memory-bound by 80x
3. **No inter-string parallelism**: batch path runs strings sequentially on the host

Target use cases span cuDF column filtering (millions of short strings), text editor search (few medium strings), and genomic/log analysis (few very long strings). The design must cover all regimes, handle variable-length strings within a batch, support cascading multi-regex pipelines, and provide a comprehensive correctness/performance test harness.

## 1. Performance Ceiling Analysis

### Current bottleneck breakdown (L=16M, Blelloch)

| Component | Time (est.) |
|-----------|-------------|
| Kernel launch overhead (51 launches x ~5us) | ~0.25 ms |
| H2D copy (L ints = 64 MB via PCIe/NVLink) | ~1 ms |
| Gather kernel (L x 256B reads from HBM) | ~0.8 ms |
| D2D copy (orig -> work, 4 GB) | ~0.8 ms |
| Upsweep (24 steps, ~2L matmuls total) | ~16 ms |
| Downsweep (24 steps, ~2L matmuls total) | ~16 ms |
| Inclusive convert (L matmuls) | ~5 ms |
| D2H copy (256 bytes) | ~0.01 ms |
| **Total measured** | **~42 ms** |

Note: the CUDA event timer wraps the full `simulate()` call including the synchronous H2D memcpy. Scan steps (upsweep + downsweep + inclusive convert) dominate at ~88% of total time.

The scan steps (upsweep + downsweep + convert) are ~64% of time. Each WMMA matmul does 4,096 int8 ops on 768 bytes of data (read 2 matrices, write 1). At 4.8 TB/s HBM bandwidth, the theoretical minimum time for ~4L matrix reads/writes at 256 bytes each is:

    4 * 16M * 256 / 4.8e12 = 3.4 ms

So the current ~26ms scan time is ~7.6x slower than the bandwidth limit. The gap comes from:
- Kernel launch overhead between steps
- Low SM occupancy in later upsweep / early downsweep steps (L/stride warps, collapsing to 1)
- No overlap between H2D transfer and compute

### Theoretical throughput ceilings

| Bottleneck | Throughput (single string, L=16M) |
|------------|-----------------------------------|
| Current measured | 0.4 GB/s |
| Bandwidth-optimal scan (single string) | ~4.7 GB/s |
| HBM bandwidth (pure streaming) | 4.8 TB/s (4,800 GB/s) |
| Peak int8 tensor | 1,979 TOPS |

For **many short strings** processed in parallel, the ceiling is HBM bandwidth divided by bytes touched per string. Each string of length L touches: L*4 bytes input + L*256 bytes gathered matrices + ~3*L*256 bytes scan read/write = L*772 bytes. Throughput ceiling (in input bytes/s): 4.8 TB/s / 772 = 6.2 GB/s of input characters.

With the gather optimization (fusing gather + first scan step), this improves further.

## 2. Parallelism Strategies by Regime

### R1: Many Short Strings (L <= 1024)

**Target**: cuDF column operations, log grep, firewall rule matching. Millions of strings, 10-1000 chars each.

**Strategy**: Single mega-kernel, one warp per string, sequential matmul chain (no scan).

For L <= 32, a single warp can chain L matmuls sequentially — each thread in the warp participates in one WMMA matmul, results stay in registers/shared memory between multiplications. No need for a prefix scan at all; the sequential chain is O(L) work and O(L) depth, but since L is small, depth doesn't matter. What matters is that we can run 4,224+ strings concurrently (132 SMs * 32 warps/SM).

For 32 < L <= 1024, each string gets a small intra-warp-group scan. A group of 1-4 warps cooperates: warp 0 scans positions 0..L/4-1 sequentially, warp 1 scans L/4..L/2-1, etc. Then a 2-4 element scan over partial products merges results. This is equivalent to a 2-level hierarchical scan where the first level is sequential and the second is tiny.

**Kernel design**:
```
__global__ void batch_short_strings_kernel(
    const int8_t *trans_matrices,   // [alphabet_size * 256]
    const int *all_chars,           // [total_chars] concatenated
    const int *offsets,             // [B+1] string start offsets
    int *results,                   // [B] accept/reject
    const int8_t *accept_mask,
    int start_state,
    int B                           // batch size
)
```

Variable-length support: strings are concatenated with an offset array (CSR-like format, same as cuDF's string column representation). Each warp reads `offsets[warp_id]` and `offsets[warp_id+1]` to determine its string boundaries.

**Expected throughput**: With 132 SMs, 32 warps/SM, and strings of L=100, each warp does ~100 matmuls sequentially. At ~50ns per WMMA matmul (including memory), one string takes ~5us. GPU processes 4,224 strings every 5us = 844M strings/s = 84.4 GB/s of input text. ~175x faster than current single-string path.

### R2: Medium Batch (L = 1K-100K, 100-10K strings)

**Target**: Document search, medium-scale text processing.

**Strategy**: Intra-block cooperative scan per string, strings across blocks.

Each string gets one or more thread blocks. Within a block, warps cooperate on a shared-memory prefix scan:
- Each warp scans a chunk of positions sequentially (e.g., 32-64 positions)
- Intra-block scan over warp-level partial products (warp-shuffle or shared memory)
- One kernel launch for all strings

For strings of length L=10K with 8 warps per block:
- Each warp handles L/8 = 1250 positions sequentially (~1250 matmuls)
- 3-step intra-block scan over 8 partial products
- Total: ~1250 + 3 = ~1253 sequential matmuls per string
- Multiple strings across blocks

**Variable-length handling**: Blocks are assigned to strings via a string-to-block mapping computed on the host. Short strings get 1 block; long strings get multiple blocks (transitioning to R3).

### R3: Few Long Strings (L = 100K-100M, 1-100 strings)

**Target**: Genome matching, full-document regex, large log file scan.

**Strategy**: Multi-block hierarchical scan with decoupled look-back.

This is the CUB-style approach adapted for non-commutative matrix multiply:

**Phase 1 — Tile-level scan**: Each block scans a tile of T positions (T=256-1024). Within the tile, the block's warps cooperate exactly as in R2. Each block produces:
- A local inclusive prefix scan for its tile
- A tile aggregate (product of all matrices in the tile)

**Phase 2 — Inter-tile coordination (decoupled look-back)**:
Instead of a separate kernel for the inter-tile scan, blocks use the decoupled look-back protocol:
1. Block i computes its tile aggregate and publishes it atomically
2. Block i then "looks back" at blocks i-1, i-2, ... to collect their inclusive prefixes
3. Once block i has the prefix from all preceding tiles, it incorporates it into its local results

This eliminates the need for multiple kernel launches. All blocks execute in a single persistent kernel.

**Non-commutative adaptation**: Standard decoupled look-back assumes commutative operations for the look-back reduction. Since matrix multiply is non-commutative but associative, the look-back must accumulate in order: block i collects prefix(i-1), then checks if prefix(i-1) already includes all preceding blocks (the "inclusive prefix" flag), or if it's just a "tile aggregate" that needs further look-back. The ordering is: `result = prefix(i-1) @ aggregate(i)` — preceding prefix on the LEFT.

**Status flags** (per tile, in global memory):
- `INVALID`: tile not yet computed
- `AGGREGATE_AVAILABLE`: tile aggregate ready, but inclusive prefix not yet known
- `PREFIX_AVAILABLE`: inclusive prefix ready (includes all prior tiles)

**Expected throughput**: For L=16M with T=512 tiles: 32K tiles, each doing 512 sequential matmuls. At 132 SMs with ~4 blocks each, all tiles process in ~60 waves. Eliminating kernel launch overhead and maintaining full SM occupancy should approach the bandwidth ceiling of ~4.7 GB/s — a ~12x improvement over current 0.4 GB/s.

### R4: Adaptive Dispatch

The runtime selects the strategy based on (B, L_max, L_total):

```
if B >= 1024 and L_max <= 1024:
    R1: batch_short_strings_kernel
elif B >= 4 and L_max <= 100K:
    R2: intra_block_cooperative_kernel
else:
    R3: decoupled_lookback_kernel (per string, possibly concurrent)
```

For **mixed-length batches** (e.g., strings ranging from 10 to 10K chars), the dispatcher can:
- Bin strings by length into R1/R2/R3 buckets
- Launch one kernel per bucket concurrently (CUDA streams)
- Or use a single R2-style kernel where each block adapts its strategy based on string length

## 3. Variable-Length String Support

### Memory layout (CSR format)

```
chars:   [a b b | a a | a b b a b b | ...]   (concatenated)
offsets: [0, 3, 5, 11, ...]                   (B+1 ints)
```

This matches cuDF's string column representation exactly.

### Warp-to-string assignment

For R1: one warp per string. Warp i reads `offsets[i]`, `offsets[i+1]`, processes `chars[offsets[i] .. offsets[i+1]-1]`.

For R2/R3: a host-side prefix sum over string lengths determines block-to-string mapping. Each block gets a (string_id, start_pos, end_pos) tuple.

### Padding elimination

Current approach pads to next power of 2. With the sequential-chain approach (R1) and hierarchical scan (R2/R3), no padding is needed — the scan operates on exact length. This saves significant memory and compute for non-power-of-2 lengths.

## 4. Cascading Regex & Mixed Operations

### 4.1 Multi-DFA Single Pass

When applying K regexes to the same strings, the naive approach runs K separate scans, each reading the input K times. The fused approach:

**K-way gather fusion**: Read each input character once, look up transition matrices for all K DFAs simultaneously, store K gathered matrix sequences.

```
__global__ void multi_dfa_gather_kernel(
    const int *chars,                   // [total_chars]
    const int8_t *trans_all,            // [K * alphabet_size * 256]
    int8_t *gathered_all,               // [K * total_positions * 256]
    const int *offsets,                 // [B+1]
    int K, int B
)
```

**Independent parallel scans**: Each (string, DFA) pair runs its own scan. With B strings and K DFAs, there are B*K independent scan tasks — more parallelism for the GPU.

**Shared memory constraint**: Each DFA's scan needs ~256 bytes per position in shared memory for the matmul. For K DFAs, that's 256K bytes. At K=4, this is 1KB per position — feasible for shared memory tiles of 32-64 positions.

**When to use**: K <= 8 DFAs with the same alphabet applied to the same strings. Beyond K=8, memory pressure makes separate passes more efficient.

### 4.2 Cascading Pipeline (regex -> op -> regex)

For chains like `filter(r1) -> extract(r2) -> match(r3)`:

**Stream compaction between stages**: After regex1 filters strings, compact the surviving string indices before running regex2. This reduces work proportional to regex1's selectivity.

```
Stage 1: regex1 on B strings -> B boolean results
Stage 2: compact (stream compaction) -> B' surviving string indices (B' <= B)
Stage 3: regex2 on B' strings (using indices into original char buffer, no data copy)
Stage 4: compact -> B'' indices
Stage 5: regex3 on B'' strings
```

**Key optimization — zero-copy substring passing**: When regex1 extracts a substring (match positions), regex2 receives (offset, length) pairs pointing into the original char buffer. No intermediate materialization. The gather kernel for regex2 reads from `chars[match_start .. match_end]` instead of copying.

**Fused filter-gather**: If regex1 is a boolean filter and regex2 runs on survivors, fuse the compaction with regex2's gather phase. A single kernel reads regex1's results, skips rejected strings, and gathers regex2's transition matrices for surviving strings.

### 4.3 Short-Circuit Evaluation

For boolean combinations of regexes:

**AND short-circuit**: Run the most selective (highest reject rate) regex first. After each stage, compact to survivors. If selectivity is unknown, estimate via DFA structure:
- Selectivity heuristic: `|accepting states| / |total states|` as rough proxy
- Or: sample 1% of strings with the CPU sequential path to estimate selectivity (~0.1ms for 1000 strings)

**OR short-circuit**: Run the least selective (highest accept rate) regex first. Remove accepted strings before running the next regex.

**Cost model for ordering**:
```
cost(DFA_i) = estimated_surviving_strings * L_avg * scan_cost_per_char(DFA_i)
```
Sort DFAs by ascending cost for AND, descending for OR. The scan cost per character depends on DFA size (number of states) since larger DFAs have denser transition matrices.

### 4.4 Mixed Regex + Literal String Match

Literal string matching (e.g., `str.contains("error")`) is much cheaper than DFA scan — it's a simple substring search (Aho-Corasick, PFAC on GPU, or even just `memchr` + verify).

**Strategy**: Always run literal matches before regex matches, since they're 10-100x faster per character and can dramatically reduce the working set.

Pipeline: `literal_filter -> compact -> regex_scan`

For patterns that are partially literal (e.g., `error: [0-9]+`), the regex compiler could extract the literal prefix "error: " and use it as a pre-filter: only strings containing "error: " proceed to the full DFA scan.

### 4.5 Multi-DFA Scan Fusion (Product DFA Avoidance)

For `regex1 AND regex2`, the theoretical approach is to construct the product DFA (Cartesian product of states). For DFAs with N1 and N2 states, the product has N1*N2 states. If N1*N2 <= 16, the product DFA fits in our 16x16 matrix and we can run a single scan — strictly better than two separate scans.

For N1*N2 > 16, we can't fit in one matrix. Options:
- **32x32 matrices**: Use 4 WMMA 16x16 tiles to emulate 32x32 matmul. Doubles memory and compute but handles up to 32 states.
- **Independent parallel scans** (the approach in 4.1): Run both DFAs separately but in a single kernel launch.

The product DFA approach is recommended when N1*N2 <= 16 (covers many practical cases: 2-state AND 3-state, 4-state AND 4-state, etc.).

## 5. Comprehensive Test Scheme

### 5.1 Correctness Tests

#### Single-string correctness (existing, extended)

| Test | Description | Purpose |
|------|-------------|---------|
| T1.1 | Known accept/reject for each DFA pattern | Basic sanity |
| T1.2 | Random strings at power-of-2 lengths (1, 2, 4, ..., 2^20) | Boundary conditions |
| T1.3 | Random strings at power-of-2 minus 1 (1, 3, 7, 15, ...) | Off-by-one in padding |
| T1.4 | All-same-character strings (L=1 to 10K) | Degenerate input |
| T1.5 | Strings that accept on last character only | Late acceptance |
| T1.6 | Strings that would accept at position k but not at end | Early accept then reject |
| T1.7 | Empty strings | Edge case |
| T1.8 | Single character strings | Minimum non-trivial |
| T1.9 | L=1 to 64, exhaustive over all binary strings (2-char alphabet) | Brute force for small L |

#### Multi-string batch correctness

| Test | Description | Purpose |
|------|-------------|---------|
| T2.1 | Uniform length, all accept | Batch baseline |
| T2.2 | Uniform length, all reject | Batch baseline |
| T2.3 | Uniform length, mixed accept/reject (50/50) | Mixed results |
| T2.4 | Uniform length, 1% accept (selective pattern) | Sparse accept |
| T2.5 | Uniform length, 99% accept (permissive pattern) | Sparse reject |
| T2.6 | **Variable length**: lengths drawn from [1, 10, 100, 1000, 10000] | CSR layout correctness |
| T2.7 | **Variable length**: lengths drawn from uniform(1, 1000) | Random variable length |
| T2.8 | **Variable length**: one string is 100K, rest are 10 chars | Extreme length variance |
| T2.9 | Batch size = 1 (degenerate batch) | Edge case |
| T2.10 | Batch size = GPU warp count (saturate SMs) | Full occupancy |
| T2.11 | Batch size = 1M (stress) | Large batch |
| T2.12 | All strings identical | Degenerate input |
| T2.13 | Strings sorted by length ascending | Ordered input |
| T2.14 | Strings sorted by length descending | Ordered input |
| T2.15 | Strings in random length order | Unordered input |

#### Early/late termination patterns

| Test | Description | Purpose |
|------|-------------|---------|
| T3.1 | String = "abb" + random(L-3) — accept happens at prefix | Early accept (DFA enters accept state early but may leave) |
| T3.2 | String = random(L-3) + "abb" — accept only at suffix | Late accept |
| T3.3 | String = "abb" repeated L/3 times — periodic re-entry to accept | Oscillating acceptance |
| T3.4 | String that visits every DFA state in sequence | Full state coverage |
| T3.5 | String that stays in start state for L-1 chars, then transitions | Late transition |
| T3.6 | Batch: half strings are T3.1 (early), half are T3.2 (late) | Mixed early/late in batch |
| T3.7 | Batch: 10% T3.1, 90% all-reject | Sparse accept with early/late mix |

#### Cascading/multi-DFA correctness

| Test | Description | Purpose |
|------|-------------|---------|
| T4.1 | Two DFAs, same strings, verify both results independently | Multi-DFA basic |
| T4.2 | regex1 AND regex2, compare with product DFA | Boolean AND correctness |
| T4.3 | regex1 OR regex2, compare with both run independently | Boolean OR correctness |
| T4.4 | Pipeline: filter(r1) then scan(r2) on survivors only | Cascading correctness |
| T4.5 | Pipeline: extract(r1) then match(r2) on substrings | Substring pass-through |
| T4.6 | Three-stage cascade: r1 -> r2 -> r3 | Deep pipeline |
| T4.7 | Literal prefix "abc" + regex `[0-9]+` | Mixed literal+regex |
| T4.8 | Multi-DFA with different alphabet sizes | Heterogeneous DFAs |

#### Cross-validation

All GPU results cross-validated against:
- CPU sequential simulation
- CPU prefix scan (numpy)
- Python `re` module (for supported patterns)

### 5.2 Performance/Stress Tests

| Test | Description | Metric |
|------|-------------|--------|
| P1 | Throughput vs L (single string): L=64 to 64M | GB/s, kernel-only and end-to-end |
| P2 | Throughput vs B (fixed L=100): B=1 to 10M | Strings/s, GB/s |
| P3 | Throughput vs B (fixed L=10K): B=1 to 100K | Strings/s, GB/s |
| P4 | Variable-length batch: L ~ Uniform(10, 1000), B=1M | GB/s, load balance efficiency |
| P5 | Variable-length batch: L ~ LogNormal(6, 2), B=100K | Skewed length distribution |
| P6 | Multi-DFA: K=1,2,4,8 DFAs, B=100K, L=100 | Scaling with DFA count |
| P7 | Cascade: 2-stage pipeline, vary selectivity of r1 from 1% to 99% | Compaction benefit |
| P8 | DFA size scaling: 2, 4, 8, 16 states, fixed L and B | State count impact |
| P9 | Kernel-only vs end-to-end timing for all regimes | Overhead breakdown |
| P10 | Memory allocation: persistent context vs per-call malloc | Allocation overhead |
| P11 | Concurrent streams: 2, 4, 8 streams with different string batches | Multi-stream scaling |

### 5.3 Regime Transition Tests

| Test | Description | Purpose |
|------|-------------|---------|
| R1 | B=1M, L=10: verify R1 strategy selected and correct | Short-string regime |
| R2 | B=1K, L=10K: verify R2 strategy selected and correct | Medium regime |
| R3 | B=1, L=10M: verify R3 strategy selected and correct | Long-string regime |
| R4 | B sweeps from 1 to 1M with L=100: measure throughput across regimes | Regime transition smoothness |
| R5 | L sweeps from 10 to 10M with B=100: measure throughput across regimes | Length transition smoothness |
| R6 | Mixed batch where optimal strategy differs per string | Adaptive dispatch |

## 6. Kernel-Only vs End-to-End Measurement

The current benchmark measures kernel time (CUDA events wrapping the full `simulate()` call including H2D copy of input chars). For the optimized design, we need two timing modes:

### Kernel-only timing

Wrap only the GPU compute kernels (gather + scan) with CUDA events, excluding all H2D/D2H transfers. This measures raw GPU compute throughput and is the right metric for:
- Comparing scan algorithms (Blelloch vs Hillis-Steele vs decoupled look-back)
- Roofline analysis
- Comparing against theoretical bandwidth ceiling

### End-to-end timing

Include H2D transfer of input, all GPU compute, and D2H transfer of results. This measures practical throughput seen by the application and is the right metric for:
- Comparing GPU vs CPU for real workloads
- Deciding crossover points
- cuDF integration benchmarks

### Overlap timing

For the pipelined version where H2D transfer overlaps with compute (using CUDA streams), measure wall-clock time from first H2D initiation to last D2H completion. This is the metric that matters for R1 (many short strings where transfer and compute can overlap).

## 7. Architecture Summary

```
Application (cuDF / text editor / CLI)
    |
    v
Dispatch Layer
    |-- estimates (B, L_max, L_total, K_dfas)
    |-- selects regime (R1/R2/R3)
    |-- handles variable-length packing (CSR format)
    |-- orders cascading DFA stages by selectivity
    |
    v
+-- R1: batch_short_strings_kernel (warp-per-string, sequential chain)
|   |-- Variable-length via offsets array
|   |-- Multi-DFA: K warps per string for K DFAs
|
+-- R2: cooperative_block_scan_kernel (block-per-string, shared-mem scan)
|   |-- Hierarchical: warps scan chunks, then intra-block merge
|   |-- Multiple strings across blocks
|
+-- R3: decoupled_lookback_kernel (multi-block-per-string)
    |-- Single persistent kernel
    |-- Non-commutative look-back protocol
    |-- Status flags: INVALID / AGGREGATE / PREFIX
    |
    v
Result Buffer (accept/reject per string, or match positions)
    |
    v
Cascade Controller
    |-- stream_compact(results, surviving_indices)
    |-- next_stage_gather(chars, surviving_indices, next_dfa)
    |-- fused_compact_gather when possible
```

## 8. Implementation Priority

1. **R1 (many short strings)**: Highest impact for cuDF use case, simplest kernel
2. **Variable-length support (CSR layout)**: Required for real workloads
3. **R3 (decoupled look-back)**: Highest theoretical speedup for long strings
4. **Multi-DFA fusion**: Needed for cascading regex
5. **R2 (cooperative block scan)**: Falls out naturally as a simpler version of R3
6. **Cascade controller with short-circuit**: Optimization layer on top
7. **Literal pre-filter fusion**: Integration with string match operations
