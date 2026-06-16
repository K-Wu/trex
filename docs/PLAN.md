# Implementation Plan: Int8 Tensor-Core Accelerated Regex Matching

## 1. Core Idea

Encode DFA transitions as int8 matrices. String matching becomes a chain of matrix multiplies composed via parallel prefix scan on GPU tensor cores, achieving **O(log L) depth** instead of O(L) sequential steps.

```
Input string:    c₀  c₁  c₂  ...  c_{L-1}
                  ↓   ↓   ↓        ↓
Transition mats: T₀  T₁  T₂  ...  T_{L-1}    (each N×N, int8)
                  ↓   ↓   ↓        ↓
Prefix scan:     T₀  T₁T₀  T₂T₁T₀  ...  T_{L-1}...T₀
                                              ↓
Final state:     (T_{L-1}...T₀) × s₀  →  check accept
```

Key insight: DFA transition matrices have exactly one 1 per column (deterministic), so all products stay in {0,1} — **no overflow, int8 is exact**.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Input                          │
│              regex pattern + input strings               │
└───────────────┬─────────────────────────────────────────┘
                │
    ┌───────────▼───────────┐
    │   Regex → NFA → DFA   │  Thompson's + subset construction + Hopcroft
    │   (offline, once)      │  Output: minimized complete DFA
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  DFA → Int8 Matrices  │  Per-char transition matrix T[c] ∈ {0,1}^{N×N}
    │  Pad N to 16          │  Convention: T[dst][src] = 1 iff δ(src,c) = dst
    └───────────┬───────────┘
                │
    ┌───────────▼────────────────────────────────────────┐
    │              GPU Pipeline                           │
    │                                                     │
    │  1. Gather: input[i] → T[input[i]]   (L matrices)  │
    │  2. Prefix scan: Hillis-Steele / Brent-Kung         │
    │     over matrix products using WMMA int8 MMA        │
    │  3. Final: prefix[L-1] × start_vec → check accept   │
    │                                                     │
    │  Each scan step = batched 16×16 tensor-core MMA     │
    │  Depth: O(log L) steps                              │
    └────────────────────────────────────────────────────┘
```

## 3. Component Breakdown

### 3.1 Regex → DFA Compiler (Python, offline)

| Step | Algorithm | Output |
|------|-----------|--------|
| Parse | Recursive descent | AST |
| NFA | Thompson's construction | ε-NFA with globally-unique state IDs |
| DFA | Subset construction | DFA (may have unreachable dead states) |
| Complete | Add explicit dead/trap state | Every (state, char) has a transition |
| Minimize | Hopcroft's algorithm | Minimal DFA |

Supported syntax: concatenation, `|`, `*`, `+`, `?`, `[a-z]`, `[^...]`, `.`, `\d`, `\w`, `\s`.

### 3.2 DFA → Int8 Transition Matrices

For a DFA with N states and alphabet Σ, produce |Σ| matrices of size N×N:

- `T[c][i][j] = 1` iff `δ(j, c) = i` (column = source, row = destination)
- Pad N to next multiple of 16 (tensor-core tile size); padded states self-loop
- `start_vec`: one-hot vector at start state
- `accept_mask`: bit mask over accept states

Property guarantees that make int8 exact:
- Each column has exactly one 1 (DFA is deterministic and complete)
- Product of two such matrices also has exactly one 1 per column
- All intermediate values ∈ {0, 1}, never overflow int8

### 3.3 GPU Kernels (CUDA, WMMA int8)

**Kernel 1 — Gather:** For each position i in the input string, copy `T[input[i]]` into a contiguous array. Simple parallel copy, memory-bound.

**Kernel 2 — Prefix Scan (core):** Hillis-Steele inclusive prefix scan:
```
for stride = 1, 2, 4, ..., < L:
    for all i ≥ stride in parallel:          ← one kernel launch per stride
        result[i] = result[i] @ result[i - stride]    ← tensor-core MMA
```
- Each step: batch of independent 16×16×16 int8 MMA operations via `wmma::mma_sync`
- `ceil(log₂ L)` steps total, each step is one kernel launch
- Work: O(L log L) matmuls. Depth: O(log L).
- Alternative: Brent-Kung (work-efficient, O(L) matmuls, same O(log L) depth) — more complex but better for large L.

**Kernel 3 — Final state:** Single matrix-vector multiply `prefix[L-1] @ start_vec`, check accept mask.

**Kernel 4 — Batched advance (throughput mode):** For many short strings of equal length, pack state vectors as columns of a wide matrix. Each position: single MMA of `T[c] × States_{16×B}`. O(L) depth but massive data parallelism.

### 3.4 CPU Reference Backends

Five backends for correctness cross-validation:

| Backend | Depth | Work | Purpose |
|---------|-------|------|---------|
| Sequential | O(L) | O(L) | Baseline, ground truth |
| Matrix-vector sequential | O(L) | O(L·N²) | Validates matrix encoding |
| Prefix scan (sequential) | O(L) | O(L·N³) | Reference prefix products |
| Prefix scan (parallel) | O(log L) steps | O(L log L · N³) | Models GPU depth, verifies scan logic |
| Batch matrix | O(L) | O(L·N²·B) | Models GPU batched mode |

## 4. Input Data Strategy

### 4.1 Synthetic Patterns (10 patterns, controlled DFA sizes)

| Pattern | Regex | States | Alphabet | Purpose |
|---------|-------|--------|----------|---------|
| abb | `(a\|b)*abb` | 4-5 | {a,b} | Classic textbook, small DFA |
| binary_div3 | `(0\|(1(01*0)*1))*` | 4 | {0,1} | Numeric, small |
| even_a | `(b*ab*ab*)*b*` | 3 | {a,b} | Minimal DFA |
| ab_star | `(ab)*` | 3-4 | {a,b} | Periodic |
| email_simple | `[a-z]+@[a-z]+\.[a-z]+` | ~8 | a-z+@. | Realistic, medium |
| hex_number | `0x[0-9a-f]+` | 5 | 0-9a-fx | Literal prefix |
| identifier | `[a-z][a-z0-9]*` | 3-4 | a-z0-9 | Large alphabet |
| fixed_keyword | `(if\|else\|while\|for\|return)` | ~16 | a-z | Near tile-size DFA |
| three_char_end | `[a-c]*abc` | ~8 | {a,b,c} | Suffix match |
| nested_alt | `((ab\|cd)(ef\|gh))+` | ~10 | a-h | Nested structure |

### 4.2 String Generation Methods

- **Random:** Uniform over the pattern's alphabet. Provides baseline match/reject ratio.
- **Matching:** Constructed to guarantee acceptance (e.g., random prefix + required suffix).
- **Adversarial:** Designed to keep DFA near accept states but reset repeatedly. Worst-case for branch prediction on CPU sequential.

### 4.3 Scaling Axes for Benchmarks

| Axis | Range | What it tests |
|------|-------|---------------|
| String length L | 64 → 16M | Prefix scan depth scaling (core claim) |
| DFA states N | 3 → 16 | Matrix size effect on MMA utilization |
| Alphabet size \|Σ\| | 2 → 36 | Gather kernel diversity |
| Batch size B | 16 → 16K | Data-parallel throughput saturation |

### 4.4 Real-World Workloads (future, not in current prototype)

- **ANMLZoo:** Standard automata benchmark suite (Snort IDS rules, protein motifs, entity recognition). Requires converting ANML XML → our DFA format.
- **Suricata/Snort rules:** Network IDS rulesets with complex regex.
- **Log parsing:** Common log format patterns (timestamps, IP addresses, HTTP status codes).

## 5. Test Plan

### 5.1 Correctness Tests (38 tests, all passing)

| Suite | Tests | What it verifies |
|-------|-------|------------------|
| DFA Compilation | 10 | Literals, alternation, Kleene star, plus, optional, char classes, ranges, state counts, completeness |
| Matrix Encoding | 7 | Dimensions, dtype, binary values, one-1-per-column invariant, product preservation, identity |
| Simulation Agreement | 6×~200 strings | All 5 backends produce identical accept/reject on 6 patterns across lengths 0-5000 |
| Prefix Scan | 5 | Single matrix, two matrices, sequential=parallel agreement, various lengths (1-128), identity chain |
| Batch Simulation | 1×200 strings | Batch matrix agrees with sequential |
| Edge Cases | 5 | Empty string, single char, 10K-char string, all-same-char, alternating pattern |
| Cross-validation | 4 | Our DFA matches Python `re` module on shared patterns |

### 5.2 GPU-Specific Tests (to implement on target hardware)

- **Bit-exactness:** CPU prefix scan output = GPU WMMA prefix scan output for same input
- **Overflow safety:** Verify all intermediate int32 accumulators ∈ {0, 1, ..., N} (N ≤ 16, so max accumulator = 16, well within int8 range after clamp)
- **Boundary lengths:** L = 1, 2, 15, 16, 17, 255, 256, 257, 1023, 1024 (power-of-2 boundaries in scan)
- **Multi-warp correctness:** L large enough that scan spans multiple warps/blocks

## 6. Benchmark Plan

### 6.1 Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| Throughput | Total input bytes / wall-clock time | GB/s |
| Latency | Wall-clock time for one string | µs |
| Scan depth | Number of kernel launches in prefix scan | count |
| MMA utilization | Fraction of theoretical peak MMA throughput achieved | % |
| Speedup | Baseline time / method time | × |

### 6.2 Baselines to Compare Against

| Baseline | Description | Expected relationship |
|----------|-------------|----------------------|
| CPU sequential | Python DFA loop | Tensor-core should win for L > ~1K |
| CPU re module | Optimized C regex engine | Strong baseline for single strings |
| ngAP (ASPLOS 2024) | State-of-art GPU automata engine | Primary GPU competitor |
| TACO 2025 SpMV | GPU SpMV-based automata | Direct matrix-method competitor |
| Hyperscan (Intel) | SIMD-optimized CPU regex | Fastest CPU engine |

### 6.3 Benchmark Suites

**Suite 1 — Throughput vs. Length:** Fixed pattern `(a|b)*abb`, vary L from 64 to 16M. Key plot: GB/s vs L showing O(log L) depth advantage materializes.

**Suite 2 — Raw Prefix Scan:** Just the scan kernel, random 16×16 permutation matrices. Isolates scan performance from gather/DFA overhead. Shows depth scaling cleanly.

**Suite 3 — DFA Size:** Fixed L=4096, vary pattern across 3-16 states. Shows whether smaller DFAs (sparse matrices, fewer effective ops) change the picture.

**Suite 4 — Batch Size:** Fixed L=512, vary batch 16 to 16K. Shows data-parallel throughput saturation curve.

**Suite 5 — Projected GPU Performance:** Roofline model comparing depth-limited (O(log L) MMA), work-limited (O(L) MMA), and memory-limited (HBM bandwidth) regimes against CPU baseline.

## 7. Execution Roadmap

### Phase 1: Done (current state) ✅

- [x] Regex → NFA → DFA compiler with minimization
- [x] DFA → int8 matrix encoder with 16-padding
- [x] 5 CPU simulation backends
- [x] CUDA kernel source (WMMA int8, compilable)
- [x] 38 correctness tests, all passing
- [x] 10-pattern synthetic data generator
- [x] CPU benchmark harness and visualization scripts
- [x] Novelty assessment with literature review

### Phase 2: GPU Bring-Up (next)

- [ ] Compile CUDA kernel on SM≥72 machine (Turing+)
- [ ] Run GPU correctness tests: compare GPU output vs CPU reference on all 38 test cases
- [ ] Debug any WMMA layout issues (row-major vs col-major, fragment load alignment)
- [ ] Validate int8→int32→int8 clamp path produces identical results to CPU numpy

### Phase 3: GPU Benchmarks

- [ ] Instrument CUDA events for per-kernel timing (gather, scan steps, final)
- [ ] Run Suites 1-4 on real hardware (A100/H100/RTX 4090)
- [ ] Measure MMA utilization via Nsight Compute
- [ ] Compare against CPU sequential and CPU re module baselines
- [ ] Generate throughput/latency/scaling charts

### Phase 4: Optimization

- [ ] **Brent-Kung scan:** Replace Hillis-Steele (O(L log L) work) with Brent-Kung (O(L) work, same O(log L) depth). Significant for large L.
- [ ] **Fused gather+first-scan-step:** Avoid materializing the full gathered array; load transition matrix and immediately combine with neighbor.
- [ ] **Shared-memory scan for small L:** For L ≤ 1024, entire scan fits in shared memory of one block — avoid global memory round-trips between steps.
- [ ] **Multi-tile for N > 16:** Block-matrix multiplication for DFAs with 17-64 states (2×2 or 4×4 tiles of 16×16 MMA).
- [ ] **NFA support:** NFA transition matrices have multiple 1s per column. Products can exceed 1, but values stay small (bounded by N). Clamp to min(val, 1) after each product for reachability semantics.

### Phase 5: Evaluation & Paper

- [ ] Benchmark against ngAP and TACO 2025 SpMV on ANMLZoo workloads
- [ ] Roofline analysis: identify whether bottleneck is compute, memory, or launch overhead
- [ ] Characterize crossover point: at what (L, N, batch_size) does tensor-core beat CUDA-core approaches
- [ ] Write up results with the novelty framing from the assessment

## 8. Key Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Small-matrix underutilization | Tensor cores designed for large GEMM; 16×16 tiles may not saturate MMA pipeline | Batch many independent scans; fuse with gather; measure actual utilization |
| Memory bandwidth bottleneck | Loading L×256 bytes of transition matrices may dominate | Shared-memory caching of transition matrices (only \|Σ\| unique ones); fused kernels |
| Launch overhead | O(log L) kernel launches, each very short for small L | Persistent kernel with grid sync; cooperative groups |
| Brent-Kung complexity | Work-efficient scan harder to implement correctly for non-commutative ops | Start with Hillis-Steele (simpler, correct); optimize later |
| NFA state explosion | Some regexes produce DFAs with >16 states | Multi-tile matmul; or fall back to NFA simulation with boolean-clamp |

## 9. File Manifest

```
tensor_regex/
├── PLAN.md                          ← this document
├── src/
│   ├── regex_to_dfa.py              ← regex→NFA→DFA compiler (376 lines)
│   ├── simulation.py                ← matrix encoder + 5 CPU backends (291 lines)
│   ├── generate_data.py             ← synthetic workload generator (330 lines)
│   └── cuda/
│       ├── tensor_core_dfa_scan.cu  ← full CUDA WMMA int8 kernel (compilable)
│       └── Makefile
├── tests/
│   └── test_correctness.py          ← 38 tests across 7 suites (401 lines)
├── benchmark.py                     ← CPU benchmark harness (407 lines)
└── visualize.py                     ← chart generation (302 lines)
```
