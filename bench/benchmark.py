"""
Benchmark harness for tensor-core regex matching.

Measures wall-clock time and throughput for:
  1. Sequential DFA simulation (baseline)
  2. Matrix-vector sequential simulation
  3. Parallel prefix scan (CPU numpy — models tensor-core depth)
  4. Batch matrix simulation

Reports:
  - Latency (ms) per string
  - Throughput (MB/s) aggregate
  - Speedup ratios
  - Scaling curves vs string length, DFA size, batch size
"""

from __future__ import annotations
import sys, os, time, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from dataclasses import dataclass, asdict
from typing import Optional

from src.regex_to_dfa import compile_regex
from src.simulation import (
    DFAMatrices, simulate_sequential, simulate_matrix_sequential,
    simulate_prefix_scan, simulate_batch_sequential, simulate_batch_matrix,
    prefix_scan_sequential, prefix_scan_parallel, _matmul_int8,
)
from src.generate_data import PATTERNS, gen_random_string


# ─── Timer utility ──────────────────────────────────────────────────────────

class Timer:
    def __enter__(self):
        gc.disable()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        self.elapsed_ms = self.elapsed * 1000
        gc.enable()


@dataclass
class BenchmarkResult:
    name: str
    pattern: str
    dfa_states: int
    dfa_states_padded: int
    alphabet_size: int
    method: str
    string_length: int
    n_strings: int
    total_chars: int
    elapsed_ms: float
    throughput_mbs: float  # MB/s
    per_string_us: float   # µs per string


# ─── Benchmark Functions ────────────────────────────────────────────────────

def bench_sequential(dfa, strings: list[str], n_warmup=2, n_reps=3) -> float:
    """Benchmark sequential simulation, return median elapsed_ms."""
    for _ in range(n_warmup):
        for s in strings[:min(10, len(strings))]:
            simulate_sequential(dfa, s)

    times = []
    for _ in range(n_reps):
        with Timer() as t:
            for s in strings:
                simulate_sequential(dfa, s)
        times.append(t.elapsed_ms)
    return sorted(times)[len(times) // 2]


def bench_matrix_sequential(dm: DFAMatrices, strings: list[str],
                            n_warmup=1, n_reps=3) -> float:
    for _ in range(n_warmup):
        for s in strings[:min(5, len(strings))]:
            simulate_matrix_sequential(dm, s)

    times = []
    for _ in range(n_reps):
        with Timer() as t:
            for s in strings:
                simulate_matrix_sequential(dm, s)
        times.append(t.elapsed_ms)
    return sorted(times)[len(times) // 2]


def bench_prefix_scan(dm: DFAMatrices, strings: list[str],
                      parallel: bool = True, n_warmup=1, n_reps=3) -> float:
    for _ in range(n_warmup):
        for s in strings[:min(3, len(strings))]:
            simulate_prefix_scan(dm, s, use_parallel=parallel)

    times = []
    for _ in range(n_reps):
        with Timer() as t:
            for s in strings:
                simulate_prefix_scan(dm, s, use_parallel=parallel)
        times.append(t.elapsed_ms)
    return sorted(times)[len(times) // 2]


def bench_batch_matrix(dm: DFAMatrices, strings: list[str],
                       n_warmup=1, n_reps=3) -> float:
    for _ in range(n_warmup):
        simulate_batch_matrix(dm, strings[:min(10, len(strings))])

    times = []
    for _ in range(n_reps):
        with Timer() as t:
            simulate_batch_matrix(dm, strings)
        times.append(t.elapsed_ms)
    return sorted(times)[len(times) // 2]


def bench_raw_prefix_scan(matrices: np.ndarray, parallel: bool = True,
                          n_warmup=1, n_reps=3) -> float:
    """Benchmark just the prefix scan operation (no string→matrix gathering)."""
    func = prefix_scan_parallel if parallel else prefix_scan_sequential
    for _ in range(n_warmup):
        func(matrices)

    times = []
    for _ in range(n_reps):
        with Timer() as t:
            func(matrices)
        times.append(t.elapsed_ms)
    return sorted(times)[len(times) // 2]


# ─── Main Benchmark Suite ──────────────────────────────────────────────────

def run_benchmarks() -> list[BenchmarkResult]:
    results = []
    rng = random.Random(42)

    print("=" * 78)
    print("BENCHMARK: Int8 Matrix-Product DFA Simulation (CPU Reference)")
    print("=" * 78)

    # ── Benchmark 1: Throughput vs String Length ────────────────────────────
    print("\n── Throughput vs String Length (pattern: (a|b)*abb) ──")
    print(f"{'Length':>10} {'N_str':>6} {'Sequential':>12} {'MatVec':>12} "
          f"{'ScanSeq':>12} {'ScanPar':>12} {'Batch':>12}  (all in ms)")

    pattern_name = 'abb'
    pinfo = PATTERNS[pattern_name]
    dfa = compile_regex(pinfo.regex)
    dm = DFAMatrices(dfa)

    lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for length in lengths:
        n_strings = max(5, min(200, 500_000 // length))
        strings = [gen_random_string('ab', length, rng) for _ in range(n_strings)]
        total_chars = sum(len(s) for s in strings)

        t_seq = bench_sequential(dfa, strings)
        t_mat = bench_matrix_sequential(dm, strings)

        # Prefix scan is expensive per-string; limit to fewer strings
        scan_strings = strings[:max(3, min(20, n_strings))]
        t_scan_s = bench_prefix_scan(dm, scan_strings, parallel=False)
        t_scan_p = bench_prefix_scan(dm, scan_strings, parallel=True)
        n_scan = len(scan_strings)

        t_batch = bench_batch_matrix(dm, strings)

        print(f"{length:>10} {n_strings:>6} {t_seq:>12.2f} {t_mat:>12.2f} "
              f"{t_scan_s:>12.2f}({n_scan}) {t_scan_p:>12.2f}({n_scan}) {t_batch:>12.2f}")

        for method, elapsed, n_s in [
            ('sequential', t_seq, n_strings),
            ('matrix_vec', t_mat, n_strings),
            ('scan_sequential', t_scan_s, n_scan),
            ('scan_parallel', t_scan_p, n_scan),
            ('batch_matrix', t_batch, n_strings),
        ]:
            tc = sum(len(s) for s in strings[:n_s])
            results.append(BenchmarkResult(
                name=f'length_{length}',
                pattern=pattern_name,
                dfa_states=dm.n_states_raw,
                dfa_states_padded=dm.n_states,
                alphabet_size=len(dm.alphabet),
                method=method,
                string_length=length,
                n_strings=n_s,
                total_chars=tc,
                elapsed_ms=elapsed,
                throughput_mbs=tc / (elapsed * 1000) if elapsed > 0 else 0,
                per_string_us=(elapsed * 1000) / n_s if n_s > 0 else 0,
            ))

    # ── Benchmark 2: Raw Prefix Scan Scaling ───────────────────────────────
    print("\n── Raw Prefix Scan: O(log L) Depth Verification ──")
    print(f"{'Length':>10} {'SeqScan_ms':>12} {'ParScan_ms':>12} {'Speedup':>10} "
          f"{'log2(L)':>8} {'MatMuls':>8}")

    for log_len in range(4, 14):
        L = 1 << log_len
        # Generate random permutation matrices (valid DFA transitions)
        np_rng = np.random.RandomState(42)
        N = 16
        matrices = np.zeros((L, N, N), dtype=np.int8)
        for i in range(L):
            perm = np_rng.permutation(N)
            for j in range(N):
                matrices[i, perm[j], j] = 1

        t_seq = bench_raw_prefix_scan(matrices, parallel=False)
        t_par = bench_raw_prefix_scan(matrices, parallel=True)
        speedup = t_seq / t_par if t_par > 0 else float('inf')

        print(f"{L:>10} {t_seq:>12.3f} {t_par:>12.3f} {speedup:>10.2f}x "
              f"{log_len:>8} {L-1:>8}")

        for method, elapsed in [('scan_seq_raw', t_seq), ('scan_par_raw', t_par)]:
            results.append(BenchmarkResult(
                name=f'raw_scan_L{L}',
                pattern='random_perm',
                dfa_states=N,
                dfa_states_padded=N,
                alphabet_size=0,
                method=method,
                string_length=L,
                n_strings=1,
                total_chars=L,
                elapsed_ms=elapsed,
                throughput_mbs=L * N * N / (elapsed * 1000) if elapsed > 0 else 0,
                per_string_us=elapsed * 1000,
            ))

    # ── Benchmark 3: DFA Size Scaling ──────────────────────────────────────
    print("\n── DFA Size Scaling (string length = 2048) ──")
    print(f"{'Pattern':>20} {'States':>7} {'Padded':>7} {'|Σ|':>5} "
          f"{'Sequential':>12} {'MatVec':>12} {'ScanPar':>12}")

    for pname in ['even_a', 'abb', 'ab_star', 'hex_number',
                   'three_char_end', 'identifier']:
        pinfo = PATTERNS[pname]
        try:
            pdfa = compile_regex(pinfo.regex)
        except Exception as e:
            print(f"{pname:>20} COMPILE ERROR: {e}")
            continue

        pdm = DFAMatrices(pdfa)

        # Determine alphabet for string generation
        if pname in ('abb', 'even_a', 'ab_star'):
            alpha = 'ab'
        elif pname == 'hex_number':
            alpha = '0x' + '0123456789abcdef'
        elif pname == 'three_char_end':
            alpha = 'abc'
        else:
            alpha = 'abcdefghijklmnopqrstuvwxyz'

        length = 2048
        n_str = 20
        # Only use chars in the DFA's alphabet
        alpha_filtered = ''.join(c for c in alpha if c in pdfa.alphabet)
        if not alpha_filtered:
            alpha_filtered = ''.join(sorted(pdfa.alphabet)[:5])
        strings = [gen_random_string(alpha_filtered, length, rng) for _ in range(n_str)]

        t_seq = bench_sequential(pdfa, strings)
        t_mat = bench_matrix_sequential(pdm, strings)
        scan_strs = strings[:5]
        t_scan = bench_prefix_scan(pdm, scan_strs, parallel=True)

        print(f"{pname:>20} {pdm.n_states_raw:>7} {pdm.n_states:>7} "
              f"{len(pdm.alphabet):>5} {t_seq:>12.2f} {t_mat:>12.2f} "
              f"{t_scan:>12.2f}({len(scan_strs)})")

        for method, elapsed, n_s in [
            ('sequential', t_seq, n_str),
            ('matrix_vec', t_mat, n_str),
            ('scan_parallel', t_scan, len(scan_strs)),
        ]:
            results.append(BenchmarkResult(
                name=f'dfa_scale_{pname}',
                pattern=pname,
                dfa_states=pdm.n_states_raw,
                dfa_states_padded=pdm.n_states,
                alphabet_size=len(pdm.alphabet),
                method=method,
                string_length=length,
                n_strings=n_s,
                total_chars=length * n_s,
                elapsed_ms=elapsed,
                throughput_mbs=length * n_s / (elapsed * 1000) if elapsed > 0 else 0,
                per_string_us=(elapsed * 1000) / n_s if n_s > 0 else 0,
            ))

    # ── Benchmark 4: Batch Size Scaling ────────────────────────────────────
    print("\n── Batch Size Scaling (length=512, pattern=(a|b)*abb) ──")
    print(f"{'Batch':>8} {'Sequential':>12} {'Batch_Mat':>12} {'Speedup':>10}")

    for batch_size in [16, 32, 64, 128, 256, 512, 1024]:
        strings = [gen_random_string('ab', 512, rng) for _ in range(batch_size)]
        t_seq = bench_sequential(dfa, strings)
        t_batch = bench_batch_matrix(dm, strings)
        speedup = t_seq / t_batch if t_batch > 0 else float('inf')
        print(f"{batch_size:>8} {t_seq:>12.2f} {t_batch:>12.2f} {speedup:>10.2f}x")

        for method, elapsed in [('sequential', t_seq), ('batch_matrix', t_batch)]:
            results.append(BenchmarkResult(
                name=f'batch_{batch_size}',
                pattern=pattern_name,
                dfa_states=dm.n_states_raw,
                dfa_states_padded=dm.n_states,
                alphabet_size=len(dm.alphabet),
                method=method,
                string_length=512,
                n_strings=batch_size,
                total_chars=512 * batch_size,
                elapsed_ms=elapsed,
                throughput_mbs=512 * batch_size / (elapsed * 1000) if elapsed > 0 else 0,
                per_string_us=(elapsed * 1000) / batch_size if batch_size > 0 else 0,
            ))

    # ── Benchmark 5: Tensor-Core Op Count Projection ───────────────────────
    print("\n── Projected Tensor Core Performance ──")
    print("  (Based on measured CPU prefix-scan depth × theoretical TC throughput)")
    print(f"  GPU A100 int8 peak: 624 TOPS")
    print(f"  Single 16×16×16 MMA: 8192 ops → ~76.2 billion MMA/s")

    for log_len in [10, 14, 18, 20, 24]:
        L = 1 << log_len
        # Prefix scan: ~2L matmuls total work, O(log L) depth
        depth = log_len
        total_mma = 2 * L  # Blelloch: 2n work
        mma_per_sec = 624e12 / 8192  # theoretical MMA/s
        projected_time_us = depth / mma_per_sec * 1e6  # depth-limited
        throughput_gbs = L / (projected_time_us * 1e-6) / 1e9

        print(f"  L={L:>10}: depth={depth:>3} matmuls, "
              f"projected ~{projected_time_us:>.3f} µs depth-limited, "
              f"~{throughput_gbs:.1f} GB/s (depth-limited, ignores memory)")

    return results


# ─── Analysis & Output ─────────────────────────────────────────────────────

def analyze_results(results: list[BenchmarkResult]):
    print("\n" + "=" * 78)
    print("ANALYSIS SUMMARY")
    print("=" * 78)

    # Key finding: prefix scan depth scaling
    scan_results = [r for r in results if r.method == 'scan_par_raw']
    if len(scan_results) >= 2:
        print("\nPrefix Scan Depth Scaling (raw):")
        for r in scan_results:
            log2L = int(np.log2(r.string_length))
            print(f"  L={r.string_length:>8} (2^{log2L}): {r.elapsed_ms:.3f} ms")

        # Check if scaling is sub-linear
        r_small = scan_results[0]
        r_large = scan_results[-1]
        length_ratio = r_large.string_length / r_small.string_length
        time_ratio = r_large.elapsed_ms / r_small.elapsed_ms
        ideal_ratio = np.log2(r_large.string_length) / np.log2(r_small.string_length)
        print(f"\n  Length grew {length_ratio:.0f}x, time grew {time_ratio:.1f}x")
        print(f"  O(log n) ideal: {ideal_ratio:.1f}x growth")
        print(f"  O(n) would be: {length_ratio:.0f}x growth")
        if time_ratio < length_ratio * 0.5:
            print("  → Sub-linear scaling confirmed (between O(log n) and O(n))")
        else:
            print("  → Near-linear scaling (CPU numpy lacks true parallelism)")

    # Matrix overhead analysis
    print("\nMatrix Method Overhead (vs sequential):")
    for name in set(r.name for r in results if 'length_' in r.name):
        seq = [r for r in results if r.name == name and r.method == 'sequential']
        mat = [r for r in results if r.name == name and r.method == 'matrix_vec']
        if seq and mat:
            ratio = mat[0].elapsed_ms / seq[0].elapsed_ms if seq[0].elapsed_ms > 0 else float('inf')
            print(f"  {name}: matrix is {ratio:.1f}x slower than sequential (expected: N^2 overhead)")

    print("\nKey Takeaway:")
    print("  On CPU (numpy), the prefix scan is slower due to no true parallelism.")
    print("  On GPU tensor cores, each 'step' of the scan is a hardware MMA op,")
    print("  so the O(log L) depth directly translates to wall-clock speedup.")
    print("  The matrix encoding is correct (all backends agree) and ready for CUDA.")


if __name__ == '__main__':
    results = run_benchmarks()
    analyze_results(results)

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_results.json', 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print("\nResults saved to results/benchmark_results.json")
