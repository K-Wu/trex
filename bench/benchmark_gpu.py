"""
GPU benchmark harness for tensor-core regex matching.

Compares GPU (tensor-core prefix scan) against CPU backends
across string length, DFA size, and batch size axes.
"""

from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from dataclasses import dataclass, asdict

from src.regex_to_dfa import compile_regex
from src.simulation import (
    DFAMatrices, simulate_sequential, simulate_prefix_scan,
)
from src.generate_data import PATTERNS
from src.gpu_bridge import GPUSimulator


@dataclass
class BenchResult:
    suite: str
    pattern_name: str
    dfa_states: int
    alphabet_size: int
    string_length: int
    method: str
    elapsed_ms: float
    throughput_mbs: float
    correct: bool


def bench_throughput_vs_length(gpu: GPUSimulator):
    """Suite 1: Throughput vs string length for (a|b)*abb."""
    pat = PATTERNS['abb']
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    alpha = sorted(dfa.alphabet)
    results = []

    lengths = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

    for L in lengths:
        random.seed(42)
        s = ''.join(random.choice(alpha) for _ in range(L))

        # CPU sequential
        iters = max(1, min(100, 500000 // max(L, 1)))
        start = time.perf_counter()
        for _ in range(iters):
            r_seq = simulate_sequential(dfa, s)
        elapsed = (time.perf_counter() - start) / iters
        results.append(BenchResult(
            'throughput_vs_length', pat.name, dm.n_states_raw, len(alpha),
            L, 'cpu_sequential', elapsed * 1000,
            L / (elapsed * 1e6) if elapsed > 0 else 0, True
        ))

        # CPU prefix scan
        iters_scan = max(1, min(20, 50000 // max(L, 1)))
        start = time.perf_counter()
        for _ in range(iters_scan):
            r_scan = simulate_prefix_scan(dm, s)
        elapsed = (time.perf_counter() - start) / iters_scan
        results.append(BenchResult(
            'throughput_vs_length', pat.name, dm.n_states_raw, len(alpha),
            L, 'cpu_prefix_scan', elapsed * 1000,
            L / (elapsed * 1e6) if elapsed > 0 else 0, r_scan == r_seq
        ))

        # GPU
        gpu_iters = max(1, min(50, 200000 // max(L, 1)))
        # Warmup
        for _ in range(3):
            gpu.simulate(dm, s)
        start = time.perf_counter()
        for _ in range(gpu_iters):
            r_gpu = gpu.simulate(dm, s)
        elapsed = (time.perf_counter() - start) / gpu_iters
        results.append(BenchResult(
            'throughput_vs_length', pat.name, dm.n_states_raw, len(alpha),
            L, 'gpu_tensor_core', elapsed * 1000,
            L / (elapsed * 1e6) if elapsed > 0 else 0, r_gpu == r_seq
        ))

        print(f"  L={L:>6}: cpu_seq={results[-3].throughput_mbs:.1f} MB/s, "
              f"cpu_scan={results[-2].throughput_mbs:.1f} MB/s, "
              f"gpu={results[-1].throughput_mbs:.1f} MB/s")

    return results


def bench_dfa_size_scaling(gpu: GPUSimulator):
    """Suite 2: DFA size effects on throughput."""
    L = 4096
    pattern_names = ['even_a', 'ab_star', 'binary_div3', 'abb',
                     'hex_number', 'identifier', 'fixed_keyword']
    results = []

    for name in pattern_names:
        if name not in PATTERNS:
            continue
        pat = PATTERNS[name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        alpha = sorted(dfa.alphabet)

        random.seed(42)
        s = ''.join(random.choice(alpha) for _ in range(L))

        for method_name, method_fn in [
            ('cpu_sequential', lambda: simulate_sequential(dfa, s)),
            ('gpu_tensor_core', lambda: gpu.simulate(dm, s)),
        ]:
            # Warmup
            for _ in range(3):
                method_fn()
            iters = 50
            start = time.perf_counter()
            for _ in range(iters):
                r = method_fn()
            elapsed = (time.perf_counter() - start) / iters

            results.append(BenchResult(
                'dfa_size_scaling', name, dm.n_states_raw, len(alpha),
                L, method_name, elapsed * 1000,
                L / (elapsed * 1e6) if elapsed > 0 else 0, True
            ))

        cpu_ms = results[-2].elapsed_ms
        gpu_ms = results[-1].elapsed_ms
        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        print(f"  {name:>16} ({dm.n_states_raw:>2} states): "
              f"cpu={cpu_ms:.3f}ms gpu={gpu_ms:.3f}ms speedup={speedup:.2f}x")

    return results


def bench_batch_scaling(gpu: GPUSimulator):
    """Suite 3: Batch size scaling."""
    pat = PATTERNS['abb']
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    alpha = sorted(dfa.alphabet)
    L = 1024
    results = []

    batch_sizes = [1, 4, 16, 64, 256, 1024]

    for B in batch_sizes:
        random.seed(42)
        strings = [''.join(random.choice(alpha) for _ in range(L)) for _ in range(B)]

        # CPU sequential batch
        start = time.perf_counter()
        for s in strings:
            simulate_sequential(dfa, s)
        cpu_elapsed = time.perf_counter() - start
        results.append(BenchResult(
            'batch_scaling', pat.name, dm.n_states_raw, len(alpha),
            L, 'cpu_sequential', cpu_elapsed * 1000,
            B * L / (cpu_elapsed * 1e6) if cpu_elapsed > 0 else 0, True
        ))

        # GPU batch
        for _ in range(2):
            gpu.simulate_batch(dm, strings)
        start = time.perf_counter()
        gpu.simulate_batch(dm, strings)
        gpu_elapsed = time.perf_counter() - start
        results.append(BenchResult(
            'batch_scaling', pat.name, dm.n_states_raw, len(alpha),
            L, 'gpu_tensor_core', gpu_elapsed * 1000,
            B * L / (gpu_elapsed * 1e6) if gpu_elapsed > 0 else 0, True
        ))

        speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else 0
        print(f"  batch={B:>5}: cpu={cpu_elapsed*1000:.1f}ms "
              f"gpu={gpu_elapsed*1000:.1f}ms speedup={speedup:.2f}x")

    return results


def main():
    print("=== GPU Benchmark: Tensor-Core DFA Scan ===\n")

    try:
        gpu = GPUSimulator()
    except Exception as e:
        print(f"GPU not available: {e}")
        return

    all_results = []

    print("Suite 1: Throughput vs String Length")
    all_results.extend(bench_throughput_vs_length(gpu))

    print("\nSuite 2: DFA Size Scaling (L=4096)")
    all_results.extend(bench_dfa_size_scaling(gpu))

    print("\nSuite 3: Batch Size Scaling (L=1024)")
    all_results.extend(bench_batch_scaling(gpu))

    # Save results
    os.makedirs('results', exist_ok=True)
    out_path = 'results/gpu_benchmark_results.json'
    with open(out_path, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
