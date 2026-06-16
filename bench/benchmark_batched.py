"""Throughput benchmarks for batched state-vector evolution (Config A).

Compares against monoid R1 (best existing) and v4 prefix scan (baseline).
Measures: kernel-only throughput, end-to-end including batch prep, and
effective tensor core utilization.
"""
import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices
from src.generate_data import PATTERNS


def _random_strings(alphabet, n, length, seed=42):
    rng = random.Random(seed)
    alpha = sorted(alphabet)
    return [''.join(rng.choice(alpha) for _ in range(length)) for _ in range(n)]


def bench_batched_vs_monoid():
    """P1: Throughput scaling with B for batched evolution vs monoid."""
    print("=" * 80)
    print("P1: Throughput vs Batch Size (Config A Batched vs Monoid R1)")
    print("=" * 80)

    pat = PATTERNS['abb']
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)

    try:
        from src.gpu_bridge_batched import BatchedGPUSimulator
        batched_sim = BatchedGPUSimulator()
    except Exception as e:
        print(f"Batched GPU not available: {e}")
        return

    try:
        from src.monoid import compute_monoid
        from src.gpu_bridge_monoid import MonoidGPUSimulator
        md = compute_monoid(dm)
        monoid_sim = MonoidGPUSimulator()
    except Exception:
        md = None
        monoid_sim = None

    L = 512
    batch_sizes = [64, 256, 1024, 4096, 16384, 65536]

    print(f"\nPattern: {pat.regex}, L={L}")
    print(f"{'B':>8}  {'Batched kern':>14}  {'Batched total':>14}  "
          f"{'Monoid kern':>14}  {'Monoid total':>14}  {'Speedup(kern)':>14}")
    print("-" * 95)

    for B in batch_sizes:
        strings = _random_strings(dfa.alphabet, B, L, seed=42)

        # Batched evolution
        eng_b = batched_sim.create_engine(dm, max_B=B + 128, max_L=L + 64)
        eng_b.simulate_batch(strings[:min(64, B)])  # warmup
        _, bk, bt = eng_b.simulate_batch_timed(strings)
        b_gchars_k = B * L / (bk * 1e6) if bk > 0 else float('inf')
        b_gchars_t = B * L / (bt * 1e6) if bt > 0 else float('inf')
        eng_b.destroy()

        # Monoid
        mk_str, mt_str, speedup_str = "N/A", "N/A", "N/A"
        if monoid_sim and md:
            eng_m = monoid_sim.create_engine(
                md, dm,
                max_total_chars=B * L + 4096,
                max_batch=B + 128
            )
            eng_m.simulate_batch(strings[:min(64, B)])  # warmup
            _, mk, mt = eng_m.simulate_batch_timed(strings)
            m_gchars_k = B * L / (mk * 1e6) if mk > 0 else float('inf')
            m_gchars_t = B * L / (mt * 1e6) if mt > 0 else float('inf')
            mk_str = f"{m_gchars_k:.2f} Gc/s"
            mt_str = f"{m_gchars_t:.2f} Gc/s"
            if m_gchars_k > 0 and m_gchars_k != float('inf'):
                speedup_str = f"{b_gchars_k / m_gchars_k:.2f}x"
            eng_m.destroy()

        print(f"{B:>8}  {b_gchars_k:>11.2f} Gc/s  {b_gchars_t:>11.2f} Gc/s  "
              f"{mk_str:>14}  {mt_str:>14}  {speedup_str:>14}")


def bench_length_scaling():
    """P2: Throughput scaling with L."""
    print("\n" + "=" * 80)
    print("P2: Throughput vs String Length (Config A, B=16384)")
    print("=" * 80)

    pat = PATTERNS['abb']
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)

    try:
        from src.gpu_bridge_batched import BatchedGPUSimulator
        sim = BatchedGPUSimulator()
    except Exception as e:
        print(f"Not available: {e}")
        return

    B = 16384
    lengths = [32, 128, 512, 2048, 8192]

    print(f"\nPattern: {pat.regex}, B={B}")
    print(f"{'L':>8}  {'Kernel Gchar/s':>15}  {'Total Gchar/s':>15}  "
          f"{'Kernel ms':>10}  {'Total ms':>10}")
    print("-" * 70)

    for L in lengths:
        strings = _random_strings(dfa.alphabet, B, L, seed=42)
        eng = sim.create_engine(dm, max_B=B + 128, max_L=L + 64)
        eng.simulate_batch(strings[:64])  # warmup
        _, kern_ms, total_ms = eng.simulate_batch_timed(strings)

        total_chars = B * L
        gk = total_chars / (kern_ms * 1e6) if kern_ms > 0 else 0
        gt = total_chars / (total_ms * 1e6) if total_ms > 0 else 0

        print(f"{L:>8}  {gk:>12.2f} Gc/s  {gt:>12.2f} Gc/s  "
              f"{kern_ms:>10.3f}  {total_ms:>10.3f}")
        eng.destroy()


def bench_tensor_utilization():
    """P3: Effective INT8 TOPS vs H200 peak (3,958 TOPS)."""
    print("\n" + "=" * 80)
    print("P3: Tensor Core Utilization Estimate")
    print("=" * 80)

    pat = PATTERNS['abb']
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    N = dm.n_states

    try:
        from src.gpu_bridge_batched import BatchedGPUSimulator
        sim = BatchedGPUSimulator()
    except Exception as e:
        print(f"Not available: {e}")
        return

    sigma = len(dm.alphabet)
    B = 65536
    L = 512

    strings = _random_strings(dfa.alphabet, B, L, seed=42)
    eng = sim.create_engine(dm, max_B=B + 128, max_L=L + 64)
    eng.simulate_batch(strings[:64])  # warmup
    _, kern_ms, total_ms = eng.simulate_batch_timed(strings)

    # FLOPs: per position, sigma MMAs per tile, B/16 tiles
    # Each MMA: 16×16×16 int8 multiply-adds → 2×16³ = 8192 ops
    tiles = B // 16
    mmas_per_position = sigma * tiles
    ops_per_mma = 2 * 16 * 16 * 16
    total_ops = mmas_per_position * ops_per_mma * L
    tflops = total_ops / (kern_ms * 1e-3) / 1e12

    total_chars = B * L
    gchars_kern = total_chars / (kern_ms * 1e6)
    gchars_total = total_chars / (total_ms * 1e6)

    print(f"\n  N={N}, |Σ|={sigma}, B={B}, L={L}")
    print(f"  WMMA tiles: {tiles}")
    print(f"  MMAs per position: {mmas_per_position} ({sigma} per tile × {tiles} tiles)")
    print(f"  Total ops: {total_ops / 1e9:.1f} Gops")
    print(f"  Kernel time: {kern_ms:.3f} ms")
    print(f"  Kernel throughput: {gchars_kern:.2f} Gchar/s")
    print(f"  Total throughput: {gchars_total:.2f} Gchar/s (incl. H2D transfer)")
    print(f"  Effective: {tflops:.1f} TFLOP/s")
    print(f"  H200 INT8 peak: 3,958 TOPS")
    print(f"  Utilization: {tflops / 3958 * 100:.2f}%")
    print(f"\n  vs v4 prefix scan (21 TFLOP/s, 0.53%): "
          f"{tflops / 21:.1f}x more tensor core work")

    eng.destroy()


def bench_packed_scaling():
    """P4: Multi-pattern scaling with PackedEngine."""
    print("\n" + "=" * 80)
    print("P4: Multi-Pattern Scaling (PackedEngine Config C)")
    print("=" * 80)

    from src.packed_engine import PackedEngine

    patterns = [
        "(a|b)*abb",
        "(aa|b)*",
        "(a|b)*a(a|b)",
        "b(a|b)*",
    ]

    B = 1000
    L = 256

    for P in [1, 2, 4]:
        regexes = patterns[:P]
        pe = PackedEngine(regexes)
        info = pe.config_info

        strings = _random_strings("ab", B, L, seed=42)
        results, timing = pe.match_batch_timed(strings)

        total_chars = B * L * P
        elapsed_ms = timing.get('total_ms', timing.get('total_seconds', 0) * 1000)
        throughput = total_chars / (elapsed_ms * 1e6) if elapsed_ms > 0 else 0

        print(f"  P={P:2d} patterns, NP={info['NP']:4d} | "
              f"{elapsed_ms:.1f} ms | "
              f"{throughput:.3f} Gchar/s (CPU, {P} patterns × {B} strings × {L} chars)")


if __name__ == '__main__':
    bench_batched_vs_monoid()
    bench_length_scaling()
    bench_tensor_utilization()
    bench_packed_scaling()
