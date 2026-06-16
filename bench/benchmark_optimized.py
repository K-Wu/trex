"""
Comparative benchmark: all CPU backends + GPU monoid vs GPU v4.

bench_cpu_backends():
    Tests sequential, monoid, and k-gram-monoid backends across patterns and
    (B, L) grid.  Asserts correctness, prints throughput in Mchar/s.

bench_gpu_monoid():
    Compares GPU monoid scan vs GPU v4 matrix scan across (B, L) grid.
    Prints speedup.
"""

from __future__ import annotations
import sys, os, time, json, gc, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential
from src.monoid import compute_monoid, simulate_monoid
from src.kgram import precompute_kgrams, simulate_kgram_monoid, auto_k
from src.generate_data import PATTERNS


# ─── Timer ──────────────────────────────────────────────────────────────────

class Timer:
    def __enter__(self):
        gc.disable()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed_ms = (time.perf_counter() - self._t) * 1000
        gc.enable()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _alphabet_for(pattern_name: str, dfa_alphabet) -> str:
    """Return a suitable generation alphabet for a pattern."""
    if pattern_name in ('abb', 'even_a', 'ab_star'):
        return 'ab'
    elif pattern_name == 'binary_div3':
        return '01'
    else:
        return ''.join(sorted(dfa_alphabet))


def _gen_strings(alphabet: str, B: int, L: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return [''.join(rng.choice(alphabet) for _ in range(L)) for _ in range(B)]


def _throughput_mchars(total_chars: int, elapsed_ms: float) -> float:
    """Compute throughput in Mchar/s."""
    if elapsed_ms <= 0:
        return 0.0
    return total_chars / (elapsed_ms * 1e3)  # chars / (ms * 1000) = Mchar/s


def _bench_fn(fn, n_warmup: int = 2, n_reps: int = 3) -> float:
    """Return median elapsed_ms over n_reps runs (with n_warmup warmup calls)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_reps):
        with Timer() as t:
            fn()
        times.append(t.elapsed_ms)
    return sorted(times)[n_reps // 2]


# ─── CPU Backends Benchmark ──────────────────────────────────────────────────

def bench_cpu_backends() -> list[dict]:
    """
    Compare sequential, monoid, and k-gram monoid backends across patterns
    and a (B, L) grid.
    """
    patterns = ['abb', 'even_a', 'binary_div3', 'ab_star']
    B_grid = [100, 1000, 10_000]
    L_grid = [32, 128, 512, 2048]

    all_results: list[dict] = []

    print("=" * 80)
    print("CPU BACKENDS: sequential vs monoid vs k-gram-monoid")
    print("=" * 80)

    for pname in patterns:
        regex = PATTERNS[pname].regex
        dfa = compile_regex(regex)
        dm = DFAMatrices(dfa)
        alphabet = _alphabet_for(pname, dfa.alphabet)

        # Precompute monoid
        md = compute_monoid(dm)
        monoid_available = md is not None
        if monoid_available:
            k = auto_k(len(dfa.alphabet))
            kg = precompute_kgrams(dm, k, md)
        else:
            k = None
            kg = None

        print(f"\nPattern: {pname!r}  |  DFA states: {dm.n_states_raw}  |  "
              f"Alphabet: {alphabet!r}  |  "
              f"Monoid size: {md.size if md else 'N/A'}  |  "
              f"k={k}")
        print(f"  {'B':>8}  {'L':>6}  {'seq_Mcs':>10}  {'mono_Mcs':>10}  {'kgram_Mcs':>10}  {'correct':>8}")

        for B in B_grid:
            for L in L_grid:
                strings = _gen_strings(alphabet, B, L)
                total_chars = B * L

                # 1. Sequential (baseline)
                seq_results = []
                seq_ms = _bench_fn(
                    lambda: seq_results.__setitem__(slice(None),
                        [simulate_sequential(dfa, s) for s in strings])
                )
                # Re-run once to capture results list
                seq_results = [simulate_sequential(dfa, s) for s in strings]

                seq_mcs = _throughput_mchars(total_chars, seq_ms)

                # 2. Monoid
                mono_mcs = float('nan')
                mono_results = None
                if monoid_available:
                    mr = []
                    mono_ms = _bench_fn(
                        lambda: mr.__setitem__(slice(None),
                            [simulate_monoid(md, dm, s) for s in strings])
                    )
                    mono_results = [simulate_monoid(md, dm, s) for s in strings]
                    mono_mcs = _throughput_mchars(total_chars, mono_ms)

                # 3. k-gram monoid
                kgram_mcs = float('nan')
                kgram_results = None
                if monoid_available and kg is not None:
                    kr = []
                    kgram_ms = _bench_fn(
                        lambda: kr.__setitem__(slice(None),
                            [simulate_kgram_monoid(kg, md, dm, s) for s in strings])
                    )
                    kgram_results = [simulate_kgram_monoid(kg, md, dm, s) for s in strings]
                    kgram_mcs = _throughput_mchars(total_chars, kgram_ms)

                # Correctness check
                correct = True
                if mono_results is not None:
                    if mono_results != seq_results:
                        correct = False
                        print(f"  *** MONOID MISMATCH for {pname} B={B} L={L} ***")
                if kgram_results is not None:
                    if kgram_results != seq_results:
                        correct = False
                        print(f"  *** KGRAM MISMATCH for {pname} B={B} L={L} ***")

                assert correct, f"Correctness failure for pattern {pname!r} B={B} L={L}"

                print(f"  {B:>8}  {L:>6}  {seq_mcs:>10.2f}  {mono_mcs:>10.2f}  {kgram_mcs:>10.2f}  {'OK' if correct else 'FAIL':>8}")

                row = {
                    'benchmark': 'cpu_backends',
                    'pattern': pname,
                    'dfa_states': dm.n_states_raw,
                    'monoid_size': md.size if md else None,
                    'kgram_k': k,
                    'B': B,
                    'L': L,
                    'total_chars': total_chars,
                    'sequential_mchars_s': seq_mcs,
                    'monoid_mchars_s': mono_mcs if monoid_available else None,
                    'kgram_monoid_mchars_s': kgram_mcs if (monoid_available and kg is not None) else None,
                    'correct': correct,
                }
                all_results.append(row)

    return all_results


# ─── GPU Monoid vs v4 Benchmark ─────────────────────────────────────────────

def bench_gpu_monoid() -> list[dict]:
    """
    Compare GPU monoid scan vs GPU v4 matrix scan.
    Skips if total chars > 2^23 or if GPU libraries are unavailable.
    """
    all_results: list[dict] = []

    try:
        from src.gpu_bridge_monoid import MonoidGPUSimulator
        from src.gpu_bridge_v4 import ParallelGPUSimulator
    except Exception as e:
        print(f"\n[GPU benchmark skipped — import error: {e}]")
        return all_results

    try:
        monoid_sim = MonoidGPUSimulator()
    except Exception as e:
        print(f"\n[GPU monoid benchmark skipped — MonoidGPUSimulator unavailable: {e}]")
        return all_results

    try:
        v4_sim = ParallelGPUSimulator()
    except Exception as e:
        print(f"\n[GPU v4 benchmark skipped — ParallelGPUSimulator unavailable: {e}]")
        return all_results

    patterns = ['abb', 'even_a']
    B_grid = [1000, 10_000, 100_000]
    L_grid = [32, 128, 512]
    MAX_TOTAL = 1 << 23  # 8 M chars

    print("\n" + "=" * 80)
    print("GPU: monoid scan vs v4 matrix scan")
    print("=" * 80)

    for pname in patterns:
        regex = PATTERNS[pname].regex
        dfa = compile_regex(regex)
        dm = DFAMatrices(dfa)
        alphabet = _alphabet_for(pname, dfa.alphabet)

        md = compute_monoid(dm)
        if md is None:
            print(f"  [skipping {pname}: monoid too large]")
            continue

        print(f"\nPattern: {pname!r}  DFA states: {dm.n_states_raw}  Monoid size: {md.size}")
        print(f"  {'B':>8}  {'L':>6}  {'monoid_kern_ms':>15}  {'v4_kern_ms':>13}  {'speedup':>9}")

        for B in B_grid:
            for L in L_grid:
                total = B * L
                if total > MAX_TOTAL:
                    continue

                strings = _gen_strings(alphabet, B, L)

                # Create engines
                try:
                    mono_engine = monoid_sim.create_engine(
                        md, dm, max_total_chars=total + 1024, max_batch=B + 64
                    )
                except Exception as e:
                    print(f"  B={B} L={L}: monoid engine create failed: {e}")
                    continue

                try:
                    v4_engine = v4_sim.create_engine(
                        dm, max_total_chars=total + 1024, max_batch=B + 64
                    )
                except Exception as e:
                    mono_engine.destroy()
                    print(f"  B={B} L={L}: v4 engine create failed: {e}")
                    continue

                try:
                    # Warmup
                    for _ in range(2):
                        mono_engine.simulate_batch_timed(strings)
                        v4_engine.simulate_batch_timed(strings)

                    # Measure
                    _, monoid_kern_ms, _ = mono_engine.simulate_batch_timed(strings)
                    _, v4_kern_ms, _ = v4_engine.simulate_batch_timed(strings)

                    speedup = v4_kern_ms / monoid_kern_ms if monoid_kern_ms > 0 else float('inf')

                    print(f"  {B:>8}  {L:>6}  {monoid_kern_ms:>15.3f}  {v4_kern_ms:>13.3f}  {speedup:>9.2f}x")

                    all_results.append({
                        'benchmark': 'gpu_monoid_vs_v4',
                        'pattern': pname,
                        'dfa_states': dm.n_states_raw,
                        'monoid_size': md.size,
                        'B': B,
                        'L': L,
                        'total_chars': total,
                        'monoid_kernel_ms': monoid_kern_ms,
                        'v4_kernel_ms': v4_kern_ms,
                        'speedup_v4_over_monoid': speedup,
                    })

                except Exception as e:
                    print(f"  B={B} L={L}: benchmark error: {e}")

                finally:
                    mono_engine.destroy()
                    v4_engine.destroy()

    return all_results


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cpu_results = bench_cpu_backends()
    gpu_results = bench_gpu_monoid()

    all_results = cpu_results + gpu_results

    os.makedirs('results', exist_ok=True)
    out_path = 'results/optimized_benchmarks.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_path}  ({len(all_results)} rows)")
