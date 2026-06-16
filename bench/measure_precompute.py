"""Measure CPU-side preprocessing costs for each optimization path."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices
from src.monoid import compute_monoid
from src.kgram import precompute_kgrams, auto_k
from src.nfa_matrices import compile_nfa_matrices
from src.generate_data import PATTERNS


def measure(fn, repeats=20):
    """Return (result, median_ms)."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return result, times[len(times) // 2]


def measure_precompute_costs():
    rows = []
    pattern_names = ['abb', 'even_a', 'binary_div3', 'ab_star', 'hex_number', 'identifier']

    for name in pattern_names:
        pat = PATTERNS[name]
        regex = pat.regex

        # 1. Regex → DFA
        dfa, t_dfa = measure(lambda: compile_regex(regex))

        # 2. DFA → DFAMatrices
        dm, t_dm = measure(lambda: DFAMatrices(dfa))

        # 3. DFA → Monoid
        md, t_monoid = measure(lambda: compute_monoid(dm))

        # 4. Monoid → k-gram table
        alpha_size = len(dm.alphabet)
        k = auto_k(alpha_size)
        if md is not None:
            kg, t_kgram = measure(lambda: precompute_kgrams(dm, k, monoid=md))
        else:
            kg, t_kgram = None, 0.0

        # 5. k-gram matrix mode (no monoid)
        kg_mat, t_kgram_mat = measure(lambda: precompute_kgrams(dm, k, monoid=None))

        # 6. Regex → NFA matrices
        nm, t_nfa = measure(lambda: compile_nfa_matrices(regex))

        # 7. GPU engine init (monoid) — single measurement (global state)
        t_gpu_init = 0.0
        try:
            from src.gpu_bridge_monoid import MonoidGPUSimulator
            sim = MonoidGPUSimulator()
            if md is not None:
                t0 = time.perf_counter()
                eng = sim.create_engine(md, dm)
                t_gpu_init = (time.perf_counter() - t0) * 1000
                eng.destroy()
        except Exception:
            pass

        # 8. GPU engine init (v4) — single measurement (global state)
        t_v4_init = 0.0
        try:
            from src.gpu_bridge_v4 import ParallelGPUSimulator
            v4_sim = ParallelGPUSimulator()
            t0 = time.perf_counter()
            eng = v4_sim.create_engine(dm)
            t_v4_init = (time.perf_counter() - t0) * 1000
            eng.destroy()
        except Exception:
            pass

        # 9. Char-to-monoid mapping cost for a batch
        # Simulate _prepare_batch overhead
        import random, numpy as np
        random.seed(42)
        alpha = sorted(dfa.alphabet)
        test_strings = [''.join(random.choice(alpha) for _ in range(1000)) for _ in range(1000)]
        total_chars = sum(len(s) for s in test_strings)

        if md is not None:
            c2m = md.char_to_monoid
            identity = md.identity_idx
            def map_chars():
                chars = np.zeros(total_chars, dtype=np.uint16)
                pos = 0
                for s in test_strings:
                    for ch in s:
                        chars[pos] = c2m.get(ch, identity)
                        pos += 1
                return chars
            _, t_char_map = measure(map_chars, repeats=5)
        else:
            t_char_map = 0.0

        row = {
            'pattern': name,
            'regex': regex,
            'dfa_states': dfa.n_states,
            'alphabet_size': alpha_size,
            'monoid_size': md.size if md else None,
            'kgram_k': k,
            'nfa_states': nm.n_states_raw if nm else None,
            't_regex_to_dfa_ms': round(t_dfa, 3),
            't_dfa_to_matrices_ms': round(t_dm, 3),
            't_monoid_compute_ms': round(t_monoid, 3),
            't_kgram_monoid_ms': round(t_kgram, 3),
            't_kgram_matrix_ms': round(t_kgram_mat, 3),
            't_nfa_matrices_ms': round(t_nfa, 3),
            't_gpu_monoid_init_ms': round(t_gpu_init, 3),
            't_gpu_v4_init_ms': round(t_v4_init, 3),
            't_char_map_1M_ms': round(t_char_map, 3),
            'total_monoid_pipeline_ms': round(t_dfa + t_dm + t_monoid + t_kgram + t_gpu_init, 3),
            'total_v4_pipeline_ms': round(t_dfa + t_dm + t_v4_init, 3),
            'total_nfa_pipeline_ms': round(t_nfa, 3),
        }
        rows.append(row)

        print(f"\n=== {name} ({regex}) ===")
        print(f"  DFA: {dfa.n_states} states, |Σ|={alpha_size}, monoid M={md.size if md else 'N/A'}, k={k}")
        print(f"  NFA: {nm.n_states_raw} raw states (padded to {nm.n_states})")
        print(f"  ─── Preprocessing times (median of 20 runs) ───")
        print(f"  regex → DFA:           {t_dfa:8.3f} ms")
        print(f"  DFA → matrices:        {t_dm:8.3f} ms")
        print(f"  monoid compute:        {t_monoid:8.3f} ms")
        print(f"  k-gram (monoid, k={k:2d}): {t_kgram:8.3f} ms")
        print(f"  k-gram (matrix, k={k:2d}): {t_kgram_mat:8.3f} ms")
        print(f"  NFA matrices:          {t_nfa:8.3f} ms")
        print(f"  GPU monoid engine init:{t_gpu_init:8.3f} ms")
        print(f"  GPU v4 engine init:    {t_v4_init:8.3f} ms")
        print(f"  char→monoid 1M chars:  {t_char_map:8.3f} ms")
        print(f"  ─── Total pipeline costs ───")
        print(f"  Monoid+kgram+GPU:      {t_dfa + t_dm + t_monoid + t_kgram + t_gpu_init:8.3f} ms")
        print(f"  V4 baseline+GPU:       {t_dfa + t_dm + t_v4_init:8.3f} ms")
        print(f"  NFA path:              {t_nfa:8.3f} ms")

    return rows


if __name__ == '__main__':
    rows = measure_precompute_costs()

    # Also measure how preprocessing amortizes over batch sizes
    print("\n\n=== AMORTIZATION ANALYSIS ===")
    print("How many characters must be processed for preprocessing to be < 10% of total?")

    from src.gpu_bridge_monoid import MonoidGPUSimulator
    import random
    sim = MonoidGPUSimulator()

    for name in ['abb', 'even_a']:
        pat = PATTERNS[name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        md = compute_monoid(dm)
        alpha = sorted(dfa.alphabet)
        k = auto_k(len(alpha))
        kg = precompute_kgrams(dm, k, monoid=md)

        # Measure total precompute cost
        t0 = time.perf_counter()
        _dfa = compile_regex(pat.regex)
        _dm = DFAMatrices(_dfa)
        _md = compute_monoid(_dm)
        _kg = precompute_kgrams(_dm, k, monoid=_md)
        engine = sim.create_engine(_md, _dm)
        t_precompute = (time.perf_counter() - t0) * 1000

        # Measure kernel throughput at different batch sizes
        for total_chars_target in [10_000, 100_000, 1_000_000, 10_000_000]:
            L = 512
            B = max(1, total_chars_target // L)
            random.seed(42)
            strings = [''.join(random.choice(alpha) for _ in range(L)) for _ in range(B)]
            actual_total = B * L

            # Warmup
            engine.simulate_batch(strings)
            _, kern_ms, total_ms = engine.simulate_batch_timed(strings)

            precompute_pct = t_precompute / (t_precompute + total_ms) * 100

            print(f"  {name}: {actual_total/1e6:.1f}M chars (B={B}, L={L}) | "
                  f"precompute={t_precompute:.1f}ms  kernel+xfer={total_ms:.2f}ms | "
                  f"precompute is {precompute_pct:.1f}% of total")

        engine.destroy()

    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    outpath = f'results/precompute_costs_{timestamp}.json'
    with open(outpath, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {outpath}")
