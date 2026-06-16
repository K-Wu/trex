"""
NCU profiling harness — launches each kernel configuration in isolation
so ncu can capture clean metrics.

Usage:
    ncu --set full -o profile_monoid_r1 python bench/ncu_profile.py monoid_r1
    ncu --set full -o profile_monoid_r3 python bench/ncu_profile.py monoid_r3
    ncu --set full -o profile_v4_r1     python bench/ncu_profile.py v4_r1
    ncu --set full -o profile_v4_r3     python bench/ncu_profile.py v4_r3
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices
from src.monoid import compute_monoid
from src.generate_data import PATTERNS


def setup_pattern(name='abb'):
    pat = PATTERNS[name]
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    md = compute_monoid(dm)
    alpha = sorted(dfa.alphabet)
    return dfa, dm, md, alpha


def run_monoid_r1():
    """Many short strings — triggers R1 warp-per-string kernel."""
    from src.gpu_bridge_monoid import MonoidGPUSimulator
    _, dm, md, alpha = setup_pattern('abb')
    sim = MonoidGPUSimulator()

    B, L = 100_000, 128
    random.seed(42)
    strings = [''.join(random.choice(alpha) for _ in range(L)) for _ in range(B)]

    engine = sim.create_engine(md, dm, max_total_chars=B * L + 1, max_batch=B + 1)
    # Warmup
    engine.simulate_batch(strings)
    # Profiled run
    results = engine.simulate_batch(strings)
    engine.destroy()
    print(f"monoid_r1: B={B} L={L} accept_rate={sum(results)/len(results):.3f}")


def run_monoid_r3():
    """Single long string — triggers R3 decoupled look-back kernel."""
    from src.gpu_bridge_monoid import MonoidGPUSimulator
    _, dm, md, alpha = setup_pattern('abb')
    sim = MonoidGPUSimulator()

    L = 4_000_000
    random.seed(42)
    s = ''.join(random.choice(alpha) for _ in range(L))

    engine = sim.create_engine(md, dm, max_total_chars=L + 1, max_batch=2)
    engine.simulate_batch([s])
    results = engine.simulate_batch([s])
    engine.destroy()
    print(f"monoid_r3: L={L} result={results[0]}")


def run_v4_r1():
    """Many short strings — triggers v4 R1 warp-per-string MMA kernel."""
    from src.gpu_bridge_v4 import ParallelGPUSimulator
    _, dm, _, alpha = setup_pattern('abb')
    sim = ParallelGPUSimulator()

    B, L = 100_000, 128
    random.seed(42)
    strings = [''.join(random.choice(alpha) for _ in range(L)) for _ in range(B)]

    engine = sim.create_engine(dm, max_total_chars=B * L + 1, max_batch=B + 1)
    engine.simulate_batch(strings)
    results = engine.simulate_batch(strings)
    engine.destroy()
    print(f"v4_r1: B={B} L={L} accept_rate={sum(results)/len(results):.3f}")


def run_v4_r3():
    """Single long string — triggers v4 R3 decoupled look-back MMA kernel."""
    from src.gpu_bridge_v4 import ParallelGPUSimulator
    _, dm, _, alpha = setup_pattern('abb')
    sim = ParallelGPUSimulator()

    L = 4_000_000
    random.seed(42)
    s = ''.join(random.choice(alpha) for _ in range(L))

    engine = sim.create_engine(dm, max_total_chars=L + 1, max_batch=2)
    engine.simulate_batch([s])
    results = engine.simulate_batch([s])
    engine.destroy()
    print(f"v4_r3: L={L} result={results[0]}")


MODES = {
    'monoid_r1': run_monoid_r1,
    'monoid_r3': run_monoid_r3,
    'v4_r1': run_v4_r1,
    'v4_r3': run_v4_r3,
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(MODES.keys())}>")
        sys.exit(1)
    MODES[sys.argv[1]]()
