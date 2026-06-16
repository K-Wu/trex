"""
Comprehensive tests for v4 parallel DFA engine.

Covers: R1 warp-per-string, R3 decoupled look-back, adaptive dispatch,
variable-length strings, early/late acceptance, cross-validation against CPU.
"""

import sys, os, random, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential, simulate_prefix_scan
from src.generate_data import PATTERNS


def _engine_available():
    try:
        from src.gpu_bridge_v4 import ParallelGPUSimulator
        ParallelGPUSimulator()
        return True
    except Exception:
        return False

skip_no_engine = pytest.mark.skipif(
    not _engine_available(),
    reason="v4 parallel engine not available"
)


@pytest.fixture(scope="module")
def simulator():
    from src.gpu_bridge_v4 import ParallelGPUSimulator
    return ParallelGPUSimulator()


# ─── T1: Single-string correctness ────────────────────────────────────────

@skip_no_engine
class TestSingleString:
    PATTERN_NAMES = ['abb', 'binary_div3', 'even_a', 'ab_star',
                     'hex_number', 'identifier']

    @pytest.mark.parametrize("pattern_name", PATTERN_NAMES)
    def test_known_accept_reject(self, simulator, pattern_name):
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=1 << 20, max_batch=1024)

        alpha = sorted(dfa.alphabet)
        random.seed(42)
        for _ in range(20):
            L = random.randint(1, 200)
            s = ''.join(random.choice(alpha) for _ in range(L))
            cpu = simulate_sequential(dfa, s)
            gpu = engine.simulate_single(s)
            assert gpu == cpu, f"{pattern_name}: L={L} cpu={cpu} gpu={gpu}"

        engine.destroy()

    def test_empty_strings(self, simulator):
        for name in ['abb', 'even_a']:
            pat = PATTERNS[name]
            dfa = compile_regex(pat.regex)
            dm = DFAMatrices(dfa)
            engine = simulator.create_engine(dm)
            cpu = simulate_sequential(dfa, "")
            gpu = engine.simulate_single("")
            assert gpu == cpu, f"{name}: empty string mismatch"
            engine.destroy()

    def test_single_char(self, simulator):
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm)
        for ch in sorted(dfa.alphabet):
            cpu = simulate_sequential(dfa, ch)
            gpu = engine.simulate_single(ch)
            assert gpu == cpu
        engine.destroy()

    @pytest.mark.parametrize("L", [2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64,
                                    127, 128, 255, 256, 511, 512, 1023, 1024])
    def test_power_of_2_boundaries(self, simulator, L):
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=2048)
        alpha = sorted(dfa.alphabet)
        random.seed(L)
        s = ''.join(random.choice(alpha) for _ in range(L))
        cpu = simulate_sequential(dfa, s)
        gpu = engine.simulate_single(s)
        assert gpu == cpu, f"L={L}"
        engine.destroy()


# ─── T2: Batch correctness ────────────────────────────────────────────────

@skip_no_engine
class TestBatchCorrectness:
    def _run_batch(self, simulator, strings, pattern_name='abb'):
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        total = sum(len(s) for s in strings)
        engine = simulator.create_engine(
            dm, max_total_chars=max(total + 1, 1024), max_batch=len(strings) + 1)

        gpu_results = engine.simulate_batch(strings)
        cpu_results = [simulate_sequential(dfa, s) for s in strings]
        engine.destroy()
        return gpu_results, cpu_results

    def test_uniform_length_all_random(self, simulator):
        random.seed(42)
        alpha = ['a', 'b']
        strings = [''.join(random.choice(alpha) for _ in range(100))
                    for _ in range(500)]
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu

    def test_uniform_all_accept(self, simulator):
        strings = ['aabb' for _ in range(100)]
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu

    def test_uniform_all_reject(self, simulator):
        strings = ['ab' for _ in range(100)]
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu

    def test_mixed_accept_reject(self, simulator):
        strings = ['abb', 'ab', 'aabb', 'b', 'babb', 'aa', 'ababb']
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu

    def test_variable_lengths(self, simulator):
        random.seed(123)
        alpha = ['a', 'b']
        lengths = [1, 5, 10, 50, 100, 200, 500, 1000]
        strings = [''.join(random.choice(alpha) for _ in range(random.choice(lengths)))
                    for _ in range(200)]
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu

    def test_extreme_length_variance(self, simulator):
        random.seed(77)
        alpha = ['a', 'b']
        strings = [''.join(random.choice(alpha) for _ in range(10))
                    for _ in range(50)]
        strings.append(''.join(random.choice(alpha) for _ in range(5000)))
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu

    def test_batch_with_empty_strings(self, simulator):
        strings = ['', 'abb', '', 'ab', '']
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu

    def test_all_identical(self, simulator):
        strings = ['ababb'] * 200
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu

    def test_large_batch(self, simulator):
        random.seed(999)
        alpha = ['a', 'b']
        strings = [''.join(random.choice(alpha) for _ in range(50))
                    for _ in range(10000)]
        gpu, cpu = self._run_batch(simulator, strings)
        assert gpu == cpu


# ─── T3: Early/late acceptance patterns ───────────────────────────────────

@skip_no_engine
class TestEarlyLateAccept:
    def test_late_accept_suffix(self, simulator):
        """Accept only because of 'abb' at the end."""
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=1 << 20, max_batch=1024)

        random.seed(42)
        for L in [10, 50, 100, 500, 1000]:
            s = ''.join(random.choice(['a', 'b']) for _ in range(L - 3)) + 'abb'
            assert engine.simulate_single(s) == simulate_sequential(dfa, s)
        engine.destroy()

    def test_early_accept_then_reject(self, simulator):
        """'abb' at start but string continues and may not end in accept state."""
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=1 << 20, max_batch=1024)

        random.seed(43)
        for L in [10, 50, 100, 500]:
            s = 'abb' + ''.join(random.choice(['a', 'b']) for _ in range(L - 3))
            assert engine.simulate_single(s) == simulate_sequential(dfa, s)
        engine.destroy()

    def test_oscillating_acceptance(self, simulator):
        """'abb' repeated — periodically enters accept state."""
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=1 << 20, max_batch=1024)

        for reps in [1, 5, 10, 50, 100]:
            s = 'abb' * reps
            assert engine.simulate_single(s) == simulate_sequential(dfa, s)
        engine.destroy()

    def test_mixed_early_late_batch(self, simulator):
        """Batch with mix of early-accept, late-accept, and random strings."""
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        random.seed(44)
        strings = []
        L = 100
        for i in range(300):
            if i % 3 == 0:
                s = ''.join(random.choice(['a', 'b']) for _ in range(L - 3)) + 'abb'
            elif i % 3 == 1:
                s = 'abb' + ''.join(random.choice(['a', 'b']) for _ in range(L - 3))
            else:
                s = ''.join(random.choice(['a', 'b']) for _ in range(L))
            strings.append(s)

        total = sum(len(s) for s in strings)
        engine = simulator.create_engine(dm, max_total_chars=total + 1,
                                         max_batch=len(strings) + 1)
        gpu = engine.simulate_batch(strings)
        cpu = [simulate_sequential(dfa, s) for s in strings]
        assert gpu == cpu
        engine.destroy()


# ─── T4: Timing (smoke test that timing works) ───────────────────────────

@skip_no_engine
class TestTiming:
    def test_batch_timed(self, simulator):
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=1 << 20, max_batch=2048)

        random.seed(42)
        strings = [''.join(random.choice(['a', 'b']) for _ in range(100))
                    for _ in range(1000)]
        results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
        assert len(results) == 1000
        assert kern_ms > 0
        assert total_ms >= kern_ms
        engine.destroy()


# ─── T5: Cross-validation with Python re ──────────────────────────────────

@skip_no_engine
class TestCrossValidateRe:
    @pytest.mark.parametrize("regex,test_strings", [
        (r'(a|b)*abb', ['abb', 'aabb', 'ab', 'babb', 'a', 'b', '']),
        (r'(ab)*', ['', 'ab', 'abab', 'a', 'aba', 'b']),
    ])
    def test_matches_python_re(self, simulator, regex, test_strings):
        dfa = compile_regex(regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=1024, max_batch=64)

        for s in test_strings:
            gpu = engine.simulate_single(s)
            py_match = bool(re.fullmatch(regex, s))
            assert gpu == py_match, f"regex={regex} s={s!r} gpu={gpu} re={py_match}"
        engine.destroy()


# ─── T6: Long strings (exercises R3 decoupled look-back via adaptive) ────

@skip_no_engine
class TestLongStrings:
    @pytest.mark.parametrize("L", [10000, 100000, 1000000])
    def test_long_cross_validate(self, simulator, L):
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=L + 1, max_batch=4)

        random.seed(L)
        alpha = sorted(dfa.alphabet)
        s = ''.join(random.choice(alpha) for _ in range(L))
        cpu = simulate_sequential(dfa, s)
        gpu = engine.simulate_single(s)
        assert gpu == cpu, f"L={L}"
        engine.destroy()

    def test_long_accept_suffix(self, simulator):
        L = 50000
        pat = PATTERNS['abb']
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        engine = simulator.create_engine(dm, max_total_chars=L + 1, max_batch=4)

        random.seed(42)
        s = ''.join(random.choice(['a', 'b']) for _ in range(L - 3)) + 'abb'
        cpu = simulate_sequential(dfa, s)
        gpu = engine.simulate_single(s)
        assert gpu == cpu and gpu == True
        engine.destroy()
