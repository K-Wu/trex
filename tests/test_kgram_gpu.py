"""
Tests for the k-gram TC GPU engine (src/gpu_bridge_kgram.py).

Cross-validates against sequential CPU simulation.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest

from src.generate_data import PATTERNS
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential


def _kgram_gpu_available():
    try:
        from src.gpu_bridge_kgram import KGramGPUSimulator
        KGramGPUSimulator()
        return True
    except Exception:
        return False


skip_no_gpu = pytest.mark.skipif(
    not _kgram_gpu_available(), reason="K-gram GPU lib not available"
)

_ALPHABETS = {
    "abb": "ab",
    "binary_div3": "01",
    "even_a": "ab",
    "ab_star": "ab",
    "hex_number": "0123456789abcdefx",
    "identifier": "abcdefghijklmnopqrstuvwxyz0123456789",
}


def _random_strings(pattern_name: str, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    alpha = _ALPHABETS[pattern_name]
    lengths = [rng.randint(0, 50) for _ in range(n)]
    return ["".join(rng.choice(alpha) for _ in range(L)) for L in lengths]


def _sequential_results(pattern_name: str, strings: list) -> list:
    dfa = compile_regex(PATTERNS[pattern_name].regex)
    return [simulate_sequential(dfa, s) for s in strings]


@skip_no_gpu
class TestKGramGPUCorrectness:

    @pytest.mark.parametrize("pattern_name", ["abb", "binary_div3", "even_a", "ab_star"])
    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_matches_sequential(self, pattern_name, k):
        from src.gpu_bridge_kgram import KGramGPUSimulator

        strings = _random_strings(pattern_name, 200, seed=k * 100 + hash(pattern_name) % 1000)
        expected = _sequential_results(pattern_name, strings)

        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=k)
        got = engine.simulate_batch(strings)
        engine.destroy()

        mismatches = [
            (i, strings[i], expected[i], got[i])
            for i in range(len(strings))
            if got[i] != expected[i]
        ]
        assert not mismatches, (
            f"[{pattern_name}, k={k}] {len(mismatches)} mismatches: {mismatches[:5]}"
        )

    def test_empty_strings(self):
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS["abb"]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=4)
        results = engine.simulate_batch(["", "", ""])
        engine.destroy()

        expected = simulate_sequential(dfa, "")
        assert results == [expected] * 3

    def test_mixed_lengths(self):
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS["even_a"]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        rng = random.Random(77)
        alpha = "ab"
        strings = ["".join(rng.choice(alpha) for _ in range(L))
                    for L in [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 256]]
        expected = [simulate_sequential(dfa, s) for s in strings]

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=4)
        got = engine.simulate_batch(strings)
        engine.destroy()

        assert got == expected

    def test_large_k(self):
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS["abb"]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        strings = _random_strings("abb", 100, seed=999)
        expected = _sequential_results("abb", strings)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=16)
        got = engine.simulate_batch(strings)
        engine.destroy()

        assert got == expected

    @pytest.mark.parametrize("pattern_name", ["hex_number", "identifier"])
    def test_larger_alphabet(self, pattern_name):
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        strings = _random_strings(pattern_name, 100, seed=55)
        expected = _sequential_results(pattern_name, strings)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=2)
        got = engine.simulate_batch(strings)
        engine.destroy()

        assert got == expected


@skip_no_gpu
class TestKGramGPUTiming:

    def test_timed_returns_tuple(self):
        from src.gpu_bridge_kgram import KGramGPUSimulator

        pat = PATTERNS["abb"]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)

        sim = KGramGPUSimulator()
        engine = sim.create_engine(dm, k=4)
        strings = _random_strings("abb", 50, seed=11)
        results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
        engine.destroy()

        assert isinstance(results, list)
        assert len(results) == 50
        assert all(isinstance(r, bool) for r in results)
        assert isinstance(kern_ms, float)
        assert isinstance(total_ms, float)
        assert kern_ms >= 0.0
        assert total_ms >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
