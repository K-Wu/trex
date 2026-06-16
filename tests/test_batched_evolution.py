"""
Tests for batched state-vector evolution (simulate_batched_cpu).

Cross-validates against simulate_sequential to ensure the batched
column-parallel algorithm produces identical results.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import random
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential
from src.generate_data import PATTERNS, gen_random_string
from src.batched_evolution import simulate_batched_cpu


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_alphabet(pattern_name: str) -> str:
    """Return the alphabet for a given pattern name."""
    if pattern_name in ('abb', 'even_a', 'ab_star'):
        return 'ab'
    elif pattern_name == 'binary_div3':
        return '01'
    elif pattern_name == 'hex_number':
        return '0123456789abcdefx'
    elif pattern_name == 'identifier':
        return 'abcdefghijklmnopqrstuvwxyz0123456789'
    else:
        return 'abcdefgh'


def _cross_validate(dfa, dm, strings):
    """Run both backends and assert they agree on every string."""
    expected = [simulate_sequential(dfa, s) for s in strings]
    batched = simulate_batched_cpu(dm, strings)
    assert len(batched) == len(expected), (
        f"Length mismatch: batched={len(batched)} expected={len(expected)}"
    )
    for i, (e, b) in enumerate(zip(expected, batched)):
        assert e == b, (
            f"Mismatch at index {i}, string '{strings[i][:60]}...' "
            f"(len={len(strings[i])}): sequential={e} batched={b}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Binary alphabet patterns (parametrized)
# ═══════════════════════════════════════════════════════════════════════════

class TestBinaryPatterns:

    @pytest.mark.parametrize("pattern_name", ["abb", "even_a", "binary_div3", "ab_star"])
    def test_binary_patterns(self, pattern_name):
        """200 random strings x 128 chars, cross-validated against sequential."""
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        alphabet = _get_alphabet(pattern_name)
        rng = random.Random(42)
        strings = [gen_random_string(alphabet, 128, rng) for _ in range(200)]
        _cross_validate(dfa, dm, strings)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Larger alphabet patterns
# ═══════════════════════════════════════════════════════════════════════════

class TestLargerAlphabet:

    @pytest.mark.parametrize("pattern_name", ["hex_number", "identifier"])
    def test_larger_alphabet(self, pattern_name):
        """100 random strings x 64 chars, cross-validated against sequential."""
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        alphabet = _get_alphabet(pattern_name)
        rng = random.Random(99)
        strings = [gen_random_string(alphabet, 64, rng) for _ in range(100)]
        _cross_validate(dfa, dm, strings)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Single-string tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleString:

    def test_single_string_accepts(self):
        """'abb' should be accepted by (a|b)*abb."""
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        result = simulate_batched_cpu(dm, ["abb"])
        assert result == [True]

    def test_single_string_rejects(self):
        """'ab' should be rejected by (a|b)*abb."""
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        result = simulate_batched_cpu(dm, ["ab"])
        assert result == [False]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_batch(self):
        """Empty list of strings returns empty list."""
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        result = simulate_batched_cpu(dm, [])
        assert result == []

    def test_empty_strings(self):
        """Empty strings should all reject for (a|b)*abb (start is not accepting)."""
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        result = simulate_batched_cpu(dm, ["", "", ""])
        assert result == [False, False, False]

    def test_empty_strings_accepting_start(self):
        """Empty strings should accept for a* (start state is accepting)."""
        dfa = compile_regex("a*")
        dm = DFAMatrices(dfa)
        result = simulate_batched_cpu(dm, ["", "", ""])
        assert result == [True, True, True]

    def test_variable_length_strings(self):
        """Mix of different lengths should be handled correctly via padding."""
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        strings = ["abb", "a", "aabb", "", "babb", "ab", "ababababb", "b"]
        _cross_validate(dfa, dm, strings)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Scale tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScale:

    def test_large_batch(self):
        """1000 strings x 256 chars, cross-validated against sequential."""
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        rng = random.Random(77)
        strings = [gen_random_string('ab', 256, rng) for _ in range(1000)]
        _cross_validate(dfa, dm, strings)

    def test_boolean_threshold_no_overflow(self):
        """50 strings x 2000 chars -- long enough to overflow int8 without threshold."""
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        rng = random.Random(88)
        strings = [gen_random_string('ab', 2000, rng) for _ in range(50)]
        _cross_validate(dfa, dm, strings)


# ═══════════════════════════════════════════════════════════════════════════
# GPU Tests: Batched Evolution via CUDA
# ═══════════════════════════════════════════════════════════════════════════

def _gpu_available():
    """Check if the batched evolution GPU engine is available."""
    try:
        from src.gpu_bridge_batched import BatchedGPUSimulator
        BatchedGPUSimulator()
        return True
    except Exception:
        return False


def _sequential_results(dfa, strings):
    """Reference results from sequential simulation."""
    return [simulate_sequential(dfa, s) for s in strings]


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available or libbatched_evolution.so not found")
class TestBatchedEvolutionGPU:

    @pytest.mark.parametrize("pattern_name", ["abb", "even_a", "binary_div3", "ab_star"])
    def test_gpu_matches_cpu_binary(self, pattern_name):
        """200 random strings x 128 chars, cross-validated against sequential."""
        from src.gpu_bridge_batched import BatchedGPUSimulator
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        alphabet = _get_alphabet(pattern_name)
        rng = random.Random(42)
        strings = [gen_random_string(alphabet, 128, rng) for _ in range(200)]

        sim = BatchedGPUSimulator()
        engine = sim.create_engine(dm)
        gpu_results = engine.simulate_batch(strings)
        engine.destroy()

        expected = _sequential_results(dfa, strings)
        assert len(gpu_results) == len(expected)
        for i, (e, g) in enumerate(zip(expected, gpu_results)):
            assert e == g, (
                f"Mismatch at index {i}, string '{strings[i][:60]}...' "
                f"(len={len(strings[i])}): sequential={e} gpu={g}"
            )

    @pytest.mark.parametrize("pattern_name", ["hex_number", "identifier"])
    def test_gpu_matches_cpu_general(self, pattern_name):
        """100 random strings x 64 chars, cross-validated against sequential."""
        from src.gpu_bridge_batched import BatchedGPUSimulator
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        alphabet = _get_alphabet(pattern_name)
        rng = random.Random(99)
        strings = [gen_random_string(alphabet, 64, rng) for _ in range(100)]

        sim = BatchedGPUSimulator()
        engine = sim.create_engine(dm)
        gpu_results = engine.simulate_batch(strings)
        engine.destroy()

        expected = _sequential_results(dfa, strings)
        assert len(gpu_results) == len(expected)
        for i, (e, g) in enumerate(zip(expected, gpu_results)):
            assert e == g, (
                f"Mismatch at index {i}, string '{strings[i][:60]}...' "
                f"(len={len(strings[i])}): sequential={e} gpu={g}"
            )

    def test_gpu_variable_length(self):
        """Mix of lengths including empty strings should be handled correctly."""
        from src.gpu_bridge_batched import BatchedGPUSimulator
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        strings = ["abb", "a", "aabb", "", "babb", "ab", "ababababb", "b",
                    "", "abb", "bb", "aab", "abbb", "", "bbbabb"]

        sim = BatchedGPUSimulator()
        engine = sim.create_engine(dm)
        gpu_results = engine.simulate_batch(strings)
        engine.destroy()

        expected = _sequential_results(dfa, strings)
        assert len(gpu_results) == len(expected)
        for i, (e, g) in enumerate(zip(expected, gpu_results)):
            assert e == g, (
                f"Mismatch at index {i}, string '{strings[i]}': "
                f"sequential={e} gpu={g}"
            )

    def test_gpu_large_batch(self):
        """4096 strings x 256 chars, cross-validated against sequential."""
        from src.gpu_bridge_batched import BatchedGPUSimulator
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        rng = random.Random(77)
        strings = [gen_random_string('ab', 256, rng) for _ in range(4096)]

        sim = BatchedGPUSimulator()
        engine = sim.create_engine(dm)
        gpu_results = engine.simulate_batch(strings)
        engine.destroy()

        expected = _sequential_results(dfa, strings)
        assert len(gpu_results) == len(expected)
        mismatches = [(i, expected[i], gpu_results[i]) for i in range(len(expected))
                      if expected[i] != gpu_results[i]]
        assert not mismatches, (
            f"{len(mismatches)} mismatches out of {len(expected)} strings. "
            f"First 5: {mismatches[:5]}"
        )

    def test_gpu_timed(self):
        """Verify timed interface returns valid types."""
        from src.gpu_bridge_batched import BatchedGPUSimulator
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        rng = random.Random(42)
        strings = [gen_random_string('ab', 64, rng) for _ in range(100)]

        sim = BatchedGPUSimulator()
        engine = sim.create_engine(dm)
        results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
        engine.destroy()

        assert isinstance(results, list)
        assert len(results) == len(strings)
        assert all(isinstance(r, bool) for r in results)
        assert isinstance(kern_ms, float)
        assert isinstance(total_ms, float)
        assert kern_ms >= 0.0
        assert total_ms >= 0.0

        # Cross-validate results
        expected = _sequential_results(dfa, strings)
        assert results == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
