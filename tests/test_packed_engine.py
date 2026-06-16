"""
Tests for PackedEngine: multi-pattern matching via block-diagonal DFA packing.

Cross-validates against simulate_sequential to ensure that the packed
block-diagonal evolution produces identical results to individual per-pattern
simulation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import random
from src.regex_to_dfa import compile_regex
from src.simulation import simulate_sequential
from src.generate_data import gen_random_string
from src.packed_engine import PackedEngine


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cross_validate(regexes: list[str], strings: list[str]):
    """
    Cross-validate PackedEngine results against per-pattern sequential simulation.

    For each pattern and each string, verify the packed result matches the
    individual DFA simulation.
    """
    # Ground truth: compile each regex individually and simulate
    dfas = [compile_regex(r) for r in regexes]
    expected = []
    for dfa in dfas:
        expected.append([simulate_sequential(dfa, s) for s in strings])

    # PackedEngine results
    engine = PackedEngine(regexes)
    packed_results = engine.match_batch(strings)

    assert len(packed_results) == len(regexes), (
        f"Pattern count mismatch: packed={len(packed_results)} expected={len(regexes)}"
    )
    for p_idx in range(len(regexes)):
        assert len(packed_results[p_idx]) == len(strings), (
            f"String count mismatch for pattern {p_idx}: "
            f"packed={len(packed_results[p_idx])} expected={len(strings)}"
        )
        for s_idx in range(len(strings)):
            assert packed_results[p_idx][s_idx] == expected[p_idx][s_idx], (
                f"Mismatch: pattern[{p_idx}]='{regexes[p_idx]}', "
                f"string[{s_idx}]='{strings[s_idx][:60]}...' "
                f"(len={len(strings[s_idx])}): "
                f"packed={packed_results[p_idx][s_idx]} "
                f"expected={expected[p_idx][s_idx]}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Two-pattern test
# ═══════════════════════════════════════════════════════════════════════════

class TestTwoPatterns:

    def test_two_patterns(self):
        """Pack (a|b)*abb + (aa|b)*, 100 strings x 64 chars, cross-validate."""
        regexes = ["(a|b)*abb", "(aa|b)*"]
        rng = random.Random(42)
        strings = [gen_random_string('ab', 64, rng) for _ in range(100)]
        _cross_validate(regexes, strings)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Four-pattern test
# ═══════════════════════════════════════════════════════════════════════════

class TestFourPatterns:

    def test_four_patterns(self):
        """Pack 4 binary patterns, 200 strings x 128 chars, cross-validate."""
        regexes = [
            "(a|b)*abb",
            "(aa|b)*",
            "(ab)*",
            "(b*ab*ab*)*b*",   # even number of a's
        ]
        rng = random.Random(99)
        strings = [gen_random_string('ab', 128, rng) for _ in range(200)]
        _cross_validate(regexes, strings)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Single-pattern test
# ═══════════════════════════════════════════════════════════════════════════

class TestSinglePattern:

    def test_single_pattern(self):
        """Pack just 1 pattern, verify matches individual engine."""
        regexes = ["(a|b)*abb"]
        rng = random.Random(77)
        strings = [gen_random_string('ab', 64, rng) for _ in range(50)]
        _cross_validate(regexes, strings)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Variable-length strings
# ═══════════════════════════════════════════════════════════════════════════

class TestVariableLength:

    def test_variable_length(self):
        """Mix of string lengths including empty."""
        regexes = ["(a|b)*abb", "(aa|b)*"]
        strings = [
            "",
            "a",
            "ab",
            "abb",
            "aabb",
            "babb",
            "b",
            "aa",
            "bb",
            "ababababb",
            "",
            "aabbaabb",
            "bbbabb",
            "aaaa",
            "bbbb",
        ]
        _cross_validate(regexes, strings)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Config info
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigInfo:

    def test_config_info(self):
        """Check n_patterns, NP fields exist."""
        regexes = ["(a|b)*abb", "(aa|b)*", "(ab)*"]
        engine = PackedEngine(regexes)
        info = engine.config_info
        assert 'n_patterns' in info
        assert info['n_patterns'] == 3
        assert 'NP' in info
        assert info['NP'] > 0
        # NP should be padded to multiple of 16
        assert info['NP'] % 16 == 0
        # Should have per-pattern state counts
        assert 'state_counts' in info
        assert len(info['state_counts']) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 6. Empty batch
# ═══════════════════════════════════════════════════════════════════════════

class TestEmptyBatch:

    def test_empty_batch(self):
        """Empty string list returns correct structure."""
        regexes = ["(a|b)*abb", "(aa|b)*"]
        engine = PackedEngine(regexes)
        results = engine.match_batch([])
        # Should return a list of P empty lists
        assert len(results) == 2
        assert results[0] == []
        assert results[1] == []


# ═══════════════════════════════════════════════════════════════════════════
# 7. Timed interface
# ═══════════════════════════════════════════════════════════════════════════

class TestTimedInterface:

    def test_match_batch_timed(self):
        """Verify timed interface returns results and timing dict."""
        regexes = ["(a|b)*abb", "(aa|b)*"]
        engine = PackedEngine(regexes)
        rng = random.Random(42)
        strings = [gen_random_string('ab', 32, rng) for _ in range(20)]
        results, timing = engine.match_batch_timed(strings)
        assert len(results) == 2
        assert len(results[0]) == 20
        assert isinstance(timing, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
