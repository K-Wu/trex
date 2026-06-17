"""
Tests for src/optimized_engine.py — OptimizedEngine unified API.

Test classes
------------
TestOptimizedEngineAutoSelect   — auto-selection picks correct backend
TestOptimizedEngineCorrectness  — all backends agree with sequential on random strings
TestOptimizedEngineTiming       — match_batch_timed returns valid results + timing dict
TestAllConfigsSameResult        — all configs produce identical results
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest

from src.generate_data import PATTERNS
from src.regex_to_dfa import compile_regex
from src.simulation import simulate_sequential
from src.optimized_engine import OptimizedEngine


# ─── Helpers ────────────────────────────────────────────────────────────────

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
    lengths = [rng.randint(0, 20) for _ in range(n)]
    return ["".join(rng.choice(alpha) for _ in range(L)) for L in lengths]


def _sequential_results(pattern_name: str, strings: list) -> list:
    dfa = compile_regex(PATTERNS[pattern_name].regex)
    return [simulate_sequential(dfa, s) for s in strings]


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestOptimizedEngineAutoSelect
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizedEngineAutoSelect:
    """Verify auto-selection chooses expected backends for known patterns."""

    def test_small_dfa_selects_monoid(self):
        """'(a|b)*abb' is tiny — auto-select should pick DFA + monoid backend."""
        eng = OptimizedEngine(PATTERNS["abb"].regex)
        info = eng.config_info
        assert info["representation"] == "dfa", (
            f"Expected DFA representation, got {info['representation']}"
        )
        assert info["scan_backend"] in ("monoid", "monoid+kgram"), (
            f"Expected monoid or monoid+kgram backend, got {info['scan_backend']}"
        )

    def test_config_info_has_required_fields(self):
        """config_info must contain all required keys."""
        required_keys = {
            "representation",
            "scan_backend",
            "alphabet_size",
            "kgram_k",
            "selection_reason",
        }
        eng = OptimizedEngine(PATTERNS["abb"].regex)
        info = eng.config_info
        missing = required_keys - set(info.keys())
        assert not missing, f"config_info missing keys: {missing}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestOptimizedEngineCorrectness
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizedEngineCorrectness:
    """Cross-validate OptimizedEngine against sequential simulation."""

    @pytest.mark.parametrize("pattern_name", ["abb", "binary_div3", "even_a", "ab_star"])
    def test_auto_matches_sequential(self, pattern_name):
        """Auto-selected backend must agree with sequential on 100 random strings."""
        strings = _random_strings(pattern_name, 100, seed=1)
        expected = _sequential_results(pattern_name, strings)

        eng = OptimizedEngine(PATTERNS[pattern_name].regex)
        got = eng.match_batch(strings)

        mismatches = [
            (i, strings[i], expected[i], got[i])
            for i in range(len(strings))
            if got[i] != expected[i]
        ]
        assert not mismatches, (
            f"[{pattern_name}] auto backend mismatches on {len(mismatches)} strings: "
            f"{mismatches[:5]}"
        )

    @pytest.mark.parametrize("cfg", ["monoid", "monoid+kgram", "baseline"])
    def test_forced_config_matches_sequential(self, cfg):
        """Forced DFA-based configs must match sequential on 100 random strings."""
        pattern_name = "abb"
        strings = _random_strings(pattern_name, 100, seed=2)
        expected = _sequential_results(pattern_name, strings)

        eng = OptimizedEngine(PATTERNS[pattern_name].regex, config=cfg)
        got = eng.match_batch(strings)

        mismatches = [
            (i, strings[i], expected[i], got[i])
            for i in range(len(strings))
            if got[i] != expected[i]
        ]
        assert not mismatches, (
            f"[config={cfg}] mismatches on {len(mismatches)} strings: {mismatches[:5]}"
        )

    def test_nfa_config_matches_sequential(self):
        """NFA config must match sequential on 50 random strings."""
        pattern_name = "abb"
        strings = _random_strings(pattern_name, 50, seed=3)
        expected = _sequential_results(pattern_name, strings)

        eng = OptimizedEngine(PATTERNS[pattern_name].regex, config="nfa")
        got = eng.match_batch(strings)

        mismatches = [
            (i, strings[i], expected[i], got[i])
            for i in range(len(strings))
            if got[i] != expected[i]
        ]
        assert not mismatches, (
            f"[config=nfa] mismatches on {len(mismatches)} strings: {mismatches[:5]}"
        )

    @pytest.mark.parametrize("pattern_name", ["hex_number", "identifier"])
    def test_larger_alphabet(self, pattern_name):
        """Larger-alphabet patterns should also match sequential on 50 strings."""
        strings = _random_strings(pattern_name, 50, seed=4)
        expected = _sequential_results(pattern_name, strings)

        eng = OptimizedEngine(PATTERNS[pattern_name].regex)
        got = eng.match_batch(strings)

        mismatches = [
            (i, strings[i], expected[i], got[i])
            for i in range(len(strings))
            if got[i] != expected[i]
        ]
        assert not mismatches, (
            f"[{pattern_name}] auto backend mismatches on {len(mismatches)} strings: "
            f"{mismatches[:5]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestOptimizedEngineTiming
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizedEngineTiming:
    """Verify match_batch_timed returns correct structure."""

    def test_timed_returns_dict(self):
        """match_batch_timed must return (list[bool], dict) with timing keys."""
        eng = OptimizedEngine(PATTERNS["abb"].regex)
        strings = _random_strings("abb", 20, seed=5)

        result = eng.match_batch_timed(strings)

        assert isinstance(result, tuple), "Expected a 2-tuple"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"

        results, timing = result

        assert isinstance(results, list), "First element must be list"
        assert len(results) == len(strings), (
            f"Results length {len(results)} != strings length {len(strings)}"
        )
        assert all(isinstance(r, bool) for r in results), "All results must be bool"

        assert isinstance(timing, dict), "Second element must be dict"
        required_timing_keys = {"total_seconds", "per_string_seconds", "n_strings"}
        missing = required_timing_keys - set(timing.keys())
        assert not missing, f"timing dict missing keys: {missing}"

        assert timing["n_strings"] == len(strings)
        assert timing["total_seconds"] >= 0.0
        assert timing["per_string_seconds"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestAllConfigsSameResult
# ═══════════════════════════════════════════════════════════════════════════

class TestAllConfigsSameResult:
    """All configs must produce identical results for the same inputs."""

    def test_all_configs_agree(self):
        """None, 'monoid', 'monoid+kgram', 'baseline', 'nfa', 'monoid+gpu' must agree on 200 strings."""
        pattern_name = "abb"
        strings = _random_strings(pattern_name, 200, seed=99)
        regex = PATTERNS[pattern_name].regex

        configs = [None, "monoid", "monoid+kgram", "baseline", "nfa"]
        if _gpu_available():
            configs.append("monoid+gpu")
        if _kgram_gpu_available():
            configs.append("kgram+gpu")
        results_per_config = {}

        for cfg in configs:
            eng = OptimizedEngine(regex, config=cfg)
            results_per_config[cfg] = eng.match_batch(strings)

        # Compare every config against the baseline (sequential)
        ref_results = results_per_config["baseline"]
        for cfg in configs:
            got = results_per_config[cfg]
            mismatches = [
                (i, strings[i], ref_results[i], got[i])
                for i in range(len(strings))
                if got[i] != ref_results[i]
            ]
            assert not mismatches, (
                f"Config {cfg!r} disagrees with 'baseline' on "
                f"{len(mismatches)} strings: {mismatches[:5]}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5. GPU helper and TestOptimizedEngineGPU
# ═══════════════════════════════════════════════════════════════════════════

def _gpu_available():
    try:
        from src.gpu_bridge_monoid import MonoidGPUSimulator
        MonoidGPUSimulator()
        return True
    except Exception:
        return False


skip_no_gpu = pytest.mark.skipif(not _gpu_available(), reason="GPU not available")


def _kgram_gpu_available():
    try:
        from src.gpu_bridge_kgram import KGramGPUSimulator
        KGramGPUSimulator()
        return True
    except Exception:
        return False


@skip_no_gpu
class TestOptimizedEngineGPU:
    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_gpu_monoid_matches_cpu(self, pattern_name):
        pat = PATTERNS[pattern_name]

        cpu_engine = OptimizedEngine(pat.regex, config="monoid")
        gpu_engine = OptimizedEngine(pat.regex, config="monoid+gpu")

        random.seed(42)
        alpha = sorted(compile_regex(pat.regex).alphabet)
        strings = [''.join(random.choice(alpha) for _ in range(random.randint(0, 200)))
                    for _ in range(200)]

        cpu_results = cpu_engine.match_batch(strings)
        gpu_results = gpu_engine.match_batch(strings)
        assert gpu_results == cpu_results

    def test_gpu_long_string(self):
        engine = OptimizedEngine('(a|b)*abb', config="monoid+gpu")

        random.seed(77)
        s = ''.join(random.choice('ab') for _ in range(100000))
        dfa = compile_regex('(a|b)*abb')
        expected = simulate_sequential(dfa, s)
        got = engine.match_batch([s])[0]
        assert got == expected


# ═══════════════════════════════════════════════════════════════════════════
# 6. Batched Evolution GPU Integration
# ═══════════════════════════════════════════════════════════════════════════

def _batched_gpu_available():
    try:
        from src.gpu_bridge_batched import BatchedGPUSimulator
        BatchedGPUSimulator()
        return True
    except Exception:
        return False


skip_no_batched_gpu = pytest.mark.skipif(
    not _batched_gpu_available(), reason="Batched GPU not available"
)


@skip_no_batched_gpu
class TestBatchedEvolutionIntegration:

    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_batched_gpu_matches_baseline(self, pattern_name):
        pat = PATTERNS[pattern_name]
        engine_base = OptimizedEngine(pat.regex, config="baseline")
        engine_batched = OptimizedEngine(pat.regex, config="batched+gpu")

        strings = _random_strings(pattern_name, 500, seed=42)
        expected = engine_base.match_batch(strings)
        actual = engine_batched.match_batch(strings)
        assert actual == expected, f"batched+gpu mismatch for {pattern_name}"

    def test_batched_gpu_config_info(self):
        engine = OptimizedEngine("(a|b)*abb", config="batched+gpu")
        info = engine.config_info
        assert info["scan_backend"] == "batched+gpu"

    def test_batched_gpu_timed(self):
        engine = OptimizedEngine("(a|b)*abb", config="batched+gpu")
        strings = _random_strings("abb", 200, seed=42)
        results, timing = engine.match_batch_timed(strings)
        assert isinstance(results, list)
        assert len(results) == 200
        assert "kernel_ms" in timing
        assert "total_ms" in timing

    @pytest.mark.parametrize("pattern_name", ["hex_number", "identifier"])
    def test_batched_gpu_larger_alphabet(self, pattern_name):
        pat = PATTERNS[pattern_name]
        engine_base = OptimizedEngine(pat.regex, config="baseline")
        engine_batched = OptimizedEngine(pat.regex, config="batched+gpu")

        strings = _random_strings(pattern_name, 200, seed=99)
        expected = engine_base.match_batch(strings)
        actual = engine_batched.match_batch(strings)
        assert actual == expected, f"batched+gpu mismatch for {pattern_name}"


# ═══════════════════════════════════════════════════════════════════════════
# 7. K-gram GPU Integration
# ═══════════════════════════════════════════════════════════════════════════

skip_no_kgram_gpu = pytest.mark.skipif(
    not _kgram_gpu_available(), reason="K-gram GPU not available"
)


@skip_no_kgram_gpu
class TestKGramGPUIntegration:

    @pytest.mark.parametrize("pattern_name",
        ['abb', 'binary_div3', 'even_a', 'ab_star'])
    def test_kgram_gpu_matches_baseline(self, pattern_name):
        pat = PATTERNS[pattern_name]
        engine_base = OptimizedEngine(pat.regex, config="baseline")
        engine_kgram = OptimizedEngine(pat.regex, config="kgram+gpu")

        strings = _random_strings(pattern_name, 500, seed=42)
        expected = engine_base.match_batch(strings)
        actual = engine_kgram.match_batch(strings)
        assert actual == expected, f"kgram+gpu mismatch for {pattern_name}"

    def test_kgram_gpu_config_info(self):
        engine = OptimizedEngine("(a|b)*abb", config="kgram+gpu")
        info = engine.config_info
        assert info["scan_backend"] == "kgram+gpu"
        assert info["kgram_k"] is not None
        assert info["kgram_k"] >= 1

    def test_kgram_gpu_timed(self):
        engine = OptimizedEngine("(a|b)*abb", config="kgram+gpu")
        strings = _random_strings("abb", 200, seed=42)
        results, timing = engine.match_batch_timed(strings)
        assert isinstance(results, list)
        assert len(results) == 200
        assert "kernel_ms" in timing
        assert "total_ms" in timing

    @pytest.mark.parametrize("pattern_name", ["hex_number", "identifier"])
    def test_kgram_gpu_larger_alphabet(self, pattern_name):
        pat = PATTERNS[pattern_name]
        engine_base = OptimizedEngine(pat.regex, config="baseline")
        engine_kgram = OptimizedEngine(pat.regex, config="kgram+gpu")

        strings = _random_strings(pattern_name, 200, seed=99)
        expected = engine_base.match_batch(strings)
        actual = engine_kgram.match_batch(strings)
        assert actual == expected, f"kgram+gpu mismatch for {pattern_name}"
