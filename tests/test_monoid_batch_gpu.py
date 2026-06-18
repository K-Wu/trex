"""
GPU tests for the monoid batch engine — cross-validate against simulate_monoid.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
import numpy as np

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential
from src.generate_data import PATTERNS
from src.monoid import compute_monoid, simulate_monoid


def _make_dm(pattern_name):
    pat = PATTERNS[pattern_name]
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    return dm, dfa


def _get_alphabet(pattern_name):
    if pattern_name in ("abb", "even_a", "ab_star"):
        return "ab"
    if pattern_name == "binary_div3":
        return "01"
    if pattern_name == "hex_number":
        return "0123456789abcdefx"
    if pattern_name == "identifier":
        return "abcdefghijklmnopqrstuvwxyz0123456789"
    return "abcdefgh"


def _monoid_batch_gpu_available():
    try:
        from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator
        MonoidBatchGPUSimulator()
        return True
    except Exception:
        return False


skip_no_gpu = pytest.mark.skipif(
    not _monoid_batch_gpu_available(),
    reason="monoid batch GPU engine not available"
)


@pytest.fixture(scope="module")
def simulator():
    from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator
    return MonoidBatchGPUSimulator()


GPU_PATTERNS = ["abb", "binary_div3", "even_a", "ab_star",
                "hex_number", "identifier"]


@skip_no_gpu
class TestMonoidBatchGPU:

    def test_batch_cross_validate(self, simulator):
        for pattern_name in GPU_PATTERNS:
            dm, dfa = _make_dm(pattern_name)
            md = compute_monoid(dm)
            assert md is not None
            assert md.size <= 255

            engine = simulator.create_engine(md, dm)
            alphabet = _get_alphabet(pattern_name)
            rng = random.Random(hash(pattern_name) & 0xFFFFFFFF)

            strings = [""]
            for _ in range(499):
                length = rng.randint(0, 200)
                strings.append("".join(rng.choice(alphabet) for _ in range(length)))

            gpu_results = engine.simulate_batch(strings)
            cpu_results = [simulate_monoid(md, dm, s) for s in strings]

            mismatches = [
                (s, cpu, gpu)
                for s, cpu, gpu in zip(strings, cpu_results, gpu_results)
                if cpu != gpu
            ]
            assert not mismatches, (
                f"GPU vs CPU mismatches for '{pattern_name}' "
                f"({len(mismatches)}/500):\n" +
                "\n".join(f"  '{s[:40]}' exp={e} got={g}"
                          for s, e, g in mismatches[:10])
            )
            engine.destroy()

    def test_empty_batch(self, simulator):
        dm, _ = _make_dm("abb")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm)
        assert engine.simulate_batch([]) == []
        engine.destroy()

    def test_all_empty_strings(self, simulator):
        dm, dfa = _make_dm("even_a")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm)
        results = engine.simulate_batch(["", "", ""])
        expected = simulate_sequential(dfa, "")
        assert all(r == expected for r in results)
        engine.destroy()

    def test_timed(self, simulator):
        dm, _ = _make_dm("abb")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm)
        rng = random.Random(99)
        strings = ["".join(rng.choice("ab") for _ in range(100)) for _ in range(1000)]
        results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
        assert len(results) == 1000
        assert kern_ms > 0
        assert total_ms >= kern_ms
        engine.destroy()

    def test_variable_lengths(self, simulator):
        dm, dfa = _make_dm("abb")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm)
        rng = random.Random(77)

        strings = []
        for i in range(200):
            length = rng.randint(0, 500)
            strings.append("".join(rng.choice("ab") for _ in range(length)))

        gpu_results = engine.simulate_batch(strings)
        cpu_results = [simulate_sequential(dfa, s) for s in strings]
        assert gpu_results == cpu_results
        engine.destroy()

    def test_long_string_prefix(self, simulator):
        dm, dfa = _make_dm("even_a")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm, max_total_chars=2_000_000)
        rng = random.Random(42)

        for L in [200_000, 1_000_000]:
            s = "".join(rng.choice("ab") for _ in range(L))
            gpu_result = engine.simulate_batch([s])
            cpu_result = [simulate_monoid(md, dm, s)]
            assert gpu_result == cpu_result, f"L={L}: gpu={gpu_result}, cpu={cpu_result}"
        engine.destroy()

    def test_few_long_strings_prefix(self, simulator):
        dm, dfa = _make_dm("abb")
        md = compute_monoid(dm)
        engine = simulator.create_engine(md, dm, max_total_chars=4_000_000)
        rng = random.Random(88)

        strings = ["".join(rng.choice("ab") for _ in range(500_000))
                    for _ in range(4)]
        gpu_results = engine.simulate_batch(strings)
        cpu_results = [simulate_monoid(md, dm, s) for s in strings]
        assert gpu_results == cpu_results
        engine.destroy()
