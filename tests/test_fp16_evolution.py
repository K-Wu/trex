import pytest
import random
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices
from src.gpu_bridge_fp16_evolution import FP16EvolutionGPUSimulator


@pytest.fixture
def fp16_engine():
    dfa = compile_regex("(a|b)*abb")
    dm = DFAMatrices(dfa)
    sim = FP16EvolutionGPUSimulator()
    engine = sim.create_engine(dm)
    yield engine, dfa
    engine.destroy()


def test_basic_correctness(fp16_engine):
    engine, dfa = fp16_engine
    strings = ["abb", "aabb", "babb", "ab", "ba", ""]
    expected = [dfa.simulate(s) for s in strings]
    results = engine.simulate_batch(strings)
    assert results == expected


def test_long_strings(fp16_engine):
    engine, dfa = fp16_engine
    random.seed(42)
    strings = ["".join(random.choice("ab") for _ in range(1000)) for _ in range(100)]
    expected = [dfa.simulate(s) for s in strings]
    results = engine.simulate_batch(strings)
    assert results == expected


def test_timed_dispatch(fp16_engine):
    engine, dfa = fp16_engine
    strings = ["abb", "aabb", "babb"]
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    expected = [dfa.simulate(s) for s in strings]
    assert results == expected
    assert kern_ms >= 0
    assert total_ms >= 0


def test_empty_batch(fp16_engine):
    engine, dfa = fp16_engine
    results = engine.simulate_batch([])
    assert results == []


def test_cross_engine_validation():
    """Compare FP16 TC evolution results against sequential DFA simulation.

    The FP16 TC kernel requires sigma==2 and N==16 (padded), so we only
    test binary-alphabet regexes whose DFA fits in 16 states.
    """
    regexes = [
        "(a|b)*abb",
        "(a|b)*a(a|b)",
        "a*b*",
        "(ab|ba)*",
        "(a|b)(a|b)*b",
    ]
    random.seed(123)
    sim = FP16EvolutionGPUSimulator()

    for regex in regexes:
        dfa = compile_regex(regex)
        dm = DFAMatrices(dfa)
        engine = sim.create_engine(dm, max_total_chars=1 << 22, max_batch=200)

        alphabet = list(dfa.alphabet)
        strings = []
        for _ in range(200):
            length = random.randint(0, 500)
            strings.append("".join(random.choice(alphabet) for _ in range(length)))

        gpu_results = engine.simulate_batch(strings)
        cpu_results = [dfa.simulate(s) for s in strings]

        mismatches = sum(g != c for g, c in zip(gpu_results, cpu_results))
        assert mismatches == 0, \
            f"Regex {regex!r}: {mismatches}/{len(strings)} mismatches"

        engine.destroy()
