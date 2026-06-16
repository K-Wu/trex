"""
Tests for src/monoid.py — Transition Monoid (Python precompute).

Test classes
------------
TestMonoidCompute    — monoid size, closure, compose matches matmul,
                       char_to_monoid mapping, accept_table, identity element
TestMonoidSimulate   — cross-validate simulate_monoid against simulate_sequential
                       for patterns: abb, binary_div3, even_a, ab_star,
                       hex_number, identifier  (50 random strings each)
TestMonoidSizeGuard  — compute_monoid(dm, max_size=2) returns None
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
import numpy as np

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, _matmul_int8, simulate_sequential
from src.generate_data import PATTERNS
from src.monoid import compute_monoid, simulate_monoid, MonoidData


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_dm(pattern_name: str) -> tuple[DFAMatrices, object]:
    pat = PATTERNS[pattern_name]
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    return dm, dfa


def _get_alphabet(pattern_name: str) -> str:
    if pattern_name in ("abb", "even_a", "ab_star"):
        return "ab"
    if pattern_name == "binary_div3":
        return "01"
    if pattern_name == "hex_number":
        return "0123456789abcdefx"
    if pattern_name == "identifier":
        return "abcdefghijklmnopqrstuvwxyz0123456789"
    return "abcdefgh"


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestMonoidCompute
# ═══════════════════════════════════════════════════════════════════════════

class TestMonoidCompute:
    """Unit tests for compute_monoid correctness."""

    @pytest.fixture(params=["abb", "binary_div3", "even_a", "ab_star",
                            "hex_number", "identifier"])
    def monoid_fixture(self, request):
        name = request.param
        dm, dfa = _make_dm(name)
        md = compute_monoid(dm)
        assert md is not None, f"compute_monoid returned None for '{name}'"
        return name, dm, dfa, md

    # ── Sanity / structure ─────────────────────────────────────────────────

    def test_monoid_not_none(self, monoid_fixture):
        _, _, _, md = monoid_fixture
        assert md is not None

    def test_size_field_matches_elements(self, monoid_fixture):
        _, _, _, md = monoid_fixture
        assert md.size == len(md.elements)

    def test_elements_dtype(self, monoid_fixture):
        _, dm, _, md = monoid_fixture
        for mat in md.elements:
            assert mat.dtype == np.int8
            assert mat.shape == (dm.n_states, dm.n_states)

    def test_compose_table_shape(self, monoid_fixture):
        _, _, _, md = monoid_fixture
        assert md.compose_table.shape == (md.size, md.size)
        assert md.compose_table.dtype == np.uint16

    def test_accept_table_shape(self, monoid_fixture):
        _, _, _, md = monoid_fixture
        assert md.accept_table.shape == (md.size,)
        assert md.accept_table.dtype == bool

    def test_monoid_size_reasonable(self, monoid_fixture):
        """For small DFAs the monoid should be well under 65536."""
        _, _, _, md = monoid_fixture
        assert md.size < 1000, f"Unexpectedly large monoid: {md.size}"

    # ── Identity element ───────────────────────────────────────────────────

    def test_identity_index_valid(self, monoid_fixture):
        _, dm, _, md = monoid_fixture
        assert 0 <= md.identity_idx < md.size

    def test_identity_is_eye(self, monoid_fixture):
        _, dm, _, md = monoid_fixture
        I = md.elements[md.identity_idx]
        expected = np.eye(dm.n_states, dtype=np.int8)
        assert np.array_equal(I, expected), "Identity element is not the identity matrix"

    def test_identity_left(self, monoid_fixture):
        """I ∘ m == m  for all m in the monoid."""
        _, _, _, md = monoid_fixture
        ii = md.identity_idx
        for j in range(md.size):
            assert md.compose_table[ii, j] == j, \
                f"identity_left failed for j={j}: got {md.compose_table[ii, j]}"

    def test_identity_right(self, monoid_fixture):
        """m ∘ I == m  for all m in the monoid."""
        _, _, _, md = monoid_fixture
        ii = md.identity_idx
        for i in range(md.size):
            assert md.compose_table[i, ii] == i, \
                f"identity_right failed for i={i}: got {md.compose_table[i, ii]}"

    # ── char_to_monoid ─────────────────────────────────────────────────────

    def test_char_to_monoid_covers_alphabet(self, monoid_fixture):
        _, dm, _, md = monoid_fixture
        for ch in dm.alphabet:
            assert ch in md.char_to_monoid, f"char '{ch}' missing from char_to_monoid"

    def test_char_to_monoid_valid_index(self, monoid_fixture):
        _, _, _, md = monoid_fixture
        for ch, idx in md.char_to_monoid.items():
            assert 0 <= idx < md.size, \
                f"char '{ch}' maps to out-of-range index {idx}"

    def test_char_matrix_matches_dm_matrix(self, monoid_fixture):
        """The matrix at char_to_monoid[ch] must equal dm.matrices[ch]."""
        _, dm, _, md = monoid_fixture
        for ch in dm.alphabet:
            idx = md.char_to_monoid[ch]
            assert np.array_equal(md.elements[idx], dm.matrices[ch]), \
                f"Matrix mismatch for char '{ch}'"

    # ── Closure property ───────────────────────────────────────────────────

    def test_compose_table_closed(self, monoid_fixture):
        """compose_table[i, j] must be a valid index for all i, j."""
        _, _, _, md = monoid_fixture
        assert np.all(md.compose_table < md.size), \
            "compose_table contains out-of-range index"

    def test_compose_matches_matmul(self, monoid_fixture):
        """compose_table[i, j] == index of elements[i] @ elements[j]."""
        _, _, _, md = monoid_fixture
        key_to_idx = {mat.tobytes(): k for k, mat in enumerate(md.elements)}

        # Check a random subset (avoid O(M²) for large monoids)
        rng = random.Random(42)
        pairs = [(i, j)
                 for i in range(md.size)
                 for j in range(md.size)]
        if len(pairs) > 2000:
            pairs = rng.sample(pairs, 2000)

        for i, j in pairs:
            product = _matmul_int8(md.elements[i], md.elements[j])
            expected_idx = key_to_idx[product.tobytes()]
            got_idx = int(md.compose_table[i, j])
            assert got_idx == expected_idx, (
                f"compose_table[{i},{j}]={got_idx} but matmul gives index {expected_idx}"
            )

    def test_associativity(self, monoid_fixture):
        """(a ∘ b) ∘ c == a ∘ (b ∘ c) for random triples."""
        _, _, _, md = monoid_fixture
        rng = random.Random(7)
        ct = md.compose_table
        for _ in range(300):
            a, b, c = (rng.randrange(md.size) for _ in range(3))
            left = int(ct[int(ct[a, b]), c])
            right = int(ct[a, int(ct[b, c])])
            assert left == right, f"Associativity failed for ({a},{b},{c})"

    # ── accept_table ───────────────────────────────────────────────────────

    def test_identity_accept_matches_empty_string(self, monoid_fixture):
        """accept_table[identity_idx] should match whether '' is accepted."""
        _, dm, dfa, md = monoid_fixture
        expected = dfa.simulate("")
        got = bool(md.accept_table[md.identity_idx])
        assert got == expected, \
            f"accept_table[identity_idx]={got} but dfa.simulate('')={expected}"

    def test_accept_table_char_matrices(self, monoid_fixture):
        """accept_table for a single-char matrix matches single-char DFA."""
        _, dm, dfa, md = monoid_fixture
        for ch in dm.alphabet:
            idx = md.char_to_monoid[ch]
            expected = dfa.simulate(ch)
            got = bool(md.accept_table[idx])
            assert got == expected, \
                f"accept_table for '{ch}' is {got}, dfa.simulate('{ch}')={expected}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestMonoidSimulate
# ═══════════════════════════════════════════════════════════════════════════

SIMULATE_PATTERNS = [
    "abb", "binary_div3", "even_a", "ab_star", "hex_number", "identifier",
]

class TestMonoidSimulate:
    """Cross-validate simulate_monoid against simulate_sequential."""

    @pytest.mark.parametrize("pattern_name", SIMULATE_PATTERNS)
    def test_cross_validate_random_strings(self, pattern_name):
        dm, dfa = _make_dm(pattern_name)
        md = compute_monoid(dm)
        assert md is not None

        alphabet = _get_alphabet(pattern_name)
        rng = random.Random(hash(pattern_name) & 0xFFFFFFFF)

        mismatches = []
        n_tested = 0

        # Test empty string
        got = simulate_monoid(md, dm, "")
        exp = simulate_sequential(dfa, "")
        if got != exp:
            mismatches.append(("", exp, got))
        n_tested += 1

        # 50 random strings of varying lengths
        for _ in range(50):
            length = rng.randint(0, 30)
            s = "".join(rng.choice(alphabet) for _ in range(length))
            got = simulate_monoid(md, dm, s)
            exp = simulate_sequential(dfa, s)
            if got != exp:
                mismatches.append((s, exp, got))
            n_tested += 1

        assert not mismatches, (
            f"simulate_monoid vs simulate_sequential mismatches for '{pattern_name}' "
            f"({len(mismatches)}/{n_tested} failed):\n" +
            "\n".join(f"  '{s}' expected={e} got={g}" for s, e, g in mismatches[:10])
        )

    @pytest.mark.parametrize("pattern_name", SIMULATE_PATTERNS)
    def test_empty_string(self, pattern_name):
        dm, dfa = _make_dm(pattern_name)
        md = compute_monoid(dm)
        assert md is not None
        assert simulate_monoid(md, dm, "") == simulate_sequential(dfa, "")

    @pytest.mark.parametrize("pattern_name", SIMULATE_PATTERNS)
    def test_single_char_strings(self, pattern_name):
        dm, dfa = _make_dm(pattern_name)
        md = compute_monoid(dm)
        assert md is not None
        for ch in dm.alphabet:
            got = simulate_monoid(md, dm, ch)
            exp = simulate_sequential(dfa, ch)
            assert got == exp, \
                f"Single char '{ch}' on '{pattern_name}': monoid={got} seq={exp}"

    @pytest.mark.parametrize("pattern_name", SIMULATE_PATTERNS)
    def test_longer_strings(self, pattern_name):
        """Longer strings (50–200 chars) also match."""
        dm, dfa = _make_dm(pattern_name)
        md = compute_monoid(dm)
        assert md is not None

        alphabet = _get_alphabet(pattern_name)
        rng = random.Random(999 + hash(pattern_name) & 0xFFFFFF)
        for _ in range(20):
            length = rng.randint(50, 200)
            s = "".join(rng.choice(alphabet) for _ in range(length))
            got = simulate_monoid(md, dm, s)
            exp = simulate_sequential(dfa, s)
            assert got == exp, \
                f"'{pattern_name}' len={length}: monoid={got} seq={exp}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestMonoidSizeGuard
# ═══════════════════════════════════════════════════════════════════════════

class TestMonoidSizeGuard:
    """compute_monoid must return None when the monoid exceeds max_size."""

    def test_returns_none_for_tiny_max_size(self):
        """Any DFA with alphabet ≥ 1 char will exceed max_size=2."""
        dm, _ = _make_dm("abb")
        result = compute_monoid(dm, max_size=2)
        assert result is None, \
            f"Expected None for max_size=2, got MonoidData with size={result and result.size}"

    def test_returns_none_for_max_size_zero(self):
        dm, _ = _make_dm("even_a")
        result = compute_monoid(dm, max_size=0)
        assert result is None

    def test_returns_none_for_max_size_one(self):
        dm, _ = _make_dm("binary_div3")
        result = compute_monoid(dm, max_size=1)
        assert result is None

    def test_returns_data_for_sufficient_max_size(self):
        """With a generous max_size, result must be non-None."""
        dm, _ = _make_dm("abb")
        result = compute_monoid(dm, max_size=65536)
        assert result is not None

    def test_returns_data_when_max_size_equals_monoid_size(self):
        """Boundary: max_size exactly equals the monoid size should succeed."""
        dm, _ = _make_dm("ab_star")
        md_full = compute_monoid(dm)
        assert md_full is not None
        # Recompute with max_size == exact monoid size
        md_exact = compute_monoid(dm, max_size=md_full.size)
        assert md_exact is not None
        assert md_exact.size == md_full.size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
