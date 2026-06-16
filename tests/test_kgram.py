"""
Tests for src/kgram.py — k-Gram Precomputation.

Test classes
------------
TestKGramAutoK          — auto_k correctness for standard alphabet sizes
TestKGramMonoidMode     — monoid-mode precomputation and simulation
TestKGramMatrixMode     — matrix-mode precomputation correctness
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import itertools
import pytest
import numpy as np

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, _matmul_int8, simulate_sequential
from src.generate_data import PATTERNS
from src.monoid import compute_monoid, MonoidData
from src.kgram import (
    auto_k,
    KGramTable,
    precompute_kgrams,
    simulate_kgram_monoid,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_abb():
    """Return (dm, dfa, md) for the 'abb' pattern (binary alphabet)."""
    pat = PATTERNS["abb"]
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    md = compute_monoid(dm)
    assert md is not None
    return dm, dfa, md


def _random_gram(alphabet, k, rng):
    return tuple(rng.choice(alphabet) for _ in range(k))


def _random_string(alphabet, length, rng):
    return "".join(rng.choice(alphabet) for _ in range(length))


def _manual_monoid_compose(md: MonoidData, dm: DFAMatrices, gram: tuple) -> int:
    """Manually compose a k-gram into a monoid index (left-to-right)."""
    acc = md.identity_idx
    for ch in gram:
        c_idx = md.char_to_monoid[ch]
        acc = int(md.compose_table[c_idx, acc])
    return acc


def _manual_matrix_compose(dm: DFAMatrices, gram: tuple) -> np.ndarray:
    """Manually compose a k-gram into a matrix (left-to-right, newer on left)."""
    acc = dm.identity_matrix()
    for ch in gram:
        acc = _matmul_int8(dm.matrices[ch], acc)
    return acc


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestKGramAutoK
# ═══════════════════════════════════════════════════════════════════════════

class TestKGramAutoK:

    def test_auto_k_binary(self):
        """auto_k(2) == 16  (2^16 = 65536 == max_entries)."""
        assert auto_k(2) == 16

    def test_auto_k_byte(self):
        """auto_k(256) == 2  (256^2 = 65536 == max_entries)."""
        assert auto_k(256) == 2

    def test_auto_k_small(self):
        """auto_k(16) == 4  (16^4 = 65536 == max_entries)."""
        assert auto_k(16) == 4

    def test_auto_k_trivial_alphabet(self):
        """auto_k(1) == 1 and auto_k(0) == 1 (degenerate cases)."""
        assert auto_k(1) == 1
        assert auto_k(0) == 1

    def test_auto_k_custom_max_entries(self):
        """auto_k(2, max_entries=16) == 4  (2^4 = 16)."""
        assert auto_k(2, max_entries=16) == 4

    def test_auto_k_monotone_in_alphabet_size(self):
        """Larger alphabet → smaller or equal k."""
        prev_k = auto_k(2)
        for sigma in [4, 8, 16, 32, 64, 128, 256]:
            k = auto_k(sigma)
            assert k <= prev_k, (
                f"auto_k({sigma})={k} > auto_k of smaller alphabet = {prev_k}"
            )
            prev_k = k


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestKGramMonoidMode
# ═══════════════════════════════════════════════════════════════════════════

class TestKGramMonoidMode:
    """Tests for precompute_kgrams in monoid mode (monoid is not None)."""

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_kgram_matches_sequential_compose(self, k):
        """For 1000 random k-grams, lookup() must match manual sequential compose."""
        dm, dfa, md = _make_abb()
        kg = precompute_kgrams(dm, k, monoid=md)

        alphabet = dm.alphabet
        rng = random.Random(42 + k)

        mismatches = []
        for _ in range(1000):
            gram = _random_gram(alphabet, k, rng)
            got = kg.lookup(gram)
            expected = _manual_monoid_compose(md, dm, gram)
            if got != expected:
                mismatches.append((gram, expected, got))

        assert not mismatches, (
            f"k={k}: {len(mismatches)} mismatches in monoid lookup:\n" +
            "\n".join(
                f"  gram={g} expected={e} got={got}"
                for g, e, got in mismatches[:10]
            )
        )

    @pytest.mark.parametrize("k", [2, 4])
    def test_simulate_with_kgram(self, k):
        """100 random strings cross-validated against simulate_sequential."""
        dm, dfa, md = _make_abb()
        kg = precompute_kgrams(dm, k, monoid=md)

        alphabet = dm.alphabet
        rng = random.Random(7 + k)

        mismatches = []
        for _ in range(100):
            length = rng.randint(0, 64)
            s = _random_string(alphabet, length, rng)
            got = simulate_kgram_monoid(kg, md, dm, s)
            expected = simulate_sequential(dfa, s)
            if got != expected:
                mismatches.append((s, expected, got))

        assert not mismatches, (
            f"k={k}: {len(mismatches)} simulate mismatches:\n" +
            "\n".join(
                f"  '{s}' expected={e} got={g}"
                for s, e, g in mismatches[:10]
            )
        )

    @pytest.mark.parametrize("tail_len", [0, 1, 2, 3])
    def test_tail_handling(self, tail_len):
        """Strings of length 20+tail_len are handled correctly (tests tail logic)."""
        k = 4   # with k=4, tail_len chars remain after 5 full grams
        dm, dfa, md = _make_abb()
        kg = precompute_kgrams(dm, k, monoid=md)

        alphabet = dm.alphabet
        rng = random.Random(99 + tail_len)

        mismatches = []
        for _ in range(200):
            s = _random_string(alphabet, 20 + tail_len, rng)
            got = simulate_kgram_monoid(kg, md, dm, s)
            expected = simulate_sequential(dfa, s)
            if got != expected:
                mismatches.append((s, expected, got))

        assert not mismatches, (
            f"tail_len={tail_len} k={k}: {len(mismatches)} mismatches:\n" +
            "\n".join(
                f"  '{s}' expected={e} got={g}"
                for s, e, g in mismatches[:10]
            )
        )

    def test_empty_string(self):
        """Empty string must return md.accept_table[md.identity_idx]."""
        dm, dfa, md = _make_abb()
        kg = precompute_kgrams(dm, 4, monoid=md)
        got = simulate_kgram_monoid(kg, md, dm, "")
        expected = simulate_sequential(dfa, "")
        assert got == expected

    def test_kgram_table_size(self):
        """KGramTable should have exactly sigma^k entries."""
        dm, dfa, md = _make_abb()
        k = 3
        kg = precompute_kgrams(dm, k, monoid=md)
        sigma = len(dm.alphabet)
        assert len(kg._monoid_table) == sigma ** k

    def test_all_kgrams_valid_monoid_index(self):
        """Every entry in the monoid table must be a valid monoid index."""
        dm, dfa, md = _make_abb()
        k = 3
        kg = precompute_kgrams(dm, k, monoid=md)
        for key, val in kg._monoid_table.items():
            assert 0 <= val < md.size, (
                f"key={key} maps to out-of-range monoid index {val}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestKGramMatrixMode
# ═══════════════════════════════════════════════════════════════════════════

class TestKGramMatrixMode:
    """Tests for precompute_kgrams in matrix mode (monoid=None)."""

    @pytest.mark.parametrize("k", [2, 4])
    def test_kgram_matrix_matches_composition(self, k):
        """For 200 random k-grams, lookup_matrix() must match manual composition."""
        dm, dfa, md = _make_abb()
        kg = precompute_kgrams(dm, k, monoid=None)

        alphabet = dm.alphabet
        rng = random.Random(13 + k)

        mismatches = []
        for _ in range(200):
            gram = _random_gram(alphabet, k, rng)
            got = kg.lookup_matrix(gram)
            expected = _manual_matrix_compose(dm, gram)
            if not np.array_equal(got, expected):
                mismatches.append(gram)

        assert not mismatches, (
            f"k={k}: {len(mismatches)} matrix mismatches for grams: "
            f"{mismatches[:5]}"
        )

    def test_matrix_table_size(self):
        """Matrix table should have exactly sigma^k entries."""
        dm, dfa, md = _make_abb()
        k = 3
        kg = precompute_kgrams(dm, k, monoid=None)
        sigma = len(dm.alphabet)
        assert len(kg._matrix_table) == sigma ** k

    def test_identity_gram_gives_identity_matrix(self):
        """A k-gram of only identity-like operations — verify the special case
        that the table stores correct matrices by checking a single known gram."""
        dm, dfa, md = _make_abb()
        k = 2
        kg = precompute_kgrams(dm, k, monoid=None)
        alphabet = dm.alphabet

        # For every 2-gram, check the matrix equals manual composition
        for gram in itertools.product(alphabet, repeat=k):
            got = kg.lookup_matrix(gram)
            expected = _manual_matrix_compose(dm, gram)
            assert np.array_equal(got, expected), (
                f"Matrix mismatch for gram {gram}"
            )

    @pytest.mark.parametrize("k", [2, 4])
    def test_matrix_mode_gives_correct_acceptance(self, k):
        """Using matrix-mode k-gram table to simulate a string gives correct result."""
        dm, dfa, md = _make_abb()
        kg = precompute_kgrams(dm, k, monoid=None)

        alphabet = dm.alphabet
        rng = random.Random(77 + k)

        mismatches = []
        for _ in range(100):
            length = rng.randint(k, 5 * k)
            # Make length a multiple of k for a clean test
            length = (length // k) * k
            s = _random_string(alphabet, length, rng)

            # Simulate using matrix-mode k-grams
            acc = dm.identity_matrix()
            for pos in range(0, length, k):
                gram = tuple(s[pos:pos + k])
                mat = kg.lookup_matrix(gram)
                acc = _matmul_int8(mat, acc)

            # Apply to start vector
            final = acc.astype(np.int32) @ dm.start_vec.astype(np.int32)
            got = dm.check_accept(final.astype(np.int8))
            expected = simulate_sequential(dfa, s)

            if got != expected:
                mismatches.append((s, expected, got))

        assert not mismatches, (
            f"k={k}: {len(mismatches)} matrix-mode simulation mismatches:\n" +
            "\n".join(
                f"  '{s}' expected={e} got={g}"
                for s, e, g in mismatches[:10]
            )
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
