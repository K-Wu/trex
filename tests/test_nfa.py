"""
Tests for NFA path — matrix export (Task 5).

Covers:
  1. Matrix shape and dtype
  2. Boolean values in matrices
  3. Cross-validation of simulate_nfa against simulate_sequential (DFA)
  4. NFA state count is linear (not exponential)
  5. Cross-validation against Python re.fullmatch
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re
import random
import pytest
import numpy as np

from src.nfa_matrices import NFAMatrices, compile_nfa_matrices, simulate_nfa
from src.regex_to_dfa import compile_regex
from src.simulation import simulate_sequential


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _random_strings(alphabet: str, count: int, max_len: int = 10,
                    seed: int = 42) -> list[str]:
    """Generate random strings over alphabet."""
    rng = random.Random(seed)
    strings: list[str] = ['']  # always include empty string
    for _ in range(count - 1):
        length = rng.randint(0, max_len)
        strings.append(''.join(rng.choice(alphabet) for _ in range(length)))
    return strings


# ═══════════════════════════════════════════════════════════════════════════
# TestNFAMatrices
# ═══════════════════════════════════════════════════════════════════════════

class TestNFAMatrices:

    # ── 1. Matrix shape ──────────────────────────────────────────────────────

    def test_simple_pattern(self):
        """compile_nfa_matrices produces square matrices of size n_states × n_states."""
        nm = compile_nfa_matrices('(a|b)*abb')
        N = nm.n_states
        for ch, T in nm.matrices.items():
            assert T.shape == (N, N), (
                f"Matrix for '{ch}' has shape {T.shape}, expected ({N}, {N})"
            )
        # Also check matrix_stack shape
        assert nm.matrix_stack.shape == (len(nm.alphabet), N, N)

    # ── 2. Boolean values ────────────────────────────────────────────────────

    def test_nfa_matrices_are_boolean(self):
        """All values in every transition matrix must be 0 or 1."""
        nm = compile_nfa_matrices('(a|b)*abb')
        for ch, T in nm.matrices.items():
            unique_vals = set(np.unique(T).tolist())
            assert unique_vals.issubset({0, 1}), (
                f"Matrix for '{ch}' contains non-boolean values: {unique_vals}"
            )

    # ── 3. Cross-validation with DFA sequential simulation ──────────────────

    @pytest.mark.parametrize("pattern", [
        '(a|b)*abb',
        '(ab)*',
        '(b*ab*ab*)*b*',
    ])
    def test_nfa_simulate_matches_dfa(self, pattern):
        """simulate_nfa must agree with simulate_sequential (DFA) on 50 random strings."""
        nm = compile_nfa_matrices(pattern)
        dfa = compile_regex(pattern)

        # Use only the NFA alphabet so both engines see the same inputs
        alphabet = ''.join(sorted(nm.nfa.alphabet)) or 'ab'
        strings = _random_strings(alphabet, 50, max_len=12, seed=hash(pattern) & 0xFFFF)

        mismatches = []
        for s in strings:
            nfa_result = simulate_nfa(nm, s)
            dfa_result = simulate_sequential(dfa, s)
            if nfa_result != dfa_result:
                mismatches.append((s, nfa_result, dfa_result))

        assert not mismatches, (
            f"simulate_nfa vs simulate_sequential mismatches for '{pattern}':\n"
            + '\n'.join(
                f"  '{s}': nfa={nfa_r} dfa={dfa_r}"
                for s, nfa_r, dfa_r in mismatches[:10]
            )
        )

    # ── 4. NFA state count is linear ────────────────────────────────────────

    def test_nfa_state_count_linear(self):
        """NFA state count should be well under 50 for '(a|b)*abb'."""
        nm = compile_nfa_matrices('(a|b)*abb')
        assert nm.n_states_raw < 50, (
            f"NFA has {nm.n_states_raw} raw states — expected < 50"
        )

    # ── 5. Cross-validation with Python re.fullmatch ─────────────────────────

    @pytest.mark.parametrize("pattern", [
        '(a|b)*abb',
        '[a-z]+',
        '(ab)*',
        'a(b|c)*d',
    ])
    def test_cross_validate_python_re(self, pattern):
        """simulate_nfa must agree with re.fullmatch on 30 random strings."""
        nm = compile_nfa_matrices(pattern)

        # Build alphabet from NFA; fall back to lower-case letters
        alphabet = ''.join(sorted(nm.nfa.alphabet)) if nm.nfa.alphabet else 'abcdefghijklmnopqrstuvwxyz'
        strings = _random_strings(alphabet, 30, max_len=10, seed=hash(pattern) & 0xFFFF)

        py_pat = re.compile(f'^(?:{pattern})$')
        mismatches = []
        for s in strings:
            nfa_result = simulate_nfa(nm, s)
            py_result = bool(py_pat.match(s))
            if nfa_result != py_result:
                mismatches.append((s, nfa_result, py_result))

        assert not mismatches, (
            f"simulate_nfa vs re.fullmatch mismatches for '{pattern}':\n"
            + '\n'.join(
                f"  '{s}': nfa={nfa_r} py={py_r}"
                for s, nfa_r, py_r in mismatches[:10]
            )
        )
