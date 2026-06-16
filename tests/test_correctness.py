"""
Correctness tests for the tensor-core regex matching system.

Tests:
  1. Regex → DFA compilation (state counts, acceptance)
  2. Matrix encoding (transition matrices are valid)
  3. Simulation backends agree (sequential = matrix = prefix scan)
  4. Edge cases (empty strings, single chars, long strings)
  5. All pattern library entries
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import re as python_re
from src.regex_to_dfa import compile_regex, RegexParser, nfa_to_dfa, complete_dfa, minimize_dfa
from src.simulation import (
    DFAMatrices, simulate_sequential, simulate_matrix_sequential,
    simulate_prefix_scan, simulate_batch_matrix, simulate_batch_sequential,
    prefix_scan_sequential, prefix_scan_parallel, _matmul_int8,
)
from src.generate_data import PATTERNS, gen_random_string
import random


# ═══════════════════════════════════════════════════════════════════════════
# 1. DFA Compilation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDFACompilation:

    def test_simple_literal(self):
        dfa = compile_regex("abc")
        assert dfa.simulate("abc")
        assert not dfa.simulate("ab")
        assert not dfa.simulate("abcd")
        assert not dfa.simulate("")

    def test_alternation(self):
        dfa = compile_regex("a|b")
        assert dfa.simulate("a")
        assert dfa.simulate("b")
        assert not dfa.simulate("c")
        assert not dfa.simulate("ab")
        assert not dfa.simulate("")

    def test_kleene_star(self):
        dfa = compile_regex("a*")
        assert dfa.simulate("")
        assert dfa.simulate("a")
        assert dfa.simulate("aaa")
        assert not dfa.simulate("b")
        assert not dfa.simulate("ab")

    def test_plus(self):
        dfa = compile_regex("a+")
        assert not dfa.simulate("")
        assert dfa.simulate("a")
        assert dfa.simulate("aaa")
        assert not dfa.simulate("b")

    def test_optional(self):
        dfa = compile_regex("ab?c")
        assert dfa.simulate("ac")
        assert dfa.simulate("abc")
        assert not dfa.simulate("abbc")

    def test_char_class(self):
        dfa = compile_regex("[abc]")
        assert dfa.simulate("a")
        assert dfa.simulate("b")
        assert dfa.simulate("c")
        assert not dfa.simulate("d")
        assert not dfa.simulate("ab")

    def test_char_range(self):
        dfa = compile_regex("[a-d]+")
        assert dfa.simulate("abcd")
        assert dfa.simulate("a")
        assert not dfa.simulate("e")
        assert not dfa.simulate("")

    def test_abb_pattern(self):
        dfa = compile_regex("(a|b)*abb")
        positive = ["abb", "aabb", "babb", "ababb", "aababb", "bbbabb"]
        negative = ["ab", "ba", "a", "b", "", "abc", "abba", "ababab"]
        for s in positive:
            assert dfa.simulate(s), f"Should match: '{s}'"
        for s in negative:
            assert not dfa.simulate(s), f"Should not match: '{s}'"

    def test_dfa_state_count_upper_bound(self):
        """DFA should not have an unreasonable number of states."""
        dfa = compile_regex("(a|b)*abb")
        # Minimized DFA for (a|b)*abb has exactly 5 states (incl. dead)
        assert dfa.n_states <= 6, f"Too many states: {dfa.n_states}"

    def test_completeness(self):
        """After completion, every (state, char) pair has a transition."""
        dfa = compile_regex("(a|b)*abb")
        for s in range(dfa.n_states):
            for ch in dfa.alphabet:
                assert ch in dfa.transitions.get(s, {}), \
                    f"Missing transition: state {s}, char '{ch}'"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Matrix Encoding Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMatrixEncoding:

    def setup_method(self):
        self.dfa = compile_regex("(a|b)*abb")
        self.dm = DFAMatrices(self.dfa)

    def test_matrix_dimensions(self):
        """Matrices should be padded to multiple of 16."""
        assert self.dm.n_states >= self.dm.n_states_raw
        assert self.dm.n_states % 16 == 0
        for ch in self.dm.alphabet:
            T = self.dm.get_matrix_for_char(ch)
            assert T.shape == (self.dm.n_states, self.dm.n_states)

    def test_matrix_dtype(self):
        for ch in self.dm.alphabet:
            T = self.dm.get_matrix_for_char(ch)
            assert T.dtype == np.int8

    def test_matrix_values_binary(self):
        """Transition matrices should contain only 0 and 1."""
        for ch in self.dm.alphabet:
            T = self.dm.get_matrix_for_char(ch)
            assert set(np.unique(T)).issubset({0, 1})

    def test_dfa_one_successor_per_column(self):
        """For a complete DFA, each column has exactly one 1 (one successor)."""
        for ch in self.dm.alphabet:
            T = self.dm.get_matrix_for_char(ch)
            col_sums = T.sum(axis=0)
            for j in range(self.dm.n_states):
                assert col_sums[j] == 1, \
                    f"Column {j} of T['{ch}'] has sum {col_sums[j]}, expected 1"

    def test_matrix_product_preserves_one_per_column(self):
        """Product of DFA transition matrices has exactly one 1 per column."""
        T_a = self.dm.get_matrix_for_char('a')
        T_b = self.dm.get_matrix_for_char('b')
        product = _matmul_int8(T_b, T_a)
        col_sums = product.sum(axis=0)
        for j in range(self.dm.n_states):
            assert col_sums[j] == 1, \
                f"Product column {j} has sum {col_sums[j]}, expected 1"

    def test_start_vector(self):
        assert self.dm.start_vec.sum() == 1
        assert self.dm.start_vec[self.dfa.start] == 1

    def test_identity_matrix(self):
        I = self.dm.identity_matrix()
        for ch in self.dm.alphabet:
            T = self.dm.get_matrix_for_char(ch)
            assert np.array_equal(_matmul_int8(T, I), T)
            assert np.array_equal(_matmul_int8(I, T), T)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Simulation Backend Agreement Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSimulationAgreement:
    """All simulation backends must produce identical results."""

    @pytest.fixture(params=['abb', 'binary_div3', 'even_a', 'ab_star',
                            'hex_number', 'identifier'])
    def pattern_fixture(self, request):
        pinfo = PATTERNS[request.param]
        dfa = compile_regex(pinfo.regex)
        dm = DFAMatrices(dfa)
        return request.param, pinfo, dfa, dm

    def _get_test_strings(self, pattern_name):
        rng = random.Random(123)
        if pattern_name in ('abb', 'even_a', 'ab_star'):
            alphabet = 'ab'
        elif pattern_name == 'binary_div3':
            alphabet = '01'
        elif pattern_name == 'hex_number':
            alphabet = '0123456789abcdefx'
        elif pattern_name == 'identifier':
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
        else:
            alphabet = 'abcdefgh'

        strings = ['']  # empty string
        # Short strings (exhaustive for small alphabets)
        if len(alphabet) <= 4:
            for l in range(1, 6):
                for _ in range(min(50, len(alphabet) ** l)):
                    strings.append(gen_random_string(alphabet, l, rng))
        # Medium strings
        for l in [10, 50, 100, 500]:
            for _ in range(20):
                strings.append(gen_random_string(alphabet, l, rng))
        # Long strings
        for l in [1000, 5000]:
            for _ in range(5):
                strings.append(gen_random_string(alphabet, l, rng))
        return strings

    def test_all_backends_agree(self, pattern_fixture):
        pname, pinfo, dfa, dm = pattern_fixture
        strings = self._get_test_strings(pname)

        for s in strings:
            # Filter to strings within alphabet
            if not all(c in dfa.alphabet for c in s):
                continue

            r_seq = simulate_sequential(dfa, s)
            r_mat = simulate_matrix_sequential(dm, s)
            r_scan_s = simulate_prefix_scan(dm, s, use_parallel=False)
            r_scan_p = simulate_prefix_scan(dm, s, use_parallel=True)

            assert r_seq == r_mat == r_scan_s == r_scan_p, (
                f"Disagreement on pattern '{pname}', string '{s[:50]}...' (len={len(s)}): "
                f"seq={r_seq} mat={r_mat} scan_seq={r_scan_s} scan_par={r_scan_p}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Prefix Scan Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPrefixScan:

    def test_single_matrix(self):
        """Prefix scan of a single matrix returns that matrix."""
        M = np.eye(16, dtype=np.int8)
        M[0, 1] = 1; M[0, 0] = 0  # non-trivial
        result = prefix_scan_parallel(M.reshape(1, 16, 16))
        assert np.array_equal(result[0], M)

    def test_two_matrices(self):
        """Prefix scan of [A, B] returns [A, B@A]."""
        N = 16
        A = np.eye(N, dtype=np.int8)
        B = np.eye(N, dtype=np.int8)
        # Permutation: A swaps 0↔1, B swaps 1↔2
        A[0, 0] = 0; A[1, 1] = 0; A[0, 1] = 1; A[1, 0] = 1
        B[1, 1] = 0; B[2, 2] = 0; B[1, 2] = 1; B[2, 1] = 1

        matrices = np.stack([A, B])
        result_par = prefix_scan_parallel(matrices)
        result_seq = prefix_scan_sequential(matrices)

        assert np.array_equal(result_par[0], A)
        assert np.array_equal(result_seq[0], A)
        expected_BA = _matmul_int8(B, A)
        assert np.array_equal(result_par[1], expected_BA)
        assert np.array_equal(result_seq[1], expected_BA)

    def test_parallel_equals_sequential(self):
        """Parallel and sequential prefix scan produce identical results."""
        rng = np.random.RandomState(42)
        N = 16
        L = 64
        # Generate random permutation matrices (valid DFA transitions)
        matrices = np.zeros((L, N, N), dtype=np.int8)
        for i in range(L):
            perm = rng.permutation(N)
            for j in range(N):
                matrices[i, perm[j], j] = 1

        result_seq = prefix_scan_sequential(matrices)
        result_par = prefix_scan_parallel(matrices)
        assert np.array_equal(result_seq, result_par), \
            f"Mismatch at {np.where(result_seq != result_par)}"

    def test_various_lengths(self):
        """Prefix scan works for various lengths (power of 2 and non-power)."""
        rng = np.random.RandomState(99)
        N = 16
        for L in [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 128]:
            matrices = np.zeros((L, N, N), dtype=np.int8)
            for i in range(L):
                perm = rng.permutation(N)
                for j in range(N):
                    matrices[i, perm[j], j] = 1
            result_seq = prefix_scan_sequential(matrices)
            result_par = prefix_scan_parallel(matrices)
            assert np.array_equal(result_seq, result_par), \
                f"Mismatch for L={L}"

    def test_identity_chain(self):
        """Prefix scan of identity matrices returns identities."""
        N = 16
        L = 32
        I = np.eye(N, dtype=np.int8)
        matrices = np.tile(I, (L, 1, 1))
        result = prefix_scan_parallel(matrices)
        for i in range(L):
            assert np.array_equal(result[i], I)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Batch Simulation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchSimulation:

    def test_batch_agrees_with_sequential(self):
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        rng = random.Random(42)
        strings = [gen_random_string('ab', rng.randint(0, 100), rng) for _ in range(200)]

        results_seq = simulate_batch_sequential(dfa, strings)
        results_mat = simulate_batch_matrix(dm, strings)
        assert results_seq == results_mat


# ═══════════════════════════════════════════════════════════════════════════
# 6. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_string(self):
        dfa = compile_regex("a*")
        dm = DFAMatrices(dfa)
        assert simulate_sequential(dfa, "") == True
        assert simulate_prefix_scan(dm, "") == True

    def test_single_char(self):
        dfa = compile_regex("a")
        dm = DFAMatrices(dfa)
        assert simulate_prefix_scan(dm, "a") == True
        assert simulate_prefix_scan(dm, "b") == False

    def test_very_long_string(self):
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)
        rng = random.Random(42)
        # 10K chars ending in 'abb'
        s = gen_random_string('ab', 9997, rng) + 'abb'
        r_seq = simulate_sequential(dfa, s)
        r_scan = simulate_prefix_scan(dm, s, use_parallel=True)
        assert r_seq == r_scan == True

    def test_all_same_char(self):
        dfa = compile_regex("a+")
        dm = DFAMatrices(dfa)
        s = "a" * 1000
        assert simulate_prefix_scan(dm, s) == True
        s = "b" * 1000
        assert simulate_prefix_scan(dm, s) == False

    def test_alternating_pattern(self):
        dfa = compile_regex("(ab)*")
        dm = DFAMatrices(dfa)
        assert simulate_prefix_scan(dm, "ababab") == True
        assert simulate_prefix_scan(dm, "ababa") == False
        assert simulate_prefix_scan(dm, "") == True


# ═══════════════════════════════════════════════════════════════════════════
# 7. Cross-validation with Python re module
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossValidation:
    """
    Cross-validate our DFA against Python's re module for patterns
    that both support (full-string match semantics).
    """

    @pytest.mark.parametrize("regex,test_strings", [
        ("abc", ["abc", "ab", "abcd", "xabc", ""]),
        ("a|b", ["a", "b", "c", "ab", ""]),
        ("(ab)+", ["ab", "abab", "ababab", "a", "aba", ""]),
        ("[0-9]+", ["123", "0", "abc", "1a", ""]),
    ])
    def test_matches_python_re(self, regex, test_strings):
        dfa = compile_regex(regex)
        dm = DFAMatrices(dfa)
        py_pattern = python_re.compile(f"^({regex})$")

        for s in test_strings:
            if not all(c in dfa.alphabet for c in s):
                continue
            expected = bool(py_pattern.match(s))
            got_seq = simulate_sequential(dfa, s)
            got_scan = simulate_prefix_scan(dm, s)
            assert got_seq == expected, f"Sequential mismatch on '{s}': got {got_seq}, expected {expected}"
            assert got_scan == expected, f"Scan mismatch on '{s}': got {got_scan}, expected {expected}"


# ═══════════════════════════════════════════════════════════════════════════
# 8. GPU cross-validation (requires compiled CUDA kernel)
# ═══════════════════════════════════════════════════════════════════════════

def _gpu_available():
    try:
        from src.gpu_bridge import GPUSimulator
        GPUSimulator()
        return True
    except Exception:
        return False

@pytest.mark.skipif(not _gpu_available(), reason="GPU or libdfa_scan.so not available")
class TestGPUCrossValidation:
    """Cross-validate GPU tensor-core results against CPU backends."""

    @pytest.fixture(autouse=True)
    def setup_gpu(self):
        from src.gpu_bridge import GPUSimulator
        self.gpu = GPUSimulator()

    @pytest.mark.parametrize("pattern_name", [
        "abb", "binary_div3", "even_a", "ab_star", "hex_number", "identifier",
    ])
    def test_gpu_agrees_with_cpu(self, pattern_name):
        pat = PATTERNS[pattern_name]
        dfa = compile_regex(pat.regex)
        dm = DFAMatrices(dfa)
        alpha = sorted(dfa.alphabet)

        random.seed(42 + hash(pattern_name))
        for length in [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            for _ in range(5):
                s = ''.join(random.choice(alpha) for _ in range(length))
                cpu_seq = simulate_sequential(dfa, s)
                cpu_scan = simulate_prefix_scan(dm, s)
                gpu_result = self.gpu.simulate(dm, s)
                assert cpu_seq == cpu_scan == gpu_result, (
                    f"Mismatch on pattern={pattern_name} len={length}: "
                    f"seq={cpu_seq} scan={cpu_scan} gpu={gpu_result}"
                )

    def test_gpu_long_strings(self):
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)

        random.seed(999)
        for length in [2048, 4096, 8192]:
            s = ''.join(random.choice('ab') for _ in range(length))
            cpu = simulate_sequential(dfa, s)
            gpu = self.gpu.simulate(dm, s)
            assert cpu == gpu, f"Long string mismatch at L={length}"

            # Also test known-accepting strings
            s_accept = s[:-3] + 'abb'
            cpu_a = simulate_sequential(dfa, s_accept)
            gpu_a = self.gpu.simulate(dm, s_accept)
            assert cpu_a == gpu_a == True, f"Known-accept mismatch at L={length}"

    def test_gpu_batch(self):
        dfa = compile_regex("(a|b)*abb")
        dm = DFAMatrices(dfa)

        random.seed(777)
        strings = [''.join(random.choice('ab') for _ in range(100)) for _ in range(50)]
        cpu_results = [simulate_sequential(dfa, s) for s in strings]
        gpu_results = self.gpu.simulate_batch(dm, strings)
        assert cpu_results == gpu_results


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
