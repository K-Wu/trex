"""
PackedEngine: multi-pattern matching via block-diagonal DFA packing.

Given P regex patterns, constructs a single block-diagonal DFA where each
pattern's transition matrices occupy a diagonal block. The combined state
dimension NP = sum(Ni_padded) allows all patterns to be evaluated via a
single batched state-vector evolution.

Algorithm:
  1. Compile each regex to a DFA and build per-pattern DFAMatrices.
  2. Compute a unified alphabet across all patterns.
  3. Construct NP x NP block-diagonal transition matrices.
  4. Build a combined start vector and per-pattern accept masks.
  5. Evolve all strings simultaneously through the packed DFA.
  6. Check acceptance per-pattern using individual accept masks.
"""

from __future__ import annotations
import time
import numpy as np
from src.regex_to_dfa import compile_regex, DFA
from src.simulation import DFAMatrices


class PackedEngine:
    """
    Multi-pattern regex matcher using block-diagonal DFA packing.

    Each pattern's DFA occupies a diagonal block in the combined transition
    matrices. A single batched evolution step processes all patterns at once.
    """

    def __init__(self, regexes: list[str]):
        """Compile multiple regex patterns into a packed block-diagonal DFA."""
        self._regexes = regexes
        self._n_patterns = len(regexes)

        # Compile each pattern individually
        self._dfas: list[DFA] = []
        self._dfa_matrices: list[DFAMatrices] = []
        for regex in regexes:
            dfa = compile_regex(regex)
            dm = DFAMatrices(dfa)
            self._dfas.append(dfa)
            self._dfa_matrices.append(dm)

        # Build unified alphabet
        self._build_unified_alphabet()

        # Build block-diagonal structures
        self._build_block_diagonal()

    def _build_unified_alphabet(self):
        """Build unified alphabet and char_to_idx across all patterns."""
        all_chars: set[str] = set()
        for dm in self._dfa_matrices:
            all_chars.update(dm.alphabet)
        self._unified_alphabet = sorted(all_chars)
        self._unified_char_to_idx = {c: i for i, c in enumerate(self._unified_alphabet)}

    def _build_block_diagonal(self):
        """Construct block-diagonal transition matrices and state vectors."""
        # Compute offsets and total state dimension
        self._state_counts = [dm.n_states for dm in self._dfa_matrices]
        self._offsets = []
        offset = 0
        for n in self._state_counts:
            self._offsets.append(offset)
            offset += n
        NP_raw = offset

        # Pad total to multiple of 16
        self._NP = ((NP_raw + 15) // 16) * 16
        NP = self._NP

        sigma = len(self._unified_alphabet)

        # Build block-diagonal matrix stack: shape (sigma, NP, NP)
        self._matrix_stack = np.zeros((sigma, NP, NP), dtype=np.int8)

        for c_idx, ch in enumerate(self._unified_alphabet):
            T_packed = self._matrix_stack[c_idx]
            for p, dm in enumerate(self._dfa_matrices):
                off = self._offsets[p]
                n = self._state_counts[p]
                if ch in dm.matrices:
                    # Copy pattern p's transition matrix into its block
                    T_packed[off:off + n, off:off + n] = dm.matrices[ch]
                else:
                    # Character not in pattern's alphabet -> identity (stay)
                    for i in range(n):
                        T_packed[off + i, off + i] = 1

            # Padded rows/cols beyond NP_raw get identity (self-loop)
            for i in range(NP_raw, NP):
                T_packed[i, i] = 1

        # Build combined start vector
        self._start_vec = np.zeros(NP, dtype=np.int8)
        for p, dm in enumerate(self._dfa_matrices):
            off = self._offsets[p]
            self._start_vec[off + dm.dfa.start] = 1

        # Build per-pattern accept masks
        self._accept_masks = []
        for p, dm in enumerate(self._dfa_matrices):
            mask = np.zeros(NP, dtype=np.int8)
            off = self._offsets[p]
            for s in dm.dfa.accept_states:
                mask[off + s] = 1
            self._accept_masks.append(mask)

    def match_batch(self, strings: list[str]) -> list[list[bool]]:
        """
        Match all strings against all patterns.

        Returns results[pattern_idx][string_idx].
        """
        B = len(strings)
        P = self._n_patterns
        NP = self._NP
        sigma = len(self._unified_alphabet)
        identity_idx = sigma  # index for identity (padding) matrix

        if B == 0:
            return [[] for _ in range(P)]

        # Find L_max
        L_max = max(len(s) for s in strings) if strings else 0

        if L_max == 0:
            # All strings are empty; check if start state is in accept for each pattern
            results = []
            for p in range(P):
                accept = self._accept_masks[p].astype(np.int32)
                start = self._start_vec.astype(np.int32)
                is_accept = bool(np.dot(accept, start) > 0)
                results.append([is_accept] * B)
            return results

        # Build position-contiguous input array
        # input_arr[t, j] = char index at position t of string j
        input_arr = np.full((L_max, B), identity_idx, dtype=np.int32)
        for j, s in enumerate(strings):
            for t, ch in enumerate(s):
                idx = self._unified_char_to_idx.get(ch)
                if idx is not None:
                    input_arr[t, j] = idx
                # Characters not in unified alphabet get identity

        # Extend matrix_stack with identity
        identity = np.eye(NP, dtype=np.int8)
        matrices = np.concatenate(
            [self._matrix_stack, identity.reshape(1, NP, NP)], axis=0
        )  # shape: (sigma+1, NP, NP)

        # Initialize state matrix S[NP, B]
        S = np.zeros((NP, B), dtype=np.int32)
        # Set start states for all patterns
        for i in range(NP):
            if self._start_vec[i]:
                S[i, :] = 1

        # Evolve through each position
        for t in range(L_max):
            S_new = np.zeros((NP, B), dtype=np.int32)
            chars_at_t = input_arr[t, :]  # shape (B,)

            # Group columns by character index for batched matmul
            for c in range(sigma + 1):
                col_mask = (chars_at_t == c)
                if not np.any(col_mask):
                    continue
                col_indices = np.where(col_mask)[0]
                T_c = matrices[c].astype(np.int32)
                S_new[:, col_indices] = T_c @ S[:, col_indices]

            # Boolean threshold: clamp to [0, 1]
            np.minimum(S_new, 1, out=S_new)
            S = S_new

        # Check acceptance per-pattern
        results = []
        for p in range(P):
            accept = self._accept_masks[p].astype(np.int32)  # shape (NP,)
            accept_scores = accept @ S  # shape (B,)
            pattern_results = [bool(accept_scores[j] > 0) for j in range(B)]
            results.append(pattern_results)

        return results

    def match_batch_timed(self, strings: list[str]) -> tuple[list[list[bool]], dict]:
        """Like match_batch but with timing breakdown."""
        t0 = time.perf_counter()
        results = self.match_batch(strings)
        t1 = time.perf_counter()
        timing = {
            'total_ms': (t1 - t0) * 1000.0,
            'n_strings': len(strings),
            'n_patterns': self._n_patterns,
            'NP': self._NP,
        }
        return results, timing

    @property
    def config_info(self) -> dict:
        """Returns metadata: n_patterns, state counts, NP, etc."""
        return {
            'n_patterns': self._n_patterns,
            'state_counts': list(self._state_counts),
            'offsets': list(self._offsets),
            'NP': self._NP,
            'unified_alphabet_size': len(self._unified_alphabet),
            'regexes': list(self._regexes),
        }
