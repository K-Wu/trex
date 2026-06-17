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

    Set use_gpu=True to dispatch through the CUDA multi-tile kernel.
    """

    def __init__(self, regexes: list[str], use_gpu: bool = False):
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

        # GPU engine (lazy-init)
        self._gpu_engine = None
        if use_gpu:
            self._init_gpu()

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

    def _init_gpu(self, max_B: int = 65536, max_L: int = 4096):
        from src.gpu_bridge_batched import BatchedGPUSimulator, BatchedEvolutionEngine
        import ctypes
        sim = BatchedGPUSimulator()

        NP = self._NP
        sigma = len(self._unified_alphabet)

        trans = np.ascontiguousarray(self._matrix_stack, dtype=np.int8)
        sv = np.ascontiguousarray(self._start_vec, dtype=np.int8)
        # Combined accept mask (union of all patterns) — used by single-pattern fast path
        combined_accept = np.zeros(NP, dtype=np.int8)
        for mask in self._accept_masks:
            combined_accept |= mask

        self._gpu_engine = BatchedEvolutionEngine(
            sim.lib, None,
            max_B=max_B, max_L=max_L,
            start_vec=sv,
            accept_mask=combined_accept,
            trans_matrices=trans,
            N=NP, sigma=sigma,
            char_to_idx=self._unified_char_to_idx,
        )
        self._gpu_lib = sim.lib

        # Set up dispatch_multi ctypes signature
        if not hasattr(sim.lib, '_packed_multi_setup'):
            sim.lib.batched_engine_dispatch_multi.restype = ctypes.c_int
            sim.lib.batched_engine_dispatch_multi.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),   # input
                ctypes.c_int,                     # B
                ctypes.c_int,                     # L
                ctypes.POINTER(ctypes.c_int8),    # accept_masks
                ctypes.c_int,                     # n_patterns
                ctypes.POINTER(ctypes.c_int),     # results
                ctypes.POINTER(ctypes.c_float),   # kernel_ms
                ctypes.POINTER(ctypes.c_float),   # total_ms
            ]
            sim.lib._packed_multi_setup = True

        # Pre-build stacked accept masks: [P, NP] contiguous
        self._gpu_accept_masks = np.ascontiguousarray(
            np.stack(self._accept_masks), dtype=np.int8
        )

    def match_batch(self, strings: list[str]) -> list[list[bool]]:
        """
        Match all strings against all patterns.

        Returns results[pattern_idx][string_idx].
        """
        if self._gpu_engine is not None:
            return self._match_batch_gpu(strings)
        return self._match_batch_cpu(strings)

    def _match_batch_gpu(self, strings: list[str]) -> list[list[bool]]:
        import ctypes
        B = len(strings)
        P = self._n_patterns

        if B == 0:
            return [[] for _ in range(P)]

        L_max = max(len(s) for s in strings) if strings else 0
        if L_max == 0:
            results = []
            for p in range(P):
                accept = self._accept_masks[p].astype(np.int32)
                start = self._start_vec.astype(np.int32)
                is_accept = bool(np.dot(accept, start) > 0)
                results.append([is_accept] * B)
            return results

        input_data, B_padded, L_max = self._gpu_engine._prepare_batch(strings)

        flat_results = np.zeros(P * B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self._gpu_lib.batched_engine_dispatch_multi(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, L_max,
            self._gpu_accept_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            P,
            flat_results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"batched_engine_dispatch_multi failed with code {rc}")

        results = []
        for p in range(P):
            pattern_results = [bool(flat_results[p * B + j]) for j in range(B)]
            results.append(pattern_results)
        return results

    def _match_batch_cpu(self, strings: list[str]) -> list[list[bool]]:
        """CPU fallback for match_batch."""
        B = len(strings)
        P = self._n_patterns
        NP = self._NP
        sigma = len(self._unified_alphabet)
        identity_idx = sigma

        if B == 0:
            return [[] for _ in range(P)]

        L_max = max(len(s) for s in strings) if strings else 0

        if L_max == 0:
            results = []
            for p in range(P):
                accept = self._accept_masks[p].astype(np.int32)
                start = self._start_vec.astype(np.int32)
                is_accept = bool(np.dot(accept, start) > 0)
                results.append([is_accept] * B)
            return results

        input_arr = np.full((L_max, B), identity_idx, dtype=np.int32)
        for j, s in enumerate(strings):
            for t, ch in enumerate(s):
                idx = self._unified_char_to_idx.get(ch)
                if idx is not None:
                    input_arr[t, j] = idx

        identity = np.eye(NP, dtype=np.int8)
        matrices = np.concatenate(
            [self._matrix_stack, identity.reshape(1, NP, NP)], axis=0
        )

        S = np.zeros((NP, B), dtype=np.int32)
        for i in range(NP):
            if self._start_vec[i]:
                S[i, :] = 1

        for t in range(L_max):
            S_new = np.zeros((NP, B), dtype=np.int32)
            chars_at_t = input_arr[t, :]

            for c in range(sigma + 1):
                col_mask = (chars_at_t == c)
                if not np.any(col_mask):
                    continue
                col_indices = np.where(col_mask)[0]
                T_c = matrices[c].astype(np.int32)
                S_new[:, col_indices] = T_c @ S[:, col_indices]

            np.minimum(S_new, 1, out=S_new)
            S = S_new

        results = []
        for p in range(P):
            accept = self._accept_masks[p].astype(np.int32)
            accept_scores = accept @ S
            pattern_results = [bool(accept_scores[j] > 0) for j in range(B)]
            results.append(pattern_results)

        return results

    def match_batch_timed(self, strings: list[str]) -> tuple[list[list[bool]], dict]:
        """Like match_batch but with timing breakdown."""
        if self._gpu_engine is not None:
            return self._match_batch_timed_gpu(strings)
        t0 = time.perf_counter()
        results = self._match_batch_cpu(strings)
        t1 = time.perf_counter()
        timing = {
            'total_ms': (t1 - t0) * 1000.0,
            'n_strings': len(strings),
            'n_patterns': self._n_patterns,
            'NP': self._NP,
        }
        return results, timing

    def _match_batch_timed_gpu(self, strings: list[str]) -> tuple[list[list[bool]], dict]:
        import ctypes
        B = len(strings)
        P = self._n_patterns

        if B == 0:
            return [[] for _ in range(P)], {
                'kernel_ms': 0.0, 'total_ms': 0.0,
                'n_strings': 0, 'n_patterns': P, 'NP': self._NP,
            }

        L_max = max(len(s) for s in strings) if strings else 0
        if L_max == 0:
            results = []
            for p in range(P):
                accept = self._accept_masks[p].astype(np.int32)
                start = self._start_vec.astype(np.int32)
                is_accept = bool(np.dot(accept, start) > 0)
                results.append([is_accept] * B)
            return results, {
                'kernel_ms': 0.0, 'total_ms': 0.0,
                'n_strings': B, 'n_patterns': P, 'NP': self._NP,
            }

        input_data, B_padded, L_max = self._gpu_engine._prepare_batch(strings)

        flat_results = np.zeros(P * B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self._gpu_lib.batched_engine_dispatch_multi(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, L_max,
            self._gpu_accept_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            P,
            flat_results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"batched_engine_dispatch_multi failed with code {rc}")

        results = []
        for p in range(P):
            pattern_results = [bool(flat_results[p * B + j]) for j in range(B)]
            results.append(pattern_results)

        timing = {
            'kernel_ms': kern_ms.value,
            'total_ms': total_ms.value,
            'n_strings': B,
            'n_patterns': P,
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
