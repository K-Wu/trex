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

    def __init__(self, regexes: list[str], use_gpu: bool = False,
                 gpu_mode: str = 'auto'):
        """Compile multiple regex patterns into a packed block-diagonal DFA.

        gpu_mode: 'auto' (sparse if all patterns fit in one 16-state tile, else dense),
                  'sparse' (force sparse kernel), 'dense' (force dense multitile kernel)
        """
        self._regexes = regexes
        self._n_patterns = len(regexes)
        self._gpu_mode = gpu_mode

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
        self._sparse_gpu = False
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

    def _can_use_sparse(self) -> bool:
        """Check if all patterns have exactly 16 padded states (one WMMA tile)."""
        sigma = len(self._unified_alphabet)
        if sigma != 2:
            return False
        return all(n == 16 for n in self._state_counts)

    def _init_gpu(self, max_B: int = 65536, max_L: int = 4096):
        from src.gpu_bridge_batched import BatchedGPUSimulator, BatchedEvolutionEngine
        import ctypes
        sim = BatchedGPUSimulator()

        NP = self._NP
        sigma = len(self._unified_alphabet)
        P = self._n_patterns

        # Decide sparse vs dense
        use_sparse = False
        if self._gpu_mode == 'sparse':
            use_sparse = True
        elif self._gpu_mode == 'dense':
            use_sparse = False
        else:  # auto
            use_sparse = self._can_use_sparse()

        self._gpu_lib = sim.lib

        if use_sparse:
            self._init_gpu_sparse(sim.lib, max_B, max_L)
            return

        trans = np.ascontiguousarray(self._matrix_stack, dtype=np.int8)
        sv = np.ascontiguousarray(self._start_vec, dtype=np.int8)
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

    def _init_gpu_sparse(self, lib, max_B: int, max_L: int):
        import ctypes
        P = self._n_patterns
        sigma = len(self._unified_alphabet)

        # Extract diagonal T-blocks: [P, sigma, 16, 16]
        T_diag = np.zeros((P, sigma, 16, 16), dtype=np.int8)
        for p in range(P):
            off = self._offsets[p]
            for c in range(sigma):
                T_diag[p, c] = self._matrix_stack[c, off:off+16, off:off+16]

        # Per-pattern start vectors: [P, 16]
        start_vecs = np.zeros((P, 16), dtype=np.int8)
        for p in range(P):
            off = self._offsets[p]
            start_vecs[p] = self._start_vec[off:off+16]

        # Per-pattern accept masks: [P, 16]
        accept_masks_sp = np.zeros((P, 16), dtype=np.int8)
        for p in range(P):
            off = self._offsets[p]
            accept_masks_sp[p] = self._accept_masks[p][off:off+16]

        T_diag = np.ascontiguousarray(T_diag, dtype=np.int8)
        start_vecs = np.ascontiguousarray(start_vecs, dtype=np.int8)
        accept_masks_sp = np.ascontiguousarray(accept_masks_sp, dtype=np.int8)

        # Set up ctypes signatures
        if not hasattr(lib, '_sparse_setup'):
            lib.batched_engine_init_sparse.restype = ctypes.c_int
            lib.batched_engine_init_sparse.argtypes = [
                ctypes.c_int,                     # P
                ctypes.c_int,                     # sigma
                ctypes.POINTER(ctypes.c_int8),    # T_diag
                ctypes.POINTER(ctypes.c_int8),    # start_vecs
                ctypes.POINTER(ctypes.c_int8),    # accept_masks
                ctypes.c_int,                     # max_B
                ctypes.c_int,                     # max_L
            ]
            lib.batched_engine_dispatch_sparse.restype = ctypes.c_int
            lib.batched_engine_dispatch_sparse.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),   # input
                ctypes.c_int,                     # B
                ctypes.c_int,                     # L
                ctypes.POINTER(ctypes.c_int),     # results
                ctypes.POINTER(ctypes.c_float),   # kernel_ms
                ctypes.POINTER(ctypes.c_float),   # total_ms
            ]
            lib.batched_prepare_input.restype = None
            lib.batched_prepare_input.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
            lib._sparse_setup = True

        rc = lib.batched_engine_init_sparse(
            P, sigma,
            T_diag.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            start_vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            accept_masks_sp.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            max_B, max_L,
        )
        if rc != 0:
            raise RuntimeError(f"batched_engine_init_sparse failed: {rc}")

        # Build char_to_idx for prepare_input
        self._sp_identity_idx = sigma
        self._sp_char_to_idx = np.full(256, -1, dtype=np.int32)
        for ch, idx in self._unified_char_to_idx.items():
            self._sp_char_to_idx[ord(ch)] = idx

        self._sparse_gpu = True
        self._gpu_engine = True  # sentinel so match_batch uses GPU path

    def match_batch(self, strings: list[str]) -> list[list[bool]]:
        """
        Match all strings against all patterns.

        Returns results[pattern_idx][string_idx].
        """
        if self._gpu_engine is not None:
            return self._match_batch_gpu(strings)
        return self._match_batch_cpu(strings)

    def _prepare_batch_sparse(self, strings: list[str]):
        """Prepare input for sparse dispatch (reuses batched_prepare_input C func)."""
        import ctypes
        B = len(strings)
        L_max = max(len(s) for s in strings) if strings else 0
        B_padded = ((B + 63) // 64) * 64

        strings_concat = "".join(strings).encode("latin-1")
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)

        output = np.zeros(L_max * B_padded, dtype=np.uint8)
        self._gpu_lib.batched_prepare_input(
            strings_concat,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, B_padded, L_max,
            self._sp_char_to_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._sp_identity_idx,
        )
        return output, B_padded, L_max

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

        if self._sparse_gpu:
            return self._match_batch_gpu_sparse(strings, B, L_max)

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

    def _match_batch_gpu_sparse(self, strings, B, L_max):
        import ctypes
        P = self._n_patterns
        input_data, B_padded, L_max = self._prepare_batch_sparse(strings)

        flat_results = np.zeros(P * B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self._gpu_lib.batched_engine_dispatch_sparse(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, L_max,
            flat_results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"batched_engine_dispatch_sparse failed: {rc}")

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

        flat_results = np.zeros(P * B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        if self._sparse_gpu:
            input_data, B_padded, L_max = self._prepare_batch_sparse(strings)
            rc = self._gpu_lib.batched_engine_dispatch_sparse(
                input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                B, L_max,
                flat_results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.byref(kern_ms),
                ctypes.byref(total_ms),
            )
            if rc != 0:
                raise RuntimeError(f"dispatch_sparse failed: {rc}")
        else:
            input_data, B_padded, L_max = self._gpu_engine._prepare_batch(strings)
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
                raise RuntimeError(f"dispatch_multi failed: {rc}")

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
            'backend': ('gpu-sparse' if self._sparse_gpu else 'gpu')
                       if self._gpu_engine is not None else 'cpu',
        }
