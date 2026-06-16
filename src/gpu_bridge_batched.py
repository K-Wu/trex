"""
Python bridge to the CUDA batched state-vector evolution engine via ctypes.

The batched evolution engine processes B strings simultaneously using a
position-contiguous layout: input[pos][batch_col] = char_index. It applies
transition matrices column-by-column using tensor-core int8 matmuls.

Usage:
    from src.gpu_bridge_batched import BatchedGPUSimulator
    sim = BatchedGPUSimulator()
    engine = sim.create_engine(dm, max_B=65536, max_L=4096)
    results = engine.simulate_batch(["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    engine.destroy()
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.simulation import DFAMatrices


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libbatched_evolution.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libbatched_evolution.so not found at {base}. Run 'make' first."
    )


class BatchedEvolutionEngine:
    """Wraps a persistent GPU engine context for one DFA using batched evolution."""

    def __init__(self, lib, dm: DFAMatrices,
                 max_B: int = 65536, max_L: int = 4096):
        self.lib = lib
        self.dm = dm

        N = dm.n_states
        sigma = len(dm.alphabet)
        start_state = dm.dfa.start

        # Ensure contiguous int8 arrays for transition matrices and accept mask
        trans = np.ascontiguousarray(dm.matrix_stack, dtype=np.int8)
        accept = np.ascontiguousarray(dm.accept_mask, dtype=np.int8)

        rc = self.lib.batched_engine_init(
            N,
            sigma,
            trans.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            start_state,
            max_B,
            max_L,
        )
        if rc != 0:
            raise RuntimeError(f"batched_engine_init failed with code {rc}")

        # Build char_to_idx lookup: int32[256], default to identity_idx (sigma)
        self._identity_idx = sigma
        self._char_to_idx = np.full(256, -1, dtype=np.int32)
        for ch, idx in dm.char_to_idx.items():
            self._char_to_idx[ord(ch)] = idx

    def destroy(self):
        self.lib.batched_engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        """Convert variable-length strings to position-contiguous uint8 layout.

        Returns (output, B_padded, L_max) where output is a uint8 array of
        shape [L_max * B_padded] in position-contiguous order.
        """
        B = len(strings)
        L_max = max(len(s) for s in strings) if strings else 0

        # Pad B to next multiple of 64 (COLS_PER_BLOCK)
        B_padded = ((B + 63) // 64) * 64

        # Concatenate all strings into one bytes buffer
        strings_concat = "".join(strings).encode("latin-1")

        # Build CSR offsets: int32[B+1]
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)

        # Allocate output: uint8[L_max * B_padded]
        output = np.zeros(L_max * B_padded, dtype=np.uint8)

        # Call C function for fast transpose + char mapping
        self.lib.batched_prepare_input(
            strings_concat,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B,
            B_padded,
            L_max,
            self._char_to_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._identity_idx,
        )

        return output, B_padded, L_max

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        """Run batch of variable-length strings, return list of accept/reject."""
        if not strings:
            return []

        B = len(strings)

        # Handle all-empty strings case
        L_max = max(len(s) for s in strings)
        if L_max == 0:
            is_accept = self.dm.check_accept(self.dm.start_vec)
            return [is_accept] * B

        input_data, B_padded, L_max = self._prepare_batch(strings)

        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.batched_engine_dispatch(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B,
            L_max,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"batched_engine_dispatch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        """Like simulate_batch but also returns (results, kernel_ms, total_ms)."""
        if not strings:
            return [], 0.0, 0.0

        B = len(strings)

        # Handle all-empty strings case
        L_max = max(len(s) for s in strings)
        if L_max == 0:
            is_accept = self.dm.check_accept(self.dm.start_vec)
            return [is_accept] * B, 0.0, 0.0

        input_data, B_padded, L_max = self._prepare_batch(strings)

        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.batched_engine_dispatch(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B,
            L_max,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"batched_engine_dispatch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class BatchedGPUSimulator:
    """Factory for BatchedEvolutionEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        # Set ctypes signatures
        self.lib.batched_engine_init.restype = ctypes.c_int
        self.lib.batched_engine_init.argtypes = [
            ctypes.c_int,                     # N
            ctypes.c_int,                     # sigma
            ctypes.POINTER(ctypes.c_int8),    # trans (sigma x N x N)
            ctypes.POINTER(ctypes.c_int8),    # accept (N)
            ctypes.c_int,                     # start_state
            ctypes.c_int,                     # max_B
            ctypes.c_int,                     # max_L
        ]

        self.lib.batched_engine_destroy.restype = None
        self.lib.batched_engine_destroy.argtypes = []

        self.lib.batched_engine_dispatch.restype = ctypes.c_int
        self.lib.batched_engine_dispatch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # input (L x B_padded)
            ctypes.c_int,                     # B
            ctypes.c_int,                     # L
            ctypes.POINTER(ctypes.c_int),     # results (B)
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        self.lib.batched_engine_device_check.restype = ctypes.c_int
        self.lib.batched_engine_device_check.argtypes = []

        self.lib.batched_prepare_input.restype = None
        self.lib.batched_prepare_input.argtypes = [
            ctypes.c_char_p,                  # strings_concat
            ctypes.POINTER(ctypes.c_int),     # offsets (B+1)
            ctypes.POINTER(ctypes.c_uint8),   # output (L x B_padded)
            ctypes.c_int,                     # B
            ctypes.c_int,                     # B_padded
            ctypes.c_int,                     # L
            ctypes.POINTER(ctypes.c_int),     # char_to_idx (256)
            ctypes.c_int,                     # identity_idx
        ]

        rc = self.lib.batched_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, dm: DFAMatrices,
                      max_B: int = 65536,
                      max_L: int = 4096) -> BatchedEvolutionEngine:
        return BatchedEvolutionEngine(self.lib, dm, max_B, max_L)
