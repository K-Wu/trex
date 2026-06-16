"""
Python bridge to the CUDA tensor-core DFA scan kernel via ctypes.

Usage:
    from src.gpu_bridge import GPUSimulator
    sim = GPUSimulator()
    result = sim.simulate(dfa_matrices, "abb")
"""

from __future__ import annotations
import ctypes
import os
import numpy as np
from pathlib import Path
from src.simulation import DFAMatrices


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libdfa_scan.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libdfa_scan.so not found at {base}. Run 'make' first."
    )


class GPUSimulator:
    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.gpu_simulate_dfa.restype = ctypes.c_int
        self.lib.gpu_simulate_dfa.argtypes = [
            ctypes.c_int,                              # n_states
            ctypes.c_int,                              # alphabet_size
            ctypes.c_int,                              # start_state
            ctypes.POINTER(ctypes.c_int8),             # accept_mask[16]
            ctypes.POINTER(ctypes.c_int8),             # trans_matrices
            ctypes.POINTER(ctypes.c_int),              # input_chars[L]
            ctypes.c_int,                              # L
        ]

        self.lib.gpu_simulate_dfa_batch.restype = None
        self.lib.gpu_simulate_dfa_batch.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.lib.gpu_device_check.restype = ctypes.c_int
        rc = self.lib.gpu_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support int8 WMMA (needs SM >= 7.2)")

    def simulate(self, dm: DFAMatrices, input_str: str) -> bool:
        if not input_str:
            return dm.check_accept(dm.start_vec)

        char_indices = np.array(
            [dm.char_to_idx.get(ch, 0) for ch in input_str], dtype=np.int32
        )
        accept = dm.accept_mask.copy()
        trans = dm.matrix_stack.copy()
        trans_flat = np.ascontiguousarray(trans.reshape(-1))

        result = self.lib.gpu_simulate_dfa(
            dm.n_states_raw,
            len(dm.alphabet),
            dm.dfa.start,
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            trans_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            char_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(input_str),
        )
        return result != 0

    def simulate_batch(self, dm: DFAMatrices, strings: list[str]) -> list[bool]:
        if not strings:
            return []

        lengths = set(len(s) for s in strings)
        if len(lengths) != 1:
            return [self.simulate(dm, s) for s in strings]

        L = len(strings[0])
        if L == 0:
            accept = dm.check_accept(dm.start_vec)
            return [accept] * len(strings)

        batch_size = len(strings)
        all_chars = np.zeros(batch_size * L, dtype=np.int32)
        for b, s in enumerate(strings):
            for j, ch in enumerate(s):
                all_chars[b * L + j] = dm.char_to_idx.get(ch, 0)

        accept = dm.accept_mask.copy()
        trans_flat = np.ascontiguousarray(dm.matrix_stack.reshape(-1))
        results = np.zeros(batch_size, dtype=np.int32)

        self.lib.gpu_simulate_dfa_batch(
            dm.n_states_raw,
            len(dm.alphabet),
            dm.dfa.start,
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            trans_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            all_chars.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            batch_size,
            L,
        )
        return [bool(r) for r in results]
