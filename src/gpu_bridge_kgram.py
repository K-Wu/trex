"""
Python bridge to the CUDA k-gram TC evolution engine via ctypes.

The k-gram engine precomputes σ^k product matrices and processes k characters
per WMMA MMA call in single-string-per-warp mode.

Usage:
    from src.gpu_bridge_kgram import KGramGPUSimulator
    sim = KGramGPUSimulator()
    engine = sim.create_engine(dm, k=8)
    results = engine.simulate_batch(["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    engine.destroy()
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.simulation import DFAMatrices
from src.kgram import precompute_kgrams


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libkgram_evolution.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libkgram_evolution.so not found at {base}. Run 'make' first."
    )


STRINGS_PER_BLOCK = 4


class KGramGPUEngine:
    """Wraps a persistent GPU engine context for k-gram TC evolution."""

    def __init__(self, lib, dm: DFAMatrices, k: int,
                 max_B: int = 65536, max_L: int = 4096):
        self.lib = lib
        self.dm = dm
        self.k = k
        self.N = dm.n_states
        self.sigma = len(dm.alphabet)

        kg = precompute_kgrams(dm, k, monoid=None)
        n_entries = self.sigma ** k

        T_kgram = np.zeros((n_entries, self.N, self.N), dtype=np.int8)
        for key, mat in kg._matrix_table.items():
            T_kgram[key] = mat
        T_kgram = np.ascontiguousarray(T_kgram)

        T_base = np.ascontiguousarray(dm.matrix_stack, dtype=np.int8)

        accept = np.ascontiguousarray(dm.accept_mask, dtype=np.int8)
        start = np.zeros(self.N, dtype=np.int8)
        start[dm.dfa.start] = 1

        rc = self.lib.kgram_engine_init(
            T_kgram.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            T_base.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            start.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self.N, self.sigma, k, n_entries,
            max_B, max_L,
        )
        if rc != 0:
            raise RuntimeError(f"kgram_engine_init failed with code {rc}")

        self._identity_idx = self.sigma
        self._char_to_idx = np.full(256, -1, dtype=np.int32)
        for ch, idx in dm.char_to_idx.items():
            self._char_to_idx[ord(ch)] = idx

    def destroy(self):
        self.lib.kgram_engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        B = len(strings)
        L_max = max(len(s) for s in strings) if strings else 0

        B_padded = ((B + STRINGS_PER_BLOCK - 1) // STRINGS_PER_BLOCK) * STRINGS_PER_BLOCK

        strings_concat = "".join(strings).encode("latin-1")
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)

        output = np.zeros(L_max * B_padded, dtype=np.uint8)

        self.lib.kgram_prepare_input(
            strings_concat,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, B_padded, L_max,
            self._char_to_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._identity_idx,
        )

        return output, B_padded, L_max

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        if not strings:
            return []

        B = len(strings)
        L_max = max(len(s) for s in strings)
        if L_max == 0:
            is_accept = self.dm.check_accept(self.dm.start_vec)
            return [is_accept] * B

        input_data, B_padded, L_max = self._prepare_batch(strings)

        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.kgram_engine_dispatch(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, L_max,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"kgram_engine_dispatch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        if not strings:
            return [], 0.0, 0.0

        B = len(strings)
        L_max = max(len(s) for s in strings)
        if L_max == 0:
            is_accept = self.dm.check_accept(self.dm.start_vec)
            return [is_accept] * B, 0.0, 0.0

        input_data, B_padded, L_max = self._prepare_batch(strings)

        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.kgram_engine_dispatch(
            input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B, L_max,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"kgram_engine_dispatch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class KGramGPUSimulator:
    """Factory for KGramGPUEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.kgram_engine_device_check.restype = ctypes.c_int
        self.lib.kgram_engine_device_check.argtypes = []

        self.lib.kgram_engine_init.restype = ctypes.c_int
        self.lib.kgram_engine_init.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]

        self.lib.kgram_engine_destroy.restype = None
        self.lib.kgram_engine_destroy.argtypes = []

        self.lib.kgram_engine_dispatch.restype = ctypes.c_int
        self.lib.kgram_engine_dispatch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        self.lib.kgram_prepare_input.restype = None
        self.lib.kgram_prepare_input.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]

        rc = self.lib.kgram_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, dm: DFAMatrices, k: int,
                      max_B: int = 65536,
                      max_L: int = 4096) -> KGramGPUEngine:
        return KGramGPUEngine(self.lib, dm, k, max_B, max_L)
