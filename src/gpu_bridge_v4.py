"""
Python bridge to the v4 parallel DFA engine via ctypes.

Supports:
  - Variable-length string batches (CSR format)
  - Kernel-only and end-to-end timing
  - Adaptive dispatch (R1 warp-per-string / R3 decoupled look-back)

Usage:
    from src.gpu_bridge_v4 import ParallelGPUSimulator
    sim = ParallelGPUSimulator()
    engine = sim.create_engine(dm, max_total_chars=1<<20, max_batch=10000)
    results = engine.simulate_batch(dm, ["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(dm, strings)
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.simulation import DFAMatrices


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libparallel_engine.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libparallel_engine.so not found at {base}. Run 'make' first."
    )


class ParallelEngine:
    """Wraps a persistent GPU engine context for one DFA."""

    def __init__(self, lib, dm: DFAMatrices,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.dm = dm

        accept = dm.accept_mask.copy()
        trans_flat = np.ascontiguousarray(dm.matrix_stack.reshape(-1))

        rc = self.lib.engine_init(
            dm.n_states_raw,
            len(dm.alphabet),
            dm.dfa.start,
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            trans_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            max_total_chars,
            max_batch,
        )
        if rc != 0:
            raise RuntimeError(f"engine_init failed with code {rc}")

    def destroy(self):
        self.lib.engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        """Convert strings to CSR format (chars array + offsets array)."""
        B = len(strings)
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)
        total_chars = int(offsets[B])

        chars = np.zeros(total_chars, dtype=np.int32)
        c2i = self.dm.char_to_idx
        pos = 0
        for s in strings:
            for ch in s:
                chars[pos] = c2i.get(ch, 0)
                pos += 1

        return chars, offsets, total_chars

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        """Run batch of variable-length strings, return list of accept/reject."""
        if not strings:
            return []

        B = len(strings)
        chars, offsets, total_chars = self._prepare_batch(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.engine_dispatch_batch(
            chars.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"engine_dispatch_batch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        """Like simulate_batch but also returns (results, kernel_ms, total_ms)."""
        if not strings:
            return [], 0.0, 0.0

        B = len(strings)
        chars, offsets, total_chars = self._prepare_batch(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.engine_dispatch_batch(
            chars.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"engine_dispatch_batch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value

    def simulate_single(self, input_str: str) -> bool:
        """Convenience for single string."""
        if not input_str:
            return self.dm.check_accept(self.dm.start_vec)
        return self.simulate_batch([input_str])[0]


class ParallelGPUSimulator:
    """Factory for ParallelEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.engine_init.restype = ctypes.c_int
        self.lib.engine_init.argtypes = [
            ctypes.c_int,                   # n_states
            ctypes.c_int,                   # alphabet_size
            ctypes.c_int,                   # start_state
            ctypes.POINTER(ctypes.c_int8),  # accept_mask
            ctypes.POINTER(ctypes.c_int8),  # trans_matrices
            ctypes.c_int,                   # max_total_chars
            ctypes.c_int,                   # max_batch
        ]

        self.lib.engine_destroy.restype = None
        self.lib.engine_destroy.argtypes = []

        self.lib.engine_dispatch_batch.restype = ctypes.c_int
        self.lib.engine_dispatch_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # chars
            ctypes.POINTER(ctypes.c_int),   # offsets
            ctypes.POINTER(ctypes.c_int),   # results
            ctypes.c_int,                   # B
            ctypes.c_int,                   # total_chars
            ctypes.POINTER(ctypes.c_float), # kernel_ms
            ctypes.POINTER(ctypes.c_float), # total_ms
        ]

        self.lib.engine_dispatch_single.restype = ctypes.c_int
        self.lib.engine_dispatch_single.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # chars
            ctypes.c_int,                   # L
            ctypes.POINTER(ctypes.c_float), # kernel_ms
            ctypes.POINTER(ctypes.c_float), # total_ms
        ]

        self.lib.engine_device_check.restype = ctypes.c_int
        rc = self.lib.engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support int8 WMMA (needs SM >= 7.2)")

    def create_engine(self, dm: DFAMatrices,
                      max_total_chars: int = 1 << 22,
                      max_batch: int = 1 << 18) -> ParallelEngine:
        return ParallelEngine(self.lib, dm, max_total_chars, max_batch)
