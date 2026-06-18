"""
Python bridge to the prefix compose GPU engine via ctypes.

The prefix compose engine uses warp-shuffle-based function map composition
instead of matrix multiplication. Each DFA transition is an N-entry map,
composed via O(N) gathers instead of O(N^3) matmul.

Usage:
    from src.gpu_bridge_prefix_compose import PrefixComposeGPUSimulator
    sim = PrefixComposeGPUSimulator()
    engine = sim.create_engine(dm)
    results = engine.simulate_batch(["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    engine.destroy()
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.simulation import DFAMatrices, precompute_tmap


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libprefix_compose.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libprefix_compose.so not found at {base}. Run 'make' first."
    )


class PrefixComposeEngine:
    """Wraps a persistent GPU engine context for prefix compose dispatch."""

    def __init__(self, lib, dm: DFAMatrices,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.dm = dm

        tmap = np.ascontiguousarray(precompute_tmap(dm))
        accept = np.zeros(dm.n_states, dtype=np.uint8)
        for s in dm.dfa.accept_states:
            accept[s] = 1

        rc = self.lib.prefix_engine_init(
            tmap.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            dm.dfa.start,
            dm.n_states,
            32,  # K = BLOCK_K
            max_total_chars,
            max_batch,
        )
        if rc != 0:
            raise RuntimeError(f"prefix_engine_init failed with code {rc}")

    def destroy(self):
        self.lib.prefix_engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        B = len(strings)
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)
        total_chars = int(offsets[B])
        if total_chars > 0:
            raw_concat = np.frombuffer(
                "".join(strings).encode("latin-1"), dtype=np.uint8
            ).copy()
        else:
            raw_concat = np.zeros(1, dtype=np.uint8)
        return raw_concat, offsets, total_chars

    def simulate_batch(self, strings: list[str]) -> list[bool]:
        if not strings:
            return []

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        B = len(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.prefix_engine_dispatch(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"prefix_engine dispatch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        if not strings:
            return [], 0.0, 0.0

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        B = len(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.prefix_engine_dispatch(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"prefix_engine dispatch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class PrefixComposeGPUSimulator:
    """Factory for PrefixComposeEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.prefix_engine_device_check.restype = ctypes.c_int
        self.lib.prefix_engine_device_check.argtypes = []

        self.lib.prefix_engine_init.restype = ctypes.c_int
        self.lib.prefix_engine_init.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # tmap
            ctypes.POINTER(ctypes.c_uint8),   # accept
            ctypes.c_int,                     # start_state
            ctypes.c_int,                     # N
            ctypes.c_int,                     # K
            ctypes.c_int,                     # max_total_chars
            ctypes.c_int,                     # max_batch
        ]

        self.lib.prefix_engine_destroy.restype = None
        self.lib.prefix_engine_destroy.argtypes = []

        self.lib.prefix_engine_dispatch.restype = ctypes.c_int
        self.lib.prefix_engine_dispatch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # raw_concat
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.c_int,                     # B
            ctypes.c_int,                     # total_chars
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        rc = self.lib.prefix_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, dm: DFAMatrices,
                      max_total_chars: int = 1 << 22,
                      max_batch: int = 1 << 18) -> PrefixComposeEngine:
        return PrefixComposeEngine(self.lib, dm,
                                   max_total_chars, max_batch)
