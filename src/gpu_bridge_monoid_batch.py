"""
Python bridge to the monoid batch GPU engine via ctypes.

The monoid batch engine processes strings by sending raw bytes to the GPU,
where a fused compose-table lookup replaces O(N³) matrix multiplication
with O(1) shared-memory reads per character.

Usage:
    from src.gpu_bridge_monoid_batch import MonoidBatchGPUSimulator
    sim = MonoidBatchGPUSimulator()
    engine = sim.create_engine(md, dm)
    results = engine.simulate_batch(["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    engine.destroy()
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.monoid import MonoidData, precompute_batch_tables
from src.simulation import DFAMatrices


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libmonoid_batch.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libmonoid_batch.so not found at {base}. Run 'make' first."
    )


class MonoidBatchEngine:
    """Wraps a persistent GPU engine context for monoid batch dispatch."""

    def __init__(self, lib, md: MonoidData, dm: DFAMatrices,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.md = md

        tables = precompute_batch_tables(md, dm)
        self._M = tables['M']
        self._sigma_ext = tables['sigma_ext']
        self._identity = tables['identity_idx']

        char_compose = np.ascontiguousarray(tables['char_compose'])
        raw_char_map = np.ascontiguousarray(tables['raw_char_map'])
        accept = np.ascontiguousarray(tables['accept'])
        monoid_compose = np.ascontiguousarray(tables['monoid_compose'])

        rc = self.lib.monoid_batch_engine_init(
            self._M,
            self._sigma_ext,
            self._identity,
            char_compose.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            raw_char_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            monoid_compose.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            max_total_chars,
            max_batch,
        )
        if rc != 0:
            raise RuntimeError(f"monoid_batch_engine_init failed with code {rc}")

    def destroy(self):
        self.lib.monoid_batch_engine_destroy()

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

        B = len(strings)
        L_max = max((len(s) for s in strings), default=0)
        if L_max == 0:
            is_accept = bool(self.md.accept_table[self.md.identity_idx])
            return [is_accept] * B

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        use_prefix = (B <= 128 and L_max > 100_000)
        dispatch_fn = (self.lib.monoid_batch_engine_dispatch_prefix
                       if use_prefix
                       else self.lib.monoid_batch_engine_dispatch)

        rc = dispatch_fn(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"monoid_batch dispatch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        if not strings:
            return [], 0.0, 0.0

        B = len(strings)
        L_max = max((len(s) for s in strings), default=0)
        if L_max == 0:
            is_accept = bool(self.md.accept_table[self.md.identity_idx])
            return [is_accept] * B, 0.0, 0.0

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        use_prefix = (B <= 128 and L_max > 100_000)
        dispatch_fn = (self.lib.monoid_batch_engine_dispatch_prefix
                       if use_prefix
                       else self.lib.monoid_batch_engine_dispatch)

        rc = dispatch_fn(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"monoid_batch dispatch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class MonoidBatchGPUSimulator:
    """Factory for MonoidBatchEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.monoid_batch_engine_device_check.restype = ctypes.c_int
        self.lib.monoid_batch_engine_device_check.argtypes = []

        self.lib.monoid_batch_engine_init.restype = ctypes.c_int
        self.lib.monoid_batch_engine_init.argtypes = [
            ctypes.c_int,                     # M
            ctypes.c_int,                     # sigma_ext
            ctypes.c_int,                     # identity
            ctypes.POINTER(ctypes.c_uint8),   # char_compose
            ctypes.POINTER(ctypes.c_uint8),   # raw_char_map
            ctypes.POINTER(ctypes.c_uint8),   # accept
            ctypes.POINTER(ctypes.c_uint8),   # monoid_compose
            ctypes.c_int,                     # max_total_chars
            ctypes.c_int,                     # max_batch
        ]

        self.lib.monoid_batch_engine_destroy.restype = None
        self.lib.monoid_batch_engine_destroy.argtypes = []

        self.lib.monoid_batch_engine_dispatch.restype = ctypes.c_int
        self.lib.monoid_batch_engine_dispatch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # raw_concat
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.c_int,                     # B
            ctypes.c_int,                     # total_chars
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        self.lib.monoid_batch_engine_dispatch_prefix.restype = ctypes.c_int
        self.lib.monoid_batch_engine_dispatch_prefix.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # raw_concat
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.c_int,                     # B
            ctypes.c_int,                     # total_chars
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        rc = self.lib.monoid_batch_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, md: MonoidData, dm: DFAMatrices,
                      max_total_chars: int = 1 << 22,
                      max_batch: int = 1 << 18) -> MonoidBatchEngine:
        return MonoidBatchEngine(self.lib, md, dm,
                                max_total_chars, max_batch)
