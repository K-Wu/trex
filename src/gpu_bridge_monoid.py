"""
Python bridge to the monoid-based GPU DFA engine via ctypes.

The monoid engine performs DFA simulation by composing transition-monoid
elements rather than raw character transitions, which is much more
efficient for small monoids.

Usage:
    from src.gpu_bridge_monoid import MonoidGPUSimulator
    sim = MonoidGPUSimulator()
    engine = sim.create_engine(md, dm, max_total_chars=1<<20, max_batch=10000)
    results = engine.simulate_batch(["abb", "ab", "aabb"])
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
"""

from __future__ import annotations
import ctypes
import numpy as np
from pathlib import Path
from src.monoid import MonoidData
from src.simulation import DFAMatrices


def _find_lib():
    base = Path(__file__).parent.parent / "build" / "libmonoid_scan.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libmonoid_scan.so not found at {base}. Run 'make' first."
    )


class MonoidEngine:
    """Wraps a persistent GPU engine context for one monoid-based DFA."""

    def __init__(self, lib, md: MonoidData,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.md = md

        # Flatten compose_table to contiguous uint16 array
        compose_flat = np.ascontiguousarray(md.compose_table.reshape(-1), dtype=np.uint16)
        # Accept table as uint8 (C side uses uint8_t)
        accept = np.ascontiguousarray(md.accept_table.astype(np.uint8))

        rc = self.lib.monoid_engine_init(
            md.size,
            md.identity_idx,
            compose_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            max_total_chars,
            max_batch,
        )
        if rc != 0:
            raise RuntimeError(f"monoid_engine_init failed with code {rc}")

    def destroy(self):
        self.lib.monoid_engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        """Convert strings to CSR format using monoid indices.

        Unlike the raw DFA bridge, we map characters to monoid indices
        via md.char_to_monoid (not to alphabet indices). Unknown characters
        map to the identity element.
        """
        md = self.md
        B = len(strings)
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)
        total_chars = int(offsets[B])

        # chars holds monoid indices as uint16
        chars = np.zeros(total_chars, dtype=np.uint16)
        c2m = md.char_to_monoid
        identity = md.identity_idx
        pos = 0
        for s in strings:
            for ch in s:
                chars[pos] = c2m.get(ch, identity)
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

        rc = self.lib.monoid_engine_dispatch_batch(
            chars.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"monoid_engine_dispatch_batch failed with code {rc}")

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

        rc = self.lib.monoid_engine_dispatch_batch(
            chars.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"monoid_engine_dispatch_batch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class MonoidGPUSimulator:
    """Factory for MonoidEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.monoid_engine_init.restype = ctypes.c_int
        self.lib.monoid_engine_init.argtypes = [
            ctypes.c_int,                     # monoid_size
            ctypes.c_int,                     # identity_idx
            ctypes.POINTER(ctypes.c_uint16),  # compose_table (flattened)
            ctypes.POINTER(ctypes.c_uint8),   # accept_table
            ctypes.c_int,                     # max_total_chars
            ctypes.c_int,                     # max_batch
        ]

        self.lib.monoid_engine_destroy.restype = None
        self.lib.monoid_engine_destroy.argtypes = []

        self.lib.monoid_engine_dispatch_batch.restype = ctypes.c_int
        self.lib.monoid_engine_dispatch_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),  # chars (monoid indices, uint16)
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.c_int,                     # B
            ctypes.c_int,                     # total_chars
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        self.lib.monoid_engine_device_check.restype = ctypes.c_int
        self.lib.monoid_engine_device_check.argtypes = []

        rc = self.lib.monoid_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, md: MonoidData, dm: DFAMatrices,
                      max_total_chars: int = 1 << 22,
                      max_batch: int = 1 << 18) -> MonoidEngine:
        return MonoidEngine(self.lib, md, max_total_chars, max_batch)
