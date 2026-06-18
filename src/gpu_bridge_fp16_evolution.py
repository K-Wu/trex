"""
Python bridge to the FP16 Tensor Core evolution GPU engine via ctypes.

The FP16 evolution engine converts DFA transition matrices to FP16,
uses Tensor Core MMA for state-vector evolution, and runs batched
string acceptance checks on the GPU.

Usage:
    from src.gpu_bridge_fp16_evolution import FP16EvolutionGPUSimulator
    sim = FP16EvolutionGPUSimulator()
    engine = sim.create_engine(dm)
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
    base = Path(__file__).parent.parent / "build" / "libfp16_evolution.so"
    if base.exists():
        return str(base)
    raise FileNotFoundError(
        f"libfp16_evolution.so not found at {base}. Run 'make' first."
    )


class FP16EvolutionEngine:
    """Wraps a persistent GPU engine context for FP16 TC evolution dispatch."""

    def __init__(self, lib, dm: DFAMatrices,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.dm = dm

        N = dm.n_states
        sigma = len(dm.alphabet)

        # Build T_matrices[sigma][N][N] as float32 from dm.matrices
        T_matrices = np.zeros((sigma, N, N), dtype=np.float32)
        for ch in dm.alphabet:
            c_idx = dm.char_to_idx[ch]
            T_matrices[c_idx] = dm.matrices[ch].astype(np.float32)
        T_matrices = np.ascontiguousarray(T_matrices)

        # Build accept_mask[N] as float32
        accept_mask = np.zeros(N, dtype=np.float32)
        for s in dm.dfa.accept_states:
            accept_mask[s] = 1.0
        accept_mask = np.ascontiguousarray(accept_mask)

        # Build start_vec[N] as float32 one-hot at dm.dfa.start
        start_vec = np.zeros(N, dtype=np.float32)
        start_vec[dm.dfa.start] = 1.0
        start_vec = np.ascontiguousarray(start_vec)

        # max_L: reasonable per-string length cap
        max_L = max(max_total_chars // max(max_batch, 1), 1)

        rc = self.lib.fp16_engine_init(
            T_matrices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            accept_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            start_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, sigma, max_batch, max_L,
        )
        if rc != 0:
            raise RuntimeError(f"fp16_engine_init failed with code {rc}")

    def destroy(self):
        self.lib.fp16_engine_destroy()

    def _prepare_batch(self, strings: list[str]):
        B = len(strings)
        offsets = np.zeros(B + 1, dtype=np.int32)
        for i, s in enumerate(strings):
            offsets[i + 1] = offsets[i] + len(s)
        total_chars = int(offsets[B])
        if total_chars > 0:
            # Map characters to char indices using dm.char_to_idx
            char_to_idx = self.dm.char_to_idx
            concat_str = "".join(strings)
            raw_concat = np.array(
                [char_to_idx[c] for c in concat_str],
                dtype=np.uint8,
            )
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

        rc = self.lib.fp16_engine_dispatch(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"fp16_engine dispatch failed with code {rc}")

        return [bool(r) for r in results]

    def simulate_batch_timed(self, strings: list[str]):
        if not strings:
            return [], 0.0, 0.0

        raw_concat, offsets, total_chars = self._prepare_batch(strings)
        B = len(strings)
        results = np.zeros(B, dtype=np.int32)
        kern_ms = ctypes.c_float(0)
        total_ms = ctypes.c_float(0)

        rc = self.lib.fp16_engine_dispatch(
            raw_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B, total_chars,
            ctypes.byref(kern_ms),
            ctypes.byref(total_ms),
        )
        if rc != 0:
            raise RuntimeError(f"fp16_engine dispatch failed with code {rc}")

        return [bool(r) for r in results], kern_ms.value, total_ms.value


class FP16EvolutionGPUSimulator:
    """Factory for FP16EvolutionEngine instances."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_lib()
        self.lib = ctypes.CDLL(path)

        self.lib.fp16_engine_device_check.restype = ctypes.c_int
        self.lib.fp16_engine_device_check.argtypes = []

        self.lib.fp16_engine_init.restype = ctypes.c_int
        self.lib.fp16_engine_init.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # T_matrices
            ctypes.POINTER(ctypes.c_float),   # accept_mask
            ctypes.POINTER(ctypes.c_float),   # start_vec
            ctypes.c_int,                     # N
            ctypes.c_int,                     # sigma
            ctypes.c_int,                     # max_B
            ctypes.c_int,                     # max_L
        ]

        self.lib.fp16_engine_destroy.restype = None
        self.lib.fp16_engine_destroy.argtypes = []

        self.lib.fp16_engine_set_variant.restype = ctypes.c_int
        self.lib.fp16_engine_set_variant.argtypes = [ctypes.c_int]

        self.lib.fp16_engine_dispatch.restype = ctypes.c_int
        self.lib.fp16_engine_dispatch.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # raw_concat
            ctypes.POINTER(ctypes.c_int),     # offsets
            ctypes.POINTER(ctypes.c_int),     # results
            ctypes.c_int,                     # B
            ctypes.c_int,                     # total_chars
            ctypes.POINTER(ctypes.c_float),   # kernel_ms
            ctypes.POINTER(ctypes.c_float),   # total_ms
        ]

        rc = self.lib.fp16_engine_device_check()
        if rc == -1:
            raise RuntimeError("No CUDA device found")
        if rc == -2:
            raise RuntimeError("GPU does not support SM >= 7.x")

    def create_engine(self, dm: DFAMatrices,
                      max_total_chars: int = 1 << 22,
                      max_batch: int = 1 << 18) -> FP16EvolutionEngine:
        return FP16EvolutionEngine(self.lib, dm,
                                   max_total_chars, max_batch)
