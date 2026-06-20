# TREX

**T**ensor-core accelerated **R**egular **EX**pression engine for NVIDIA Hopper GPUs.

TREX formulates finite-automaton simulation as iterated matrix-vector multiply (`s' = T[c] * s`) and accelerates it with WMMA/wgmma tensor core instructions on H100/H200. It supports DFA (N=16) and NFA (N=64) with both Boolean and probabilistic state representations.

## Key results

| Kernel | Domain | Peak throughput |
|--------|--------|-----------------|
| Monoid batch (K-gram fused) | DFA, N≤16 | 1500-2500 Gc/s (HBM-bound) |
| Bit-parallel V4 | Boolean NFA, N=64 | 185 Gc/s |
| Batched WMMA V3 | DFA, N=16 | 116 Gc/s |
| TC-cached V3 | Boolean NFA, N=64 | 12.5 Gc/s |
| Probabilistic V5 | Real-valued NFA, N=64 | 178 TFLOPS |
| wgmma RS chain | NFA, N=64, latency | 82 cy/pos (2.6x over V3) |

## Project structure

```
cuda/           CUDA kernels (12 .cu files, see report below)
src/            Python engine, DFA/NFA construction, ctypes GPU bridges
tests/          Test suite
bench/          Benchmarking scripts
docs/           Design docs and reports
```

## Building

Requires CUDA 12+ with sm_90 (Hopper). Adjust `SM_ARCH` in the Makefile for other architectures.

```bash
make all        # build all kernels
make test       # run built-in CUDA tests
```

## Usage

```python
from src.optimized_engine import OptimizedEngine

engine = OptimizedEngine(pattern="ab*c", config="auto")
results = engine.match(strings)
```

## Kernel guide

See [docs/kernel_report.md](docs/kernel_report.md) for a comprehensive report covering how each kernel variant frames the problem, measured performance, and a decision tree for selecting the right kernel.
