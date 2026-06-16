# Makefile for tensor-core DFA scan
#
# Adjust SM_ARCH for your GPU:
#   sm_75  = Turing (RTX 2080, T4)
#   sm_80  = Ampere (A100)
#   sm_86  = Ampere (RTX 3090)
#   sm_89  = Ada Lovelace (RTX 4090, L4)
#   sm_90  = Hopper (H100, H200)

NVCC       = nvcc
SM_ARCH    ?= sm_90
NVCC_FLAGS = -O3 -arch=$(SM_ARCH) --use_fast_math -std=c++17

CUDA_DIR   = cuda
BUILD_DIR  = build

SRC    = $(CUDA_DIR)/tensor_core_dfa_scan.cu
EXE    = $(BUILD_DIR)/dfa_scan
LIB    = $(BUILD_DIR)/libdfa_scan.so

all: $(BUILD_DIR) $(EXE) $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(EXE): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

test-py:
	python -m pytest tests/ -v

test-gpu: $(EXE)
	./$(EXE)

test-all: test-gpu test-py

bench-cpu:
	python bench/benchmark.py

bench-gpu: $(LIB)
	python bench/benchmark_gpu.py

eval: $(LIB)
	python bench/visualize_gpu.py

bench-all: bench-cpu bench-gpu

.PHONY: all clean test-py test-gpu test-all bench-cpu bench-gpu eval bench-all
