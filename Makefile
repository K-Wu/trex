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

SRC_V4 = $(CUDA_DIR)/parallel_dfa_engine.cu
EXE_V4 = $(BUILD_DIR)/parallel_engine
LIB_V4 = $(BUILD_DIR)/libparallel_engine.so

SRC_MONOID = $(CUDA_DIR)/monoid_scan.cu
EXE_MONOID = $(BUILD_DIR)/monoid_scan
LIB_MONOID = $(BUILD_DIR)/libmonoid_scan.so

SRC_PROFILE = $(CUDA_DIR)/profile_kernels.cu
EXE_PROFILE = $(BUILD_DIR)/profile_kernels

SRC_BATCHED = $(CUDA_DIR)/batched_evolution.cu
EXE_BATCHED = $(BUILD_DIR)/batched_evolution
LIB_BATCHED = $(BUILD_DIR)/libbatched_evolution.so

SRC_KGRAM = $(CUDA_DIR)/kgram_evolution.cu
EXE_KGRAM = $(BUILD_DIR)/kgram_evolution
LIB_KGRAM = $(BUILD_DIR)/libkgram_evolution.so

SRC_MONOID_BATCH = $(CUDA_DIR)/monoid_batch.cu
EXE_MONOID_BATCH = $(BUILD_DIR)/monoid_batch
LIB_MONOID_BATCH = $(BUILD_DIR)/libmonoid_batch.so

SRC_PREFIX = $(CUDA_DIR)/prefix_compose.cu
EXE_PREFIX = $(BUILD_DIR)/prefix_compose
LIB_PREFIX = $(BUILD_DIR)/libprefix_compose.so

SRC_FP16 = $(CUDA_DIR)/fp16_evolution.cu
EXE_FP16 = $(BUILD_DIR)/fp16_evolution
LIB_FP16 = $(BUILD_DIR)/libfp16_evolution.so

SRC_NFA_TC = $(CUDA_DIR)/nfa_tc_evolution.cu
EXE_NFA_TC = $(BUILD_DIR)/nfa_tc_evolution
LIB_NFA_TC = $(BUILD_DIR)/libnfa_tc_evolution.so

all: $(BUILD_DIR) $(EXE) $(LIB) $(EXE_V4) $(LIB_V4) $(EXE_MONOID) $(LIB_MONOID) $(EXE_BATCHED) $(LIB_BATCHED) $(EXE_KGRAM) $(LIB_KGRAM) $(EXE_MONOID_BATCH) $(LIB_MONOID_BATCH) $(EXE_PREFIX) $(LIB_PREFIX) $(EXE_FP16) $(LIB_FP16) $(EXE_NFA_TC) $(LIB_NFA_TC)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(EXE): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_V4): $(SRC_V4) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_V4): $(SRC_V4) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_MONOID): $(SRC_MONOID) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_MONOID): $(SRC_MONOID) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_BATCHED): $(SRC_BATCHED) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_BATCHED): $(SRC_BATCHED) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_KGRAM): $(SRC_KGRAM) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_KGRAM): $(SRC_KGRAM) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_MONOID_BATCH): $(SRC_MONOID_BATCH) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_MONOID_BATCH): $(SRC_MONOID_BATCH) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_PREFIX): $(SRC_PREFIX) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_PREFIX): $(SRC_PREFIX) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_FP16): $(SRC_FP16) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_FP16): $(SRC_FP16) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_NFA_TC): $(SRC_NFA_TC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_NFA_TC): $(SRC_NFA_TC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<

$(EXE_PROFILE): $(SRC_PROFILE) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -lineinfo -o $@ $<

profile: $(EXE_PROFILE)
	./$(EXE_PROFILE)

clean:
	rm -rf $(BUILD_DIR)

test-py:
	python -m pytest tests/ -v

test-gpu: $(EXE)
	./$(EXE)

test-v4: $(EXE_V4)
	./$(EXE_V4)

test-monoid: $(EXE_MONOID)
	./$(EXE_MONOID)

test-batched: $(EXE_BATCHED)
	./$(EXE_BATCHED)

test-kgram: $(EXE_KGRAM)
	./$(EXE_KGRAM)

test-monoid-batch: $(EXE_MONOID_BATCH)
	./$(EXE_MONOID_BATCH)

test-prefix: $(EXE_PREFIX)
	./$(EXE_PREFIX)

test-fp16: $(EXE_FP16)
	./$(EXE_FP16)

test-nfa-tc: $(EXE_NFA_TC)
	./$(EXE_NFA_TC)

test-all: test-gpu test-py

bench-cpu:
	python bench/benchmark.py

bench-gpu: $(LIB)
	python bench/benchmark_gpu.py

eval: $(LIB)
	python bench/visualize_gpu.py

bench-all: bench-cpu bench-gpu

.PHONY: all clean test-py test-gpu test-v4 test-monoid test-batched test-kgram test-monoid-batch test-prefix test-fp16 test-nfa-tc test-all bench-cpu bench-gpu eval bench-all profile
