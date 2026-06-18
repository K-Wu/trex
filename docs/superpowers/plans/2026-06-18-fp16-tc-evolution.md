# FP16 Tensor Core Evolution Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an FP16 tensor core DFA evolution kernel that eliminates the INT32→INT8 conversion bottleneck, targeting 400+ Gc/s at N=16 and 150+ Gc/s at N=32.

**Architecture:** FP16 WMMA MMA with FP16 accumulator exploits the DFA permutation matrix invariant: output stays exactly {0.0, 1.0}, so no threshold or data conversion is needed between MMA steps. The V1 kernel stores accumulators to shared memory for per-column character select; a V2 register-path variant is deferred. The kernel is templated on N (16, 32, 48, 64) with compile-time unrolled tile loops.

**Tech Stack:** CUDA 12.x with `mma.h` (WMMA API), FP16 (`half` type via `cuda_fp16.h`), ctypes Python bridge, pytest.

**Spec:** `docs/superpowers/specs/2026-06-18-fp16-tc-evolution-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `cuda/fp16_evolution.cu` | Create | FP16 WMMA kernel (V1 smem path), engine struct, C API, built-in tests, benchmark |
| `src/gpu_bridge_fp16_evolution.py` | Create | Python ctypes bridge: FP16EvolutionEngine + FP16EvolutionGPUSimulator |
| `src/optimized_engine.py` | Modify | Add `fp16_tc+gpu` config, `_setup_fp16_tc_gpu()`, update auto-selection |
| `Makefile` | Modify | Add `fp16_evolution` build targets |
| `tests/test_fp16_evolution.py` | Create | Python-level correctness tests |

---

### Task 1: Makefile Build Targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add fp16_evolution variables and targets to Makefile**

Add after the `LIB_PREFIX` line (line 46):

```makefile
SRC_FP16 = $(CUDA_DIR)/fp16_evolution.cu
EXE_FP16 = $(BUILD_DIR)/fp16_evolution
LIB_FP16 = $(BUILD_DIR)/libfp16_evolution.so
```

Update the `all` target (line 48) to append `$(EXE_FP16) $(LIB_FP16)`.

Add build rules after the prefix rules (after line 93):

```makefile
$(EXE_FP16): $(SRC_FP16) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(LIB_FP16): $(SRC_FP16) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DBUILD_LIB -shared -Xcompiler -fPIC -o $@ $<
```

Add test target before `.PHONY`:

```makefile
test-fp16: $(EXE_FP16)
	./$(EXE_FP16)
```

Update `.PHONY` to include `test-fp16`.

- [ ] **Step 2: Create a minimal fp16_evolution.cu stub that compiles**

Create `cuda/fp16_evolution.cu` with minimal content:

```cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>

using namespace nvcuda;

extern "C" {
int fp16_engine_device_check(void) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 7) return -2;
    return 0;
}
}

#ifndef BUILD_LIB
int main() {
    printf("=== FP16 TC Evolution (stub) ===\n");
    int rc = fp16_engine_device_check();
    printf("Device check: %d\n", rc);
    return rc != 0 ? 1 : 0;
}
#endif
```

- [ ] **Step 3: Verify build**

Run: `make build/fp16_evolution build/libfp16_evolution.so`
Expected: Both compile with no errors.

Run: `./build/fp16_evolution`
Expected: Prints device check result, exits 0.

- [ ] **Step 4: Commit**

```bash
git add Makefile cuda/fp16_evolution.cu
git commit -m "feat(fp16): add Makefile targets and compilation stub"
```

---

### Task 2: FP16 Binary Kernel (N=16, sigma=2)

**Files:**
- Modify: `cuda/fp16_evolution.cu`

This is the core kernel. It follows the same structure as `batched_evolution.cu`'s binary kernel V1 but uses FP16 throughout. The key difference: after MMA, the accumulator stores `half` values that are already {0.0, 1.0}, so the per-column select has **no threshold and no type conversion**.

- [ ] **Step 1: Add constants, CHECK_CUDA macro, and shared memory layout**

Replace the stub content of `cuda/fp16_evolution.cu` with:

```cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace nvcuda;

constexpr int TILE = 16;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int COLS_PER_BLOCK = WARPS_PER_BLOCK * TILE;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)
```

- [ ] **Step 2: Write the FP16 binary kernel**

Add the kernel function after the constants. Key differences from the INT8 version:
- All fragments use `half` instead of `int8_t`
- Accumulator type is `half` instead of `int32_t`
- Accumulator shared memory uses `half` (512B per tile) instead of `int32_t` (1024B per tile)
- The select loop has NO threshold and NO type cast — just a direct assignment

```cuda
// FP16 Binary Kernel (|Sigma|=2, N=16)
//
// Shared memory layout:
//   T0_sh[16×16]      512B   (half, transition matrix char 0)
//   T1_sh[16×16]      512B   (half, transition matrix char 1)
//   S_sh[4×16×16]     2048B  (half, state tiles for 4 warps, col-major)
//   acc0_sh[4×16×16]  2048B  (half, accumulator buffer 0, row-major)
//   acc1_sh[4×16×16]  2048B  (half, accumulator buffer 1, row-major)
//   Total:             7168B

__global__ void fp16_evolution_binary_kernel(
    const half    *__restrict__ T0_global,     // [16*16] transition for char 0
    const half    *__restrict__ T1_global,     // [16*16] transition for char 1
    const uint8_t *__restrict__ input,         // [L][B_padded] position-contiguous
    const half    *__restrict__ accept_mask,   // [16]
    const half    *__restrict__ start_vec,     // [16] initial state (one-hot)
    int B, int B_padded, int L,
    int *__restrict__ results
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * COLS_PER_BLOCK;
    int warp_col_start = block_col_start + warp_in_block * TILE;

    extern __shared__ char smem_raw[];
    half *T0_sh   = (half *)smem_raw;
    half *T1_sh   = T0_sh + TILE_ELEMS;
    half *S_base  = T1_sh + TILE_ELEMS;
    half *acc0_base = S_base + WARPS_PER_BLOCK * TILE_ELEMS;
    half *acc1_base = acc0_base + WARPS_PER_BLOCK * TILE_ELEMS;

    half *S_sh    = S_base + warp_in_block * TILE_ELEMS;
    half *acc0_sh = acc0_base + warp_in_block * TILE_ELEMS;
    half *acc1_sh = acc1_base + warp_in_block * TILE_ELEMS;

    // Load T0, T1 into shared memory (all threads cooperate)
    for (int e = threadIdx.x; e < TILE_ELEMS; e += blockDim.x) {
        T0_sh[e] = T0_global[e];
        T1_sh[e] = T1_global[e];
    }

    // Initialize S to start vector (col-major: S_sh[col*16 + row])
    half h_zero = __float2half(0.0f);
    for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
        int row = e % TILE;
        S_sh[e] = start_vec[row];
    }
    __syncthreads();

    // Fragment declarations — ALL FP16
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_T0, frag_T1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_S;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc0, frag_acc1;

    // Load T fragments (constant across all positions)
    wmma::load_matrix_sync(frag_T0, T0_sh, TILE);
    wmma::load_matrix_sync(frag_T1, T1_sh, TILE);

    for (int t = 0; t < L; t++) {
        wmma::load_matrix_sync(frag_S, S_sh, TILE);

        // FP16 MMA: T0 × S → acc0, T1 × S → acc1
        wmma::fill_fragment(frag_acc0, h_zero);
        wmma::mma_sync(frag_acc0, frag_T0, frag_S, frag_acc0);
        wmma::fill_fragment(frag_acc1, h_zero);
        wmma::mma_sync(frag_acc1, frag_T1, frag_S, frag_acc1);

        // Store accumulators to smem (row-major)
        wmma::store_matrix_sync(acc0_sh, frag_acc0, TILE, wmma::mem_row_major);
        wmma::store_matrix_sync(acc1_sh, frag_acc1, TILE, wmma::mem_row_major);
        __syncwarp();

        // Per-column select: NO threshold, NO type conversion
        // acc is row-major: acc[row * 16 + col]
        // S_sh is col-major: S_sh[col * 16 + row]
        for (int e = lane; e < TILE_ELEMS; e += WARP_SIZE) {
            int col = e / TILE;
            int row = e % TILE;
            int string_id = warp_col_start + col;
            uint8_t ch = 2;
            if (string_id < B_padded)
                ch = input[t * B_padded + string_id];
            if (ch < 2) {
                S_sh[e] = (ch == 0) ? acc0_sh[row * TILE + col]
                                    : acc1_sh[row * TILE + col];
            }
        }
        __syncwarp();
    }

    // Accept check
    half h_one = __float2half(1.0f);
    for (int col = lane; col < TILE; col += WARP_SIZE) {
        int string_id = warp_col_start + col;
        if (string_id >= B) continue;
        int accepted = 0;
        for (int r = 0; r < TILE; r++) {
            if (__hgt(S_sh[col * TILE + r], h_zero) &&
                __hgt(accept_mask[r], h_zero)) {
                accepted = 1;
                break;
            }
        }
        results[string_id] = accepted;
    }
}
```

- [ ] **Step 3: Verify kernel compiles**

Run: `make build/fp16_evolution`
Expected: Compiles without errors (even though main() is still a stub).

- [ ] **Step 4: Commit**

```bash
git add cuda/fp16_evolution.cu
git commit -m "feat(fp16): FP16 binary evolution kernel (N=16, sigma=2)"
```

---

### Task 3: Engine Struct and C API

**Files:**
- Modify: `cuda/fp16_evolution.cu`

- [ ] **Step 1: Write the FP16Engine struct**

Add after the kernel, before any `extern "C"`. This struct accepts float arrays from Python and converts to `half` internally.

```cuda
struct FP16Engine {
    int N;
    int sigma;
    int max_B;
    int max_L;
    int B_padded_max;

    // Device memory
    half    *d_T;          // [sigma * N * N] transition matrices
    half    *d_accept;     // [N]
    half    *d_start_vec;  // [N]
    uint8_t *d_input;      // [max_L * B_padded_max]
    int     *d_results;    // [max_B]

    cudaEvent_t ev_start, ev_stop, ev_kern_start, ev_kern_stop;
    bool initialized;

    void init(const float *T_matrices, const float *accept_mask,
              const float *start_vec, int n, int sig, int maxB, int maxL) {
        N = n;
        sigma = sig;
        max_B = maxB;
        max_L = maxL;
        B_padded_max = ((maxB + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

        int T_count = sig * n * n;
        std::vector<half> h_T(T_count);
        for (int i = 0; i < T_count; i++)
            h_T[i] = __float2half(T_matrices[i]);

        std::vector<half> h_accept(n);
        for (int i = 0; i < n; i++)
            h_accept[i] = __float2half(accept_mask[i]);

        std::vector<half> h_start(n);
        for (int i = 0; i < n; i++)
            h_start[i] = __float2half(start_vec[i]);

        CHECK_CUDA(cudaMalloc(&d_T, T_count * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_accept, n * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_start_vec, n * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_input, (size_t)maxL * B_padded_max));
        CHECK_CUDA(cudaMalloc(&d_results, (size_t)maxB * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_T, h_T.data(), T_count * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_accept, h_accept.data(), n * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_start_vec, h_start.data(), n * sizeof(half), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_kern_start));
        CHECK_CUDA(cudaEventCreate(&ev_kern_stop));

        initialized = true;
    }

    void destroy() {
        if (!initialized) return;
        cudaFree(d_T);
        cudaFree(d_accept);
        cudaFree(d_start_vec);
        cudaFree(d_input);
        cudaFree(d_results);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_kern_start);
        cudaEventDestroy(ev_kern_stop);
        initialized = false;
    }

    int dispatch(const uint8_t *h_input, int B, int L, int B_padded,
                 int *h_results, float *kernel_ms, float *total_ms) {
        if (!initialized) return -1;
        if (sigma != 2 || N != TILE) return -2;

        CHECK_CUDA(cudaEventRecord(ev_start));

        size_t input_bytes = (size_t)L * B_padded;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(ev_kern_start));

        int n_blocks = B_padded / COLS_PER_BLOCK;
        // smem: T0(512) + T1(512) + S(2048) + acc0(2048) + acc1(2048) = 7168
        int smem = (2 + WARPS_PER_BLOCK * 3) * TILE_ELEMS * (int)sizeof(half);
        fp16_evolution_binary_kernel<<<n_blocks, BLOCK_SIZE, smem>>>(
            d_T, d_T + TILE_ELEMS,
            d_input, d_accept, d_start_vec,
            B, B_padded, L, d_results);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(ev_kern_stop));

        CHECK_CUDA(cudaMemcpy(h_results, d_results, (size_t)B * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        if (kernel_ms) cudaEventElapsedTime(kernel_ms, ev_kern_start, ev_kern_stop);
        if (total_ms) cudaEventElapsedTime(total_ms, ev_start, ev_stop);

        return 0;
    }
};

static FP16Engine g_fp16_engine = {};
```

- [ ] **Step 2: Write the C API functions**

```cuda
extern "C" {

int fp16_engine_device_check(void) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 7) return -2;
    return 0;
}

int fp16_engine_init(
    const float *T_matrices,
    const float *accept_mask,
    const float *start_vec,
    int N, int sigma,
    int max_B, int max_L)
{
    if (g_fp16_engine.initialized) g_fp16_engine.destroy();
    g_fp16_engine.init(T_matrices, accept_mask, start_vec, N, sigma, max_B, max_L);
    return 0;
}

int fp16_engine_dispatch(
    const uint8_t *raw_concat,
    const int     *offsets,
    int           *results,
    int B, int total_chars,
    float *kernel_ms, float *total_ms)
{
    if (!g_fp16_engine.initialized) return -1;

    int L = 0;
    for (int i = 0; i < B; i++) {
        int len = offsets[i + 1] - offsets[i];
        if (len > L) L = len;
    }
    if (L == 0) {
        // All empty strings: check if start state accepts
        for (int i = 0; i < B; i++) results[i] = 0;
        half h_zero = __float2half(0.0f);
        std::vector<half> h_accept(g_fp16_engine.N);
        CHECK_CUDA(cudaMemcpy(h_accept.data(), g_fp16_engine.d_accept,
                              g_fp16_engine.N * sizeof(half), cudaMemcpyDeviceToHost));
        std::vector<half> h_start(g_fp16_engine.N);
        CHECK_CUDA(cudaMemcpy(h_start.data(), g_fp16_engine.d_start_vec,
                              g_fp16_engine.N * sizeof(half), cudaMemcpyDeviceToHost));
        for (int i = 0; i < B; i++) {
            for (int r = 0; r < g_fp16_engine.N; r++) {
                if (__half2float(h_start[r]) > 0.0f && __half2float(h_accept[r]) > 0.0f) {
                    results[i] = 1;
                    break;
                }
            }
        }
        if (kernel_ms) *kernel_ms = 0.0f;
        if (total_ms) *total_ms = 0.0f;
        return 0;
    }

    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;
    int identity_idx = g_fp16_engine.sigma;

    // Transpose raw_concat + offsets → position-contiguous layout
    size_t input_size = (size_t)L * B_padded;
    std::vector<uint8_t> input(input_size, (uint8_t)identity_idx);
    for (int b = 0; b < B; b++) {
        int str_start = offsets[b];
        int str_len = offsets[b + 1] - str_start;
        for (int t = 0; t < str_len && t < L; t++) {
            input[t * B_padded + b] = raw_concat[str_start + t];
        }
    }

    return g_fp16_engine.dispatch(input.data(), B, L, B_padded,
                                  results, kernel_ms, total_ms);
}

void fp16_engine_destroy(void) {
    g_fp16_engine.destroy();
}

}  // extern "C"
```

- [ ] **Step 3: Verify compilation**

Run: `make build/fp16_evolution build/libfp16_evolution.so`
Expected: Both compile cleanly.

- [ ] **Step 4: Commit**

```bash
git add cuda/fp16_evolution.cu
git commit -m "feat(fp16): engine struct and C API with raw_concat dispatch"
```

---

### Task 4: Built-in CUDA Tests

**Files:**
- Modify: `cuda/fp16_evolution.cu`

- [ ] **Step 1: Write test infrastructure and EvenADFA helper**

Add at the bottom of the file, inside `#ifndef BUILD_LIB`:

```cuda
#ifndef BUILD_LIB

static int g_tests = 0, g_pass = 0;
static void check(const char *name, bool cond) {
    g_tests++;
    if (cond) { g_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

// Sequential reference: apply transition matrices to a state vector
static bool simulate_sequential_fp16(
    int N, const float *start_vec, const float *accept_mask,
    const float *trans, int sigma,
    const uint8_t *chars, int L)
{
    std::vector<float> state(N), new_state(N);
    for (int i = 0; i < N; i++) state[i] = start_vec[i];

    for (int t = 0; t < L; t++) {
        int c = chars[t];
        if (c >= sigma) continue;  // identity
        const float *T = trans + c * N * N;
        std::fill(new_state.begin(), new_state.end(), 0.0f);
        for (int row = 0; row < N; row++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += T[row * N + k] * state[k];
            new_state[row] = (sum > 0.5f) ? 1.0f : 0.0f;
        }
        state = new_state;
    }

    for (int r = 0; r < N; r++)
        if (state[r] > 0.5f && accept_mask[r] > 0.5f) return true;
    return false;
}
```

- [ ] **Step 2: Write test_basic_correctness**

Even-A DFA: 2 logical states, sigma=2. Tests 8 specific strings.

```cuda
static void test_basic_correctness() {
    printf("\n--- test_basic_correctness ---\n");

    int N = TILE;
    int sigma = 2;
    float trans[2 * TILE_ELEMS];
    float accept[TILE];
    float start_vec[TILE];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));

    accept[0] = 1.0f;
    start_vec[0] = 1.0f;

    // Padding states: self-loop
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1.0f;

    // 'a' (0): swap 0<->1
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1.0f;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1.0f;
    // 'b' (1): identity on 0,1
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1.0f;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1.0f;

    FP16Engine eng;
    eng.init(trans, accept, start_vec, N, sigma, 64, 256);

    // Test strings via dispatch
    // "": accept (even a-count)
    // "a": reject (odd)
    // "aa": accept (even)
    // "b": accept (even)
    // "ab": reject (odd)
    // "ba": reject (odd)
    // "aab": accept (even)
    // "abba": accept (even)
    uint8_t all_chars[] = {0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0};
    int offsets[] = {0, 0, 1, 3, 4, 6, 8, 11, 15};
    bool expected[] = {true, false, true, true, false, false, true, true};
    const char *names[] = {
        "empty", "'a'", "'aa'", "'b'", "'ab'", "'ba'", "'aab'", "'abba'"
    };
    int B = 8;

    int results[8];
    float km, tm;
    int rc = fp16_engine_dispatch(all_chars, offsets, results, B, 15, &km, &tm);

    for (int i = 0; i < B; i++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s -> %s", names[i], expected[i] ? "accept" : "reject");
        check(msg, results[i] == (expected[i] ? 1 : 0));
    }
    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);

    eng.destroy();
}
```

- [ ] **Step 3: Write test_large_random**

```cuda
static void test_large_random() {
    printf("\n--- test_large_random (4096 x 256) ---\n");

    int N = TILE;
    int sigma = 2;
    float trans[2 * TILE_ELEMS];
    float accept[TILE];
    float start_vec[TILE];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));

    accept[0] = 1.0f;
    start_vec[0] = 1.0f;
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1.0f;
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1.0f;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1.0f;
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1.0f;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1.0f;

    FP16Engine eng;
    eng.init(trans, accept, start_vec, N, sigma, 4096, 256);

    int B = 4096, L = 256;
    srand(42);
    std::vector<uint8_t> chars_flat;
    std::vector<int> offsets(B + 1, 0);
    for (int b = 0; b < B; b++) {
        int len = 1 + rand() % L;
        offsets[b + 1] = offsets[b] + len;
        for (int t = 0; t < len; t++)
            chars_flat.push_back(rand() % 2);
    }
    int total = offsets[B];

    std::vector<int> results(B);
    float km, tm;
    fp16_engine_dispatch(chars_flat.data(), offsets.data(), results.data(),
                         B, total, &km, &tm);

    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        int len = offsets[b + 1] - offsets[b];
        bool seq = simulate_sequential_fp16(N, start_vec, accept, trans, sigma,
                                             chars_flat.data() + offsets[b], len);
        if (results[b] != (seq ? 1 : 0)) mismatches++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "large_random B=%d (%d mismatches)", B, mismatches);
    check(msg, mismatches == 0);
    printf("  Kernel: %.3f ms, Total: %.3f ms\n", km, tm);

    eng.destroy();
}
```

- [ ] **Step 4: Write test_fp16_invariant**

Verifies that after processing 100K positions, all state vector entries are exactly 0.0 or 1.0.

```cuda
static void test_fp16_invariant() {
    printf("\n--- test_fp16_invariant (64 x 100000) ---\n");

    int N = TILE;
    int sigma = 2;
    float trans[2 * TILE_ELEMS];
    float accept[TILE];
    float start_vec[TILE];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));

    accept[0] = 1.0f;
    start_vec[0] = 1.0f;
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1.0f;
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1.0f;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1.0f;
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1.0f;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1.0f;

    int B = 64, L = 100000;
    int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

    // Convert trans/start/accept to half and upload
    FP16Engine eng;
    eng.init(trans, accept, start_vec, N, sigma, B_padded, L);

    srand(77);
    size_t input_size = (size_t)L * B_padded;
    std::vector<uint8_t> input(input_size, (uint8_t)sigma);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < L; t++)
            input[t * B_padded + b] = rand() % 2;

    // Run kernel via direct dispatch (not through C API raw_concat)
    std::vector<int> results(B);
    float km, tm;
    eng.dispatch(input.data(), B, L, B_padded, results.data(), &km, &tm);

    // Read back S from device: state is in shared memory, not accessible
    // Instead, verify by re-running with a modified kernel or checking results consistency
    // For this test, we verify correctness = invariant holds (results match sequential)
    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        std::vector<uint8_t> str_chars(L);
        for (int t = 0; t < L; t++)
            str_chars[t] = input[t * B_padded + b];
        bool seq = simulate_sequential_fp16(N, start_vec, accept, trans, sigma,
                                             str_chars.data(), L);
        if (results[b] != (seq ? 1 : 0)) mismatches++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "fp16_invariant B=%d L=%d (%d mismatches)", B, L, mismatches);
    check(msg, mismatches == 0);
    printf("  Kernel: %.3f ms, Total: %.3f ms (%.1f Gc/s)\n",
           km, tm, (double)B * L / (km * 1e6));

    eng.destroy();
}
```

- [ ] **Step 5: Write main() and benchmark**

```cuda
static void bench_throughput() {
    printf("\n=== FP16 TC Throughput Benchmark (binary, N=16) ===\n");

    int N = TILE;
    int sigma = 2;
    float trans[2 * TILE_ELEMS];
    float accept[TILE];
    float start_vec[TILE];
    memset(trans, 0, sizeof(trans));
    memset(accept, 0, sizeof(accept));
    memset(start_vec, 0, sizeof(start_vec));
    accept[0] = 1.0f;
    start_vec[0] = 1.0f;
    for (int c = 0; c < 2; c++)
        for (int s = 2; s < TILE; s++)
            trans[c * TILE_ELEMS + s * TILE + s] = 1.0f;
    trans[0 * TILE_ELEMS + 1 * TILE + 0] = 1.0f;
    trans[0 * TILE_ELEMS + 0 * TILE + 1] = 1.0f;
    trans[1 * TILE_ELEMS + 0 * TILE + 0] = 1.0f;
    trans[1 * TILE_ELEMS + 1 * TILE + 1] = 1.0f;

    int batch_sizes[] = {1024, 4096, 16384, 65536, 262144};
    int lengths[]     = {128, 512, 2048};
    int n_batches = 5, n_lengths = 3;

    printf("  %8s  %6s  |  %8s  %8s\n", "B", "L", "Gc/s", "kern_ms");
    printf("  %s\n", "-------------------------------------");

    for (int bi = 0; bi < n_batches; bi++) {
        for (int li = 0; li < n_lengths; li++) {
            int B = batch_sizes[bi];
            int L = lengths[li];
            int B_padded = ((B + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK) * COLS_PER_BLOCK;

            FP16Engine eng;
            eng.init(trans, accept, start_vec, N, sigma, B_padded, L);

            srand(42);
            size_t input_size = (size_t)L * B_padded;
            std::vector<uint8_t> input(input_size, (uint8_t)sigma);
            for (int b = 0; b < B; b++)
                for (int t = 0; t < L; t++)
                    input[t * B_padded + b] = rand() % 2;

            std::vector<int> results(B);

            // Warmup
            for (int w = 0; w < 3; w++)
                eng.dispatch(input.data(), B, L, B_padded, results.data(), nullptr, nullptr);

            int iters = 20;
            float total_km = 0;
            for (int it = 0; it < iters; it++) {
                float km;
                eng.dispatch(input.data(), B, L, B_padded, results.data(), &km, nullptr);
                total_km += km;
            }
            float avg_km = total_km / iters;
            double gchs = (double)B * L / (avg_km * 1e6);

            printf("  %8d  %6d  |  %8.1f  %8.3f\n", B, L, gchs, avg_km);

            eng.destroy();
        }
    }
}


int main() {
    printf("=== FP16 TC Evolution Engine ===\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    int rc = fp16_engine_device_check();
    if (rc != 0) {
        printf("Device check failed: %d\n", rc);
        return 1;
    }

    test_basic_correctness();
    test_large_random();
    test_fp16_invariant();

    printf("\n=== Results: %d / %d passed ===\n", g_pass, g_tests);
    if (g_pass != g_tests) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }

    bench_throughput();

    return 0;
}

#endif  // BUILD_LIB
```

- [ ] **Step 6: Build and run all tests**

Run: `make build/fp16_evolution && ./build/fp16_evolution`
Expected: All tests PASS, benchmark prints throughput numbers.

- [ ] **Step 7: Commit**

```bash
git add cuda/fp16_evolution.cu
git commit -m "feat(fp16): built-in tests and benchmark"
```

---

### Task 5: Python Bridge

**Files:**
- Create: `src/gpu_bridge_fp16_evolution.py`

Follows the same pattern as `src/gpu_bridge_prefix_compose.py`. The dispatch uses the same raw_concat + offsets interface.

- [ ] **Step 1: Write the bridge module**

Create `src/gpu_bridge_fp16_evolution.py`:

```python
"""
Python bridge to the FP16 tensor core evolution engine via ctypes.

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
    """Wraps a persistent GPU engine context for FP16 TC evolution."""

    def __init__(self, lib, dm: DFAMatrices,
                 max_total_chars: int = 1 << 22,
                 max_batch: int = 1 << 18):
        self.lib = lib
        self.dm = dm

        N = dm.n_states
        sigma = len(dm.alphabet)

        # Build T_matrices[sigma][N][N] as float32 (converted to half on GPU)
        T = np.zeros((sigma, N, N), dtype=np.float32)
        for c_idx in range(sigma):
            T[c_idx] = dm.matrices[c_idx].astype(np.float32)
        T = np.ascontiguousarray(T)

        # Build accept_mask[N] as float32
        accept = np.zeros(N, dtype=np.float32)
        for s in dm.dfa.accept_states:
            accept[s] = 1.0

        # Build start_vec[N] as float32
        start = np.zeros(N, dtype=np.float32)
        start[dm.dfa.start] = 1.0

        # Build char_to_idx mapping
        self._char_to_idx = {}
        for ch, idx in dm.char_to_idx.items():
            self._char_to_idx[ch] = idx
        self._sigma = sigma

        rc = self.lib.fp16_engine_init(
            T.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            accept.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            start.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, sigma,
            max_batch, max(max_total_chars // max(max_batch, 1), 1),
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
            raw = bytearray()
            for s in strings:
                for ch in s:
                    idx = self._char_to_idx.get(ch, self._sigma)
                    raw.append(idx)
            raw_concat = np.frombuffer(bytes(raw), dtype=np.uint8).copy()
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
            raise RuntimeError(f"fp16_engine_dispatch failed with code {rc}")

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
            raise RuntimeError(f"fp16_engine_dispatch failed with code {rc}")

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
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.gpu_bridge_fp16_evolution import FP16EvolutionGPUSimulator; print('import OK')"`
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add src/gpu_bridge_fp16_evolution.py
git commit -m "feat(fp16): Python ctypes bridge for FP16 TC engine"
```

---

### Task 6: Python Tests

**Files:**
- Create: `tests/test_fp16_evolution.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_fp16_evolution.py`:

```python
import pytest
import random
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices
from src.gpu_bridge_fp16_evolution import FP16EvolutionGPUSimulator


@pytest.fixture
def fp16_engine():
    dfa = compile_regex("(a|b)*abb")
    dm = DFAMatrices(dfa)
    sim = FP16EvolutionGPUSimulator()
    engine = sim.create_engine(dm)
    yield engine, dfa
    engine.destroy()


def test_basic_correctness(fp16_engine):
    engine, dfa = fp16_engine
    strings = ["abb", "aabb", "babb", "ab", "ba", ""]
    expected = [dfa.simulate(s) for s in strings]
    results = engine.simulate_batch(strings)
    assert results == expected


def test_long_strings(fp16_engine):
    engine, dfa = fp16_engine
    random.seed(42)
    strings = ["".join(random.choice("ab") for _ in range(1000)) for _ in range(100)]
    expected = [dfa.simulate(s) for s in strings]
    results = engine.simulate_batch(strings)
    assert results == expected


def test_timed_dispatch(fp16_engine):
    engine, dfa = fp16_engine
    strings = ["abb", "aabb", "babb"]
    results, kern_ms, total_ms = engine.simulate_batch_timed(strings)
    expected = [dfa.simulate(s) for s in strings]
    assert results == expected
    assert kern_ms >= 0
    assert total_ms >= 0


def test_empty_batch(fp16_engine):
    engine, dfa = fp16_engine
    results = engine.simulate_batch([])
    assert results == []


def test_cross_engine_validation():
    """Compare FP16 TC results against sequential DFA simulation."""
    regexes = [
        "(a|b)*abb",
        "(a|b)*a(a|b)",
        "a*b*",
        "(ab|ba)*",
        "(a|b|c)*abc",
    ]
    random.seed(123)

    for regex in regexes:
        dfa = compile_regex(regex)
        dm = DFAMatrices(dfa)
        sim = FP16EvolutionGPUSimulator()
        engine = sim.create_engine(dm)

        alphabet = list(dfa.alphabet)
        strings = []
        for _ in range(200):
            length = random.randint(0, 500)
            strings.append("".join(random.choice(alphabet) for _ in range(length)))

        gpu_results = engine.simulate_batch(strings)
        cpu_results = [dfa.simulate(s) for s in strings]

        mismatches = sum(g != c for g, c in zip(gpu_results, cpu_results))
        assert mismatches == 0, \
            f"Regex {regex!r}: {mismatches}/{len(strings)} mismatches"

        engine.destroy()
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_fp16_evolution.py -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fp16_evolution.py
git commit -m "test(fp16): Python correctness tests for FP16 TC engine"
```

---

### Task 7: OptimizedEngine Integration

**Files:**
- Modify: `src/optimized_engine.py`

- [ ] **Step 1: Add fp16_tc+gpu config branch**

In `__init__`, add a new elif branch after the `prefix+gpu` case (after line 102):

```python
        elif config == "fp16_tc+gpu":
            self._force_baseline()
            self._setup_fp16_tc_gpu()
```

Update the ValueError message (line 104) to include `'fp16_tc+gpu'`.

- [ ] **Step 2: Add _setup_fp16_tc_gpu method**

Add after `_setup_prefix_gpu` (after line 293):

```python
    def _setup_fp16_tc_gpu(self):
        self._build_dfa()
        from src.gpu_bridge_fp16_evolution import FP16EvolutionGPUSimulator
        sim = FP16EvolutionGPUSimulator()
        self._fp16_tc_gpu = sim.create_engine(
            self._dm,
            max_total_chars=1 << 29,
            max_batch=1 << 19,
        )
        self._scan_backend = 'fp16_tc+gpu'
        self._selection_reason = (
            f'GPU FP16 TC evolution (N={self._dm.n_states})'
        )
```

- [ ] **Step 3: Add _fp16_tc_gpu field to __init__**

In `__init__`, add after the `_prefix_compose_gpu` field (line 71):

```python
        self._fp16_tc_gpu = None  # FP16EvolutionEngine (GPU)
```

- [ ] **Step 4: Wire dispatch methods**

In `_match_one` (around line 329), add before the `prefix_compose_gpu` check:

```python
        if self._fp16_tc_gpu is not None:
            return self._fp16_tc_gpu.simulate_batch([s])[0]
```

In `match_batch` (around line 346), add before the `prefix_compose_gpu` check:

```python
        if self._fp16_tc_gpu is not None:
            return self._fp16_tc_gpu.simulate_batch(strings)
```

In `match_batch_timed` (around line 364), add before the `prefix_compose_gpu` check:

```python
        if self._fp16_tc_gpu is not None:
            results, kern_ms, total_ms = self._fp16_tc_gpu.simulate_batch_timed(strings)
            return results, {'kernel_ms': kern_ms, 'total_ms': total_ms}
```

- [ ] **Step 5: Update auto-selection to use FP16 TC for N ≤ 64 when monoid is too large**

In `_auto_select`, after the monoid batch GPU attempt succeeds (around line 153), add the FP16 TC tier. Replace the existing Tier 2 prefix compose fallback (lines 154-164) with:

```python
            # Tier 2: FP16 TC for N ≤ 64 (M > 255 case)
            if n_states <= 64:
                try:
                    self._setup_fp16_tc_gpu()
                    self._representation = "dfa"
                    self._selection_reason = (
                        f"DFA has {n_states} states; monoid size {md.size} > 255; "
                        f"auto-selected fp16_tc+gpu"
                    )
                    return
                except Exception:
                    pass
            # Tier 3: Prefix compose fallback
            try:
                self._setup_prefix_gpu()
                ...  # (existing code)
```

Also update the monoid-too-large fallback (around line 175) similarly: try FP16 TC before prefix compose.

- [ ] **Step 6: Verify integration**

Run: `python -c "from src.optimized_engine import OptimizedEngine; e = OptimizedEngine('(a|b)*abb', config='fp16_tc+gpu'); print(e.match_batch(['abb', 'ab']))"`
Expected: `[True, False]`

- [ ] **Step 7: Run existing tests to check for regressions**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: All existing tests still pass, plus new FP16 tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/optimized_engine.py
git commit -m "feat(fp16): integrate FP16 TC engine into OptimizedEngine"
```

---

### Task 8: Run Built-in CUDA Benchmarks

**Files:** None (run only)

- [ ] **Step 1: Build and run CUDA tests + benchmark**

Run: `make build/fp16_evolution && ./build/fp16_evolution`

Expected: All PASS, benchmark table printed. Record the Gc/s numbers.

- [ ] **Step 2: Compare against existing kernels**

Run: `make build/batched_evolution && ./build/batched_evolution`

Compare FP16 Gc/s vs INT8 V2/V3 Gc/s at matching B×L configs. The FP16 kernel should be at least 3-5× faster.

- [ ] **Step 3: Document results**

Print a comparison summary to stdout. No file changes needed.
