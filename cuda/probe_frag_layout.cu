#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
using namespace nvcuda;

__global__ void probe_acc_layout() {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag;
    for (int i = 0; i < frag.num_elements; i++)
        frag.x[i] = lane * 100 + i;

    __shared__ int32_t buf[256];
    wmma::store_matrix_sync(buf, frag, 16, wmma::mem_row_major);
    __syncwarp();

    if (lane == 0) {
        printf("Accumulator fragment layout (row-major store):\n");
        printf("  num_elements = %d\n", frag.num_elements);
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int val = buf[r * 16 + c];
                int t = val / 100, e = val % 100;
                printf("  (%2d,%2d) -> thread %2d elem %d\n", r, c, t, e);
            }
        }
    }
}

__global__ void probe_matb_layout() {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    // Probe matrix_b by loading known pattern and inspecting fragment elements
    // S is 16x16 int8 col-major: S[col*16 + row] = row*16 + col (unique value)
    __shared__ int8_t S_probe[256];
    for (int e = lane; e < 256; e += 32) {
        int row = e % 16;
        int col = e / 16;
        S_probe[e] = (int8_t)(row * 16 + col);  // encode (row,col) as row*16+col
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b;
    wmma::load_matrix_sync(frag_b, S_probe, 16);

    if (lane == 0)
        printf("\nMatrix_b fragment layout (col-major load, ldm=16):\n");
    if (lane == 0)
        printf("  num_elements = %d\n", frag_b.num_elements);

    for (int t = 0; t < 32; t++) {
        if (lane == t) {
            printf("  thread %2d:", t);
            for (int i = 0; i < frag_b.num_elements; i++) {
                uint8_t val = (uint8_t)frag_b.x[i];
                int row = val / 16;
                int col = val % 16;
                printf("  [%d]=(r%d,c%d)", i, row, col);
            }
            printf("\n");
        }
        __syncwarp();
    }

    // Also check if fragment x[] is contiguous in memory as packed int32
    if (lane == 0) {
        printf("\n  Raw bytes per thread (first 4 threads):\n");
    }
    for (int t = 0; t < 4; t++) {
        if (lane == t) {
            printf("  thread %d:", t);
            const uint8_t *raw = reinterpret_cast<const uint8_t*>(&frag_b.x[0]);
            for (int i = 0; i < frag_b.num_elements; i++) {
                printf(" 0x%02x", raw[i]);
            }
            printf("\n");
        }
        __syncwarp();
    }
}

__global__ void probe_mata_layout() {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    // Probe matrix_a by loading known pattern
    __shared__ int8_t T_probe[256];
    for (int e = lane; e < 256; e += 32) {
        int row = e / 16;   // row-major
        int col = e % 16;
        T_probe[e] = (int8_t)(row * 16 + col);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a;
    wmma::load_matrix_sync(frag_a, T_probe, 16);

    if (lane == 0)
        printf("\nMatrix_a fragment layout (row-major load, ldm=16):\n");
    if (lane == 0)
        printf("  num_elements = %d\n", frag_a.num_elements);

    for (int t = 0; t < 4; t++) {
        if (lane == t) {
            printf("  thread %2d:", t);
            for (int i = 0; i < frag_a.num_elements; i++) {
                uint8_t val = (uint8_t)frag_a.x[i];
                int row = val / 16;
                int col = val % 16;
                printf("  [%d]=(r%d,c%d)", i, row, col);
            }
            printf("\n");
        }
        __syncwarp();
    }
}

__global__ void probe_fp16_matb_layout() {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    __shared__ half S_fp16[256];
    for (int e = lane; e < 256; e += 32) {
        int row = e % 16;
        int col = e / 16;
        S_fp16[e] = __float2half((float)(row * 16 + col));
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::load_matrix_sync(frag_b, S_fp16, 16);

    if (lane == 0)
        printf("\nFP16 Matrix_b fragment layout (col-major, m16n16k16):\n");
    if (lane == 0)
        printf("  num_elements = %d\n", frag_b.num_elements);

    for (int t = 0; t < 32; t++) {
        if (lane == t) {
            printf("  thread %2d:", t);
            for (int i = 0; i < frag_b.num_elements; i++) {
                int val = (int)__half2float(frag_b.x[i]);
                int row = val / 16;
                int col = val % 16;
                printf("  [%d]=(r%d,c%d)", i, row, col);
            }
            printf("\n");
        }
        __syncwarp();
    }
}

__global__ void probe_fp16_acc_layout() {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag;
    for (int i = 0; i < frag.num_elements; i++)
        frag.x[i] = __float2half((float)(lane * 100 + i));

    __shared__ half buf[256];
    wmma::store_matrix_sync(buf, frag, 16, wmma::mem_row_major);
    __syncwarp();

    if (lane == 0) {
        printf("\nFP16 Accumulator fragment layout (m16n16k16):\n");
        printf("  num_elements = %d\n", frag.num_elements);
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int val = (int)__half2float(buf[r * 16 + c]);
                int t = val / 100, e = val % 100;
                printf("  (%2d,%2d) -> thread %2d elem %d\n", r, c, t, e);
            }
        }
    }
}

/*
 * Benchmark FP16 WMMA serial chain with smem round-trip.
 * Compute S = T × S repeatedly, measuring cycles/step.
 * This is the baseline for comparison with wgmma RS chain.
 */
__global__ void bench_wmma_fp16_chain(int num_steps)
{
    int lane = threadIdx.x;
    if (lane >= 32) return;

    // T = cyclic shift permutation (same as wgmma benchmark)
    __shared__ half T_smem[256];  // row-major
    __shared__ half S_smem[256];  // col-major
    for (int e = lane; e < 256; e += 32) {
        int r = e / 16, c = e % 16;
        T_smem[e] = __float2half((c == ((r + 1) % 16)) ? 1.0f : 0.0f);
    }
    for (int e = lane; e < 256; e += 32) {
        int r = e % 16, c = e / 16;
        S_smem[e] = __float2half((r == c) ? 1.0f : 0.0f);  // identity, col-major
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_t;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_s;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::load_matrix_sync(frag_t, T_smem, 16);

    // Warmup
    for (int i = 0; i < 10; i++) {
        wmma::load_matrix_sync(frag_s, S_smem, 16);
        wmma::fill_fragment(frag_c, __float2half(0.0f));
        wmma::mma_sync(frag_c, frag_t, frag_s, frag_c);
        wmma::store_matrix_sync(S_smem, frag_c, 16, wmma::mem_col_major);
    }
    __syncwarp();

    // Reset S = identity
    for (int e = lane; e < 256; e += 32) {
        int r = e % 16, c = e / 16;
        S_smem[e] = __float2half((r == c) ? 1.0f : 0.0f);
    }
    __syncwarp();

    uint64_t start, end;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));

    for (int i = 0; i < num_steps; i++) {
        wmma::load_matrix_sync(frag_s, S_smem, 16);
        wmma::fill_fragment(frag_c, __float2half(0.0f));
        wmma::mma_sync(frag_c, frag_t, frag_s, frag_c);
        wmma::store_matrix_sync(S_smem, frag_c, 16, wmma::mem_col_major);
    }

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
    __syncwarp();

    if (lane == 0) {
        float ns = (float)(end - start);
        float ns_per_step = ns / (float)num_steps;
        float clock_ghz = 1.98f;
        float cycles_per_step = ns_per_step * clock_ghz;
        printf("\n=== FP16 WMMA chain benchmark (smem round-trip) ===\n");
        printf("Steps: %d, threads: 32 (1 warp)\n", num_steps);
        printf("Per step: %.1f ns = %.1f cycles\n", ns_per_step, cycles_per_step);
        printf("Compare: wgmma RS chain = ~70 cycles/step (128 threads)\n");
    }
}

int main() {
    probe_fp16_acc_layout<<<1, 32>>>();
    probe_fp16_matb_layout<<<1, 32>>>();
    cudaDeviceSynchronize();

    bench_wmma_fp16_chain<<<1, 32>>>(10000);
    cudaDeviceSynchronize();
    return 0;
}
