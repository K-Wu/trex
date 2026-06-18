#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
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
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int val = buf[r * 16 + c];
                int t = val / 100, e = val % 100;
                printf("  (%2d,%2d) -> thread %2d elem %d\n", r, c, t, e);
            }
        }
    }

    // Summary: group by column to show which threads own each column
    if (lane == 0) {
        printf("\nPer-column thread ownership:\n");
        for (int c = 0; c < 16; c++) {
            printf("  col %2d: ", c);
            for (int r = 0; r < 16; r++) {
                int val = buf[r * 16 + c];
                printf("t%d.%d ", val/100, val%100);
            }
            printf("\n");
        }
    }
}

int main() {
    probe_acc_layout<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
