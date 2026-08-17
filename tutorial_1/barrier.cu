#include <stdio.h>
#include <cuda_runtime.h>

__device__ void grid_barrier(int* global_counter, int expected_blocks) {
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(global_counter, 1);

        while (atomicAdd(global_counter, 0) < expected_blocks) {
            // wait
        }
    }
    // Sync the block again so all threads wait for Thread 0 to finish spinning
    __syncthreads();
}

__global__ void syncKernel(int* d_barrier_counter) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    printf("Block %d, Thread %d: Reached Checkpoint A (Before Barrier)\n", bid, tid);
    grid_barrier(d_barrier_counter, gridDim.x);

    printf("Block %d, Thread %d: Reached Checkpoint B (After Barrier)\n", bid, tid);
}

int main() {
    int num_blocks = 2;
    int threads_per_block = 4;

    int* d_barrier_counter;
    cudaMalloc((void**)&d_barrier_counter, sizeof(int));
    
    cudaMemset(d_barrier_counter, 0, sizeof(int));

    printf("Launching Kernel with %d blocks and %d threads per block...\n\n", num_blocks, threads_per_block);

    syncKernel<<<num_blocks, threads_per_block>>>(d_barrier_counter);
    cudaDeviceSynchronize();

    cudaFree(d_barrier_counter);

    printf("\nKernel execution completed successfully.\n");
    return 0;
}