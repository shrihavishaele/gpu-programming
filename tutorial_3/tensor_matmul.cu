// 64x64 Matrix Multiplication using Tensor Cores (16x16 tiles)
// Compile: nvcc -arch=sm_75 -o tensor_matmul tensor_matmul.cu

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int N = 64;
const int TILES = N / WMMA_M; // 4 tiles per dimension

// Each warp computes one 16x16 output tile of C
__global__ void tensorCoreMatMul(half *A, half *B, float *C, int M_dim, int N_dim, int K_dim) {
    int tileRow = blockIdx.x;
    int tileCol = blockIdx.y;

    if (tileRow * WMMA_M >= M_dim || tileCol * WMMA_N >= N_dim) return;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> aFrag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> bFrag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    fill_fragment(cFrag, 0.0f);

    // C[i,j] = sum over k: A[i,k] * B[k,j]
    for (int k = 0; k < TILES; k++) {
        half *aPtr = A + tileRow * WMMA_M * K_dim + k * WMMA_K;
        half *bPtr = B + k * WMMA_K * N_dim + tileCol * WMMA_N;

        load_matrix_sync(aFrag, aPtr, K_dim);
        load_matrix_sync(bFrag, bPtr, N_dim);
        mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    float *cPtr = C + tileRow * WMMA_M * N_dim + tileCol * WMMA_N;
    store_matrix_sync(cPtr, cFrag, N_dim, mem_row_major);
}

// CPU reference for verification
void cpuMatMul(half *A, half *B, float *C_ref, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < dim; k++) {
                sum += __half2float(A[i * dim + k]) * __half2float(B[k * dim + j]);
            }
            C_ref[i * dim + j] = sum;
        }
    }
}

int main() {
    printf("=== 64x64 Matrix Multiplication using Tensor Cores ===\n");
    printf("Tile size: %dx%d | Tiles per dim: %d | Total tiles: %d\n\n", WMMA_M, WMMA_N, TILES, TILES * TILES);

    size_t sizeHalf  = N * N * sizeof(half);
    size_t sizeFloat = N * N * sizeof(float);

    half  *h_A     = (half *)malloc(sizeHalf);
    half  *h_B     = (half *)malloc(sizeHalf);
    float *h_C     = (float *)malloc(sizeFloat);
    float *h_C_ref = (float *)malloc(sizeFloat);

    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = __float2half((float)(rand() % 5));
        h_B[i] = __float2half((float)(rand() % 5));
    }

    half  *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, sizeHalf);
    cudaMalloc(&d_B, sizeHalf);
    cudaMalloc(&d_C, sizeFloat);

    cudaMemcpy(d_A, h_A, sizeHalf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeHalf, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeFloat);

    // Grid (4,4) = one block per tile, Block (32) = one warp for WMMA
    dim3 grid(TILES, TILES);
    dim3 block(32, 1);

    tensorCoreMatMul<<<grid, block>>>(d_A, d_B, d_C, N, N, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeFloat, cudaMemcpyDeviceToHost);

    // Verify against CPU
    cpuMatMul(h_A, h_B, h_C_ref, N);

    int errors = 0;
    float maxDiff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > 1.0f) {
            if (errors < 10) {
                printf("  MISMATCH at C[%d][%d]: GPU=%.2f, CPU=%.2f\n",
                       i / N, i % N, h_C[i], h_C_ref[i]);
            }
            errors++;
        }
    }

    printf("\n--- Results ---\n");
    if (errors == 0)
        printf("PASSED! All elements match (max diff: %.4f)\n", maxDiff);
    else
        printf("FAILED! %d mismatches (max diff: %.4f)\n", errors, maxDiff);

    printf("\nTop-left 4x4 of C:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            printf("%8.1f", h_C[i * N + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
