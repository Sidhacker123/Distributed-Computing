
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>

#define TILE_WIDTH 32

__global__ void matrixMultiplyShared(float* A, float* B, float* C, int width) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        ds_A[ty][tx] = A[Row * width + ph * TILE_WIDTH + tx];
        ds_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    C[Row * width + Col] = Pvalue;
}

void fillMatrix(float* mat, int size) {
    for (int i = 0; i < size; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

void profile_cuda_event(cudaEvent_t start, cudaEvent_t stop, const char* label) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%s: %.2f ms\n", label, milliseconds);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int matrixSize = N * N;
    size_t bytes = matrixSize * sizeof(float);

    float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
    cudaMallocHost(&h_A, bytes);
    cudaMallocHost(&h_B, bytes);
    cudaMallocHost(&h_C, bytes);

    if (rank == 0) {
        fillMatrix(h_A, matrixSize);
        fillMatrix(h_B, matrixSize);
    }

    // MPI Ring Broadcast: send A, B to all ranks
    MPI_Request reqs[2];
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            MPI_Isend(h_A, matrixSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(h_B, matrixSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &reqs[1]);
        }
    } else {
        MPI_Recv(h_A, matrixSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_B, matrixSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(N / TILE_WIDTH, N / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiplyShared<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    profile_cuda_event(start, stop, "Kernel execution");

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    if (rank == 0) {
        std::cout << "Sample C[0][0] = " << h_C[0] << std::endl;
    }

    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    MPI_Finalize();
    return 0;
}
