// GPU matrix multiplication kernel using tiling and shared memory
// Computes C = A * B (assuming A is MxK, B is KxN, C is MxN)
#define TILE_SIZE 16

extern "C" __global__
void matrixMultiplyShared(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate global thread coordinates
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;
    // Loop over tiles of the K dimension
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Load a tile of A and B into shared memory
        if (row < M && (t + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (t + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if ((t + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        // Compute this tile's partial product
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    // Write result to C
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

