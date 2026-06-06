#include "kernels.cuh"

#include <cuda_runtime.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// Naive GEMM — one thread computes one output element
// ---------------------------------------------------------------------------
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

void matmul_naive(const float* dA, const float* dB, float* dC,
                  int M, int N, int K, int block_dim) {
    const dim3 block(block_dim, block_dim);
    const dim3 grid((N + block_dim - 1) / block_dim,
                    (M + block_dim - 1) / block_dim);
    matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}

// ---------------------------------------------------------------------------
// Tiled GEMM — shared-memory tile blocking, TILE_DIM × TILE_DIM threads/block
//
// Templated on TILE_DIM so the compiler sees a compile-time constant for the
// shared-memory array sizes.  Explicit instantiations below cover tile dims
// 8, 16, and 32.
// ---------------------------------------------------------------------------
template <int TILE_DIM>
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    __shared__ float tile_a[TILE_DIM][TILE_DIM];
    __shared__ float tile_b[TILE_DIM][TILE_DIM];

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE_DIM + threadIdx.x;
        const int b_row = t * TILE_DIM + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Explicit instantiations for every supported tile dimension
template __global__ void matmul_tiled_kernel<8>(
    const float*, const float*, float*, int, int, int);
template __global__ void matmul_tiled_kernel<16>(
    const float*, const float*, float*, int, int, int);
template __global__ void matmul_tiled_kernel<32>(
    const float*, const float*, float*, int, int, int);

// Runtime dispatch: choose the template instantiation matching tile_dim
void matmul_tiled(const float* dA, const float* dB, float* dC,
                  int M, int N, int K, int tile_dim) {
#define LAUNCH_TILED(T)                                                    \
    case T: {                                                              \
        const dim3 block(T, T);                                            \
        const dim3 grid((N + T - 1) / T, (M + T - 1) / T);               \
        matmul_tiled_kernel<T><<<grid, block>>>(dA, dB, dC, M, N, K);     \
        break;                                                             \
    }
    switch (tile_dim) {
        LAUNCH_TILED(8)
        LAUNCH_TILED(16)
        LAUNCH_TILED(32)
        default:
            std::fprintf(stderr,
                "[matmul_tiled] unsupported tile_dim=%d — use 8, 16, or 32\n",
                tile_dim);
    }
#undef LAUNCH_TILED
}
