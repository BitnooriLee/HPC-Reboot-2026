#pragma once

#include <cuda_runtime.h>

// Naive GEMM: C = A · B  (row-major, M×K · K×N → M×N)
// block_dim : side length of the 2-D thread block (block_dim × block_dim threads).
//             Must satisfy block_dim * block_dim ≤ GPU max threads-per-block (1024).
void matmul_naive(const float* dA, const float* dB, float* dC,
                  int M, int N, int K, int block_dim = 16);

// Tiled shared-memory GEMM.
// tile_dim  : shared-memory tile side length; also sets the thread-block dimensions.
//             Must be one of {8, 16, 32}.
void matmul_tiled(const float* dA, const float* dB, float* dC,
                  int M, int N, int K, int tile_dim = 16);
