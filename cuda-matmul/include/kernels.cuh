#pragma once

#include <cstddef>

constexpr int TILE_DIM = 16;

void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K);

void matmul_tiled(const float* A, const float* B, float* C,
                  int M, int N, int K);
