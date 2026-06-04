#include "kernels.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

float gflops(long long flops, float seconds) {
    return static_cast<float>(flops) / seconds / 1e9f;
}

float time_kernel(void (*launch)(const float*, const float*, float*, int, int, int),
                  const float* dA, const float* dB, float* dC,
                  int M, int N, int K, int repeats) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    launch(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        launch(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 1000.0f / repeats;
}

}  // namespace

int main(int argc, char** argv) {
    std::string output = "../results/cuda-matmul/matmul_timing.csv";
    int warmup = 2;
    int repeats = 10;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        }
    }

    const std::vector<int> sizes = {512, 1024, 2048, 4096};

    FILE* fp = std::fopen(output.c_str(), "w");
    if (!fp) {
        std::perror("fopen");
        return 1;
    }
    std::fprintf(fp, "M,N,K,kernel,seconds,gflops\n");

    for (int n : sizes) {
        const int M = n, N = n, K = n;
        const size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
        const size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
        const size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMalloc(&dA, bytes_a);
        cudaMalloc(&dB, bytes_b);
        cudaMalloc(&dC, bytes_c);

        const long long flops = 2LL * M * N * K;

        for (int w = 0; w < warmup; ++w) {
            matmul_naive(dA, dB, dC, M, N, K);
        }
        cudaDeviceSynchronize();

        const float t_naive =
            time_kernel(matmul_naive, dA, dB, dC, M, N, K, repeats);
        const float t_tiled =
            time_kernel(matmul_tiled, dA, dB, dC, M, N, K, repeats);

        std::fprintf(fp, "%d,%d,%d,naive,%.6f,%.3f\n", M, N, K, t_naive,
                     gflops(flops, t_naive));
        std::fprintf(fp, "%d,%d,%d,tiled,%.6f,%.3f\n", M, N, K, t_tiled,
                     gflops(flops, t_tiled));

        std::printf("N=%d  naive %.2f GFLOPS  tiled %.2f GFLOPS\n", n,
                    gflops(flops, t_naive), gflops(flops, t_tiled));

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    std::fclose(fp);
    std::printf("Wrote %s\n", output.c_str());
    return 0;
}
