// CUDA matrix-multiply benchmark
// Kernels  : naive (global memory) vs tiled (shared memory)
// Timing   : CUDA events, averaged over --repeats launches
// Correctness: GPU output vs double-precision CPU reference

#include "kernels.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "[CUDA error] %s:%d  %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// CPU reference — double accumulator for an accurate baseline
// ---------------------------------------------------------------------------
static void cpu_matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += static_cast<double>(A[i * K + k]) *
                     static_cast<double>(B[k * N + j]);
            C[i * N + j] = static_cast<float>(s);
        }
    }
}

// Returns {max_absolute_error, max_relative_error} over all M*N elements.
// Relative error is normalised by max(|ref[i]|, 1e-6) to handle near-zero entries.
static std::pair<float, float> max_errors(const float* ref,
                                          const float* test, int len) {
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float abs_err = std::fabsf(ref[i] - test[i]);
        const float rel_err = abs_err / (std::fabsf(ref[i]) + 1e-6f);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
    }
    return {max_abs, max_rel};
}

// ---------------------------------------------------------------------------
// CUDA-event timing helper
// The 7th integer parameter is kernel-specific (block_dim for naive, tile_dim
// for tiled), matching the function pointer type used by both launchers.
// ---------------------------------------------------------------------------
using KernelFn = void (*)(const float*, const float*, float*,
                           int, int, int, int);

static float time_kernel_ms(KernelFn fn,
                             const float* dA, const float* dB, float* dC,
                             int M, int N, int K, int param,
                             int warmup, int repeats) {
    // Warmup passes (not timed)
    for (int i = 0; i < warmup; ++i)
        fn(dA, dB, dC, M, N, K, param);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < repeats; ++i)
        fn(dA, dB, dC, M, N, K, param);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return ms / static_cast<float>(repeats);
}

static float gflops_from_ms(long long flops, float ms) {
    return static_cast<float>(flops) / (ms * 1e-3f) * 1e-9f;
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------
static std::vector<int> parse_int_list(const char* s) {
    std::vector<int> out;
    std::string buf;
    for (const char* p = s; ; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!buf.empty()) {
                out.push_back(std::stoi(buf));
                buf.clear();
            }
            if (*p == '\0') break;
        } else {
            buf += *p;
        }
    }
    return out;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --output FILE     CSV output path [default: ../results/cuda-matmul/matmul_timing.csv]\n"
        "  --sizes N1,N2,…   Comma-separated square matrix sizes [default: 512,1024,2048,4096]\n"
        "  --tile-dim T      Tile dimension for tiled kernel, one of {8,16,32} [default: 16]\n"
        "  --block-dim B     Thread-block side for naive kernel [default: 16]\n"
        "  --repeats R       Timed repetitions per kernel [default: 10]\n"
        "  --warmup W        Warmup iterations [default: 2]\n"
        "  --check-max N     Skip CPU check for matrix sizes > N (0 = skip all) [default: 2048]\n",
        prog);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Defaults
    std::string output    = "../results/cuda-matmul/matmul_timing.csv";
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    int tile_dim   = 16;
    int block_dim  = 16;
    int repeats    = 10;
    int warmup     = 2;
    int check_max  = 2048;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto need_next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[error] %s requires an argument\n", a);
                std::exit(1);
            }
            return argv[++i];
        };

        if      (std::strcmp(a, "--output")    == 0) output    = need_next();
        else if (std::strcmp(a, "--sizes")     == 0) sizes     = parse_int_list(need_next());
        else if (std::strcmp(a, "--tile-dim")  == 0) tile_dim  = std::atoi(need_next());
        else if (std::strcmp(a, "--block-dim") == 0) block_dim = std::atoi(need_next());
        else if (std::strcmp(a, "--repeats")   == 0) repeats   = std::atoi(need_next());
        else if (std::strcmp(a, "--warmup")    == 0) warmup    = std::atoi(need_next());
        else if (std::strcmp(a, "--check-max") == 0) check_max = std::atoi(need_next());
        else if (std::strcmp(a, "--help")      == 0) { usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "[warn] Unknown argument: %s\n", a);
        }
    }

    // Validate tile_dim
    if (tile_dim != 8 && tile_dim != 16 && tile_dim != 32) {
        std::fprintf(stderr,
            "[error] --tile-dim must be 8, 16, or 32 (got %d)\n", tile_dim);
        return 1;
    }
    // block_dim * block_dim must not exceed GPU max threads per block (1024)
    if (block_dim < 1 || block_dim > 32) {
        std::fprintf(stderr,
            "[error] --block-dim must be in [1, 32] (got %d)\n", block_dim);
        return 1;
    }

    // Print device info
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("=== CUDA Matrix Multiply Benchmark ===\n");
    std::printf("  GPU       : %s\n", prop.name);
    std::printf("  tile_dim  : %d   (tiled kernel thread-block = %d×%d)\n",
                tile_dim, tile_dim, tile_dim);
    std::printf("  block_dim : %d   (naive kernel thread-block = %d×%d)\n",
                block_dim, block_dim, block_dim);
    std::printf("  repeats   : %d   warmup: %d\n", repeats, warmup);
    std::printf("  check_max : %d   (CPU correctness check for N ≤ this)\n",
                check_max);
    std::printf("  output    : %s\n\n", output.c_str());

    FILE* fp = std::fopen(output.c_str(), "w");
    if (!fp) {
        std::perror("fopen");
        return 1;
    }
    std::fprintf(fp, "M,N,K,kernel,tile_dim,block_dim,ms,gflops,correct\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int n : sizes) {
        const int M = n, N = n, K = n;
        const long long ne_a = static_cast<long long>(M) * K;
        const long long ne_b = static_cast<long long>(K) * N;
        const long long ne_c = static_cast<long long>(M) * N;
        const long long flops = 2LL * M * N * K;

        std::printf("--- N = %d  (%.2f M elements per matrix) ---\n",
                    n, ne_c / 1e6f);

        // Host buffers
        std::vector<float> hA(ne_a), hB(ne_b), hC_ref(ne_c), hC_gpu(ne_c);
        for (float& x : hA) x = dist(rng);
        for (float& x : hB) x = dist(rng);

        // Device buffers
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, ne_a * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, ne_b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, ne_c * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), ne_a * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), ne_b * sizeof(float),
                              cudaMemcpyHostToDevice));

        // CPU reference (only if matrix fits within check_max)
        const bool do_check = (n <= check_max);
        if (do_check) {
            std::printf("  Computing CPU reference ... ");
            std::fflush(stdout);
            cpu_matmul(hA.data(), hB.data(), hC_ref.data(), M, N, K);
            std::printf("done.\n");
        }

        // ---- Naive ----
        const float ms_naive = time_kernel_ms(
            matmul_naive, dA, dB, dC, M, N, K, block_dim, warmup, repeats);
        const float gf_naive = gflops_from_ms(flops, ms_naive);

        std::string correct_naive = "N/A";
        if (do_check) {
            CUDA_CHECK(cudaMemcpy(hC_gpu.data(), dC, ne_c * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            auto [abs_err, rel_err] = max_errors(hC_ref.data(), hC_gpu.data(),
                                                 static_cast<int>(ne_c));
            correct_naive = (rel_err < 1e-2f) ? "PASS" : "FAIL";
            std::printf("  naive  : %8.2f GFLOPS  %8.3f ms"
                        "  [%s  max_rel=%.2e  max_abs=%.2e]\n",
                        gf_naive, ms_naive,
                        correct_naive.c_str(), rel_err, abs_err);
        } else {
            std::printf("  naive  : %8.2f GFLOPS  %8.3f ms\n",
                        gf_naive, ms_naive);
        }
        std::fprintf(fp, "%d,%d,%d,naive,%d,%d,%.4f,%.3f,%s\n",
                     M, N, K, tile_dim, block_dim,
                     ms_naive, gf_naive, correct_naive.c_str());

        // ---- Tiled ----
        const float ms_tiled = time_kernel_ms(
            matmul_tiled, dA, dB, dC, M, N, K, tile_dim, warmup, repeats);
        const float gf_tiled = gflops_from_ms(flops, ms_tiled);

        std::string correct_tiled = "N/A";
        if (do_check) {
            CUDA_CHECK(cudaMemcpy(hC_gpu.data(), dC, ne_c * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            auto [abs_err, rel_err] = max_errors(hC_ref.data(), hC_gpu.data(),
                                                 static_cast<int>(ne_c));
            correct_tiled = (rel_err < 1e-2f) ? "PASS" : "FAIL";
            std::printf("  tiled  : %8.2f GFLOPS  %8.3f ms"
                        "  [%s  max_rel=%.2e  max_abs=%.2e]\n",
                        gf_tiled, ms_tiled,
                        correct_tiled.c_str(), rel_err, abs_err);
        } else {
            std::printf("  tiled  : %8.2f GFLOPS  %8.3f ms\n",
                        gf_tiled, ms_tiled);
        }
        std::fprintf(fp, "%d,%d,%d,tiled,%d,%d,%.4f,%.3f,%s\n",
                     M, N, K, tile_dim, tile_dim,
                     ms_tiled, gf_tiled, correct_tiled.c_str());

        std::printf("  speedup: %.2fx  (tiled / naive)\n\n",
                    ms_naive / ms_tiled);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    std::fclose(fp);
    std::printf("Results written to %s\n", output.c_str());
    return 0;
}
