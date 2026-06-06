#include "benchmarks.hpp"

#include <chrono>
#include <cmath>
#include <vector>

namespace {

double median_seconds(std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    const size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    return samples[mid];
}

}  // namespace

void run_flops_peak(double* out_seconds, double* out_gflops, int repeats) {
    constexpr int N = 1 << 20;
    std::vector<double> a(N, 1.0), b(N, 2.0), c(N, 0.0);

    // Warm-up
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] * b[i] + c[i];
    }

    std::vector<double> timings;
    timings.reserve(repeats);

    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            c[i] = a[i] * b[i] + c[i];
        }
        const auto t1 = std::chrono::steady_clock::now();
        timings.push_back(
            std::chrono::duration<double>(t1 - t0).count());
    }

    const double sec = median_seconds(timings);
    const double flops = 2.0 * N;  // one FMA per element
    *out_seconds = sec;
    *out_gflops = flops / sec / 1e9;
}

void run_stream_triad(double* out_seconds, double* out_gbps, int n, int repeats) {
    std::vector<double> a(n, 1.0), b(n, 2.0), c(n, 0.0);

    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + 0.5 * b[i];
    }

    std::vector<double> timings;
    timings.reserve(repeats);

    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) {
            c[i] = a[i] + 0.5 * b[i];
        }
        const auto t1 = std::chrono::steady_clock::now();
        timings.push_back(
            std::chrono::duration<double>(t1 - t0).count());
    }

    const double sec = median_seconds(timings);
    const double bytes = 3.0 * n * sizeof(double);  // read a, read b, write c
    *out_seconds = sec;
    *out_gbps = bytes / sec / 1e9;
}
