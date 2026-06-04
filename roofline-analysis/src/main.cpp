#include "benchmarks.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    std::string output = "../results/roofline-analysis/roofline_points.csv";
    int repeats = 10;
    int stream_n = 1 << 24;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--stream-n") == 0 && i + 1 < argc) {
            stream_n = std::atoi(argv[++i]);
        }
    }

    double flops_sec = 0.0, flops_gflops = 0.0;
    double stream_sec = 0.0, stream_gbps = 0.0;

    run_flops_peak(&flops_sec, &flops_gflops, repeats);
    run_stream_triad(&stream_sec, &stream_gbps, stream_n, repeats);

    // Roofline: intensity = FLOPs / bytes (approximate)
    const double flops_intensity = 2.0 / (3.0 * sizeof(double));  // FMA in registers
    const double stream_intensity = 2.0 / (3.0 * sizeof(double));  // 2 FLOPs, 3 arrays

    FILE* fp = std::fopen(output.c_str(), "w");
    if (!fp) {
        std::perror("fopen");
        return 1;
    }

    std::fprintf(fp, "kernel,intensity,gflops,gbps,is_bandwidth,seconds\n");
    std::fprintf(fp, "flops_peak,%.6e,%.3f,0,0,%.6f\n", flops_intensity, flops_gflops,
                 flops_sec);
    std::fprintf(fp, "stream_triad,%.6e,0,%.3f,1,%.6f\n", stream_intensity,
                 stream_gbps, stream_sec);
    std::fclose(fp);

    std::printf("flops_peak:  %.2f GFLOPS (%.4f s)\n", flops_gflops, flops_sec);
    std::printf("stream_triad: %.2f GB/s (%.4f s, n=%d)\n", stream_gbps, stream_sec,
                stream_n);
    std::printf("Wrote %s\n", output.c_str());
    return 0;
}
