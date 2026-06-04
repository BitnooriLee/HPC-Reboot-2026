// MPI Allreduce microbenchmark.
//
// Sweeps a range of message sizes (number of doubles) and reports median
// latency (µs) and effective bandwidth (GB/s) for MPI_Allreduce(MPI_SUM)
// on MPI_DOUBLE.  Rank 0 appends one CSV row per (size, msg_bytes) pair.
//
// Usage:
//   mpirun -np <N> ./build/allreduce_bench [--reps <R>] [--output <file>]

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Message sizes to sweep (number of double elements).
// Covers 4 B (1 elem) → 8 MB (1 M elems) — representative of gradient
// vectors from a toy model up to a shallow neural network.
constexpr int kNumSizes = 10;
constexpr int kMsgElems[kNumSizes] = {
    1,       // 8 B
    8,       // 64 B
    64,      // 512 B
    512,     // 4 KB
    4096,    // 32 KB
    32768,   // 256 KB
    131072,  // 1 MB
    524288,  // 4 MB
    1048576, // 8 MB
    4194304, // 32 MB
};

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2 == 0) ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int reps = 200;
    std::string output = "../results/mpi-training/allreduce_bench.csv";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        }
    }

    if (rank == 0) {
        std::printf("MPI Allreduce benchmark — ranks=%d  reps=%d\n", size, reps);
        std::printf("%-12s  %-14s  %-14s  %-14s\n",
                    "msg_bytes", "median_us", "min_us", "bw_gbs");
        std::printf("%s\n", std::string(58, '-').c_str());
    }

    FILE* fp = nullptr;
    if (rank == 0) {
        bool needs_header = true;
        FILE* check = std::fopen(output.c_str(), "r");
        if (check) {
            needs_header = (std::fgetc(check) == EOF);
            std::fclose(check);
        }
        fp = std::fopen(output.c_str(), "a");
        if (!fp) {
            std::perror("fopen");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (needs_header) {
            std::fprintf(fp, "ranks,msg_bytes,median_us,min_us,bandwidth_gbs\n");
        }
    }

    for (int s = 0; s < kNumSizes; ++s) {
        const int n = kMsgElems[s];
        const long msg_bytes = static_cast<long>(n) * sizeof(double);

        std::vector<double> buf(n, static_cast<double>(rank + 1));
        std::vector<double> times(reps);

        // Warm-up: 10 calls, not recorded.
        for (int w = 0; w < 10; ++w) {
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for (int r = 0; r < reps; ++r) {
            // Re-initialise so the operation is never a no-op (avoids
            // collectives being short-circuited by smart MPI implementations).
            buf.assign(n, static_cast<double>(rank + 1));

            const double t0 = MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            times[r] = MPI_Wtime() - t0;
        }

        // Collect the per-rank median time at rank 0.
        const double local_median_s = median(times);
        double global_median_s = 0.0;
        MPI_Reduce(&local_median_s, &global_median_s, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);

        double local_min_s = times[0];
        double global_min_s = 0.0;
        MPI_Reduce(&local_min_s, &global_min_s, 1, MPI_DOUBLE, MPI_MIN,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            const double median_us = global_median_s * 1e6;
            const double min_us    = global_min_s    * 1e6;
            // Allreduce transfers 2*(P-1)/P * msg_bytes per rank in a
            // ring algorithm; use msg_bytes as a conservative denominator.
            const double bw = (msg_bytes * 1e-9) / global_median_s;

            std::printf("%-12ld  %-14.2f  %-14.2f  %-14.4f\n",
                        msg_bytes, median_us, min_us, bw);
            std::fprintf(fp, "%d,%ld,%.6f,%.6f,%.6f\n",
                         size, msg_bytes, median_us, min_us, bw);
        }
    }

    if (fp) std::fclose(fp);

    MPI_Finalize();
    return 0;
}
