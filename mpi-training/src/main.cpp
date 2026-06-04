#include "trainer.hpp"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    TrainConfig cfg;
    std::string output = "../results/mpi-training/scaling.csv";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            cfg.epochs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            cfg.samples = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--features") == 0 && i + 1 < argc) {
            cfg.features = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        }
    }

    const TrainResult result = train_logistic_regression(cfg, rank, size);

    if (rank == 0) {
        FILE* check = std::fopen(output.c_str(), "r");
        const bool needs_header =
            check == nullptr || std::fgetc(check) == EOF;
        if (check) std::fclose(check);

        FILE* fp = std::fopen(output.c_str(), "a");
        if (!fp) {
            std::perror("fopen");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (needs_header) {
            std::fprintf(fp, "ranks,epochs,samples,features,seconds,final_loss\n");
        }

        std::fprintf(fp, "%d,%d,%d,%d,%.6f,%.6f\n", size, result.epochs,
                     result.samples, result.features, result.total_seconds,
                     result.final_loss);
        std::fclose(fp);

        std::printf("MPI ranks=%d  time=%.3fs  loss=%.4f  → %s\n", size,
                    result.total_seconds, result.final_loss, output.c_str());
    }

    MPI_Finalize();
    return 0;
}
