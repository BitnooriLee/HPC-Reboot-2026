#include "trainer.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

}  // namespace

TrainResult train_logistic_regression(const TrainConfig& cfg, int rank, int size) {
    const int local_samples = (cfg.samples + size - 1) / size;
    const int start = rank * local_samples;
    const int end = std::min(start + local_samples, cfg.samples);
    const int count = std::max(0, end - start);

    std::vector<double> X(static_cast<size_t>(count) * cfg.features);
    std::vector<double> y(count);

    uint64_t rng = cfg.seed + static_cast<uint64_t>(rank);
    for (int i = 0; i < count; ++i) {
        double dot = 0.0;
        for (int f = 0; f < cfg.features; ++f) {
            const double v = (splitmix64(rng) % 1000) / 500.0 - 1.0;
            X[static_cast<size_t>(i) * cfg.features + f] = v;
            dot += v;
        }
        y[i] = dot > 0.0 ? 1.0 : 0.0;
    }

    std::vector<double> weights(cfg.features, 0.0);
    double final_loss = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::vector<double> grad(cfg.features, 0.0);
        double loss = 0.0;

        for (int i = 0; i < count; ++i) {
            double logit = 0.0;
            for (int f = 0; f < cfg.features; ++f) {
                logit += weights[f] * X[static_cast<size_t>(i) * cfg.features + f];
            }
            const double p = sigmoid(logit);
            const double err = p - y[i];
            loss += -y[i] * std::log(p + 1e-12) -
                    (1.0 - y[i]) * std::log(1.0 - p + 1e-12);
            for (int f = 0; f < cfg.features; ++f) {
                grad[f] += err * X[static_cast<size_t>(i) * cfg.features + f];
            }
        }

        const int grad_count = static_cast<int>(grad.size());
        MPI_Allreduce(MPI_IN_PLACE, grad.data(), grad_count, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        double global_loss = loss;
        MPI_Allreduce(MPI_IN_PLACE, &global_loss, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        final_loss = global_loss / cfg.samples;

        const double scale = cfg.learning_rate / cfg.samples;
        for (int f = 0; f < cfg.features; ++f) {
            weights[f] -= scale * grad[f];
        }
    }

    const double t1 = MPI_Wtime();

    TrainResult result;
    result.total_seconds = t1 - t0;
    result.final_loss = final_loss;
    result.epochs = cfg.epochs;
    result.samples = cfg.samples;
    result.features = cfg.features;
    return result;
}
