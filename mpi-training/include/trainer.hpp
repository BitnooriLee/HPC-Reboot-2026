#pragma once

#include <vector>

struct TrainConfig {
    int epochs = 20;
    int samples = 65536;
    int features = 128;
    double learning_rate = 0.01;
    unsigned seed = 42;
};

struct TrainResult {
    double total_seconds = 0.0;
    double final_loss = 0.0;
    int epochs = 0;
    int samples = 0;
    int features = 0;
};

TrainResult train_logistic_regression(const TrainConfig& cfg, int rank, int size);
