#pragma once

struct RooflinePoint {
    const char* name;
    double intensity;   // FLOPs per byte moved
    double gflops;      // or GB/s for bandwidth-dominated
    bool is_bandwidth;  // true → plot on bandwidth axis
};

void run_flops_peak(double* out_seconds, double* out_gflops, int repeats);
void run_stream_triad(double* out_seconds, double* out_gbps, int n, int repeats);
