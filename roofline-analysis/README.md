# roofline-analysis

CPU micro-benchmarks to estimate **peak FLOPS** (compute-bound) and **memory bandwidth** (stream-like) for roofline charts.

## Build

```bash
make
# Optional OpenMP: edit Makefile to enable -fopenmp
```

## Run

```bash
make run
# CSV → ../results/roofline-analysis/roofline_points.csv
```

## Kernels

| Kernel | Bound | Metric |
|--------|-------|--------|
| `flops_peak` | Compute | GFLOPS from FMA chain |
| `stream_triad` | Memory | GB/s (read+read+write) |

## Plot

```bash
python ../scripts/plot_roofline.py
```

## Interpreting

Plot arithmetic intensity (FLOPs / bytes) against achievable performance. Compare application kernels as points between the compute and memory roofs.

## TODO

- [ ] GPU roofline via CUDA memcpy + GEMM
- [ ] Cache-aware intensity sweep
- [ ] LIKWID integration
