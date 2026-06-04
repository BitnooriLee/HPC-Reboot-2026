# Methodology

## Timing

- **CUDA**: CUDA events around kernel launches; exclude host allocation unless noted.
- **MPI**: `MPI_Wtime()` around the training loop; rank 0 reports aggregated stats.
- **Roofline**: wall-clock over repeated kernels; report median of N iterations.

## Correctness

- `cuda-matmul`: compare against cuBLAS or CPU reference for small N.
- `mpi-training`: loss should decrease monotonically on fixed seed (single rank).
- `roofline-analysis`: sanity-check FLOPS against theoretical peak order-of-magnitude.

## Reporting

Export raw CSV from each benchmark, then use `scripts/plot_*.py` for figures. Always record compiler flags and problem size in the CSV header comments or sidecar JSON.
