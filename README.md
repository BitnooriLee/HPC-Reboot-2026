# HPC-Reboot-2026

A clean benchmarking workspace for **C**, **C++**, and **CUDA** experiments, plus **MPI** scaling studies and **roofline** characterization. Use this repo to compare kernels, track hardware limits, and publish reproducible plots.

## Layout

| Path | Purpose |
|------|---------|
| [`cuda-matmul/`](cuda-matmul/) | GPU dense matrix multiply (naive vs tiled) |
| [`mpi-training/`](mpi-training/) | Distributed mini-batch SGD over MPI |
| [`roofline-analysis/`](roofline-analysis/) | CPU peak FLOPS and memory bandwidth probes |
| [`scripts/`](scripts/) | Python helpers to plot `results/` |
| [`results/`](results/) | CSV/JSON outputs from benchmarks (gitignored by default) |
| [`docs/`](docs/) | Notes, methodology, and run logs |

## Quick start

```bash
# CUDA matmul (requires nvcc + GPU)
make -C cuda-matmul run

# MPI training (requires mpicxx)
make -C mpi-training run NP=4

# Roofline probes (OpenMP optional)
make -C roofline-analysis run

# Plot latest results
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
python scripts/plot_matmul.py
python scripts/plot_mpi_scaling.py
python scripts/plot_roofline.py
```

## Environment

| Component | Typical toolchain |
|-----------|-------------------|
| CUDA | `nvcc` ≥ 12, CUDA-capable GPU |
| MPI | Open MPI or MPICH (`mpicxx`) |
| CPU roofline | `g++` with `-fopenmp` (optional) |
| Plotting | Python 3.10+, see `scripts/requirements.txt` |

Set `CUDA_ARCH` when building (e.g. `make CUDA_ARCH=sm_80`).

## Results workflow

1. Run a benchmark; it writes CSV under `results/<benchmark>/`.
2. Use the matching script in `scripts/` to generate figures.
3. Document hardware and compiler flags in `docs/run-log.md`.

## License

MIT — see each subproject README for benchmark-specific notes.
