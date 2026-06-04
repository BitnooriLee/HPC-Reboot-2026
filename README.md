# HPC-Reboot-2026

A structured self-study and benchmarking workspace revisiting the core topics of
**DAT400 High-Performance Computing** in preparation for
[ISC High Performance 2026](https://www.isc-hpc.com/) (Hamburg, June 2026).
The repository experiments with **CUDA kernel optimisation**, **MPI distributed
training**, and **roofline-model characterisation**, producing reproducible plots
and CSVs from every run.

---

## Motivation

DAT400 at the University of Stavanger introduced the foundational machinery of
modern HPC: GPU thread hierarchies, distributed-memory communication patterns,
the memory wall, and performance-bound reasoning. ISC 2026 is an opportunity to
engage with that material again — not as coursework, but as a practitioner
asking sharper questions:

- Where exactly does a naive CUDA GEMM leave performance on the table, and how
  much of that gap does shared-memory tiling recover?
- What is the practical strong-scaling ceiling for a synchronous gradient-reduce
  workload, and when does Amdahl's law become the binding constraint?
- What are the measured compute and bandwidth ceilings of the development
  machine, and how do application kernels sit relative to that roofline?

The goal is to arrive at ISC with concrete numbers, reproducible tooling, and a
cleaner mental model of the hardware/software interface.

---

## Repository Layout

```
HPC-Reboot-2026/
├── cuda-matmul/          # GPU dense matrix multiply (naive vs tiled)
│   ├── include/kernels.cuh
│   └── src/{kernels,main}.cu
├── mpi-training/         # Distributed logistic regression via MPI Allreduce
│   ├── include/trainer.hpp
│   └── src/{trainer,main}.cpp
├── roofline-analysis/    # CPU peak-FLOPS and memory-bandwidth probes
│   ├── include/benchmarks.hpp
│   └── src/{benchmarks,main}.cpp
├── scripts/              # Python helpers: plotting + sweep orchestration
│   ├── bench_matmul.py
│   ├── plot_matmul.py
│   ├── plot_mpi_scaling.py
│   └── plot_roofline.py
├── results/              # CSV / JSON benchmark outputs (gitignored by default)
│   └── roofline-analysis/roofline_points.csv
├── docs/
│   ├── methodology.md    # Timing, correctness, and reporting conventions
│   └── run-log.md        # Per-run hardware and compiler notes
└── Makefile              # Top-level orchestration
```

---

## Architecture

### 1 — CUDA Matrix Multiply (`cuda-matmul/`)

Two CUDA GEMM kernels benchmark the impact of the GPU memory hierarchy:

| Kernel | Strategy | Expected bottleneck |
|---|---|---|
| `matmul_naive` | One thread per output element; reads A and B directly from global memory | Global-memory bandwidth (high DRAM traffic, low reuse) |
| `matmul_tiled` | 2-D thread blocks load shared-memory tiles; each element of A and B is loaded once per tile | Shared-memory throughput and occupancy |

Both kernels compute **C = A · B** for square and rectangular row-major
matrices. Timing uses CUDA events placed around the kernel launch; host
allocation and data transfer are measured separately so device-only throughput
can be reported cleanly.

Thread-block dimensions and tile sizes are runtime parameters (`block_dim`,
`tile_dim` ∈ {8, 16, 32}), enabling a full sweep without recompilation.

### 2 — MPI Distributed Training (`mpi-training/`)

Implements synchronous data-parallel **logistic regression** over a synthetic
dataset:

1. Each MPI rank owns an equal shard of the training data.
2. Each rank computes local gradients for one mini-batch.
3. `MPI_Allreduce` averages gradients across all ranks.
4. All ranks apply the same parameter update (SGD).

This mirrors the `AllReduce`-based pattern used by frameworks such as Horovod
and PyTorch DDP, making it a clean minimal model for studying communication
overhead as a function of rank count, message size (gradient vector), and
network topology.

Timing wraps the full training loop with `MPI_Wtime()`; rank 0 collects and
reports aggregate statistics. The benchmark targets **strong-scaling
efficiency** — the training dataset is fixed while the number of ranks
increases from 1 to 2, 4, 8, …

### 3 — Roofline Analysis (`roofline-analysis/`)

Two micro-kernels bracket the CPU performance envelope:

| Kernel | Bound | What it measures |
|---|---|---|
| `flops_peak` | Compute | Sustained GFLOPS from a long FMA dependency chain |
| `stream_triad` | Memory | Achievable memory bandwidth (GB/s) via STREAM-Triad pattern |

Results are emitted as `(arithmetic_intensity, performance)` pairs which become
anchor points on the roofline chart. Application kernels can then be overlaid
as additional points to visualise where they sit relative to the compute roof
and bandwidth slope.

---

## Benchmark Methodology

### Timing

| Subsystem | Mechanism | What is excluded |
|---|---|---|
| CUDA kernels | `cudaEventRecord` around `<<<>>>` launch | Host allocation, H↔D transfers (measured separately) |
| MPI training | `MPI_Wtime()` around training loop | Rank initialisation, dataset generation |
| Roofline probes | Wall-clock (`clock_gettime`) over N=100 repetitions, report median | Cold-cache effects on first iteration |

### Correctness Checks

- **CUDA GEMM**: output compared against CPU reference for N ≤ 1024; relative
  error must be below 1 × 10⁻⁴.
- **MPI training**: on a fixed random seed with NP = 1, loss must decrease
  monotonically across epochs.
- **Roofline probes**: measured GFLOPS must be within one order of magnitude of
  the processor's published peak.

### Reproducibility

Every benchmark writes raw numbers to `results/<benchmark>/`. A sidecar JSON
records the compiler, flags, matrix size or rank count, and host hardware at
the time of the run. Figures are regenerated deterministically from those CSVs
using the scripts in `scripts/`.

---

## Quick Start

```bash
# Prerequisites: nvcc ≥ 12, mpicxx (Open MPI or MPICH), g++ with -fopenmp, Python 3.10+

# Build everything
make all

# Single-config run across all benchmarks
make run

# CUDA tile-dimension sweep (TILE ∈ {8, 16, 32} × SIZE ∈ {256…4096})
make sweep
# Equivalently:
make -C cuda-matmul sweep SWEEP_TILE_DIMS=8,16,32 SWEEP_SIZES=256,512,1024,2048,4096

# MPI strong-scaling run
make -C mpi-training run NP=8 EPOCHS=20 SAMPLES=65536 FEATURES=128

# Roofline probes
make -C roofline-analysis run

# Regenerate all figures
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
make plot
```

Override the target CUDA architecture at build time:

```bash
make CUDA_ARCH=sm_86   # Ampere (RTX 30xx / A-series)
make CUDA_ARCH=sm_90   # Hopper (H100)
```

---

## Expected Results

### CUDA Matrix Multiply

| Configuration | Expected GFLOPS (sm_75 baseline) | Notes |
|---|---|---|
| Naive, 4096×4096 | ~150–400 | Heavily memory-bound; scales weakly with N |
| Tiled (tile=16), 4096×4096 | ~2 000–5 000 | Shared-memory reuse cuts DRAM pressure |
| Tiled (tile=32), 4096×4096 | ~3 000–8 000 | Larger tile, higher register pressure |
| cuBLAS (planned) | ~10 000+ | Vendor-tuned; serves as upper bound |

Tiling is expected to deliver a **5–20× throughput improvement** over the naive
kernel at large matrix sizes, with the exact factor depending on occupancy,
shared-memory bank conflicts, and warp-level instruction-level parallelism.

### MPI Strong Scaling

Ideal (Amdahl) efficiency assumes the workload is embarrassingly parallel.
In practice, `MPI_Allreduce` latency and gradient-vector size (128 floats =
512 B per rank) are small, so near-linear scaling is expected up to ≈ 8 ranks
on a single node. Efficiency should remain above 80 % at NP = 4 and begin
degrading noticeably beyond NP = 16 due to collective overhead.

| NP | Expected speedup | Expected efficiency |
|---|---|---|
| 1 | 1.0× | 100 % |
| 2 | ~1.9× | ~95 % |
| 4 | ~3.6× | ~90 % |
| 8 | ~6.5× | ~81 % |
| 16 | ~10–12× | ~65–75 % |

### Roofline

Initial measurements on the development machine (Apple M-series / x86_64 host):

| Kernel | Arithmetic intensity (FLOPs/byte) | Performance |
|---|---|---|
| `flops_peak` | 0.083 | 5.6 GFLOPS |
| `stream_triad` | 0.083 | 72.6 GB/s |

These form the two anchor points of the roofline. The ridge point — where the
bandwidth-limited slope intersects the compute ceiling — sits at approximately
**0.077 FLOPs/byte** on this hardware, meaning any kernel with arithmetic
intensity above that value is compute-bound.

---

## Environment

| Component | Minimum | Notes |
|---|---|---|
| CUDA | `nvcc` ≥ 12.0 | Tested on sm_75 (Turing), sm_80 (Ampere) |
| MPI | Open MPI ≥ 4.0 or MPICH ≥ 3.4 | `mpicxx` must be on `PATH` |
| C++ | GCC ≥ 11 or Clang ≥ 14 | `-std=c++17`; `-fopenmp` optional |
| Python | 3.10+ | `matplotlib`, `pandas`, `numpy` (see `scripts/requirements.txt`) |

---

## Future Work

The following extensions are tracked as TODOs in the individual subproject
READMEs; this section provides the strategic context:

**CUDA**
- Add a **cuBLAS baseline** to anchor the upper performance bound and quantify
  the remaining gap between hand-tuned tiling and vendor libraries.
- Implement a **Tensor Core WMMA variant** (`wmma::mma_sync`) to explore
  mixed-precision (FP16 accumulate in FP32) on Volta/Turing/Ampere.
- Export **Nsight Compute (NCU) roofline data** for the GPU so device kernels
  can be placed on the GPU roofline alongside the CPU measurements.

**MPI**
- Add a **weak-scaling mode** where dataset size grows proportionally with rank
  count, isolating communication overhead from computation load imbalance.
- Experiment with **non-blocking collectives** (`MPI_Iallreduce`) to overlap
  gradient communication with the next forward pass.
- Swap in a **real dataset loader** (e.g. LIBSVM format) to move from synthetic
  micro-benchmarking toward an application-representative workload.

**Roofline**
- Extend probes to the **GPU** using CUDA memcpy (bandwidth) and CUBLAS GEMM
  (compute) to build a device-side roofline.
- Implement a **cache-aware intensity sweep** that gradually increases the
  working-set size to trace the L1 → L2 → LLC → DRAM bandwidth steps on the
  same chart.
- Integrate **LIKWID** hardware counters for a hardware-validated roofline
  rather than an empirical approximation.

**Infrastructure**
- Add a CI job (GitHub Actions) that builds all benchmarks on a self-hosted
  GPU runner and uploads timing CSVs as artifacts.
- Publish a `docs/isc-notes.md` summarising findings and any conference
  takeaways after ISC 2026.

---

## Results Workflow

1. Run a benchmark — it writes a CSV under `results/<benchmark>/`.
2. Record hardware and compiler flags in `docs/run-log.md`.
3. Run the matching `scripts/plot_*.py` to regenerate figures.
4. Commit CSVs and figures together so results are always in sync with the
   generating code.

---

## License

MIT — see individual subproject READMEs for benchmark-specific notes.
