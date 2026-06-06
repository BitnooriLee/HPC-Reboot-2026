# HPC-Reboot-2026

A structured self-study workspace revisiting high-performance computing across
two tracks: **Chalmers DAT400** (GPU/MPI/roofline benchmarking) and
**Stanford CS149** (parallel computing coursework). Built in preparation for
[ISC High Performance 2026](https://www.isc-hpc.com/) (Hamburg, June 2026).

---

## Repository Layout

```
HPC-Reboot-2026/
├── Chalmers DAT400/          # Benchmark experiments from DAT400 HPC course
│   ├── cuda-matmul/          #   GPU dense matrix multiply (naive vs tiled)
│   ├── mpi-training/         #   Distributed logistic regression via MPI Allreduce
│   ├── roofline-analysis/    #   CPU peak-FLOPS and memory-bandwidth probes
│   ├── scripts/              #   Python helpers: plotting + sweep orchestration
│   ├── results/              #   CSV / JSON benchmark outputs
│   ├── docs/                 #   Methodology and run logs
│   └── Makefile              #   Top-level orchestration
└── Stanford CS149/           # CS149: Parallel Computing (Stanford, Fall 2024)
    ├── asst1/                #   Performance analysis on a quad-core CPU
    ├── asst2/                #   Task execution library from scratch
    ├── asst3/                #   CUDA renderer
    ├── asst4-trainium2/      #   Programming an ML accelerator (AWS Trainium2)
    ├── asst5-kernels/        #   Custom GPU kernels
    ├── cs149gpt/             #   GPT inference with ISPC
    ├── biggraphs-ec/         #   Graph processing extra credit
    ├── intro_to_cuda/        #   CUDA introductory demos
    ├── cuda_tutorial/        #   CUDA tutorial exercises
    └── kernel-infra/         #   Kernel infrastructure utilities
```

---

## Chalmers DAT400

DAT400 High Performance Parallel Programming at Chalmers University of
Technology introduced the foundational machinery of modern HPC: GPU thread
hierarchies, distributed-memory communication patterns, the memory wall, and
performance-bound reasoning. This track revisits that material with sharper
questions:

- Where exactly does a naive CUDA GEMM leave performance on the table, and how
  much of that gap does shared-memory tiling recover?
- What is the practical strong-scaling ceiling for a synchronous gradient-reduce
  workload, and when does Amdahl's law become the binding constraint?
- What are the measured compute and bandwidth ceilings of the development
  machine, and how do application kernels sit relative to that roofline?

### Modules

#### 1 — CUDA Matrix Multiply (`cuda-matmul/`)

Two CUDA GEMM kernels benchmark the impact of the GPU memory hierarchy:

| Kernel | Strategy | Expected bottleneck |
|---|---|---|
| `matmul_naive` | One thread per output element; reads A and B directly from global memory | Global-memory bandwidth |
| `matmul_tiled` | 2-D thread blocks load shared-memory tiles; each element loaded once per tile | Shared-memory throughput and occupancy |

Thread-block dimensions and tile sizes are runtime parameters (`tile_dim` ∈ {8, 16, 32}),
enabling a full sweep without recompilation.

#### 2 — MPI Distributed Training (`mpi-training/`)

Synchronous data-parallel **logistic regression** over a synthetic dataset.
Each rank owns an equal shard; `MPI_Allreduce` averages gradients; all ranks
apply the same SGD update. Targets **strong-scaling efficiency** as rank count
increases from 1 → 2 → 4 → 8 → 16.

#### 3 — Roofline Analysis (`roofline-analysis/`)

| Kernel | Bound | What it measures |
|---|---|---|
| `flops_peak` | Compute | Sustained GFLOPS from a long FMA dependency chain |
| `stream_triad` | Memory | Achievable memory bandwidth via STREAM-Triad pattern |

Results are emitted as `(arithmetic_intensity, performance)` pairs which anchor
the roofline chart.

### Quick Start (Chalmers DAT400)

```bash
# Prerequisites: nvcc ≥ 12, mpicxx (Open MPI or MPICH), g++ -fopenmp, Python 3.10+

cd "Chalmers DAT400"

# Build everything
make all

# Single-config run across all benchmarks
make run

# CUDA tile-dimension sweep
make sweep

# MPI strong-scaling run
make -C mpi-training run NP=8 EPOCHS=20 SAMPLES=65536 FEATURES=128

# Roofline probes
make -C roofline-analysis run

# Regenerate all figures
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
make plot
```

Override target CUDA architecture:

```bash
make CUDA_ARCH=sm_86   # Ampere (RTX 30xx / A-series)
make CUDA_ARCH=sm_90   # Hopper (H100)
```

---

## Stanford CS149

[CS149: Parallel Computing](https://cs149.stanford.edu) is Stanford University's
core parallel systems course, taught by Profs. Kayvon Fatahalian and Kunle Olukotun.
Course materials and assignment starters are available at
[github.com/stanford-cs149/asst1](https://github.com/stanford-cs149) (and
corresponding repos for each assignment). This track works through the assignments
to build rigorous intuitions around SIMD/SPMD parallelism, task scheduling, GPU
programming, and hardware accelerators.

### Assignments

| # | Title | Key topics |
|---|---|---|
| [asst1](Stanford%20CS149/asst1/) | Performance Analysis on a Quad-Core CPU | Pthreads, ISPC SIMD, roofline, Mandelbrot |
| [asst2](Stanford%20CS149/asst2/) | Building a Task Execution Library | Thread pool, task graph, `ITaskSystem` interface |
| [asst3](Stanford%20CS149/asst3/) | A Simple CUDA Renderer | CUDA circles renderer, parallel prefix scan |
| [asst4](Stanford%20CS149/asst4-trainium2/) | Programming an ML Accelerator | AWS Trainium2, NeuronCore, conv2d / matmul kernels |
| [asst5](Stanford%20CS149/asst5-kernels/) | Custom GPU Kernels | Rust-based kernel authoring framework |
| [cs149gpt](Stanford%20CS149/cs149gpt/) | GPT Inference with ISPC | Attention, FlashAttention, ISPC vectorisation |

### Notes

- `intro_to_cuda/` — CUDA hello-world, convolution, matmul, and error-handling demos
- `cuda_tutorial/` — Structured CUDA tutorial exercises
- `biggraphs-ec/` — Graph processing (BFS/PageRank) extra-credit problem
- `kernel-infra/` — Shared kernel infrastructure and utilities

---

## Benchmark Methodology

### Timing

| Subsystem | Mechanism | What is excluded |
|---|---|---|
| CUDA kernels | `cudaEventRecord` around `<<<>>>` launch | Host allocation, H↔D transfers (measured separately) |
| MPI training | `MPI_Wtime()` around training loop | Rank initialisation, dataset generation |
| Roofline probes | Wall-clock (`clock_gettime`) over N=100 repetitions, report median | Cold-cache effects on first iteration |

### Correctness Checks

- **CUDA GEMM**: output compared against CPU reference for N ≤ 1024; relative error < 1×10⁻⁴.
- **MPI training**: on a fixed random seed with NP=1, loss must decrease monotonically.
- **Roofline probes**: measured GFLOPS must be within one order of magnitude of published peak.

---

## Environment

| Component | Minimum | Notes |
|---|---|---|
| CUDA | `nvcc` ≥ 12.0 | Tested on sm_75 (Turing), sm_80 (Ampere) |
| MPI | Open MPI ≥ 4.0 or MPICH ≥ 3.4 | `mpicxx` must be on `PATH` |
| C++ | GCC ≥ 11 or Clang ≥ 14 | `-std=c++17`; `-fopenmp` optional |
| Python | 3.10+ | `matplotlib`, `pandas`, `numpy` |
| ISPC | ≥ 1.20 | Required for CS149 asst1, asst3, cs149gpt |

---

## License

MIT — see individual subproject READMEs for benchmark-specific notes.
