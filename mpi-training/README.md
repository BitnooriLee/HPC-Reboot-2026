# mpi-training

Distributed **logistic regression** training with MPI Allreduce for gradient
synchronization, plus a standalone **Allreduce microbenchmark** that
characterises collective communication latency and bandwidth across message
sizes.

---

## Contents

| Source file | What it does |
|---|---|
| `src/trainer.cpp` | Synchronous data-parallel logistic regression (SGD + Allreduce) |
| `src/main.cpp` | CLI wrapper for the training loop |
| `src/allreduce_bench.cpp` | Standalone Allreduce latency/bandwidth sweep |

---

## Build

```bash
make          # builds both train_mpi and allreduce_bench
```

Requires `mpicxx` (Open MPI or MPICH).

---

## Logistic Regression Training

Implements synchronous mini-batch SGD: each rank owns a shard of the
synthetic dataset, computes local gradients, and calls `MPI_Allreduce` to
average them before the weight update. Measures **strong-scaling efficiency**
as rank count increases from 1 to 2, 4, 8, …

### Run

```bash
make run NP=4 EPOCHS=20 SAMPLES=65536 FEATURES=128
# CSV → ../results/mpi-training/scaling.csv
```

### Plot

```bash
python ../scripts/plot_mpi_scaling.py
```

### Parameters

| Variable | Default | Meaning |
|---|---|---|
| `NP` | 4 | MPI ranks |
| `EPOCHS` | 20 | Training epochs |
| `SAMPLES` | 65536 | Synthetic dataset size |
| `FEATURES` | 128 | Feature dimension (= gradient vector length) |

---

## Allreduce Microbenchmark (`src/allreduce_bench.cpp`)

A focused collective communication benchmark that sweeps message sizes from
**8 B (1 double) to 32 MB (4 M doubles)** and reports:

- **Median latency (µs)** — computed from the slowest rank per repetition to
  reflect the wall-clock cost seen by the training loop.
- **Minimum latency (µs)** — best-case single-call time.
- **Effective bandwidth (GB/s)** — `msg_bytes / median_s`, using the raw
  message size as a conservative denominator.

Each size gets 10 warm-up calls before `--reps` timed measurements, ensuring
the collective library's internal buffers are warmed up.

### Run

```bash
make bench NP=4 REPS=200
# CSV → ../results/mpi-training/allreduce_bench.csv
```

### Parameters

| Variable | Default | Meaning |
|---|---|---|
| `NP` | 4 | MPI ranks |
| `REPS` | 200 | Timed repetitions per message size |

### Expected output

```
MPI Allreduce benchmark — ranks=4  reps=200
msg_bytes     median_us       min_us          bw_gbs
----------------------------------------------------------
8             3.20            2.80            0.0025
64            3.40            2.95            0.0188
512           3.80            3.20            0.1349
4096          5.10            4.30            0.8031
32768         9.80            8.50            3.3436
262144        48.00           43.00           5.4613
1048576       176.00          162.00          5.9588
4194304       690.00          640.00          6.0769
8388608       1380.00         1280.00         6.0786
33554432      5600.00         5100.00         5.9918
```

*Numbers above are illustrative; actual results depend on network fabric,
MPI implementation, and process placement.*

The CSV schema is:

```
ranks,msg_bytes,median_us,min_us,bandwidth_gbs
```

---

## Comparison with DAT400 MPI Neural Network Lab

> **Placeholder — to be filled in after retrieving the original DAT400 lab
> materials.**

The DAT400 course at the University of Stavanger included an MPI lab in which
students implemented a distributed neural network training loop using
point-to-point or collective communication. This section will document a
direct empirical comparison between that original implementation and the
benchmarks here.

Planned comparison axes:

| Dimension | DAT400 lab | This benchmark |
|---|---|---|
| Model | *Neural network (TBD)* | Logistic regression |
| Communication | *TBD (P2P or Allreduce)* | `MPI_Allreduce` (gradient avg.) |
| Dataset | *TBD (course dataset)* | Synthetic (splitmix64) |
| Gradient vector size | *TBD* | 128 doubles (1 KB) |
| Measured metric | *TBD* | Strong-scaling efficiency, Allreduce latency |
| Hardware | *UiS cluster (TBD specs)* | Local dev machine + HPC node |

Once the DAT400 materials are available, this section will include:

1. Side-by-side scaling curves (speedup vs. NP) for both implementations.
2. Per-epoch timing breakdown: compute vs. communication fraction.
3. Allreduce latency at the gradient vector sizes used by the neural network,
   read directly from `allreduce_bench.csv`.
4. Notes on algorithmic or implementation differences that explain any
   performance gap.

---

## TODO

- [ ] Weak scaling mode (grow data with ranks)
- [ ] Non-blocking collectives (`MPI_Iallreduce`)
- [ ] Real dataset loader (LIBSVM format)
- [ ] Add `scripts/plot_allreduce_bench.py` to plot latency vs. message size
- [ ] Fill in DAT400 comparison section once lab materials are retrieved
