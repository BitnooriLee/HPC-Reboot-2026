# mpi-training

Distributed **logistic regression** training with MPI Allreduce for gradient synchronization. Measures strong-scaling efficiency vs. process count.

## Build

```bash
make
```

Requires `mpicxx` (Open MPI or MPICH).

## Run

```bash
make run NP=4 EPOCHS=20
# CSV → ../results/mpi-training/scaling.csv
```

For a scaling sweep, run multiple times with different `NP` and append rows, or use a job script.

## Plot

```bash
python ../scripts/plot_mpi_scaling.py
```

## Parameters

| Variable | Default | Meaning |
|----------|---------|---------|
| `NP` | 4 | MPI ranks |
| `EPOCHS` | 20 | Training epochs |
| `SAMPLES` | 65536 | Synthetic dataset size |
| `FEATURES` | 128 | Feature dimension |

## TODO

- [ ] Weak scaling mode (grow data with ranks)
- [ ] Non-blocking collectives
- [ ] Real dataset loader
