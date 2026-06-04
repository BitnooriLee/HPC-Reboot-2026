# Scripts

Python plotting helpers for benchmark CSVs under `results/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

## Usage

| Script | Input | Output |
|--------|-------|--------|
| `plot_matmul.py` | `results/cuda-matmul/matmul_timing.csv` | `matmul_gflops.png` |
| `plot_mpi_scaling.py` | `results/mpi-training/scaling.csv` | `mpi_scaling.png` |
| `plot_roofline.py` | `results/roofline-analysis/roofline_points.csv` | `roofline.png` |

```bash
python scripts/plot_matmul.py --csv path/to.csv --output fig.png
make plot   # from repo root
```
