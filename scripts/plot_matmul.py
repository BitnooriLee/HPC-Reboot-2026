#!/usr/bin/env python3
"""Plot CUDA matmul benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results" / "cuda-matmul" / "matmul_timing.csv"
DEFAULT_FIG = REPO_ROOT / "results" / "cuda-matmul" / "matmul_gflops.png"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: make -C cuda-matmul run"
        )
    return pd.read_csv(path)


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for kernel, group in df.groupby("kernel"):
        group = group.sort_values("N")
        ax.plot(group["N"], group["gflops"], marker="o", label=kernel)

    ax.set_xlabel("Matrix size N (M=N=K)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("CUDA Matrix Multiply — GFLOPS vs. Problem Size")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()

    df = load_csv(args.csv)
    plot(df, args.output)


if __name__ == "__main__":
    main()
