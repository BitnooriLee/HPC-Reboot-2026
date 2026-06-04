#!/usr/bin/env python3
"""Plot MPI training strong-scaling results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results" / "mpi-training" / "scaling.csv"
DEFAULT_FIG = REPO_ROOT / "results" / "mpi-training" / "mpi_scaling.png"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: make -C mpi-training run NP=1,2,4,..."
        )
    return pd.read_csv(path)


def plot(df: pd.DataFrame, out: Path) -> None:
    df = df.sort_values("ranks").drop_duplicates(subset=["ranks"], keep="last")
    t1 = df.loc[df["ranks"] == df["ranks"].min(), "seconds"].iloc[0]
    ideal = t1 / df["ranks"]
    efficiency = (t1 / df["seconds"]) / df["ranks"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(df["ranks"], df["seconds"], "o-", label="measured")
    ax.plot(df["ranks"], ideal, "--", label="ideal (linear)")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Training time (s)")
    ax.set_title("Strong scaling — wall time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(df["ranks"].astype(str), efficiency, color="steelblue")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Parallel efficiency")
    ax.set_ylim(0, 110)

    fig.suptitle("MPI Logistic Regression Training", fontsize=12)
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
    if len(df) < 2:
        print(
            "Warning: need multiple rank counts for scaling plot. "
            "Re-run with different NP= values."
        )
    plot(df, args.output)


if __name__ == "__main__":
    main()
