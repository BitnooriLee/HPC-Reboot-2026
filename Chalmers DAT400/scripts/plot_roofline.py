#!/usr/bin/env python3
"""Plot CPU roofline chart from benchmark CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results" / "roofline-analysis" / "roofline_points.csv"
DEFAULT_FIG = REPO_ROOT / "results" / "roofline-analysis" / "roofline.png"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: make -C roofline-analysis run"
        )
    return pd.read_csv(path)


def plot(df: pd.DataFrame, out: Path) -> None:
    flops_row = df[df["kernel"] == "flops_peak"].iloc[0]
    stream_row = df[df["kernel"] == "stream_triad"].iloc[0]

    peak_gflops = float(flops_row["gflops"])
    peak_gbps = float(stream_row["gbps"])
    peak_gflops_from_bw = peak_gbps * 1e9 / 1e9  # GB/s → GFLOPS at 1 byte/FLOP

    intensities = np.logspace(-2, 2, 200)
    memory_roof = np.minimum(peak_gbps * intensities, peak_gflops)
    compute_roof = np.full_like(intensities, peak_gflops)
    roof = np.minimum(memory_roof, compute_roof)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(intensities, roof, "k-", linewidth=2, label="roofline")
    ax.axhline(peak_gflops, color="gray", linestyle=":", alpha=0.6)
    ax.axhline(peak_gbps, color="gray", linestyle=":", alpha=0.6,
               label=f"bandwidth ceiling ({peak_gbps:.1f} GB/s)")

    for _, row in df.iterrows():
        intensity = max(float(row["intensity"]), 1e-3)
        if row["is_bandwidth"]:
            perf = float(row["gbps"]) * intensity
            ax.loglog(intensity, perf, "s", markersize=10, label=row["kernel"])
        else:
            ax.loglog(intensity, float(row["gflops"]), "o", markersize=10,
                      label=row["kernel"])

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title("CPU Roofline")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
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
