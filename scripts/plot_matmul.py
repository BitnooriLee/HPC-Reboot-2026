#!/usr/bin/env python3
"""Plot CUDA matmul benchmark results: GFLOPS, runtime (ms), and speedup.

Reads the sweep CSV produced by bench_matmul.py (or falls back to the
single-run matmul_timing.csv from 'make -C cuda-matmul run').

Outputs
-------
  matmul_gflops.png   — GFLOPS vs matrix size, all kernels / tile-dims
  matmul_runtime.png  — Kernel runtime (ms) vs matrix size
  matmul_speedup.png  — Speedup of tiled over naive, per tile dimension
  matmul_summary.png  — All three panels in one figure

Usage
-----
    python scripts/plot_matmul.py
    python scripts/plot_matmul.py --csv results/cuda-matmul/matmul_sweep.csv
    python scripts/plot_matmul.py --output-dir results/cuda-matmul
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS   = REPO_ROOT / "results" / "cuda-matmul"

_SWEEP_CSV  = RESULTS / "matmul_sweep.csv"
_SINGLE_CSV = RESULTS / "matmul_timing.csv"

# Colour palette: keeps naive dashed/grey, tiled solid/coloured
_NAIVE_COLOUR = "#888888"
_TILED_COLOURS = ["#2196F3", "#4CAF50", "#FF5722"]  # blue, green, red-orange
_MARKERS_NAIVE = "o"
_MARKERS_TILED = ["s", "^", "D"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path: Path | None) -> pd.DataFrame:
    """Load sweep or single-run CSV; normalise column names."""
    candidates = [path] if path else [_SWEEP_CSV, _SINGLE_CSV]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            df = pd.read_csv(candidate)
            # Normalise old format ('seconds') to new format ('ms')
            if "seconds" in df.columns and "ms" not in df.columns:
                df["ms"] = df["seconds"] * 1000.0
            # Add synthetic tile_dim / block_dim for old CSVs that lack them
            if "tile_dim" not in df.columns:
                df["tile_dim"] = 16
            if "block_dim" not in df.columns:
                df["block_dim"] = 16
            print(f"Loaded {candidate}  ({len(df)} rows)")
            return df

    tried = ", ".join(str(c) for c in candidates if c is not None)
    raise FileNotFoundError(
        f"No results CSV found. Tried: {tried}\n"
        "Run: python scripts/bench_matmul.py --build\n"
        "  or: make -C cuda-matmul run"
    )


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def _naive_label(block_dim: int) -> str:
    return f"naive  (blk {block_dim}×{block_dim})"


def _tiled_label(tile_dim: int) -> str:
    return f"tiled  (tile {tile_dim}×{tile_dim})"


def _fmt_n(x: float, _pos: int) -> str:
    return f"{int(x)}"


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------
def plot_gflops(ax: plt.Axes, df: pd.DataFrame) -> None:
    """GFLOPS vs N for every (kernel, tile_dim / block_dim) group."""
    naive = df[df["kernel"] == "naive"]
    tiled = df[df["kernel"] == "tiled"]

    for (block_dim,), grp in naive.groupby(["block_dim"]):
        grp = grp.sort_values("N")
        ax.plot(
            grp["N"], grp["gflops"],
            marker=_MARKERS_NAIVE, linestyle="--", color=_NAIVE_COLOUR,
            label=_naive_label(block_dim),
        )

    for idx, ((tile_dim,), grp) in enumerate(tiled.groupby(["tile_dim"])):
        grp = grp.sort_values("N")
        ax.plot(
            grp["N"], grp["gflops"],
            marker=_MARKERS_TILED[idx % len(_MARKERS_TILED)],
            color=_TILED_COLOURS[idx % len(_TILED_COLOURS)],
            label=_tiled_label(tile_dim),
        )

    ax.set_xlabel("Matrix size N  (M = N = K)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Peak Performance")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_n))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_runtime(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Kernel runtime (ms) vs N on a log-log scale."""
    naive = df[df["kernel"] == "naive"]
    tiled = df[df["kernel"] == "tiled"]

    for (block_dim,), grp in naive.groupby(["block_dim"]):
        grp = grp.sort_values("N")
        ax.plot(
            grp["N"], grp["ms"],
            marker=_MARKERS_NAIVE, linestyle="--", color=_NAIVE_COLOUR,
            label=_naive_label(block_dim),
        )

    for idx, ((tile_dim,), grp) in enumerate(tiled.groupby(["tile_dim"])):
        grp = grp.sort_values("N")
        ax.plot(
            grp["N"], grp["ms"],
            marker=_MARKERS_TILED[idx % len(_MARKERS_TILED)],
            color=_TILED_COLOURS[idx % len(_TILED_COLOURS)],
            label=_tiled_label(tile_dim),
        )

    ax.set_xlabel("Matrix size N  (M = N = K)")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Kernel Runtime")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_n))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.3g}")
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


def plot_speedup(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Speedup of tiled over naive (naive_ms / tiled_ms) per tile_dim."""
    naive = df[df["kernel"] == "naive"]
    tiled = df[df["kernel"] == "tiled"]

    if naive.empty or tiled.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Use the smallest block_dim naive run as the baseline (typically the only one)
    baseline_bdim = int(naive["block_dim"].min())
    naive_ms = (
        naive[naive["block_dim"] == baseline_bdim]
        .sort_values("N")
        .set_index("N")["ms"]
    )

    for idx, ((tile_dim,), grp) in enumerate(tiled.groupby(["tile_dim"])):
        grp = grp.sort_values("N").set_index("N")
        shared_n = naive_ms.index.intersection(grp.index)
        if shared_n.empty:
            continue
        speedup = naive_ms.loc[shared_n] / grp.loc[shared_n, "ms"]
        ax.plot(
            speedup.index, speedup.values,
            marker=_MARKERS_TILED[idx % len(_MARKERS_TILED)],
            color=_TILED_COLOURS[idx % len(_TILED_COLOURS)],
            label=_tiled_label(tile_dim),
        )

    # Reference line at speedup = 1×
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="1× (no gain)")
    ax.set_xlabel("Matrix size N  (M = N = K)")
    ax.set_ylabel("Speedup  (naive ms / tiled ms)")
    ax.set_title("Tiled Speedup over Naive")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_n))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help=(
            "Input CSV path. "
            f"Defaults to {_SWEEP_CSV} (falls back to {_SINGLE_CSV})"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS,
        help=f"Directory for output PNG files [default: {RESULTS}]",
    )
    parser.add_argument("--dpi", type=int, default=150,
                        help="Figure DPI [default: 150]")
    args = parser.parse_args()

    df = load_csv(args.csv)
    out = args.output_dir

    # ---- Summary figure (3 panels) ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "CUDA Matrix Multiply — Performance Sweep",
        fontsize=13, fontweight="bold",
    )
    plot_gflops(axes[0], df)
    plot_runtime(axes[1], df)
    plot_speedup(axes[2], df)
    fig.tight_layout()
    _save(fig, out / "matmul_summary.png", args.dpi)
    plt.close(fig)

    # ---- Individual figures -------------------------------------------------
    plots = [
        ("matmul_gflops.png",  "GFLOPS vs Matrix Size",        plot_gflops),
        ("matmul_runtime.png", "Kernel Runtime vs Matrix Size", plot_runtime),
        ("matmul_speedup.png", "Tiled Speedup over Naive",      plot_speedup),
    ]
    for fname, title, fn in plots:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fn(ax2, df)
        fig2.suptitle(title, fontsize=11, fontweight="bold")
        fig2.tight_layout()
        _save(fig2, out / fname, args.dpi)
        plt.close(fig2)


if __name__ == "__main__":
    main()
