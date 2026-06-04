#!/usr/bin/env python3
"""Sweep the CUDA matmul benchmark over tile dims, block dims, and matrix sizes.

Drives cuda-matmul/build/matmul_bench for every requested (tile_dim, block_dim)
pair, then merges all CSV outputs into a single file ready for plot_matmul.py.

Duplicate naive rows that appear in each tile_dim run are collapsed into one
representative row per (M, N, K, block_dim), using the median timing.

Usage
-----
    # Build first, then sweep with defaults
    python scripts/bench_matmul.py --build

    # Custom sweep
    python scripts/bench_matmul.py \\
        --tile-dims  8,16,32   \\
        --block-dims 16        \\
        --sizes      256,512,1024,2048,4096 \\
        --repeats    10        \\
        --warmup     2         \\
        --check-max  1024      \\
        --output     results/cuda-matmul/matmul_sweep.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "cuda-matmul" / "build" / "matmul_bench"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "cuda-matmul" / "matmul_sweep.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--binary", type=Path, default=DEFAULT_BINARY,
        help="Path to the matmul_bench executable",
    )
    p.add_argument(
        "--sizes", default="256,512,1024,2048,4096",
        help="Comma-separated square matrix sizes [default: 256,512,1024,2048,4096]",
    )
    p.add_argument(
        "--tile-dims", default="8,16,32",
        help="Comma-separated tile dimensions to sweep [default: 8,16,32]",
    )
    p.add_argument(
        "--block-dims", default="16",
        help="Comma-separated naive-kernel block dims to sweep [default: 16]",
    )
    p.add_argument("--repeats",   type=int, default=10)
    p.add_argument("--warmup",    type=int, default=2)
    p.add_argument(
        "--check-max", type=int, default=1024,
        help="Skip CPU correctness check for N > this value [default: 1024]",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Destination CSV for merged results",
    )
    p.add_argument(
        "--build", action="store_true",
        help="Run 'make -C cuda-matmul' before benchmarking",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_binary() -> None:
    print("Building cuda-matmul …")
    result = subprocess.run(
        ["make", "-C", str(REPO_ROOT / "cuda-matmul")],
        check=False,
    )
    if result.returncode != 0:
        print("[error] Build failed.", file=sys.stderr)
        sys.exit(1)
    print()


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_one(
    binary: Path,
    sizes: str,
    tile_dim: int,
    block_dim: int,
    repeats: int,
    warmup: int,
    check_max: int,
    out_csv: Path,
    dry_run: bool,
) -> pd.DataFrame | None:
    cmd = [
        str(binary),
        "--sizes",     sizes,
        "--tile-dim",  str(tile_dim),
        "--block-dim", str(block_dim),
        "--repeats",   str(repeats),
        "--warmup",    str(warmup),
        "--check-max", str(check_max),
        "--output",    str(out_csv),
    ]
    tag = f"tile_dim={tile_dim}  block_dim={block_dim}  sizes={sizes}"
    print(f"  [{tag}]")
    if dry_run:
        print("    (dry-run) " + " ".join(cmd))
        return None

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[warn] Benchmark returned non-zero exit for {tag}", file=sys.stderr)
        return None
    if not out_csv.exists():
        print(f"[warn] Expected output CSV not found: {out_csv}", file=sys.stderr)
        return None
    return pd.read_csv(out_csv)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(combined: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated naive rows that appear once per tile_dim sweep run.

    Tiled rows are unique per (M, N, K, tile_dim) so they are kept as-is.
    Naive rows are unique per (M, N, K, block_dim); we keep the median timing.
    """
    naive = combined[combined["kernel"] == "naive"].copy()
    tiled = combined[combined["kernel"] == "tiled"].copy()

    if naive.empty:
        return tiled

    naive_deduped = (
        naive
        .groupby(["M", "N", "K", "kernel", "block_dim"], as_index=False)
        .agg(
            tile_dim=pd.NamedAgg("tile_dim", "first"),
            ms=pd.NamedAgg("ms", "median"),
            gflops=pd.NamedAgg("gflops", "median"),
            correct=pd.NamedAgg("correct", "first"),
        )
    )
    return pd.concat([naive_deduped, tiled], ignore_index=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.build:
        build_binary()

    if not args.dry_run and not args.binary.exists():
        print(f"[error] Binary not found: {args.binary}", file=sys.stderr)
        print("  Hint: run with --build, or: make -C cuda-matmul", file=sys.stderr)
        sys.exit(1)

    tile_dims  = [int(t) for t in args.tile_dims.split(",")]
    block_dims = [int(b) for b in args.block_dims.split(",")]
    n_runs     = len(tile_dims) * len(block_dims)

    print(f"=== matmul sweep: {n_runs} run(s) ===")
    print(f"  tile_dims  : {tile_dims}")
    print(f"  block_dims : {block_dims}")
    print(f"  sizes      : {args.sizes}")
    print(f"  repeats    : {args.repeats}   warmup: {args.warmup}")
    print(f"  check_max  : {args.check_max}")
    print()

    frames: list[pd.DataFrame] = []
    run_idx = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for tile_dim in tile_dims:
            for block_dim in block_dims:
                run_idx += 1
                tmp_csv = Path(tmpdir) / f"run_t{tile_dim}_b{block_dim}.csv"
                print(f"Run {run_idx}/{n_runs}")
                df = run_one(
                    args.binary, args.sizes,
                    tile_dim, block_dim,
                    args.repeats, args.warmup, args.check_max,
                    tmp_csv, args.dry_run,
                )
                if df is not None:
                    frames.append(df)
                print()

    if args.dry_run:
        print("[dry-run] No results to write.")
        return

    if not frames:
        print("[error] No successful benchmark runs; nothing to save.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    clean    = deduplicate(combined)
    clean    = clean.sort_values(["N", "kernel", "tile_dim", "block_dim"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)

    print(f"Sweep complete → {args.output}")
    print(f"  {len(frames)} run(s) merged  →  {len(clean)} rows\n")
    print(clean.to_string(index=False))


if __name__ == "__main__":
    main()
