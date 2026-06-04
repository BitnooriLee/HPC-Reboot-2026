# Results

Benchmark outputs land here as CSV (and optionally JSON metadata).

```
results/
├── cuda-matmul/
│   └── matmul_timing.csv
├── mpi-training/
│   └── scaling.csv
└── roofline-analysis/
    └── roofline_points.csv
```

Generated artifacts are gitignored by default. Commit only small reference CSVs if you need pinned baselines for CI.
