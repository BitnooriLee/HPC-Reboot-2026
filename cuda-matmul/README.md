# cuda-matmul

GPU dense matrix multiply micro-benchmark comparing **naive** and **tiled** CUDA kernels.

## Build

```bash
make                    # default arch sm_75
make CUDA_ARCH=sm_80    # target specific GPU
```

## Run

```bash
make run
# CSV → ../results/cuda-matmul/matmul_timing.csv
```

## Kernels

| Kernel | Description |
|--------|-------------|
| `matmul_naive` | One thread per output element |
| `matmul_tiled` | Shared-memory tile blocking |

## Plot

```bash
python ../scripts/plot_matmul.py
```

## TODO

- [ ] Add cuBLAS baseline
- [ ] Tensor Core WMMA variant
- [ ] NCU roofline export
