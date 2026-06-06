# Top-level orchestration for HPC-Reboot-2026 benchmarks

BENCHMARKS := cuda-matmul mpi-training roofline-analysis
PYTHON     ?= python3

.PHONY: all clean run sweep plot help

help:
	@echo "Targets:"
	@echo "  make all              — build all benchmarks"
	@echo "  make run              — single-config run (CUDA/MPI may be skipped)"
	@echo "  make sweep            — tile-dim / block-dim sweep + plot for cuda-matmul"
	@echo "  make plot             — regenerate all figures from existing results/"
	@echo "  make clean            — remove build artifacts"
	@echo ""
	@echo "cuda-matmul sweep overrides:"
	@echo "  SWEEP_TILE_DIMS=8,16,32   SWEEP_SIZES=256,512,1024,2048,4096"

all:
	@for d in $(BENCHMARKS); do $(MAKE) -C $$d || exit 1; done

run:
	@for d in $(BENCHMARKS); do $(MAKE) -C $$d run || true; done

sweep:
	$(MAKE) -C cuda-matmul sweep
	$(MAKE) -C cuda-matmul plot

plot:
	$(PYTHON) scripts/plot_matmul.py
	$(PYTHON) scripts/plot_mpi_scaling.py
	$(PYTHON) scripts/plot_roofline.py

clean:
	@for d in $(BENCHMARKS); do $(MAKE) -C $$d clean; done
