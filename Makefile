# Top-level orchestration for HPC-Reboot-2026 benchmarks

BENCHMARKS := cuda-matmul mpi-training roofline-analysis

.PHONY: all clean run plot help

help:
	@echo "Targets:"
	@echo "  make all    - build all benchmarks"
	@echo "  make run    - run all (CUDA/MPI may be skipped if unavailable)"
	@echo "  make plot   - generate figures from results/"
	@echo "  make clean  - remove build artifacts"

all:
	@for d in $(BENCHMARKS); do $(MAKE) -C $$d || exit 1; done

run:
	@for d in $(BENCHMARKS); do $(MAKE) -C $$d run || true; done

plot:
	python3 scripts/plot_matmul.py
	python3 scripts/plot_mpi_scaling.py
	python3 scripts/plot_roofline.py

clean:
	@for d in $(BENCHMARKS); do $(MAKE) -C $$d clean; done
