# cuda-matmul — GPU dense GEMM micro-benchmarks
NVCC       ?= nvcc
CUDA_ARCH  ?= sm_75
CXXFLAGS   := -O3 -std=c++17 --expt-relaxed-constexpr
NVCCFLAGS  := $(CXXFLAGS) -arch=$(CUDA_ARCH)
INCLUDES   := -Iinclude
SRC        := src/main.cu src/kernels.cu
BUILD_DIR  := build
BIN        := $(BUILD_DIR)/matmul_bench
RESULTS    := ../results/cuda-matmul

# Configurable benchmark parameters (override on the command line)
TILE_DIM   ?= 16          # shared-memory tile side; must be 8, 16, or 32
BLOCK_DIM  ?= 16          # naive-kernel thread-block side (1–32)
SIZES      ?= 512,1024,2048,4096
CHECK_MAX  ?= 2048        # skip CPU correctness check for N > this; 0 = skip all
REPEATS    ?= 10
WARMUP     ?= 2

# Sweep parameters (used by the 'sweep' target)
SWEEP_TILE_DIMS  ?= 8,16,32
SWEEP_BLOCK_DIMS ?= 16
SWEEP_SIZES      ?= 256,512,1024,2048,4096
SWEEP_CHECK_MAX  ?= 1024
SWEEP_CSV        := $(RESULTS)/matmul_sweep.csv
PYTHON           ?= python3

.PHONY: all clean run sweep plot dirs

all: dirs $(BIN)

dirs:
	@mkdir -p $(BUILD_DIR) $(RESULTS)

$(BIN): $(SRC) include/kernels.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRC) -o $@

run: all
	./$(BIN) \
	  --output    $(RESULTS)/matmul_timing.csv \
	  --sizes     $(SIZES)                      \
	  --tile-dim  $(TILE_DIM)                   \
	  --block-dim $(BLOCK_DIM)                  \
	  --repeats   $(REPEATS)                    \
	  --warmup    $(WARMUP)                     \
	  --check-max $(CHECK_MAX)

sweep: all
	$(PYTHON) ../scripts/bench_matmul.py \
	  --binary    $(BIN)              \
	  --sizes     $(SWEEP_SIZES)      \
	  --tile-dims $(SWEEP_TILE_DIMS)  \
	  --block-dims $(SWEEP_BLOCK_DIMS) \
	  --repeats   $(REPEATS)          \
	  --warmup    $(WARMUP)           \
	  --check-max $(SWEEP_CHECK_MAX)  \
	  --output    $(SWEEP_CSV)

plot:
	$(PYTHON) ../scripts/plot_matmul.py \
	  --csv        $(SWEEP_CSV) \
	  --output-dir $(RESULTS)

clean:
	rm -rf $(BUILD_DIR)
