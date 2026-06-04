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

.PHONY: all clean run dirs

all: dirs $(BIN)

dirs:
	@mkdir -p $(BUILD_DIR) $(RESULTS)

$(BIN): $(SRC) include/kernels.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRC) -o $@

run: all
	./$(BIN) --output $(RESULTS)/matmul_timing.csv

clean:
	rm -rf $(BUILD_DIR)
