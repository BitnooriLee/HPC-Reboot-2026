# roofline-analysis — CPU peak FLOPS and memory bandwidth probes
CXX        ?= g++
CXXFLAGS   := -O3 -std=c++17 -Wall -march=native
LDFLAGS    :=
INCLUDES   := -Iinclude
SRC        := src/main.cpp src/benchmarks.cpp
BUILD_DIR  := build
BIN        := $(BUILD_DIR)/roofline_bench
RESULTS    := ../results/roofline-analysis

# Uncomment for OpenMP parallel kernels
# CXXFLAGS += -fopenmp
# LDFLAGS  += -fopenmp

.PHONY: all clean run dirs

all: dirs $(BIN)

dirs:
	@mkdir -p $(BUILD_DIR) $(RESULTS)

$(BIN): $(SRC) include/benchmarks.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(LDFLAGS) -o $@

run: all
	./$(BIN) --output $(RESULTS)/roofline_points.csv

clean:
	rm -rf $(BUILD_DIR)
