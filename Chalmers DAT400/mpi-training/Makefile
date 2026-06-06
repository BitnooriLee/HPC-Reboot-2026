# mpi-training — distributed mini-batch SGD scaling study + Allreduce benchmark
MPICXX     ?= mpicxx
CXXFLAGS   := -O3 -std=c++17 -Wall
INCLUDES   := -Iinclude
SRC        := src/main.cpp src/trainer.cpp
BUILD_DIR  := build
BIN        := $(BUILD_DIR)/train_mpi
BENCH_BIN  := $(BUILD_DIR)/allreduce_bench
RESULTS    := ../results/mpi-training
NP         ?= 4
EPOCHS     ?= 20
SAMPLES    ?= 65536
FEATURES   ?= 128
REPS       ?= 200

.PHONY: all clean run bench dirs

all: dirs $(BIN) $(BENCH_BIN)

dirs:
	@mkdir -p $(BUILD_DIR) $(RESULTS)

$(BIN): $(SRC) include/trainer.hpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $@

$(BENCH_BIN): src/allreduce_bench.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

run: all
	mpirun -np $(NP) ./$(BIN) \
		--epochs $(EPOCHS) --samples $(SAMPLES) --features $(FEATURES) \
		--output $(RESULTS)/scaling.csv

bench: dirs $(BENCH_BIN)
	mpirun -np $(NP) ./$(BENCH_BIN) \
		--reps $(REPS) \
		--output $(RESULTS)/allreduce_bench.csv

clean:
	rm -rf $(BUILD_DIR)
