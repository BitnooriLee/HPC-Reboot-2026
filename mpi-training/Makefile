# mpi-training — distributed mini-batch SGD scaling study
MPICXX     ?= mpicxx
CXXFLAGS   := -O3 -std=c++17 -Wall
INCLUDES   := -Iinclude
SRC        := src/main.cpp src/trainer.cpp
BUILD_DIR  := build
BIN        := $(BUILD_DIR)/train_mpi
RESULTS    := ../results/mpi-training
NP         ?= 4
EPOCHS     ?= 20
SAMPLES    ?= 65536
FEATURES   ?= 128

.PHONY: all clean run dirs

all: dirs $(BIN)

dirs:
	@mkdir -p $(BUILD_DIR) $(RESULTS)

$(BIN): $(SRC) include/trainer.hpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $@

run: all
	mpirun -np $(NP) ./$(BIN) \
		--epochs $(EPOCHS) --samples $(SAMPLES) --features $(FEATURES) \
		--output $(RESULTS)/scaling.csv

clean:
	rm -rf $(BUILD_DIR)
