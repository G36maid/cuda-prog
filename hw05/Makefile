CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v
OMPFLAGS := -Xcompiler -fopenmp -lgomp

BIN     := heatDiff2d
LIB     :=

all: $(BIN)

$(BIN): heatDiff2d.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) $(OMPFLAGS) heatDiff2d.cu $(LIB)

clean:
	rm -f $(BIN)