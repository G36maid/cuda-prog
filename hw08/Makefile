CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v
OMPFLAGS := -Xcompiler -fopenmp -lgomp

BIN     := Monte_Carlo
LIB     :=

all: $(BIN)

$(BIN): Monte_Carlo.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) $(OMPFLAGS) Monte_Carlo.cu $(LIB)

clean:
	rm -f $(BIN)
