CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v
OMPFLAGS := -Xcompiler -fopenmp -lgomp

BIN     := poisson_3d
LIB     :=

all: $(BIN)

$(BIN): poisson_3d.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) $(OMPFLAGS) poisson_3d.cu $(LIB)

clean:
	rm -f $(BIN)
