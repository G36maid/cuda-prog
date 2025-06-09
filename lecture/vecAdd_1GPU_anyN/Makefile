CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v

BIN     := vecAdd_anyN
LIB     :=

all: $(BIN)

$(BIN): vecAdd_anyN.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) vecAdd_anyN.cu $(LIB)

clean:
	rm -f $(BIN)
