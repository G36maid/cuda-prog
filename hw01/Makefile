CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v

BIN     := vecAdd
LIB     :=

all: $(BIN)

$(BIN): vecAdd.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) vecAdd.cu $(LIB)

clean:
	rm -f $(BIN)
