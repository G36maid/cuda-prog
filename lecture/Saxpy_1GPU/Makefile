CUDIR     := /usr/local/nvidia

CC      := gcc
#CFLAGS  := -O3 -Wall -g
CFLAGS  := -Wall -g
CPP     := g++
#CFLAGS  := -O3 -Wall -g
CFLAGS  := -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
#NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v
NVFLAGS := -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v
LIB     := -lcublas

BIN00    := Saxpy_1gpu
BIN01    := Saxpy_1gpu_umem
BIN10    := Saxpy_1gpu_cublas
BIN11    := Saxpy_1gpu_cublas_umem

all: $(BIN00) $(BIN01) $(BIN10) $(BIN11)

$(BIN00): Saxpy_1gpu.cu
	$(NVCC) -o $(BIN00) $(NVFLAGS) Saxpy_1gpu.cu $(LIB)

$(BIN01): Saxpy_1gpu_umem.cu
	$(NVCC) -o $(BIN01) $(NVFLAGS) Saxpy_1gpu_umem.cu $(LIB)

$(BIN10): Saxpy_1gpu_cublas.cu
	$(NVCC) -o $(BIN10) $(NVFLAGS) Saxpy_1gpu_cublas.cu $(LIB)

$(BIN11): Saxpy_1gpu_cublas_umem.cu
	$(NVCC) -o $(BIN11) $(NVFLAGS) Saxpy_1gpu_cublas_umem.cu $(LIB)

clean:
	rm -f $(BIN00) $(BIN01) $(BIN10) $(BIN11)
