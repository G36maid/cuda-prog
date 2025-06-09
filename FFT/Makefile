CUDIR   := /usr/local/nvidia

CC      := g++
CFLAGS  := -O3 -Wall
NVCC    := nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v -rdc=true
#LIB     := -lcufft -lcudart
LIB     := -lcufft

BIN1    := fft_cpu
BIN2    := fft_cuFFT

all: $(BIN1) $(BIN2)

$(BIN1): fft_cpu.cpp
	$(CC) -o $(BIN1) $(CFLAGS) $<

$(BIN2): fft_cuFFT.cu
	$(NVCC) -o $(BIN2) $(NVFLAGS) $< $(LIB)

clean:
	rm -f $(BIN1) $(BIN2) *.o
