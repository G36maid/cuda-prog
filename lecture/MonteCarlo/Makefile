CUDIR     := /usr/local/nvidia

SM    := sm_61
CP    := 61

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CPPFLAGS  := -O3 -Wall -g --std=c++11
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_$(CP) -code=$(SM) -Xptxas -v -rdc=true --std=c++11
LIB     := -lgsl -lgslcblas 
#LIB     := -lgsl -lgslcblas -lcurand -lcudadevrt
#LIB     := -lgsl -lgslcblas -lcudadevrt

BIN    := metro

all: $(BIN) 

$(BIN): metro.c
	$(CPP) -o $(BIN) $(CPPFLAGS) metro.c $(LIB)

clean:
	rm -f $(BIN) 
