################################################################################
#
# Build script for project
#
################################################################################

SM    := sm_61
CP    := 61

CXX     := icc
CFLAGS  := -O3
NVCC    := nvcc
#CUFLAGS := -arch=compute_$(CP) -code=$(SM),$(SM) -O3 -m64 --compiler-options -fno-strict-aliasing -DUNIX -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler -fopenmp -rdc=true

CUFLAGS := -arch=compute_$(CP) -code=$(SM),$(SM) -O3 --compiler-options -fopenmp
#CUFLAGS := -O3 --compiler-options -fopenmp

BIN     := vecAdd_ngpu_umem
OBJ     := vecAdd_ngpu_umem.o
BIN_A   := vecAdd_ngpu_umem_A
OBJ_A   := vecAdd_ngpu_umem_A.o
LIB     :=

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

all: $(BIN) $(BIN_A)

$(BIN_A): $(OBJ_A)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN_A) $(CUFLAGS) $(OBJ_A) $(LIB)
	$(NVCC) -o $(BIN_A) $(CUFLAGS) $(OBJ_A) $(LIB)

$(BIN): $(OBJ)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
	$(NVCC) -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)

vecAdd_ngpu_umem.o:  vecAdd_ngpu_umem.cu
vecAdd_ngpu_umem_A.o:  vecAdd_ngpu_umem_A.cu

clean:
	rm -f $(OBJ) $(BIN) $(OBJ_A) $(BIN_A)

