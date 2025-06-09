################################################################################
#
# Build script for project
#
################################################################################

CXX     := icc
CFLAGS  := -O3
NVCC    := nvcc
#CUFLAGS := -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 --compiler-options -fno-strict-aliasing -DUNIX -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler -fopenmp -rdc=true

CUFLAGS := -arch=compute_61 -code=sm_61,sm_61 -O3 --compiler-options -fopenmp
#CUFLAGS := -O3 --compiler-options -fopenmp

BIN     := laplace2D_NGPU
OBJ     := laplace2D_NGPU.o
LIB     :=

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

all: $(BIN)

$(BIN): $(OBJ)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
	$(NVCC) -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)

laplace2D_NGPU.o:  laplace2D_NGPU.cu

clean:
	rm -f $(OBJ) $(BIN)

