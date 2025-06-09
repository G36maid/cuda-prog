// Saxpy: B <- alpha*A + B, for arbitrarily long vectors
// Use Unified Memory

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"          // header for CUBLAS


// Variables
float* h_A;   // host vectors
float* h_B;
float* h_D;


// functions
void RandomInit(float*, long);  


// Device code 
__global__ void Saxpy(const float alpha, const float* A, float* B, long N)
{

    long i = blockDim.x * blockIdx.x + threadIdx.x;

    while(i < N) {
      B[i] += alpha*A[i];
      i += blockDim.x * gridDim.x;  // go to the next grid
    }
    __syncthreads();
}

// Host code

int main(void)
{

    int gid;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Saxpy: B <- alpha*A + B\n");
    printf("Enter the GPU_ID: ");
    scanf("%d",&gid);
    printf("%d\n",gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    float alpha;
    printf("Enter the value of alpha: ");
    scanf("%f",&alpha);        
    printf("%f\n",alpha);        

    int N;
    printf("Enter the size of the vectors: ");
    scanf("%ld",&N);        
    printf("%ld\n",N);        
    long size = N * sizeof(float);
    
    // the timer
    cudaEvent_t start,stop;

    // Allocate input vectors h_A and h_B in unified memory

    cudaMallocManaged(&h_A, size);
    cudaMallocManaged(&h_B, size);

    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    h_D = (float*)malloc(size);   // CPU reference solution
    for(long i = 0; i < N; ++i) 
      h_D[i] = h_B[i];

    // Set the sizes of threads and blocks

    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d",&threadsPerBlock);
    printf("%d\n",threadsPerBlock);
    if( threadsPerBlock > 1024 ) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(0);
    }

//    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d",&blocksPerGrid);
    printf("%d\n",blocksPerGrid);
    if( blocksPerGrid > 2147483647 ) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      exit(0); 
    }
    printf("The number of blocks is %d\n", blocksPerGrid);

    // start the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    Saxpy <<< blocksPerGrid, threadsPerBlock >>> (alpha, h_A, h_B, N);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",2*N/(1000000.0*gputime));
    
    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // start the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    for(long i = 0; i < N; ++i) 
      h_D[i] += alpha*h_A[i];
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",2*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/gputime);

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result

    printf("Check result:\n");
    double sum=0; 
    double diff;
    for (long i = 0; i < N; ++i) {
      diff = abs(h_D[i] - h_B[i]);
      sum += diff*diff; 
      if(diff > 1.0e-15) { 
//        printf("i=%d, h_D=%.10e, h_B=%.10e \n", i, h_D[i], h_B[i]);
      }
    }
    sum = sqrt(sum);
    printf("norm(h_D - h_B)=%.15e\n\n",sum);
//
    cudaFree(h_A);
    cudaFree(h_B);
//
    free(h_D);
//
    cudaDeviceReset();
}


void RandomInit(float* data, long n)    // Allocates an array with random float entries.
{
    for (long i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


