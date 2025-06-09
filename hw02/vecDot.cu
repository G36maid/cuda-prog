#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

float* h_A;
float* h_B;
float* h_C;
float* d_A;
float* d_B;
float* d_C;

void RandomInit(float* data, int n) {
    for (int i = 0; i < n; ++i)
        data[i] = 2.0*rand()/(float)RAND_MAX - 1.0;
}

__global__ void VecDot(const float* A, const float* B, float* C, int N) {
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0;
    while (i < N) {
        temp += A[i] * B[i];
        i += blockDim.x * gridDim.x;
    }
   
    cache[cacheIndex] = temp;

    __syncthreads();

    // Parallel reduction
    int ib = blockDim.x/2;
    while (ib != 0) {
        if(cacheIndex < ib)
            cache[cacheIndex] += cache[cacheIndex + ib]; 
        __syncthreads();
        ib /= 2;
    }
    
    if(cacheIndex == 0)
        C[blockIdx.x] = cache[0];
}

double runGPUComputation(int N, int threadsPerBlock, int blocksPerGrid,
                        double &input_time, double &kernel_time, 
                        double &output_time, double &total_time) {
    int size = N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, sb);

    // Copy input data to device
    auto start_input = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    auto end_input = std::chrono::high_resolution_clock::now();

    // Execute kernel
    auto start_kernel = std::chrono::high_resolution_clock::now();
    int sm = threadsPerBlock * sizeof(float);
    VecDot<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();

    // Copy result back and cleanup
    auto start_output = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
    
    double result = 0.0;
    for(int i = 0; i < blocksPerGrid; i++) {
        result += static_cast<double>(h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    auto end_output = std::chrono::high_resolution_clock::now();

    // Calculate timings
    input_time = std::chrono::duration<double, std::milli>(end_input - start_input).count();
    kernel_time = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    output_time = std::chrono::duration<double, std::milli>(end_output - start_output).count();
    total_time = input_time + kernel_time + output_time;

    return result;
}

double computeCPUReference(float* A, float* B, int N, double &cpu_time) {
    auto start = std::chrono::high_resolution_clock::now();
    
    double result = 0.0;
    for(int i = 0; i < N; i++) {
        result += static_cast<double>(A[i] * B[i]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

void testConfiguration(int N, int threadsPerBlock, int blocksPerGrid) {
    double input_time, kernel_time, output_time, total_time;
    
    double gpu_result = runGPUComputation(N, threadsPerBlock, blocksPerGrid,
                                        input_time, kernel_time, output_time, total_time);
    
    double gflops = (2.0 * N) / (kernel_time * 1000000.0);
    
    printf("\nConfiguration: %d threads/block, %d blocks\n", threadsPerBlock, blocksPerGrid);
    printf("Input time: %.6f ms\n", input_time);
    printf("Kernel time: %.6f ms\n", kernel_time);
    printf("Output time: %.6f ms\n", output_time);
    printf("Total time: %.6f ms\n", total_time);
    printf("Performance: %.6f GFLOPS\n", gflops);
    
    double cpu_time;
    double cpu_result = computeCPUReference(h_A, h_B, N, cpu_time);
    double relative_error = fabs((cpu_result - gpu_result) / cpu_result);
    
    printf("CPU time: %.6f ms\n", cpu_time);
    printf("CPU GFLOPS: %.6f\n", (2.0 * N) / (cpu_time * 1000000.0));
    printf("Speedup vs CPU: %.2fx\n", cpu_time/total_time);
    printf("Relative Error: %.15e\n", relative_error);
}

int main() {
    int gpuID, N;
    
    // Get GPU ID
    std::cout << "Enter the GPU ID: ";
    std::cin >> gpuID;
    cudaSetDevice(gpuID);
    std::cout << "Set GPU with device ID = " << gpuID << "\n";

    // Get vector size
    std::cout << "Enter the size of the vectors: ";
    std::cin >> N;
    
    // Allocate and initialize host vectors
    int size = N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    printf("\nTesting different configurations:\n");
    printf("----------------------------------\n");

    // Test different block sizes (must be power of 2)
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    
    for (int threads : block_sizes) {
        // Calculate appropriate number of blocks
        int blocks = (N + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535; // Maximum blocks limit
        
        h_C = (float*)malloc(blocks * sizeof(float));
        testConfiguration(N, threads, blocks);
        free(h_C);
    }

    free(h_A);
    free(h_B);
    
    cudaDeviceReset();
    return 0;
}