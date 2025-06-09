#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <algorithm>

struct TestResult {
    int blockSize;
    int gridSize;
    double kernelTime;
    double totalTime;
    double gflops;
    double relativeError;
};

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

TestResult runTest(int N, int threadsPerBlock, int blocksPerGrid, const double cpu_result) {
    TestResult result;
    result.blockSize = threadsPerBlock;
    result.gridSize = blocksPerGrid;

    int size = N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, sb);
    h_C = (float*)malloc(sb);

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
    
    double gpu_result = 0.0;
    for(int i = 0; i < blocksPerGrid; i++) {
        gpu_result += static_cast<double>(h_C[i]);
    }

    auto end_output = std::chrono::high_resolution_clock::now();

    // Calculate timings
    double input_time = std::chrono::duration<double, std::milli>(end_input - start_input).count();
    result.kernelTime = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    double output_time = std::chrono::duration<double, std::milli>(end_output - start_output).count();
    result.totalTime = input_time + result.kernelTime + output_time;
    
    result.gflops = (2.0 * N) / (result.kernelTime * 1000000.0);
    result.relativeError = fabs((cpu_result - gpu_result) / cpu_result);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C);

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

void findOptimalConfiguration(int N) {
    std::vector<TestResult> results;
    
    // CPU Reference computation
    double cpu_time;
    double cpu_result = computeCPUReference(h_A, h_B, N, cpu_time);
    printf("\nCPU Reference:\n");
    printf("Time: %.6f ms\n", cpu_time);
    printf("GFLOPS: %.6f\n", (2.0 * N) / (cpu_time * 1000000.0));

    printf("\nTesting different configurations:\n");
    printf("----------------------------------\n");

    // Test different block sizes (powers of 2)
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    
    // Test different grid sizes for each block size
    int max_blocks = std::min(65535, (N + 31) / 32);
    int grid_sizes[] = {
        max_blocks,
        max_blocks / 2,
        max_blocks / 4,
        max_blocks / 8,
        max_blocks / 16
    };

    for (int block_size : block_sizes) {
        for (int grid_size : grid_sizes) {
            if (grid_size < 1) continue;
            
            TestResult result = runTest(N, block_size, grid_size, cpu_result);
            results.push_back(result);

            printf("\nConfig: %d threads/block, %d blocks\n", block_size, grid_size);
            printf("Kernel time: %.6f ms\n", result.kernelTime);
            printf("Total time: %.6f ms\n", result.totalTime);
            printf("GFLOPS: %.6f\n", result.gflops);
            printf("Relative Error: %.15e\n", result.relativeError);
        }
    }

    // Find best configuration based on kernel time
    auto best = std::min_element(results.begin(), results.end(),
        [](const TestResult& a, const TestResult& b) {
            return a.kernelTime < b.kernelTime;
        });

    printf("\n=== Optimal Configuration ===\n");
    printf("Block Size: %d\n", best->blockSize);
    printf("Grid Size: %d\n", best->gridSize);
    printf("Kernel Time: %.6f ms\n", best->kernelTime);
    printf("Total Time: %.6f ms\n", best->totalTime);
    printf("GFLOPS: %.6f\n", best->gflops);
    printf("Relative Error: %.15e\n", best->relativeError);
    printf("Speedup vs CPU: %.2fx\n", cpu_time / best->totalTime);
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

    findOptimalConfiguration(N);

    free(h_A);
    free(h_B);
    
    cudaDeviceReset();
    return 0;
}