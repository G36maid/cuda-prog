#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <omp.h>
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

// Host pointers
float* h_A;
float* h_B;
float* h_C;

// Device pointers for GPU 0
float* d_A0;
float* d_B0;
float* d_C0;

// Device pointers for GPU 1
float* d_A1;
float* d_B1;
float* d_C1;

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

void setupGPUs(int gpu0, int gpu1) {
    // Check P2P capabilities
    int can_access_peer_0_1, can_access_peer_1_0;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpu0, gpu1);
    cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpu1, gpu0);
    
    if (!can_access_peer_0_1 || !can_access_peer_1_0) {
        printf("P2P access not available between GPU %d and GPU %d\n", gpu0, gpu1);
        exit(1);
    }

    // Enable P2P access
    cudaSetDevice(gpu0);
    cudaDeviceEnablePeerAccess(gpu1, 0);
    cudaSetDevice(gpu1);
    cudaDeviceEnablePeerAccess(gpu0, 0);
}

void cleanupGPUs(int gpu0, int gpu1) {
    cudaSetDevice(gpu0);
    cudaDeviceDisablePeerAccess(gpu1);
    cudaSetDevice(gpu1);
    cudaDeviceDisablePeerAccess(gpu0);
}

TestResult runTest(int N, int threadsPerBlock, int blocksPerGrid, const double cpu_result) {
    TestResult result;
    result.blockSize = threadsPerBlock;
    result.gridSize = blocksPerGrid;

    int half_N = N / 2;
    int size_half = half_N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);

    // Allocate device memory on both GPUs
    #pragma omp parallel num_threads(2)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);

        if (gpu_id == 0) {
            cudaMalloc(&d_A0, size_half);
            cudaMalloc(&d_B0, size_half);
            cudaMalloc(&d_C0, sb);
        } else {
            cudaMalloc(&d_A1, size_half);
            cudaMalloc(&d_B1, size_half);
            cudaMalloc(&d_C1, sb);
        }
    }

    h_C = (float*)malloc(2 * sb);

    // Copy input data to devices
    auto start_input = std::chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(2)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);

        if (gpu_id == 0) {
            cudaMemcpy(d_A0, h_A, size_half, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B0, h_B, size_half, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(d_A1, h_A + half_N, size_half, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B1, h_B + half_N, size_half, cudaMemcpyHostToDevice);
        }
    }
    auto end_input = std::chrono::high_resolution_clock::now();

    // Execute kernels
    auto start_kernel = std::chrono::high_resolution_clock::now();
    int sm = threadsPerBlock * sizeof(float);
    #pragma omp parallel num_threads(2)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);

        if (gpu_id == 0) {
            VecDot<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A0, d_B0, d_C0, half_N);
        } else {
            VecDot<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A1, d_B1, d_C1, half_N);
        }
    }
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();

    // Copy results back and cleanup
    auto start_output = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C0, sb, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C + blocksPerGrid, d_C1, sb, cudaMemcpyDeviceToHost);

    double gpu_result = 0.0;
    for(int i = 0; i < 2 * blocksPerGrid; i++) {
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
    #pragma omp parallel num_threads(2)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);

        if (gpu_id == 0) {
            cudaFree(d_A0);
            cudaFree(d_B0);
            cudaFree(d_C0);
        } else {
            cudaFree(d_A1);
            cudaFree(d_B1);
            cudaFree(d_C1);
        }
    }
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
    printf("\n=== 2-GPU Vector Dot Product (N=%d) ===\n", N);
    printf("CPU Reference: Time=%.3f ms, GFLOPS=%.2f\n", 
           cpu_time, (2.0 * N) / (cpu_time * 1000000.0));
    printf("\nBlock  Grid   KTime(ms)   TTime(ms)   GFLOPS    Error\n");
    printf("--------------------------------------------------------\n");

    // Test different block sizes (powers of 2)
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // Test different grid sizes for each block size
    int max_blocks = std::min(65535, (N/2 + 31) / 32);  // N/2 because work is split between 2 GPUs
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

            printf("%4d   %4d   %8.3f    %8.3f    %6.2f    %.2e\n",
                   block_size, grid_size,
                   result.kernelTime, result.totalTime,
                   result.gflops, result.relativeError);
        }
    }

    // Find best configuration based on kernel time
    auto best = std::min_element(results.begin(), results.end(),
        [](const TestResult& a, const TestResult& b) {
            return a.kernelTime < b.kernelTime;
        });

    printf("\n=== Best Configuration ===\n");
    printf("Block: %d, Grid: %d, KTime: %.3f ms, GFLOPS: %.2f\n", 
           best->blockSize, best->gridSize, best->kernelTime, best->gflops);
    printf("Total: %.3f ms, Error: %.2e, Speedup: %.2fx\n",
           best->totalTime, best->relativeError, cpu_time / best->totalTime);
}

int main() {
    // Fixed vector size for hw04
    const int N = 40960000;
    
    // Get GPU IDs
    int gpu0, gpu1;
    std::cout << "Enter two GPU IDs (space separated): ";
    std::cin >> gpu0 >> gpu1;
    printf("\nUsing GPUs: %d and %d\n", gpu0, gpu1);

    // Setup P2P access between GPUs
    setupGPUs(gpu0, gpu1);

    // Allocate and initialize host vectors
    int size = N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    findOptimalConfiguration(N);

    // Cleanup
    free(h_A);
    free(h_B);
    cleanupGPUs(gpu0, gpu1);

    return 0;
}