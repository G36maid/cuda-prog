#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cmath>

struct TestResult {
    int blockSize;
    int gridSize;
    double kernelTime;
    double totalTime;
    double maxError;
};

// Host pointers
float *h_potential;
float *h_new_potential;
bool *h_convergence;

// Device pointers
float *d_potential;
float *d_new_potential;
bool *d_convergence;

void initializePotential(float *potential, int L) {
    int size = L * L * L;
    for (int i = 0; i < size; i++) {
        potential[i] = 0.0f;  // Initialize boundary conditions to zero
    }
    
    // Set point charge at center
    int center = L/2;
    potential[center + L * center + L * L * center] = 1.0f;
}

__global__ void poissonKernel(float *potential, float *new_potential, bool *convergence,
                             int L, float tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx > 0 && idx < L-1 && idy > 0 && idy < L-1 && idz > 0 && idz < L-1) {
        int index = idx + L * idy + L * L * idz;
        
        // Calculate new potential using 6-point stencil
        float new_val = (potential[index + 1] + potential[index - 1] +
                        potential[index + L] + potential[index - L] +
                        potential[index + L*L] + potential[index - L*L]) / 6.0f;
        
        new_potential[index] = new_val;
        
        // Check convergence
        if (fabsf(new_val - potential[index]) > tolerance) {
            *convergence = false;
        }
    }
}

TestResult runTest(int L, int threadsPerBlock, int maxIter, float tolerance) {
    TestResult result;
    result.blockSize = threadsPerBlock;
    
    int size = L * L * L * sizeof(float);
    bool h_converged;
    
    // Allocate device memory
    cudaMalloc(&d_potential, size);
    cudaMalloc(&d_new_potential, size);
    cudaMalloc(&d_convergence, sizeof(bool));
    
    // Copy initial conditions to device
    cudaMemcpy(d_potential, h_potential, size, cudaMemcpyHostToDevice);
    
    // Calculate grid dimensions
    dim3 block(threadsPerBlock, threadsPerBlock, threadsPerBlock);
    int numBlocks = (L + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(numBlocks, numBlocks, numBlocks);
    result.gridSize = numBlocks;
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Iteration loop
    int iter;
    for (iter = 0; iter < maxIter; iter++) {
        // Reset convergence flag
        h_converged = true;
        cudaMemcpy(d_convergence, &h_converged, sizeof(bool), cudaMemcpyHostToDevice);
        
        // Launch kernel
        poissonKernel<<<grid, block>>>(d_potential, d_new_potential, d_convergence, L, tolerance);
        
        // Check convergence
        cudaMemcpy(&h_converged, d_convergence, sizeof(bool), cudaMemcpyDeviceToHost);
        
        // Swap pointers
        float *temp = d_potential;
        d_potential = d_new_potential;
        d_new_potential = temp;
        
        if (h_converged) break;
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy result back to host
    cudaMemcpy(h_potential, d_potential, size, cudaMemcpyDeviceToHost);
    
    // Calculate timing and error
    result.kernelTime = std::chrono::duration<double, std::milli>(end - start).count();
    result.totalTime = result.kernelTime;  // Add memory transfer time if needed
    
    // Calculate maximum error (difference from analytical solution)
    float maxError = 0.0f;
    int center = L/2;
    for (int i = 1; i < L-1; i++) {
        for (int j = 1; j < L-1; j++) {
            for (int k = 1; k < L-1; k++) {
                if (i == center && j == center && k == center) continue;
                float r = sqrt(pow(i-center, 2) + pow(j-center, 2) + pow(k-center, 2));
                float analytical = 1.0f / (4.0f * M_PI * r);
                float numerical = h_potential[i + L*j + L*L*k];
                maxError = std::max(maxError, fabsf(numerical - analytical));
            }
        }
    }
    result.maxError = maxError;
    
    // Cleanup
    cudaFree(d_potential);
    cudaFree(d_new_potential);
    cudaFree(d_convergence);
    
    return result;
}

void findOptimalConfiguration(int L, int maxIter, float tolerance) {
    std::vector<TestResult> results;
    printf("\nTesting different configurations for L=%d:\n", L);
    printf("----------------------------------\n");
    
    // Test different block sizes (powers of 2, up to 8 threads per dimension)
    int block_sizes[] = {2, 4, 8};
    
    for (int block_size : block_sizes) {
        TestResult result = runTest(L, block_size, maxIter, tolerance);
        results.push_back(result);
        
        printf("\nConfig: %dx%dx%d threads/block, %dx%dx%d blocks\n",
               block_size, block_size, block_size,
               result.gridSize, result.gridSize, result.gridSize);
        printf("Kernel time: %.6f ms\n", result.kernelTime);
        printf("Total time: %.6f ms\n", result.totalTime);
        printf("Max Error vs Analytical: %.6e\n", result.maxError);
    }
    
    // Find best configuration based on kernel time
    auto best = std::min_element(results.begin(), results.end(),
        [](const TestResult& a, const TestResult& b) {
            return a.kernelTime < b.kernelTime;
        });
    
    printf("\n=== Optimal Configuration ===\n");
    printf("Block Size: %dx%dx%d\n", best->blockSize, best->blockSize, best->blockSize);
    printf("Grid Size: %dx%dx%d\n", best->gridSize, best->gridSize, best->gridSize);
    printf("Kernel Time: %.6f ms\n", best->kernelTime);
    printf("Total Time: %.6f ms\n", best->totalTime);
    printf("Max Error vs Analytical: %.6e\n", best->maxError);
}

int main() {
    int gpuID, L, maxIter;
    float tolerance;
    
    // Get input parameters
    std::cout << "Enter the GPU ID: ";
    std::cin >> gpuID;
    cudaSetDevice(gpuID);
    std::cout << "Set GPU with device ID = " << gpuID << "\n";
    
    std::cout << "Enter the size of the cube (L): ";
    std::cin >> L;
    
    std::cout << "Enter maximum iterations: ";
    std::cin >> maxIter;
    
    std::cout << "Enter convergence tolerance: ";
    std::cin >> tolerance;
    
    // Allocate and initialize host memory
    int size = L * L * L * sizeof(float);
    h_potential = (float*)malloc(size);
    h_new_potential = (float*)malloc(size);
    initializePotential(h_potential, L);
    
    findOptimalConfiguration(L, maxIter, tolerance);
    
    // Write potential vs distance data
    FILE *fp = fopen("potential.dat", "w");
    int center = L/2;
    for (int i = 1; i < L-1; i++) {
        float r = sqrt(pow(i-center, 2));
        float potential = h_potential[i + L*center + L*L*center];
        fprintf(fp, "%f %f\n", r, potential);
    }
    fclose(fp);
    
    free(h_potential);
    free(h_new_potential);
    
    cudaDeviceReset();
    return 0;
}