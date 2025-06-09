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
    double avgError;
    double relativeError;
    int numIterations;
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

        if (h_converged) {
            result.numIterations = iter + 1;
            break;
        }
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
    float avgError = 0.0f;
    int numPoints = 0;
    int center = L/2;

    // Calculate errors at different distances
    for (int i = 1; i < L-1; i++) {
        for (int j = 1; j < L-1; j++) {
            for (int k = 1; k < L-1; k++) {
                if (i == center && j == center && k == center) continue;
                float r = sqrt(pow(i-center, 2) + pow(j-center, 2) + pow(k-center, 2));
                float analytical = 1.0f / (4.0f * M_PI * r);
                float numerical = h_potential[i + L*j + L*L*k];
                float error = fabsf(numerical - analytical);
                maxError = std::max(maxError, error);
                avgError += error;
                numPoints++;
            }
        }
    }
    result.maxError = maxError;
    result.avgError = avgError / numPoints;
    result.relativeError = maxError / (1.0f / (4.0f * M_PI)); // Relative to maximum potential

    // Cleanup
    cudaFree(d_potential);
    cudaFree(d_new_potential);
    cudaFree(d_convergence);

    return result;
}

void findOptimalConfiguration(int L, int maxIter, float tolerance) {
    std::vector<TestResult> results;
    printf("\n=== Configuration Tests for L=%d ===\n", L);
    printf("Block  Grid   KernelTime  TotalTime   MaxError    AvgError    RelError    Iters\n");
    printf("--------------------------------------------------------------------------------\n");

    // Test different block sizes (powers of 2, up to 16 threads per dimension)
    // Note: Since this is 3D, we need to be careful with total threads per block (<=1024)
    int block_sizes[] = {2, 4, 6, 8, 10, 12, 14, 16};

    for (int block_size : block_sizes) {
        TestResult result = runTest(L, block_size, maxIter, tolerance);
        results.push_back(result);

        printf("%3d    %3d    %8.3f    %8.3f    %.2e    %.2e    %.2e    %4d\n",
               block_size, result.gridSize,
               result.kernelTime, result.totalTime,
               result.maxError, result.avgError,
               result.relativeError, result.numIterations);
    }

    // Find best configuration based on kernel time
    TestResult best = results[0];
    for(size_t i = 1; i < results.size(); i++) {
        if(results[i].kernelTime < best.kernelTime) {
            best = results[i];
        }
    }

    printf("\n=== Best Configuration ===\n");
    printf("Block: %d³  Grid: %d³  KTime: %.3fms  TTime: %.3fms  Error: %.2e\n",
           best.blockSize, best.gridSize,
           best.kernelTime, best.totalTime, best.maxError);

    // Write potential vs distance data for this L
    char filename[32];
    sprintf(filename, "potential_L%d.dat", L);
    FILE *fp = fopen(filename, "w");
    int center = L/2;
    // Write both numerical and analytical solutions
    for (int i = 1; i < L-1; i++) {
        for (int j = 1; j < L-1; j++) {
            for (int k = 1; k < L-1; k++) {
                if (i == center && j == center && k == center) continue;
                float r = sqrt(pow(i-center, 2) + pow(j-center, 2) + pow(k-center, 2));
                float numerical = h_potential[i + L*j + L*L*k];
                float analytical = 1.0f / (4.0f * M_PI * r);
                fprintf(fp, "%f %f %f\n", r, numerical, analytical);
            }
        }
    }
    fclose(fp);
}

void testAllLSizes(int maxIter, float tolerance) {
    int L_sizes[] = {8, 16, 32, 64, 128, 256};

    printf("\n============== Poisson Solver Performance Analysis ==============\n");
    printf("Parameters: MaxIter=%d, Tolerance=%.1e\n\n", maxIter, tolerance);

    for (int L : L_sizes) {
        // Allocate and initialize host memory for this L
        int size = L * L * L * sizeof(float);
        h_potential = (float*)malloc(size);
        h_new_potential = (float*)malloc(size);
        initializePotential(h_potential, L);

        // Run tests for this L
        findOptimalConfiguration(L, maxIter, tolerance);

        // Cleanup
        free(h_potential);
        free(h_new_potential);

        printf("\n------------------------------------------------------------\n");
    }
    printf("\nAnalysis Complete - Results written to potential_L*.dat files\n");
}

int main() {
    int gpuID, maxIter;
    float tolerance;

    // Get GPU ID
    std::cout << "Enter the GPU ID: ";
    std::cin >> gpuID;
    cudaSetDevice(gpuID);
    std::cout << "Set GPU with device ID = " << gpuID << "\n";

    // Get convergence parameters
    std::cout << "Enter maximum iterations: ";
    std::cin >> maxIter;

    std::cout << "Enter convergence tolerance: ";
    std::cin >> tolerance;

    // Test all L sizes
    testAllLSizes(maxIter, tolerance);

    cudaDeviceReset();
    return 0;
}
