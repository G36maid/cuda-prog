#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <vector>

// Constants for the problem
const int GRID_SIZE = 1024;  // 1024x1024 grid
const float TOP_TEMP = 400.0f;
const float OTHER_TEMP = 273.0f;

struct TestResult {
    int blockSize;
    double kernelTime;
    double totalTime;
    int iterations;
    double maxError;
};

// Host pointers
float *h_temp_current;
float *h_temp_next;

// Device pointers
float *d_temp_current;
float *d_temp_next;
bool *d_converged;

void initializeTemperature(float *temp) {
    for (int j = 0; j < GRID_SIZE; j++) {
        for (int i = 0; i < GRID_SIZE; i++) {
            if (j == 0) {  // Top edge
                temp[j * GRID_SIZE + i] = TOP_TEMP;
            } else if (j == GRID_SIZE-1 || i == 0 || i == GRID_SIZE-1) {  // Other edges
                temp[j * GRID_SIZE + i] = OTHER_TEMP;
            } else {  // Interior points
                temp[j * GRID_SIZE + i] = OTHER_TEMP;  // Initial guess
            }
        }
    }
}

__global__ void jacobi_kernel(
    float* T_new,
    const float* T_old,
    bool* converged,
    float tolerance
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < GRID_SIZE-1 && j > 0 && j < GRID_SIZE-1) {
        float new_temp = 0.25f * (
            T_old[j*GRID_SIZE + (i+1)] +
            T_old[j*GRID_SIZE + (i-1)] +
            T_old[(j+1)*GRID_SIZE + i] +
            T_old[(j-1)*GRID_SIZE + i]
        );

        T_new[j*GRID_SIZE + i] = new_temp;

        // Check convergence
        if (fabsf(new_temp - T_old[j*GRID_SIZE + i]) > tolerance) {
            *converged = false;
        }
    }
}

TestResult runTest(int blockSize, int maxIter, float tolerance) {
    TestResult result;
    result.blockSize = blockSize;

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((GRID_SIZE + blockSize - 1) / blockSize,
                   (GRID_SIZE + blockSize - 1) / blockSize);

    // Start timing
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_kernel = std::chrono::high_resolution_clock::now();

    // Main iteration loop
    int iter;
    bool h_converged;
    for (iter = 0; iter < maxIter; iter++) {
        h_converged = true;
        cudaMemcpy(d_converged, &h_converged, sizeof(bool), cudaMemcpyHostToDevice);

        jacobi_kernel<<<numBlocks, threadsPerBlock>>>(
            d_temp_next, d_temp_current, d_converged, tolerance);

        cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);

        // Swap pointers
        float *temp = d_temp_current;
        d_temp_current = d_temp_next;
        d_temp_next = temp;

        if (h_converged) break;
    }

    auto end_kernel = std::chrono::high_resolution_clock::now();

    // Copy final result back to host
    cudaMemcpy(h_temp_current, d_temp_current,
               GRID_SIZE * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    auto end_total = std::chrono::high_resolution_clock::now();

    result.kernelTime = std::chrono::duration<double, std::milli>(
        end_kernel - start_kernel).count();
    result.totalTime = std::chrono::duration<double, std::milli>(
        end_total - start_total).count();
    result.iterations = iter + 1;

    // Calculate maximum difference between iterations
    float max_diff = 0.0f;
    for (int j = 1; j < GRID_SIZE-1; j++) {
        for (int i = 1; i < GRID_SIZE-1; i++) {
            float diff = fabs(h_temp_next[j*GRID_SIZE + i] -
                            h_temp_current[j*GRID_SIZE + i]);
            max_diff = std::max(max_diff, diff);
        }
    }
    result.maxError = max_diff;

    return result;
}

void findOptimalConfiguration(int maxIter, float tolerance) {
    std::vector<TestResult> results;

    printf("\n=== Heat Diffusion Solver (1024x1024) ===\n");
    printf("Max Iterations: %d, Tolerance: %.1e\n", maxIter, tolerance);
    printf("\nBlock  KTime(ms)    TTime(ms)    Iters    MaxError\n");
    printf("----------------------------------------------------\n");

    // Test different block sizes
    int block_sizes[] = {8, 16, 32};

    for (int block_size : block_sizes) {
        TestResult result = runTest(block_size, maxIter, tolerance);
        results.push_back(result);

        printf("%4d   %9.3f   %9.3f   %6d    %.2e\n",
               block_size, result.kernelTime, result.totalTime,
               result.iterations, result.maxError);
    }

    // Find best configuration based on kernel time
    auto best = std::min_element(results.begin(), results.end(),
        [](const TestResult& a, const TestResult& b) {
            return a.kernelTime < b.kernelTime;
        });

    printf("\n=== Best Configuration ===\n");
    printf("Block Size: %d\n", best->blockSize);
    printf("Kernel Time: %.3f ms\n", best->kernelTime);
    printf("Total Time: %.3f ms\n", best->totalTime);
    printf("Iterations: %d\n", best->iterations);
    printf("Max Error: %.2e\n", best->maxError);

    // Save the temperature distribution to a file
    FILE *fp = fopen("temperature.dat", "w");
    for (int j = 0; j < GRID_SIZE; j++) {
        for (int i = 0; i < GRID_SIZE; i++) {
            fprintf(fp, "%d %d %.6f\n", i, j, h_temp_current[j*GRID_SIZE + i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main() {
    int gpuID, maxIter;
    float tolerance;

    // Get input parameters
    std::cout << "Enter the GPU ID: ";
    std::cin >> gpuID;
    cudaSetDevice(gpuID);
    std::cout << "Using GPU with device ID = " << gpuID << "\n";

    std::cout << "Enter maximum iterations: ";
    std::cin >> maxIter;

    std::cout << "Enter convergence tolerance: ";
    std::cin >> tolerance;

    // Allocate host memory
    size_t size = GRID_SIZE * GRID_SIZE * sizeof(float);
    h_temp_current = (float*)malloc(size);
    h_temp_next = (float*)malloc(size);

    // Initialize temperature distribution
    initializeTemperature(h_temp_current);
    initializeTemperature(h_temp_next);

    // Allocate device memory
    cudaMalloc(&d_temp_current, size);
    cudaMalloc(&d_temp_next, size);
    cudaMalloc(&d_converged, sizeof(bool));

    // Copy initial data to device
    cudaMemcpy(d_temp_current, h_temp_current, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_next, h_temp_next, size, cudaMemcpyHostToDevice);

    // Find optimal configuration and solve
    findOptimalConfiguration(maxIter, tolerance);

    // Cleanup
    free(h_temp_current);
    free(h_temp_next);
    cudaFree(d_temp_current);
    cudaFree(d_temp_next);
    cudaFree(d_converged);

    return 0;
}
