#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <omp.h>
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

// Device pointers for single GPU
float *d_temp_current;
float *d_temp_next;
bool *d_converged;

// Device pointers for dual GPU
float *d_temp_current0, *d_temp_current1;
float *d_temp_next0, *d_temp_next1;
bool *d_converged0, *d_converged1;

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
    float tolerance,
    int start_row,
    int end_row
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + start_row;

    if (i > 0 && i < GRID_SIZE-1 && j > start_row && j < end_row-1) {
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

TestResult runTest1GPU(int gpuID, int blockSize, int maxIter, float tolerance) {
    TestResult result;
    result.blockSize = blockSize;

    cudaSetDevice(gpuID);

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((GRID_SIZE + blockSize - 1) / blockSize,
                   (GRID_SIZE + blockSize - 1) / blockSize);

    size_t size = GRID_SIZE * GRID_SIZE * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_temp_current, size);
    cudaMalloc(&d_temp_next, size);
    cudaMalloc(&d_converged, sizeof(bool));

    // Copy initial data to device
    cudaMemcpy(d_temp_current, h_temp_current, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_next, h_temp_next, size, cudaMemcpyHostToDevice);

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
            d_temp_next, d_temp_current, d_converged, tolerance, 0, GRID_SIZE);

        cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);

        // Swap pointers
        float *temp = d_temp_current;
        d_temp_current = d_temp_next;
        d_temp_next = temp;

        if (h_converged) break;
    }

    auto end_kernel = std::chrono::high_resolution_clock::now();

    // Copy final result back to host
    cudaMemcpy(h_temp_current, d_temp_current, size, cudaMemcpyDeviceToHost);

    auto end_total = std::chrono::high_resolution_clock::now();

    // Calculate timings and error
    result.kernelTime = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    result.totalTime = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    result.iterations = iter + 1;

    float max_diff = 0.0f;
    for (int j = 1; j < GRID_SIZE-1; j++) {
        for (int i = 1; i < GRID_SIZE-1; i++) {
            float diff = fabs(h_temp_next[j*GRID_SIZE + i] - h_temp_current[j*GRID_SIZE + i]);
            max_diff = std::max(max_diff, diff);
        }
    }
    result.maxError = max_diff;

    // Cleanup
    cudaFree(d_temp_current);
    cudaFree(d_temp_next);
    cudaFree(d_converged);

    return result;
}

void setupGPUs(int gpu0, int gpu1) {
    int can_access_peer_0_1, can_access_peer_1_0;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpu0, gpu1);
    cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpu1, gpu0);

    if (!can_access_peer_0_1 || !can_access_peer_1_0) {
        printf("P2P access not available between GPU %d and GPU %d\n", gpu0, gpu1);
        exit(1);
    }

    cudaSetDevice(gpu0);
    cudaDeviceEnablePeerAccess(gpu1, 0);
    cudaSetDevice(gpu1);
    cudaDeviceEnablePeerAccess(gpu0, 0);
}

TestResult runTest2GPU(int gpu0, int gpu1, int blockSize, int maxIter, float tolerance) {
    TestResult result;
    result.blockSize = blockSize;

    setupGPUs(gpu0, gpu1);

    const int rows_per_gpu = GRID_SIZE / 2;
    size_t size_per_gpu = rows_per_gpu * GRID_SIZE * sizeof(float);
    size_t full_size = GRID_SIZE * GRID_SIZE * sizeof(float);

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((GRID_SIZE + blockSize - 1) / blockSize,
                   (rows_per_gpu + blockSize - 1) / blockSize);

    // Allocate device memory for both GPUs
    #pragma omp parallel num_threads(2)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id == 0 ? gpu0 : gpu1);

        if (gpu_id == 0) {
            cudaMalloc(&d_temp_current0, full_size);
            cudaMalloc(&d_temp_next0, full_size);
            cudaMalloc(&d_converged0, sizeof(bool));
            cudaMemcpy(d_temp_current0, h_temp_current, full_size, cudaMemcpyHostToDevice);
        } else {
            cudaMalloc(&d_temp_current1, full_size);
            cudaMalloc(&d_temp_next1, full_size);
            cudaMalloc(&d_converged1, sizeof(bool));
            cudaMemcpy(d_temp_current1, h_temp_current, full_size, cudaMemcpyHostToDevice);
        }
    }

    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_kernel = std::chrono::high_resolution_clock::now();

    int iter;
    bool h_converged0, h_converged1;
    for (iter = 0; iter < maxIter; iter++) {
        h_converged0 = h_converged1 = true;

        #pragma omp parallel num_threads(2)
        {
            int gpu_id = omp_get_thread_num();
            cudaSetDevice(gpu_id == 0 ? gpu0 : gpu1);

            if (gpu_id == 0) {
                cudaMemcpy(d_converged0, &h_converged0, sizeof(bool), cudaMemcpyHostToDevice);
                jacobi_kernel<<<numBlocks, threadsPerBlock>>>(
                    d_temp_next0, d_temp_current0, d_converged0, tolerance,
                    0, rows_per_gpu + 1);
            } else {
                cudaMemcpy(d_converged1, &h_converged1, sizeof(bool), cudaMemcpyHostToDevice);
                jacobi_kernel<<<numBlocks, threadsPerBlock>>>(
                    d_temp_next1, d_temp_current1, d_converged1, tolerance,
                    rows_per_gpu - 1, GRID_SIZE);
            }
        }

        // Exchange boundary data
        cudaMemcpyPeer(d_temp_next1 + (rows_per_gpu-1)*GRID_SIZE,
                      gpu1,
                      d_temp_next0 + (rows_per_gpu-1)*GRID_SIZE,
                      gpu0,
                      GRID_SIZE * sizeof(float));
        cudaMemcpyPeer(d_temp_next0 + rows_per_gpu*GRID_SIZE,
                      gpu0,
                      d_temp_next1 + rows_per_gpu*GRID_SIZE,
                      gpu1,
                      GRID_SIZE * sizeof(float));

        // Check convergence
        cudaMemcpy(&h_converged0, d_converged0, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_converged1, d_converged1, sizeof(bool), cudaMemcpyDeviceToHost);

        // Swap pointers
        #pragma omp parallel num_threads(2)
        {
            int gpu_id = omp_get_thread_num();
            if (gpu_id == 0) {
                float *temp = d_temp_current0;
                d_temp_current0 = d_temp_next0;
                d_temp_next0 = temp;
            } else {
                float *temp = d_temp_current1;
                d_temp_current1 = d_temp_next1;
                d_temp_next1 = temp;
            }
        }

        if (h_converged0 && h_converged1) break;
    }

    auto end_kernel = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    #pragma omp parallel num_threads(2)
    {
        int gpu_id = omp_get_thread_num();
        if (gpu_id == 0) {
            cudaMemcpy(h_temp_current, d_temp_current0,
                      (rows_per_gpu+1) * GRID_SIZE * sizeof(float),
                      cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpy(h_temp_current + rows_per_gpu * GRID_SIZE,
                      d_temp_current1 + rows_per_gpu * GRID_SIZE,
                      (rows_per_gpu) * GRID_SIZE * sizeof(float),
                      cudaMemcpyDeviceToHost);
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();

    result.kernelTime = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    result.totalTime = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    result.iterations = iter + 1;

    float max_diff = 0.0f;
    for (int j = 1; j < GRID_SIZE-1; j++) {
        for (int i = 1; i < GRID_SIZE-1; i++) {
            float diff = fabs(h_temp_next[j*GRID_SIZE + i] - h_temp_current[j*GRID_SIZE + i]);
            max_diff = std::max(max_diff, diff);
        }
    }
    result.maxError = max_diff;

    // Cleanup
    #pragma omp parallel num_threads(2)
    {
        int gpu_id = omp_get_thread_num();
        if (gpu_id == 0) {
            cudaFree(d_temp_current0);
            cudaFree(d_temp_next0);
            cudaFree(d_converged0);
        } else {
            cudaFree(d_temp_current1);
            cudaFree(d_temp_next1);
            cudaFree(d_converged1);
        }
    }

    return result;
}

void findOptimalConfiguration(bool run_both, int gpu0, int gpu1, int maxIter, float tolerance) {
    std::vector<TestResult> results_1gpu, results_2gpu;

    printf("\n=== Heat Diffusion Solver (1024x1024) ===\n");
    printf("Max Iterations: %d, Tolerance: %.1e\n", maxIter, tolerance);

    // Test different block sizes
    int block_sizes[] = {8, 16, 24, 32, 48, 64, 96, 128};

    printf("\nSingle-GPU Version (GPU %d):\n", gpu0);
    printf("Block  KTime(ms)    TTime(ms)    Iters    MaxError\n");
    printf("----------------------------------------------------\n");

    for (int block_size : block_sizes) {
        TestResult result = runTest1GPU(gpu0, block_size, maxIter, tolerance);
        results_1gpu.push_back(result);
        printf("%4d   %9.3f   %9.3f   %6d    %.2e\n",
               block_size, result.kernelTime, result.totalTime,
               result.iterations, result.maxError);
    }

    if (run_both) {
        printf("\nDual-GPU Version (GPU %d & %d):\n", gpu0, gpu1);
        printf("Block  KTime(ms)    TTime(ms)    Iters    MaxError\n");
        printf("----------------------------------------------------\n");

        for (int block_size : block_sizes) {
            TestResult result = runTest2GPU(gpu0, gpu1, block_size, maxIter, tolerance);
            results_2gpu.push_back(result);
            printf("%4d   %9.3f   %9.3f   %6d    %.2e\n",
                   block_size, result.kernelTime, result.totalTime,
                   result.iterations, result.maxError);
        }
    }

    // Find best configurations
    auto best_1gpu = std::min_element(results_1gpu.begin(), results_1gpu.end(),
        [](const TestResult& a, const TestResult& b) {
            return a.kernelTime < b.kernelTime;
        });

    printf("\n=== Best Single-GPU Configuration ===\n");
    printf("Block Size: %d\n", best_1gpu->blockSize);
    printf("Kernel Time: %.3f ms\n", best_1gpu->kernelTime);
    printf("Total Time: %.3f ms\n", best_1gpu->totalTime);
    printf("Iterations: %d\n", best_1gpu->iterations);
    printf("Max Error: %.2e\n", best_1gpu->maxError);

    if (run_both) {
        auto best_2gpu = std::min_element(results_2gpu.begin(), results_2gpu.end(),
            [](const TestResult& a, const TestResult& b) {
                return a.kernelTime < b.kernelTime;
            });

        printf("\n=== Best Dual-GPU Configuration ===\n");
        printf("Block Size: %d\n", best_2gpu->blockSize);
        printf("Kernel Time: %.3f ms\n", best_2gpu->kernelTime);
        printf("Total Time: %.3f ms\n", best_2gpu->totalTime);
        printf("Iterations: %d\n", best_2gpu->iterations);
        printf("Max Error: %.2e\n", best_2gpu->maxError);

        printf("\n=== Performance Comparison ===\n");
        printf("Speedup (Kernel): %.2fx\n", best_1gpu->kernelTime / best_2gpu->kernelTime);
        printf("Speedup (Total): %.2fx\n", best_1gpu->totalTime / best_2gpu->totalTime);
    }

    // Save the temperature distributions to files
    FILE *fp = fopen("temperature_1GPU.dat", "w");
    for (int j = 0; j < GRID_SIZE; j++) {
        for (int i = 0; i < GRID_SIZE; i++) {
            fprintf(fp, "%d %d %.6f\n", i, j, h_temp_current[j*GRID_SIZE + i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main() {
    int gpu0, gpu1, maxIter;
    float tolerance;
    bool run_both;

    // Get input parameters
    std::cout << "Run both 1-GPU and 2-GPU versions? (1/0): ";
    std::cin >> run_both;

    std::cout << "Enter first GPU ID: ";
    std::cin >> gpu0;

    if (run_both) {
        std::cout << "Enter second GPU ID: ";
        std::cin >> gpu1;
    } else {
        gpu1 = gpu0;  // Not used in 1-GPU mode
    }

    std::cout << "Enter maximum iterations: ";
    std::cin >> maxIter;

    std::cout << "Enter convergence tolerance: ";
    std::cin >> tolerance;

    // Allocate and initialize host memory
    size_t size = GRID_SIZE * GRID_SIZE * sizeof(float);
    h_temp_current = (float*)malloc(size);
    h_temp_next = (float*)malloc(size);
    initializeTemperature(h_temp_current);
    initializeTemperature(h_temp_next);

    // Find optimal configurations and compare
    findOptimalConfiguration(run_both, gpu0, gpu1, maxIter, tolerance);

    // Cleanup
    free(h_temp_current);
    free(h_temp_next);

    return 0;
}
