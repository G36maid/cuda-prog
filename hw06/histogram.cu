#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define DATA_SIZE 81920000
#define NUM_BINS 128

void generate_exponential_data(std::vector<float>& data, float lambda = 1.0f) {
    std::mt19937 rng(12345);
    std::exponential_distribution<float> exp_dist(lambda);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = exp_dist(rng);
    }
}

void cpu_histogram(const std::vector<float>& data, std::vector<int>& hist, float bin_width) {
    std::fill(hist.begin(), hist.end(), 0);
    for (size_t i = 0; i < data.size(); ++i) {
        int bin = std::min(int(data[i] / bin_width), NUM_BINS - 1);
        hist[bin]++;
    }
}

__global__ void histogram_global_kernel(const float* data, int* hist, int n, float bin_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = min(int(data[idx] / bin_width), NUM_BINS - 1);
        atomicAdd(&hist[bin], 1);
    }
}

__global__ void histogram_shared_kernel(const float* data, int* hist, int n, float bin_width) {
    __shared__ int local_hist[NUM_BINS];
    int tid = threadIdx.x;
    if (tid < NUM_BINS) local_hist[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = min(int(data[idx] / bin_width), NUM_BINS - 1);
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    if (tid < NUM_BINS) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

void print_histogram(const std::vector<int>& hist, float bin_width) {
    for (int i = 0; i < NUM_BINS; ++i) {
        float bin_start = i * bin_width;
        float bin_end = (i + 1) * bin_width;
        printf("%8.4f - %8.4f : %d\n", bin_start, bin_end, hist[i]);
    }
}

void run_cpu_histogram(const std::vector<float>& data, std::vector<int>& hist, float bin_width, double& cpu_time) {
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_histogram(data, hist, bin_width);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
}

void run_gpu_histogram_global(const std::vector<float>& data, std::vector<int>& hist, float bin_width, double& gpu_time, int block_size) {
    float *d_data;
    int *d_hist;
    cudaMalloc(&d_data, DATA_SIZE * sizeof(float));
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));
    cudaMemcpy(d_data, data.data(), DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    int grid_size = (DATA_SIZE + block_size - 1) / block_size;

    cudaEvent_t g_start, g_end;
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_end);
    cudaEventRecord(g_start);
    histogram_global_kernel<<<grid_size, block_size>>>(d_data, d_hist, DATA_SIZE, bin_width);
    cudaEventRecord(g_end);
    cudaEventSynchronize(g_end);
    float gpu_global_time = 0;
    cudaEventElapsedTime(&gpu_global_time, g_start, g_end);
    gpu_time = gpu_global_time;

    cudaMemcpy(hist.data(), d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_hist);
    cudaEventDestroy(g_start);
    cudaEventDestroy(g_end);
}

void run_gpu_histogram_shared(const std::vector<float>& data, std::vector<int>& hist, float bin_width, double& gpu_time, int block_size) {
    float *d_data;
    int *d_hist;
    cudaMalloc(&d_data, DATA_SIZE * sizeof(float));
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));
    cudaMemcpy(d_data, data.data(), DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    int grid_size = (DATA_SIZE + block_size - 1) / block_size;

    cudaEvent_t g_start, g_end;
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_end);
    cudaEventRecord(g_start);
    histogram_shared_kernel<<<grid_size, block_size>>>(d_data, d_hist, DATA_SIZE, bin_width);
    cudaEventRecord(g_end);
    cudaEventSynchronize(g_end);
    float gpu_shared_time = 0;
    cudaEventElapsedTime(&gpu_shared_time, g_start, g_end);
    gpu_time = gpu_shared_time;

    cudaMemcpy(hist.data(), d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_hist);
    cudaEventDestroy(g_start);
    cudaEventDestroy(g_end);
}

void compare_histograms(const std::vector<int>& cpu_hist, const std::vector<int>& gpu_hist, const std::vector<int>& gpu_shared_hist) {
    int cpu_vs_gpu = 0, cpu_vs_gpu_shared = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (cpu_hist[i] != gpu_hist[i]) cpu_vs_gpu++;
        if (cpu_hist[i] != gpu_shared_hist[i]) cpu_vs_gpu_shared++;
    }
    printf("CPU vs GPU (global) mismatched bins: %d\n", cpu_vs_gpu);
    printf("CPU vs GPU (shared) mismatched bins: %d\n", cpu_vs_gpu_shared);
}

int main() {
    // Generate data
    std::vector<float> h_data(DATA_SIZE);
    generate_exponential_data(h_data);

    // Find max value for binning
    float max_val = *std::max_element(h_data.begin(), h_data.end());
    float bin_width = max_val / NUM_BINS;

    // --- CPU Histogram ---
    std::vector<int> cpu_hist(NUM_BINS, 0);
    double cpu_time = 0;
    run_cpu_histogram(h_data, cpu_hist, bin_width, cpu_time);

    // --- GPU Histogram (Global/Shared) for various block sizes ---
    std::vector<int> gpu_hist(NUM_BINS, 0), gpu_shared_hist(NUM_BINS, 0);
    double best_global_time = 1e30, best_shared_time = 1e30;
    int best_global_block = 0, best_shared_block = 0;

    printf("Block  GPU_Global(ms)  GPU_Shared(ms)\n");
    printf("--------------------------------------\n");

    for (int block_size = 8; block_size <= 1024; block_size *= 2) {
        double gpu_global_time = 0, gpu_shared_time = 0;

        run_gpu_histogram_global(h_data, gpu_hist, bin_width, gpu_global_time, block_size);
        run_gpu_histogram_shared(h_data, gpu_shared_hist, bin_width, gpu_shared_time, block_size);

        printf("%4d    %12.3f    %12.3f\n", block_size, gpu_global_time, gpu_shared_time);

        if (gpu_global_time < best_global_time) {
            best_global_time = gpu_global_time;
            best_global_block = block_size;
        }
        if (gpu_shared_time < best_shared_time) {
            best_shared_time = gpu_shared_time;
            best_shared_block = block_size;
        }
    }

    printf("\nCPU Histogram Time: %.3f ms\n", cpu_time);
    printf("Optimal GPU Global Block Size: %d (%.3f ms)\n", best_global_block, best_global_time);
    printf("Optimal GPU Shared Block Size: %d (%.3f ms)\n", best_shared_block, best_shared_time);

    compare_histograms(cpu_hist, gpu_hist, gpu_shared_hist);
    // Optionally print_histogram(cpu_hist, bin_width);
    return 0;
}
