#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define DATA_SIZE 81920000
#define NUM_BINS 128

// Host-side exponential random number generator
void generate_exponential_data(std::vector<float>& data, float lambda = 1.0f) {
    std::mt19937 rng(12345);
    std::exponential_distribution<float> exp_dist(lambda);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = exp_dist(rng);
    }
}

// CPU histogram
void cpu_histogram(const std::vector<float>& data, std::vector<int>& hist, float bin_width) {
    std::fill(hist.begin(), hist.end(), 0);
    for (size_t i = 0; i < data.size(); ++i) {
        int bin = std::min(int(data[i] / bin_width), NUM_BINS - 1);
        hist[bin]++;
    }
}

// CUDA kernel: global memory histogram (atomic)
__global__ void histogram_global_kernel(const float* data, int* hist, int n, float bin_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = min(int(data[idx] / bin_width), NUM_BINS - 1);
        atomicAdd(&hist[bin], 1);
    }
}

// CUDA kernel: shared memory histogram (per-block reduction)
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

int main() {
    // Generate data
    std::vector<float> h_data(DATA_SIZE);
    generate_exponential_data(h_data);

    // Find max value for binning
    float max_val = *std::max_element(h_data.begin(), h_data.end());
    float bin_width = max_val / NUM_BINS;

    // --- CPU Histogram ---
    std::vector<int> cpu_hist(NUM_BINS, 0);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_histogram(h_data, cpu_hist, bin_width);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --- GPU Histogram (Global Memory) ---
    float *d_data;
    int *d_hist;
    std::vector<int> gpu_hist(NUM_BINS, 0);
    cudaMalloc(&d_data, DATA_SIZE * sizeof(float));
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    int block_size = 256;
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

    cudaMemcpy(gpu_hist.data(), d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // --- GPU Histogram (Shared Memory) ---
    std::vector<int> gpu_shared_hist(NUM_BINS, 0);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));
    cudaEventRecord(g_start);
    histogram_shared_kernel<<<grid_size, block_size>>>(d_data, d_hist, DATA_SIZE, bin_width);
    cudaEventRecord(g_end);
    cudaEventSynchronize(g_end);
    float gpu_shared_time = 0;
    cudaEventElapsedTime(&gpu_shared_time, g_start, g_end);

    cudaMemcpy(gpu_shared_hist.data(), d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Output Results ---
    printf("CPU Histogram Time: %.3f ms\n", cpu_time);
    printf("GPU Histogram (Global) Time: %.3f ms\n", gpu_global_time);
    printf("GPU Histogram (Shared) Time: %.3f ms\n", gpu_shared_time);

    // Optionally print histogram
    // print_histogram(cpu_hist, bin_width);

    // Compare histograms
    int cpu_vs_gpu = 0, cpu_vs_gpu_shared = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (cpu_hist[i] != gpu_hist[i]) cpu_vs_gpu++;
        if (cpu_hist[i] != gpu_shared_hist[i]) cpu_vs_gpu_shared++;
    }
    printf("CPU vs GPU (global) mismatched bins: %d\n", cpu_vs_gpu);
    printf("CPU vs GPU (shared) mismatched bins: %d\n", cpu_vs_gpu_shared);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaEventDestroy(g_start);
    cudaEventDestroy(g_end);

    return 0;
}
