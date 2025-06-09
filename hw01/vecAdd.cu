#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

void readUserInput(int &gpuID, int &N) {
    std::cout << "Enter the GPU ID: ";
    std::cin >> gpuID;
    cudaSetDevice(gpuID);
    std::cout << "Set GPU with device ID = " << gpuID << "\n";

    std::cout << "Vector Addition: C = 1/A + 1/B\n";
    std::cout << "Enter the size of the vectors: ";
    std::cin >> N;
}

void initData(float *a, float *b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void checkResult(const float *cpu, const float *gpu, int n) {
    const float epsilon = 1e-5;
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        norm += fabs(cpu[i] - gpu[i]);
    }
    printf("Check result:\nnorm(h_C - h_D)=%.15e\n", norm);
}

void addVectorsCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = 1.0f / a[i] + 1.0f / b[i];
    }
}

__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        c[idx] = 1.0f / a[idx] + 1.0f / b[idx];
    }
}

double addVectorsGPU(const float *a, const float *b, float *c, int n, int threadsPerBlock,
                     double &gpu_input_time, double &gpu_kernel_time, 
                     double &gpu_output_time, double &total_gpu_time) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    auto start_input = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    auto end_input = std::chrono::high_resolution_clock::now();

    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    auto start_kernel = std::chrono::high_resolution_clock::now();
    addKernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();

    auto start_output = std::chrono::high_resolution_clock::now();
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    auto end_output = std::chrono::high_resolution_clock::now();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    gpu_input_time = std::chrono::duration<double, std::milli>(end_input - start_input).count();
    gpu_kernel_time = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    gpu_output_time = std::chrono::duration<double, std::milli>(end_output - start_output).count();
    total_gpu_time = gpu_input_time + gpu_kernel_time + gpu_output_time;

    double gflops = 2.0 * n / (gpu_kernel_time / 1000.0) / 1e9;
    return gflops;
}

int main() {
    int gpuID, N;
    readUserInput(gpuID, N);
    
    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *c_cpu = (float *)malloc(N * sizeof(float));
    float *c_gpu = (float *)malloc(N * sizeof(float));

    initData(a, b, N);

    // CPU computation for validation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    addVectorsCPU(a, b, c_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double time_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    double cpu_gflops = 2.0 * N / (time_cpu / 1000.0) / 1e9;

    printf("\nCPU Results:\n");
    printf("Processing time for CPU: %.6f (ms)\n", time_cpu);
    printf("CPU Gflops: %.6f\n\n", cpu_gflops);

    // Test different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    double best_time = 1e9;
    int best_block_size = 0;
    double best_gflops = 0;

    printf("Testing different block sizes:\n");
    printf("------------------------------------------------\n");

    for (int threads : block_sizes) {
        double input_time, kernel_time, output_time, total_time;
        double gflops = addVectorsGPU(a, b, c_gpu, N, threads, 
                                    input_time, kernel_time, output_time, total_time);
        
        printf("Block size: %d\n", threads);
        printf("Input time: %.6f ms\n", input_time);
        printf("Kernel time: %.6f ms\n", kernel_time);
        printf("Output time: %.6f ms\n", output_time);
        printf("Total time: %.6f ms\n", total_time);
        printf("GFLOPS: %.6f\n", gflops);
        printf("------------------------\n");

        if (kernel_time < best_time) {
            best_time = kernel_time;
            best_block_size = threads;
            best_gflops = gflops;
        }
    }

    printf("\nOptimal configuration found:\n");
    printf("Block size: %d\n", best_block_size);
    printf("Kernel time: %.6f ms\n", best_time);
    printf("GFLOPS: %.6f\n", best_gflops);
    printf("Speedup vs CPU: %.2fx\n", time_cpu / best_time);

    // Validate results using optimal block size
    double input_time, kernel_time, output_time, total_time;
    addVectorsGPU(a, b, c_gpu, N, best_block_size, 
                  input_time, kernel_time, output_time, total_time);
    checkResult(c_cpu, c_gpu, N);

    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
    return 0;
}