#include <iostream> // 使用 C++ 輸入輸出
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

void readUserInput(int &gpuID, int &N, int &threadsPerBlock) {
    std::cout << "Enter the GPU ID: ";
    std::cin >> gpuID;
    cudaSetDevice(gpuID);
    std::cout << "Set GPU with device ID = " << gpuID << "\n";

    std::cout << "Vector Addition: C = 1/A + 1/B\n";
    std::cout << "Enter the size of the vectors: ";
    std::cin >> N;

    std::cout << "Enter the number of threads per block: ";
    std::cin >> threadsPerBlock;

    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "The number of blocks is " << blocks << "\n";
}

void printResults(double cpu_time, double cpu_gflops,
                  double gpu_input_time, double gpu_kernel_time,
                  double gpu_output_time, double total_gpu_time,
                  double gpu_gflops, double speedup) {
    printf("Processing time for CPU: %.6f (ms)\n", cpu_time);
    printf("CPU Gflops: %.6f\n", cpu_gflops);
    printf("Input time for GPU: %.6f (ms)\n", gpu_input_time);
    printf("Processing time for GPU: %.6f (ms)\n", gpu_kernel_time);
    printf("GPU Gflops: %.6f\n", gpu_gflops);
    printf("Output time for GPU: %.6f (ms)\n", gpu_output_time);
    printf("Total time for GPU: %.6f (ms)\n", total_gpu_time);
    printf("Speed up of GPU = %.6f\n", speedup);
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
        c[i] = 1.0f / a[i] + 1.0f / b[i]; // HW01
    }
}

__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        c[idx] = 1.0f / a[idx] + 1.0f / b[idx]; // HW01
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

    // Calculate timing information
    gpu_input_time = std::chrono::duration<double, std::milli>(end_input - start_input).count();
    gpu_kernel_time = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    gpu_output_time = std::chrono::duration<double, std::milli>(end_output - start_output).count();
    total_gpu_time = gpu_input_time + gpu_kernel_time + gpu_output_time;

    double gflops = 2.0 * n / (gpu_kernel_time / 1000.0) / 1e9;
    return gflops;
}


int main() {
    int gpuID, N, threadsPerBlock;
    readUserInput(gpuID, N, threadsPerBlock);

    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *c_cpu = (float *)malloc(N * sizeof(float));
    float *c_gpu = (float *)malloc(N * sizeof(float));

    initData(a, b, N);

    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    addVectorsCPU(a, b, c_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double time_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    double cpu_gflops = 2.0 * N / (time_cpu / 1000.0) / 1e9;

    // GPU
    double gpu_input_time, gpu_kernel_time, gpu_output_time, total_gpu_time;
    auto gpu_start_input = std::chrono::high_resolution_clock::now();
    // GPU kernel function internally should fill gpu_input_time, etc...
    double gpu_gflops = addVectorsGPU(a, b, c_gpu, N, threadsPerBlock,
                                      gpu_input_time, gpu_kernel_time, gpu_output_time, total_gpu_time);

    double speedup = time_cpu / gpu_kernel_time;

    printResults(time_cpu, cpu_gflops,
                 gpu_input_time, gpu_kernel_time, gpu_output_time,
                 total_gpu_time, gpu_gflops, speedup);
    checkResult(c_cpu, c_gpu, N);

    free(a); free(b); free(c_cpu); free(c_gpu);
    return 0;
}
