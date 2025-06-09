#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define N (1 << 24)
#define THREADS_PER_BLOCK 256

void initData(float *a, float *b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void checkResult(const float *a, const float *b, int n) {
    const float epsilon = 1e-5;
    for (int i = 0; i < n; ++i) {
        if (fabs(a[i] - b[i]) > epsilon) {
            fprintf(stderr, "Mismatch at index %d: CPU=%.6f, GPU=%.6f\n", i, a[i], b[i]);
            return;
        }
    }
    printf("Result verification passed!\n");
}

void addVectorsCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        //    c[i] = a[i] + b[i];
        c[i] = 1.0f / a[i] + 1.0f / b[i];//hw01
    }
}

__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
    //    c[idx] = a[idx] + b[idx];
        c[idx] = 1.0f / a[idx] + 1.0f / b[idx];//hw01
    }
}

void addVectorsGPU(const float *a, const float *b, float *c, int n) {
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    addKernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *c_cpu = (float *)malloc(N * sizeof(float));
    float *c_gpu = (float *)malloc(N * sizeof(float));

    initData(a, b, N);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    addVectorsCPU(a, b, c_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();
    printf("CPU Time: %.4f seconds\n", time_cpu);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    addVectorsGPU(a, b, c_gpu, N);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
    printf("GPU Time: %.4f seconds\n", time_gpu);

    checkResult(c_cpu, c_gpu, N);

    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
    return 0;
}
