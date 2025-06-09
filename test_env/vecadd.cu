// vecadd.cu

#include <iostream>

__global__
void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main() {
    int N = 1 << 20;  // 向量大小為 2^20（1048576）
    size_t size = N * sizeof(float);

    // 配置 host 記憶體
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // 初始化資料
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 2.0f;
    }

    // 配置 device 記憶體
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 複製資料到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 設定 kernel 參數並啟動 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 將結果複製回 host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 驗證結果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            success = false;
            break;
        }
    }

    std::cout << (success ? "✅ Vector addition successful!" : "❌ Vector addition failed!") << std::endl;

    // 清除記憶體
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return success ? 0 : 1;
}
