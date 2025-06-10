#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <iomanip>
#include <fstream>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUFFT_CHECK(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult code, const char *file, int line, bool abort=true)
{
    if (code != CUFFT_SUCCESS)
    {
        fprintf(stderr,"CUFFT Error: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

// 計算 3D 格點的 k 向量
__global__ void setup_k_vectors(float *kx, float *ky, float *kz, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    // 計算 k 向量 (週期邊界條件)
    float ki = (i <= N/2) ? i : i - N;
    float kj = (j <= N/2) ? j : j - N;
    float kk = (k <= N/2) ? k : k - N;

    ki *= 2.0f * M_PI / N;
    kj *= 2.0f * M_PI / N;
    kk *= 2.0f * M_PI / N;

    int idx = k * N * N + j * N + i;
    kx[idx] = ki;
    ky[idx] = kj;
    kz[idx] = kk;
}

// 在動量空間求解 Poisson 方程
__global__ void solve_poisson_3d(cufftComplex *rho_k, cufftComplex *phi_k,
                                  float *kx, float *ky, float *kz, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;

    float k2 = kx[idx] * kx[idx] + ky[idx] * ky[idx] + kz[idx] * kz[idx];

    // 避免 k=0 的奇異性
    if (i == 0 && j == 0 && k == 0) {
        phi_k[idx].x = 0.0f;
        phi_k[idx].y = 0.0f;
    } else {
        // φ(k) = -ρ(k) / k²
        phi_k[idx].x = -rho_k[idx].x / k2;
        phi_k[idx].y = -rho_k[idx].y / k2;
    }
}

// 設置點電荷源項
__global__ void setup_point_charge(cufftComplex *rho, int N, float charge = 1.0f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;

    // 在原點放置點電荷
    if (i == 0 && j == 0 && k == 0) {
        rho[idx].x = charge;
        rho[idx].y = 0.0f;
    } else {
        rho[idx].x = 0.0f;
        rho[idx].y = 0.0f;
    }
}

// 3D Poisson 求解器類別
class Poisson3DSolver {
private:
    int N;
    size_t size;
    cufftComplex *d_rho, *d_phi;
    float *d_kx, *d_ky, *d_kz;
    cufftHandle plan_forward, plan_backward;
    dim3 block_size, grid_size;

public:
    Poisson3DSolver(int grid_size_n) : N(grid_size_n) {
        size = N * N * N;

        // 分配 GPU 記憶體
        CUDA_CHECK(cudaMalloc(&d_rho, sizeof(cufftComplex) * size));
        CUDA_CHECK(cudaMalloc(&d_phi, sizeof(cufftComplex) * size));
        CUDA_CHECK(cudaMalloc(&d_kx, sizeof(float) * size));
        CUDA_CHECK(cudaMalloc(&d_ky, sizeof(float) * size));
        CUDA_CHECK(cudaMalloc(&d_kz, sizeof(float) * size));

        // 設置 CUDA 執行配置
        block_size = dim3(8, 8, 8);
        grid_size = dim3((N + block_size.x - 1) / block_size.x,
                        (N + block_size.y - 1) / block_size.y,
                        (N + block_size.z - 1) / block_size.z);

        // 創建 cuFFT 計劃
        CUFFT_CHECK(cufftPlan3d(&plan_forward, N, N, N, CUFFT_C2C));
        CUFFT_CHECK(cufftPlan3d(&plan_backward, N, N, N, CUFFT_C2C));

        // 設置 k 向量
        setup_k_vectors<<<grid_size, block_size>>>(d_kx, d_ky, d_kz, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~Poisson3DSolver() {
        cudaFree(d_rho);
        cudaFree(d_phi);
        cudaFree(d_kx);
        cudaFree(d_ky);
        cudaFree(d_kz);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_backward);
    }

    void solve_point_charge(float charge = 1.0f) {
            // 設置點電荷源項
            setup_point_charge<<<grid_size, block_size>>>(d_rho, N, charge);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 前向 FFT: ρ(r) -> ρ(k)
            CUFFT_CHECK(cufftExecC2C(plan_forward, d_rho, d_rho, CUFFT_FORWARD));

            // 在動量空間求解 Poisson 方程
            solve_poisson_3d<<<grid_size, block_size>>>(d_rho, d_phi, d_kx, d_ky, d_kz, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 逆向 FFT: φ(k) -> φ(r)
            CUFFT_CHECK(cufftExecC2C(plan_backward, d_phi, d_phi, CUFFT_INVERSE));

            // Declaration of scale_result kernel before use
            __global__ void scale_result(cufftComplex *data, float scale, int N);

            // 正規化
            float norm = 1.0f / (N * N * N);
            scale_result<<<grid_size, block_size>>>(d_phi, norm, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    std::vector<float> get_diagonal_potential() {
        std::vector<cufftComplex> h_phi(size);
        CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phi, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost));

        std::vector<float> diagonal(N);
        for (int i = 0; i < N; ++i) {
            int idx = i * N * N + i * N + i; // 對角線 (i,i,i)
            diagonal[i] = h_phi[idx].x;
        }
        return diagonal;
    }

    std::vector<float> get_x_axis_potential() {
        std::vector<cufftComplex> h_phi(size);
        CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phi, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost));

        std::vector<float> x_axis(N);
        for (int i = 0; i < N; ++i) {
            int idx = i; // x 軸 (i,0,0)
            x_axis[i] = h_phi[idx].x;
        }
        return x_axis;
    }

    double benchmark_performance() {
        auto start = std::chrono::high_resolution_clock::now();
        solve_point_charge();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

    size_t get_memory_usage() {
        return sizeof(cufftComplex) * size * 2 + sizeof(float) * size * 3;
    }
};

// 正規化結果
__global__ void scale_result(cufftComplex *data, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;
    data[idx].x *= scale;
    data[idx].y *= scale;
}

// 理論解析解 (用於驗證)
float analytical_potential(float r, float charge = 1.0f) {
    if (r < 1e-10) return 0.0f; // 避免奇異性
    return charge / (4.0f * M_PI * r);
}

// 驗證物理正確性
void verify_solution(const std::vector<float>& diagonal, const std::vector<float>& x_axis, int N) {
    std::cout << "\n=== Physical Verification ===\n";

    // 檢查對角線
    std::cout << "Diagonal potential (r = i*sqrt(3)):\n";
    std::cout << "i\tr\tNumerical\tAnalytical\tError\n";
    std::cout << std::string(60, '-') << "\n";

    for (int i = 1; i < std::min(N, 10); ++i) {
        float r = i * std::sqrt(3.0f);
        float numerical = diagonal[i];
        float analytical = analytical_potential(r);
        float error = std::abs(numerical - analytical) / analytical * 100;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << i << "\t" << r << "\t" << numerical << "\t\t"
                  << analytical << "\t\t" << error << "%\n";
    }

    // 檢查 x 軸
    std::cout << "\nX-axis potential (r = i):\n";
    std::cout << "i\tr\tNumerical\tAnalytical\tError\n";
    std::cout << std::string(60, '-') << "\n";

    for (int i = 1; i < std::min(N, 10); ++i) {
        float r = (float)i;
        float numerical = x_axis[i];
        float analytical = analytical_potential(r);
        float error = std::abs(numerical - analytical) / analytical * 100;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << i << "\t" << r << "\t" << numerical << "\t\t"
                  << analytical << "\t\t" << error << "%\n";
    }
}

// 測試最大格點大小
void test_maximum_grid_size() {
    std::cout << "\n=== Maximum Grid Size Test ===\n";

    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    std::cout << "Available GPU memory: " << free_memory / (1024*1024) << " MB\n";

    std::vector<int> test_sizes = {32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512};

    std::cout << "Grid Size\tMemory (MB)\tTime (s)\tStatus\n";
    std::cout << std::string(50, '-') << "\n";

    for (int N : test_sizes) {
        try {
            Poisson3DSolver solver(N);
            size_t memory_usage = solver.get_memory_usage();

            if (memory_usage > free_memory) {
                std::cout << N << "³\t\t" << memory_usage/(1024*1024) << "\t\t-\t\tMemory exceeded\n";
                break;
            }

            double time = solver.benchmark_performance();
            std::cout << N << "³\t\t" << memory_usage/(1024*1024) << "\t\t"
                      << std::fixed << std::setprecision(3) << time << "\t\tSuccess\n";

        } catch (...) {
            std::cout << N << "³\t\t-\t\t-\t\tFailed\n";
            break;
        }
    }
}

// 輸出結果到檔案
void save_results(const std::vector<float>& diagonal, const std::vector<float>& x_axis, int N) {
    std::ofstream file("poisson_3d_results.txt");
    file << "# 3D Poisson Equation Results (N=" << N << ")\n";
    file << "# i\tDiagonal\tX-axis\n";

    for (int i = 0; i < N; ++i) {
        file << i << "\t" << diagonal[i] << "\t" << x_axis[i] << "\n";
    }
    file.close();
}

int main() {
    std::cout << "3D Poisson Equation Solver using cuFFT\n";
    std::cout << std::string(50, '=') << "\n";

    try {
        // 檢查 GPU 資訊
        int device;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n\n";

        // 問題 1: 32³ 格點求解
        std::cout << "=== 32x32x32 Grid Solution ===\n";
        int N = 32;
        Poisson3DSolver solver(N);

        auto start = std::chrono::high_resolution_clock::now();
        solver.solve_point_charge(1.0f);
        auto end = std::chrono::high_resolution_clock::now();
        double solve_time = std::chrono::duration<double>(end - start).count();

        std::cout << "Solution time: " << solve_time << " seconds\n";

        // 獲取對角線和 x 軸的電位
        auto diagonal = solver.get_diagonal_potential();
        auto x_axis = solver.get_x_axis_potential();

        // 驗證物理正確性
        verify_solution(diagonal, x_axis, N);

        // 儲存結果
        save_results(diagonal, x_axis, N);
        std::cout << "\nResults saved to poisson_3d_results.txt\n";

        // 問題 2: 測試最大格點大小
        test_maximum_grid_size();

        std::cout << "\nAll tests completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
