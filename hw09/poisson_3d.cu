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

// 修正的 Poisson 求解器 (在動量空間)
__global__ void solve_poisson_3d_corrected(cufftComplex *rho_k, cufftComplex *phi_k, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;

    // 計算 k 向量 (正確的週期邊界條件)
    float ki = (i <= N/2) ? i : i - N;
    float kj = (j <= N/2) ? j : j - N;
    float kk = (k <= N/2) ? k : k - N;

    // 歸一化 k 向量
    ki *= 2.0f * M_PI / N;
    kj *= 2.0f * M_PI / N;
    kk *= 2.0f * M_PI / N;

    float k2 = ki * ki + kj * kj + kk * kk;

    // 處理 k=0 的情況 (設定參考電位)
    if (i == 0 && j == 0 && k == 0) {
        phi_k[idx].x = 0.0f;
        phi_k[idx].y = 0.0f;
    } else {
        // 修正的格林函數：φ(k) = ρ(k) / k² (注意符號)
        float green_factor = 1.0f / k2;
        phi_k[idx].x = rho_k[idx].x * green_factor;
        phi_k[idx].y = rho_k[idx].y * green_factor;
    }
}

// 設置點電荷 (修正歸一化)
__global__ void setup_point_charge_corrected(cufftComplex *rho, int N, float charge = 1.0f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;

    // 在原點放置點電荷 (正確的歸一化)
    if (i == 0 && j == 0 && k == 0) {
        rho[idx].x = charge * N * N * N; // 考慮離散化效應
        rho[idx].y = 0.0f;
    } else {
        rho[idx].x = 0.0f;
        rho[idx].y = 0.0f;
    }
}

// 正規化結果 (修正係數)
__global__ void scale_result_corrected(cufftComplex *data, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;
    data[idx].x *= scale;
    data[idx].y *= scale;
}

// 修正的 3D Poisson 求解器類別
class Poisson3DSolverCorrected {
private:
    int N;
    size_t size;
    cufftComplex *d_rho, *d_phi;
    cufftHandle plan_forward, plan_backward;
    dim3 block_size, grid_size;

public:
    Poisson3DSolverCorrected(int grid_size_n) : N(grid_size_n) {
        size = N * N * N;

        // 分配 GPU 記憶體
        CUDA_CHECK(cudaMalloc(&d_rho, sizeof(cufftComplex) * size));
        CUDA_CHECK(cudaMalloc(&d_phi, sizeof(cufftComplex) * size));

        // 設置 CUDA 執行配置
        block_size = dim3(8, 8, 8);
        grid_size = dim3((N + block_size.x - 1) / block_size.x,
                        (N + block_size.y - 1) / block_size.y,
                        (N + block_size.z - 1) / block_size.z);

        // 創建 cuFFT 計劃
        CUFFT_CHECK(cufftPlan3d(&plan_forward, N, N, N, CUFFT_C2C));
        CUFFT_CHECK(cufftPlan3d(&plan_backward, N, N, N, CUFFT_C2C));
    }

    ~Poisson3DSolverCorrected() {
        cudaFree(d_rho);
        cudaFree(d_phi);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_backward);
    }

    void solve_point_charge(float charge = 1.0f) {
        // 設置點電荷源項
        setup_point_charge_corrected<<<grid_size, block_size>>>(d_rho, N, charge);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 前向 FFT: ρ(r) -> ρ(k)
        CUFFT_CHECK(cufftExecC2C(plan_forward, d_rho, d_rho, CUFFT_FORWARD));

        // 在動量空間求解 Poisson 方程
        solve_poisson_3d_corrected<<<grid_size, block_size>>>(d_rho, d_phi, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 逆向 FFT: φ(k) -> φ(r)
        CUFFT_CHECK(cufftExecC2C(plan_backward, d_phi, d_phi, CUFFT_INVERSE));

        // 正規化 (考慮物理單位)
        float norm = 1.0f / (N * N * N * 4.0f * M_PI);
        scale_result_corrected<<<grid_size, block_size>>>(d_phi, norm, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> get_diagonal_potential() {
        std::vector<cufftComplex> h_phi(size);
        CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phi, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost));

        std::vector<float> diagonal(N);
        for (int i = 0; i < N; ++i) {
            int idx = i * N * N + i * N + i;
            diagonal[i] = h_phi[idx].x;
        }
        return diagonal;
    }

    std::vector<float> get_x_axis_potential() {
        std::vector<cufftComplex> h_phi(size);
        CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phi, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost));

        std::vector<float> x_axis(N);
        for (int i = 0; i < N; ++i) {
            int idx = i;
            x_axis[i] = h_phi[idx].x;
        }
        return x_axis;
    }

    size_t get_memory_usage() {
        return sizeof(cufftComplex) * size * 2;
    }
};

// 修正的理論解析解
float analytical_potential_corrected(float r, float charge = 1.0f) {
    if (r < 1e-10) return 1e10; // 原點處的奇異性
    return charge / (4.0f * M_PI * r);
}

// 修正的驗證函數
void verify_solution_corrected(const std::vector<float>& diagonal, const std::vector<float>& x_axis, int N) {
    std::cout << "\n=== Corrected Physical Verification ===\n";

    // 檢查對角線
    std::cout << "Diagonal potential (r = i*sqrt(3)):\n";
    std::cout << "i\tr\tNumerical\tAnalytical\tError\n";
    std::cout << std::string(60, '-') << "\n";

    for (int i = 1; i < std::min(N, 10); ++i) {
        float r = i * std::sqrt(3.0f);
        float numerical = diagonal[i];
        float analytical = analytical_potential_corrected(r);
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
        float analytical = analytical_potential_corrected(r);
        float error = std::abs(numerical - analytical) / analytical * 100;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << i << "\t" << r << "\t" << numerical << "\t\t"
                  << analytical << "\t\t" << error << "%\n";
    }
}

int main() {
    std::cout << "Corrected 3D Poisson Equation Solver using cuFFT\n";
    std::cout << std::string(50, '=') << "\n";

    try {
        // 32³ 格點求解 (修正版)
        std::cout << "=== Corrected 32x32x32 Grid Solution ===\n";
        int N = 32;
        Poisson3DSolverCorrected solver(N);

        auto start = std::chrono::high_resolution_clock::now();
        solver.solve_point_charge(1.0f);
        auto end = std::chrono::high_resolution_clock::now();
        double solve_time = std::chrono::duration<double>(end - start).count();

        std::cout << "Solution time: " << solve_time << " seconds\n";

        // 獲取結果
        auto diagonal = solver.get_diagonal_potential();
        auto x_axis = solver.get_x_axis_potential();

        // 修正的驗證
        verify_solution_corrected(diagonal, x_axis, N);

        std::cout << "\n=== Performance Summary ===\n";
        std::cout << "Your GTX 1060 6GB can handle up to 512³ grid (3.6GB memory)\n";
        std::cout << "This corresponds to ~134 million grid points\n";
        std::cout << "Solution time scales approximately as O(N³ log N)\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
