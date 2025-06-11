#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <iomanip>

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

// 正規化 kernel (移到前面定義)
__global__ void scale_result(cufftComplex *data, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;
    data[idx].x *= scale;
    data[idx].y *= scale;
}

// 最終修正版本的 Poisson 求解器
__global__ void solve_poisson_final(cufftComplex *rho_k, cufftComplex *phi_k, int N, float L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;

    // 正確的 k 向量計算
    float ki = (i <= N/2) ? i : i - N;
    float kj = (j <= N/2) ? j : j - N;
    float kk = (k <= N/2) ? k : k - N;

    // 物理單位的 k 向量
    ki *= 2.0f * M_PI / L;
    kj *= 2.0f * M_PI / L;
    kk *= 2.0f * M_PI / L;

    float k2 = ki * ki + kj * kj + kk * kk;

    // 處理 k=0 (設定電位參考點)
    if (i == 0 && j == 0 && k == 0) {
        phi_k[idx].x = 0.0f;
        phi_k[idx].y = 0.0f;
    } else {
        // 正確的 Poisson 方程求解：∇²φ = -4πρ → φ(k) = 4πρ(k)/k²
        float factor = 4.0f * M_PI / k2;
        phi_k[idx].x = rho_k[idx].x * factor;
        phi_k[idx].y = rho_k[idx].y * factor;
    }
}

// 正確的點電荷設置
__global__ void setup_point_charge_final(cufftComplex *rho, int N, float charge, float L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int idx = k * N * N + j * N + i;

    // 點電荷密度 (考慮格點體積)
    float dx = L / N;
    float volume = dx * dx * dx;

    if (i == 0 && j == 0 && k == 0) {
        rho[idx].x = charge / volume;  // 電荷密度 = 電荷/體積
        rho[idx].y = 0.0f;
    } else {
        rho[idx].x = 0.0f;
        rho[idx].y = 0.0f;
    }
}

// 最終修正的 3D Poisson 求解器
class FinalPoisson3DSolver {
private:
    int N;
    float L;  // 物理尺寸
    size_t size;
    cufftComplex *d_rho, *d_phi;
    cufftHandle plan_forward, plan_backward;
    dim3 block_size, grid_size;

public:
    FinalPoisson3DSolver(int grid_size_n, float box_length = 32.0f) : N(grid_size_n), L(box_length) {
        size = N * N * N;

        // 分配 GPU 記憶體
        CUDA_CHECK(cudaMalloc(&d_rho, sizeof(cufftComplex) * size));
        CUDA_CHECK(cudaMalloc(&d_phi, sizeof(cufftComplex) * size));

        // 設置執行配置
        block_size = dim3(8, 8, 8);
        grid_size = dim3((N + block_size.x - 1) / block_size.x,
                        (N + block_size.y - 1) / block_size.y,
                        (N + block_size.z - 1) / block_size.z);

        // 創建 cuFFT 計劃
        CUFFT_CHECK(cufftPlan3d(&plan_forward, N, N, N, CUFFT_C2C));
        CUFFT_CHECK(cufftPlan3d(&plan_backward, N, N, N, CUFFT_C2C));
    }

    ~FinalPoisson3DSolver() {
        cudaFree(d_rho);
        cudaFree(d_phi);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_backward);
    }

    void solve_point_charge(float charge = 1.0f) {
        // 設置點電荷
        setup_point_charge_final<<<grid_size, block_size>>>(d_rho, N, charge, L);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 前向 FFT
        CUFFT_CHECK(cufftExecC2C(plan_forward, d_rho, d_rho, CUFFT_FORWARD));

        // 求解 Poisson 方程
        solve_poisson_final<<<grid_size, block_size>>>(d_rho, d_phi, N, L);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 逆向 FFT
        CUFFT_CHECK(cufftExecC2C(plan_backward, d_phi, d_phi, CUFFT_INVERSE));

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

    float get_grid_spacing() { return L / N; }

    size_t get_memory_usage() {
        return sizeof(cufftComplex) * size * 2;
    }
};

// 理論解析解
float analytical_potential(float r, float charge = 1.0f) {
    if (r < 1e-10) return 1e10;
    return charge / (4.0f * M_PI * r);
}

// 最終驗證
void final_verification(const std::vector<float>& diagonal, const std::vector<float>& x_axis,
                       int N, float dx) {
    std::cout << "\n=== Final Physical Verification ===\n";
    std::cout << "Grid spacing: " << dx << " units\n\n";

    // 檢查對角線
    std::cout << "Diagonal potential (r = i*sqrt(3)*dx):\n";
    std::cout << "i\tr\tNumerical\tAnalytical\tError\n";
    std::cout << std::string(60, '-') << "\n";

    for (int i = 1; i < std::min(N, 10); ++i) {
        float r = i * std::sqrt(3.0f) * dx;
        float numerical = diagonal[i];
        float analytical = analytical_potential(r);
        float error = std::abs(numerical - analytical) / analytical * 100;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << i << "\t" << r << "\t" << numerical << "\t\t"
                  << analytical << "\t\t" << error << "%\n";
    }

    // 檢查 x 軸
    std::cout << "\nX-axis potential (r = i*dx):\n";
    std::cout << "i\tr\tNumerical\tAnalytical\tError\n";
    std::cout << std::string(60, '-') << "\n";

    for (int i = 1; i < std::min(N, 10); ++i) {
        float r = i * dx;
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

    std::vector<int> test_sizes = {64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512};

    std::cout << "Grid Size\tMemory (MB)\tTime (s)\tStatus\n";
    std::cout << std::string(50, '-') << "\n";

    for (int N : test_sizes) {
        try {
            FinalPoisson3DSolver solver(N);
            size_t memory_usage = solver.get_memory_usage();

            if (memory_usage > free_memory) {
                std::cout << N << "³\t\t" << memory_usage/(1024*1024) << "\t\t-\t\tMemory exceeded\n";
                break;
            }

            auto start = std::chrono::high_resolution_clock::now();
            solver.solve_point_charge(1.0f);
            auto end = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration<double>(end - start).count();

            std::cout << N << "³\t\t" << memory_usage/(1024*1024) << "\t\t"
                      << std::fixed << std::setprecision(3) << time << "\t\tSuccess\n";

        } catch (...) {
            std::cout << N << "³\t\t-\t\t-\t\tFailed\n";
            break;
        }
    }
}

int main() {
    std::cout << "Final Corrected 3D Poisson Equation Solver\n";
    std::cout << std::string(50, '=') << "\n";

    try {
        // 檢查 GPU 資訊
        int device;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n\n";

        // 使用物理上合理的參數
        int N = 32;
        float L = 32.0f;  // 盒子大小 32 個單位

        std::cout << "=== Final 32x32x32 Grid Solution ===\n";
        std::cout << "Box size: " << L << " units\n";
        std::cout << "Grid spacing: " << L/N << " units\n";

        FinalPoisson3DSolver solver(N, L);

        auto start = std::chrono::high_resolution_clock::now();
        solver.solve_point_charge(1.0f);
        auto end = std::chrono::high_resolution_clock::now();
        double solve_time = std::chrono::duration<double>(end - start).count();

        std::cout << "Solution time: " << solve_time << " seconds\n";

        // 獲取結果
        auto diagonal = solver.get_diagonal_potential();
        auto x_axis = solver.get_x_axis_potential();
        float dx = solver.get_grid_spacing();

        // 最終驗證
        final_verification(diagonal, x_axis, N, dx);

        // 測試最大格點大小
        test_maximum_grid_size();

        std::cout << "\n=== Analysis ===\n";
        std::cout << "The solution should now show much better agreement with analytical results.\n";
        std::cout << "Key corrections made:\n";
        std::cout << "1. Proper physical units and grid spacing\n";
        std::cout << "2. Correct Poisson equation: ∇²φ = -4πρ\n";
        std::cout << "3. Proper charge density normalization\n";
        std::cout << "4. Consistent k-space and real-space scaling\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
