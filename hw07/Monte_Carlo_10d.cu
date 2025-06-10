#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <omp.h>

constexpr int NDIM = 10;
constexpr int N_MIN = 2;
constexpr int N_MAX = 16;
constexpr unsigned int DEFAULT_SEED = 1234;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Device function: integrand
__device__ __host__ inline double integrand(const double* x) {
    double sum = 0.0;
    for (int i = 0; i < NDIM; ++i)
        sum += x[i] * x[i];
    return 1.0 / (1.0 + sum);
}

// CPU Monte Carlo integration
void monte_carlo_cpu(size_t N, double& mean, double& stddev, unsigned int seed = DEFAULT_SEED) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double sum = 0.0, sum2 = 0.0;
    std::vector<double> x(NDIM);

    for (size_t i = 0; i < N; ++i) {
        for (int d = 0; d < NDIM; ++d)
            x[d] = dist(rng);
        double val = integrand(x.data());
        sum += val;
        sum2 += val * val;
    }
    mean = sum / N;
    stddev = std::sqrt((sum2 / N - mean * mean) / N);
}

// CUDA kernel
__global__ void monte_carlo_kernel(
    size_t N,
    double* results,
    unsigned long long seed)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;

    double thread_sum = 0.0;
    double thread_sum2 = 0.0;

    unsigned long long local_seed = seed + tid * 7919ULL;
    for (size_t i = tid; i < N; i += total_threads) {
        double x[NDIM];
        for (int d = 0; d < NDIM; ++d) {
            local_seed = 6364136223846793005ULL * local_seed + 1;
            x[d] = (double)(local_seed & 0xFFFFFFFFFFFFULL) / (double)0x1000000000000ULL;
        }
        double val = integrand(x);
        thread_sum += val;
        thread_sum2 += val * val;
    }
    results[2 * tid] = thread_sum;
    results[2 * tid + 1] = thread_sum2;
}

// Single GPU Monte Carlo
void monte_carlo_single_gpu(size_t N, double& mean, double& stddev, unsigned long long seed = DEFAULT_SEED) {
    CUDA_CHECK(cudaSetDevice(0));

    int block_size = 256;
    int num_blocks = 256;
    int num_threads = block_size * num_blocks;

    double* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(double) * 2 * num_threads));

    monte_carlo_kernel<<<num_blocks, block_size>>>(N, d_results, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_results(2 * num_threads);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, sizeof(double) * 2 * num_threads, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_results));

    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < num_threads; ++i) {
        sum += h_results[2 * i];
        sum2 += h_results[2 * i + 1];
    }
    mean = sum / N;
    stddev = std::sqrt((sum2 / N - mean * mean) / N);
}

// Dual GPU Monte Carlo with OpenMP
void monte_carlo_dual_gpu(size_t N, double& mean, double& stddev, unsigned long long seed = DEFAULT_SEED) {
    // 分配工作負載
    size_t N1 = N / 2;
    size_t N2 = N - N1;

    // 儲存結果
    double means[2], stddevs[2];
    size_t Ns[2] = {N1, N2};

    // 設定 OpenMP 使用 2 個線程
    omp_set_num_threads(2);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int gpu_id = thread_id; // Thread 0 -> GPU 0, Thread 1 -> GPU 1

        CUDA_CHECK(cudaSetDevice(gpu_id));

        int block_size = 256;
        int num_blocks = 256;
        int num_threads = block_size * num_blocks;

        double* d_results;
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(double) * 2 * num_threads));

        monte_carlo_kernel<<<num_blocks, block_size>>>(Ns[thread_id], d_results, seed + thread_id * 1000000);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> h_results(2 * num_threads);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, sizeof(double) * 2 * num_threads, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_results));

        double sum = 0.0, sum2 = 0.0;
        for (int i = 0; i < num_threads; ++i) {
            sum += h_results[2 * i];
            sum2 += h_results[2 * i + 1];
        }
        means[thread_id] = sum / Ns[thread_id];
        stddevs[thread_id] = std::sqrt((sum2 / Ns[thread_id] - means[thread_id] * means[thread_id]) / Ns[thread_id]);
    }

    // 合併結果
    double sum1 = means[0] * N1;
    double sum2 = means[1] * N2;
    double sum2_1 = (stddevs[0] * stddevs[0] * N1 + means[0] * means[0]) * N1;
    double sum2_2 = (stddevs[1] * stddevs[1] * N2 + means[1] * means[1]) * N2;

    mean = (sum1 + sum2) / N;
    stddev = std::sqrt(((sum2_1 + sum2_2) / N - mean * mean) / N);
}

void display_system_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::cout << "System Configuration:\n";
    std::cout << "Available GPUs: " << device_count << "\n";

    for (int i = 0; i < std::min(device_count, 2); ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "GPU " << i << ": " << prop.name
                  << " (Memory: " << prop.totalGlobalMem / (1024*1024) << " MB)\n";
    }
    std::cout << "\n";
}

struct BenchmarkResult {
    size_t N;
    double cpu_mean, cpu_stddev, cpu_time;
    double single_gpu_mean, single_gpu_stddev, single_gpu_time;
    double dual_gpu_mean, dual_gpu_stddev, dual_gpu_time;
};

void run_benchmark() {
    display_system_info();

    std::vector<BenchmarkResult> results;

    std::cout << "Running Monte Carlo 10D Integration Benchmark...\n";
    std::cout << std::string(60, '-') << "\n";

    for (int n = N_MIN; n <= N_MAX; ++n) {
        size_t N = 1ULL << n;
        BenchmarkResult result;
        result.N = N;

        std::cout << "N = " << std::setw(8) << N << " | ";

        // CPU 測試
        auto start = std::chrono::high_resolution_clock::now();
        monte_carlo_cpu(N, result.cpu_mean, result.cpu_stddev, DEFAULT_SEED + n);
        auto end = std::chrono::high_resolution_clock::now();
        result.cpu_time = std::chrono::duration<double>(end - start).count();
        std::cout << "CPU: " << std::setw(8) << std::fixed << std::setprecision(4) << result.cpu_time << "s | ";

        // 單 GPU 測試
        start = std::chrono::high_resolution_clock::now();
        monte_carlo_single_gpu(N, result.single_gpu_mean, result.single_gpu_stddev, DEFAULT_SEED + n);
        end = std::chrono::high_resolution_clock::now();
        result.single_gpu_time = std::chrono::duration<double>(end - start).count();
        std::cout << "1GPU: " << std::setw(8) << result.single_gpu_time << "s | ";

        // 雙 GPU 測試
        start = std::chrono::high_resolution_clock::now();
        monte_carlo_dual_gpu(N, result.dual_gpu_mean, result.dual_gpu_stddev, DEFAULT_SEED + n);
        end = std::chrono::high_resolution_clock::now();
        result.dual_gpu_time = std::chrono::duration<double>(end - start).count();
        std::cout << "2GPU: " << std::setw(8) << result.dual_gpu_time << "s";

        results.push_back(result);
        std::cout << "\n";
    }

    // 詳細結果表格
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "Detailed Benchmark Results\n";
    std::cout << std::string(100, '=') << "\n";

    std::cout << std::setw(10) << "N"
              << std::setw(12) << "CPU (s)"
              << std::setw(12) << "1GPU (s)"
              << std::setw(12) << "2GPU (s)"
              << std::setw(12) << "1GPU Speedup"
              << std::setw(12) << "2GPU Speedup"
              << std::setw(12) << "GPU Scaling"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& result : results) {
        double single_speedup = result.cpu_time / result.single_gpu_time;
        double dual_speedup = result.cpu_time / result.dual_gpu_time;
        double gpu_scaling = result.single_gpu_time / result.dual_gpu_time;

        std::cout << std::setw(10) << result.N
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.cpu_time
                  << std::setw(12) << result.single_gpu_time
                  << std::setw(12) << result.dual_gpu_time
                  << std::setw(12) << std::setprecision(2) << single_speedup << "x"
                  << std::setw(12) << dual_speedup << "x"
                  << std::setw(12) << gpu_scaling << "x"
                  << "\n";
    }

    // 性能總結
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Performance Summary\n";
    std::cout << std::string(60, '=') << "\n";

    if (!results.empty()) {
        const auto& best = results.back(); // 最大問題規模的結果
        double efficiency = (best.single_gpu_time / best.dual_gpu_time) / 2.0 * 100;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Best Performance (N = " << best.N << "):\n";
        std::cout << "- Single GPU vs CPU: " << (best.cpu_time / best.single_gpu_time) << "x speedup\n";
        std::cout << "- Dual GPU vs CPU: " << (best.cpu_time / best.dual_gpu_time) << "x speedup\n";
        std::cout << "- Dual GPU vs Single GPU: " << (best.single_gpu_time / best.dual_gpu_time) << "x speedup\n";
        std::cout << "- Dual GPU parallel efficiency: " << efficiency << "%\n";

        // // 根據研究結果[1][3]，GPU 在蒙地卡羅計算上通常能達到 35-500 倍的加速比
        // if (best.cpu_time / best.dual_gpu_time > 50) {
        //     std::cout << "\n✓ Excellent GPU acceleration achieved!\n";
        // } else if (best.cpu_time / best.dual_gpu_time > 20) {
        //     std::cout << "\n✓ Good GPU acceleration achieved.\n";
        // } else {
        //     std::cout << "\n⚠ GPU acceleration could be improved.\n";
        // }
    }
}

int main() {
    std::cout << "Monte Carlo 10D Integration - CPU vs 1GPU vs 2GPU Benchmark\n";
    std::cout << std::string(65, '=') << "\n\n";

    try {
        run_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
