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
#include <thread>
#include <future>

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

// CPU reference Monte Carlo integration (simple sampling)
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

// CUDA kernel for simple sampling
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

// Host function to launch CUDA kernel and compute mean/stddev
void monte_carlo_gpu(int gpu_id, size_t N, double& mean, double& stddev, unsigned long long seed = DEFAULT_SEED) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

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

// Dual GPU Monte Carlo integration
void monte_carlo_dual_gpu(int gpu1_id, int gpu2_id, size_t N, double& mean, double& stddev, unsigned long long seed = DEFAULT_SEED) {
    // Split work between two GPUs
    size_t N1 = N / 2;
    size_t N2 = N - N1;

    double mean1, stddev1, mean2, stddev2;

    // Launch computations on both GPUs concurrently using threads
    auto gpu1_task = std::async(std::launch::async, [&]() {
        monte_carlo_gpu(gpu1_id, N1, mean1, stddev1, seed);
    });

    auto gpu2_task = std::async(std::launch::async, [&]() {
        monte_carlo_gpu(gpu2_id, N2, mean2, stddev2, seed + 1000000);
    });

    // Wait for both tasks to complete
    gpu1_task.wait();
    gpu2_task.wait();

    // Combine results
    double sum1 = mean1 * N1;
    double sum2 = mean2 * N2;
    double sum2_1 = (stddev1 * stddev1 * N1 + mean1 * mean1) * N1;
    double sum2_2 = (stddev2 * stddev2 * N2 + mean2 * mean2) * N2;

    mean = (sum1 + sum2) / N;
    stddev = std::sqrt(((sum2_1 + sum2_2) / N - mean * mean) / N);
}

// Get available GPU count
int get_gpu_count() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    return device_count;
}

// Display GPU information
void display_gpu_info() {
    int device_count = get_gpu_count();
    std::cout << "Available GPUs: " << device_count << "\n";

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "GPU " << i << ": " << prop.name
                  << " (Compute Capability: " << prop.major << "." << prop.minor << ")\n";
    }
    std::cout << "\n";
}

struct ResultRow {
    size_t N;
    double cpu_mean, cpu_stddev, cpu_time;
    double gpu1_mean, gpu1_stddev, gpu1_time;
    double gpu2_mean, gpu2_stddev, gpu2_time;
    double dual_gpu_mean, dual_gpu_stddev, dual_gpu_time;
};

void run_comprehensive_benchmark() {
    int gpu_count = get_gpu_count();
    if (gpu_count < 2) {
        std::cerr << "Warning: Less than 2 GPUs available. Dual GPU benchmark will use GPU 0 twice.\n";
    }

    int gpu1_id = 0;
    int gpu2_id = (gpu_count > 1) ? 1 : 0;

    display_gpu_info();

    std::vector<ResultRow> results;

    for (int n = N_MIN; n <= N_MAX; ++n) {
        size_t N = 1ULL << n;
        ResultRow row;
        row.N = N;

        std::cout << "Processing N = " << N << "..." << std::flush;

        // CPU
        auto t1 = std::chrono::high_resolution_clock::now();
        monte_carlo_cpu(N, row.cpu_mean, row.cpu_stddev, DEFAULT_SEED + n);
        auto t2 = std::chrono::high_resolution_clock::now();
        row.cpu_time = std::chrono::duration<double>(t2 - t1).count();

        // GPU 1
        auto t3 = std::chrono::high_resolution_clock::now();
        monte_carlo_gpu(gpu1_id, N, row.gpu1_mean, row.gpu1_stddev, DEFAULT_SEED + n);
        auto t4 = std::chrono::high_resolution_clock::now();
        row.gpu1_time = std::chrono::duration<double>(t4 - t3).count();

        // GPU 2 (if available)
        auto t5 = std::chrono::high_resolution_clock::now();
        monte_carlo_gpu(gpu2_id, N, row.gpu2_mean, row.gpu2_stddev, DEFAULT_SEED + n + 100);
        auto t6 = std::chrono::high_resolution_clock::now();
        row.gpu2_time = std::chrono::duration<double>(t6 - t5).count();

        // Dual GPU
        auto t7 = std::chrono::high_resolution_clock::now();
        monte_carlo_dual_gpu(gpu1_id, gpu2_id, N, row.dual_gpu_mean, row.dual_gpu_stddev, DEFAULT_SEED + n + 200);
        auto t8 = std::chrono::high_resolution_clock::now();
        row.dual_gpu_time = std::chrono::duration<double>(t8 - t7).count();

        results.push_back(row);
        std::cout << " Done\n";
    }

    // Display comprehensive results
    std::cout << "\n" << std::string(120, '=') << "\n";
    std::cout << "Monte Carlo 10D Integration Comprehensive Benchmark Results\n";
    std::cout << std::string(120, '=') << "\n\n";

    std::cout << std::setw(10) << "N"
              << std::setw(12) << "CPU Mean"
              << std::setw(12) << "CPU Time(s)"
              << std::setw(12) << "GPU1 Mean"
              << std::setw(12) << "GPU1 Time(s)"
              << std::setw(12) << "GPU2 Mean"
              << std::setw(12) << "GPU2 Time(s)"
              << std::setw(12) << "Dual Mean"
              << std::setw(12) << "Dual Time(s)"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(120, '-') << "\n";

    std::cout << std::fixed << std::setprecision(6);
    for (const auto& row : results) {
        double speedup = row.cpu_time / row.dual_gpu_time;
        std::cout << std::setw(10) << row.N
                  << std::setw(12) << row.cpu_mean
                  << std::setw(12) << row.cpu_time
                  << std::setw(12) << row.gpu1_mean
                  << std::setw(12) << row.gpu1_time
                  << std::setw(12) << row.gpu2_mean
                  << std::setw(12) << row.gpu2_time
                  << std::setw(12) << row.dual_gpu_mean
                  << std::setw(12) << row.dual_gpu_time
                  << std::setw(12) << speedup << "x"
                  << "\n";
    }

    // Performance analysis
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Performance Analysis\n";
    std::cout << std::string(60, '=') << "\n";

    if (!results.empty()) {
        const auto& last_result = results.back();
        double cpu_vs_gpu1 = last_result.cpu_time / last_result.gpu1_time;
        double cpu_vs_dual = last_result.cpu_time / last_result.dual_gpu_time;
        double gpu1_vs_dual = last_result.gpu1_time / last_result.dual_gpu_time;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "For largest problem size (N = " << last_result.N << "):\n";
        std::cout << "CPU vs Single GPU speedup: " << cpu_vs_gpu1 << "x\n";
        std::cout << "CPU vs Dual GPU speedup: " << cpu_vs_dual << "x\n";
        std::cout << "Single GPU vs Dual GPU speedup: " << gpu1_vs_dual << "x\n";
        std::cout << "Dual GPU efficiency: " << (gpu1_vs_dual / 2.0 * 100) << "%\n";
    }
}

int main() {
    std::cout << "Monte Carlo 10D Integration - Multi-GPU Benchmark\n";
    std::cout << std::string(50, '=') << "\n\n";

    try {
        run_comprehensive_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
