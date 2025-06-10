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

// Read GPU ID from stdin
int read_gpu_id_from_stdin() {
    std::string line;
    int gpu_id = 0;
    while (std::getline(std::cin, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '=' || line[0] == '#') continue;
        std::istringstream iss(line);
        int id;
        if (iss >> id) {
            gpu_id = id;
            break;
        }
    }
    return gpu_id;
}

struct ResultRow {
    size_t N;
    double cpu_mean, cpu_stddev, cpu_time;
    double gpu_mean, gpu_stddev, gpu_time;
};

void run_single_gpu_benchmark(int gpu_id) {
    std::vector<ResultRow> results;

    for (int n = N_MIN; n <= N_MAX; ++n) {
        size_t N = 1ULL << n;
        ResultRow row;
        row.N = N;

        // CPU
        auto t1 = std::chrono::high_resolution_clock::now();
        monte_carlo_cpu(N, row.cpu_mean, row.cpu_stddev, DEFAULT_SEED + n);
        auto t2 = std::chrono::high_resolution_clock::now();
        row.cpu_time = std::chrono::duration<double>(t2 - t1).count();

        // GPU
        auto t3 = std::chrono::high_resolution_clock::now();
        monte_carlo_gpu(gpu_id, N, row.gpu_mean, row.gpu_stddev, DEFAULT_SEED + n);
        auto t4 = std::chrono::high_resolution_clock::now();
        row.gpu_time = std::chrono::duration<double>(t4 - t3).count();

        results.push_back(row);
    }

    // Write results to stdout
    std::cout << "Monte Carlo 10D Integration Results\n\n";
    std::cout << "GPU ID: " << gpu_id << "\n\n";
    std::cout << std::setw(10) << "N"
              << std::setw(15) << "CPU Mean"
              << std::setw(15) << "CPU Stddev"
              << std::setw(15) << "CPU Time(s)"
              << std::setw(15) << "GPU Mean"
              << std::setw(15) << "GPU Stddev"
              << std::setw(15) << "GPU Time(s)"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    std::cout << std::fixed << std::setprecision(8);
    for (const auto& row : results) {
        std::cout << std::setw(10) << row.N
                  << std::setw(15) << row.cpu_mean
                  << std::setw(15) << row.cpu_stddev
                  << std::setw(15) << row.cpu_time
                  << std::setw(15) << row.gpu_mean
                  << std::setw(15) << row.gpu_stddev
                  << std::setw(15) << row.gpu_time
                  << "\n";
    }
}

int main() {
    int gpu_id = read_gpu_id_from_stdin();
    run_single_gpu_benchmark(gpu_id);
    return 0;
}
