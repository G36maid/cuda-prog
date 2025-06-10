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

// CPU reference Monte Carlo integration
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

// CUDA kernel for Monte Carlo integration
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

// Kernel for combining results from multiple GPUs
__global__ void combine_results_kernel(
    double* gpu1_results,
    double* gpu2_results,
    double* combined_results,
    int num_threads,
    size_t N1,
    size_t N2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        double sum1 = 0.0, sum2_1 = 0.0;
        double sum2 = 0.0, sum2_2 = 0.0;

        // Sum GPU1 results
        for (int i = 0; i < num_threads; ++i) {
            sum1 += gpu1_results[2 * i];
            sum2_1 += gpu1_results[2 * i + 1];
        }

        // Sum GPU2 results
        for (int i = 0; i < num_threads; ++i) {
            sum2 += gpu2_results[2 * i];
            sum2_2 += gpu2_results[2 * i + 1];
        }

        // Store combined results
        combined_results[0] = sum1 + sum2;           // total sum
        combined_results[1] = sum2_1 + sum2_2;      // total sum of squares
        combined_results[2] = (double)(N1 + N2);    // total N
    }
}

// Check P2P capability between GPUs
bool check_p2p_capability(int gpu1, int gpu2) {
    int can_access_peer = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, gpu1, gpu2));
    return can_access_peer != 0;
}

// Enable P2P access between GPUs
void enable_p2p_access(int gpu1, int gpu2) {
    if (check_p2p_capability(gpu1, gpu2)) {
        CUDA_CHECK(cudaSetDevice(gpu1));
        cudaError_t err = cudaDeviceEnablePeerAccess(gpu2, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            CUDA_CHECK(err);
        }

        CUDA_CHECK(cudaSetDevice(gpu2));
        err = cudaDeviceEnablePeerAccess(gpu1, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            CUDA_CHECK(err);
        }
        std::cout << "P2P access enabled between GPU " << gpu1 << " and GPU " << gpu2 << "\n";
    } else {
        std::cout << "P2P not supported between GPU " << gpu1 << " and GPU " << gpu2 << "\n";
    }
}

// Single GPU Monte Carlo
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

// P2P optimized dual GPU Monte Carlo
void monte_carlo_dual_gpu_p2p(int gpu1_id, int gpu2_id, size_t N, double& mean, double& stddev, unsigned long long seed = DEFAULT_SEED) {
    // Enable P2P access
    enable_p2p_access(gpu1_id, gpu2_id);

    size_t N1 = N / 2;
    size_t N2 = N - N1;

    int block_size = 256;
    int num_blocks = 256;
    int num_threads = block_size * num_blocks;

    // Allocate memory on both GPUs
    double *d_results1, *d_results2, *d_combined;

    // GPU1 setup
    CUDA_CHECK(cudaSetDevice(gpu1_id));
    CUDA_CHECK(cudaMalloc(&d_results1, sizeof(double) * 2 * num_threads));
    CUDA_CHECK(cudaMalloc(&d_combined, sizeof(double) * 3)); // For final results

    // GPU2 setup
    CUDA_CHECK(cudaSetDevice(gpu2_id));
    CUDA_CHECK(cudaMalloc(&d_results2, sizeof(double) * 2 * num_threads));

    // Create CUDA streams for concurrent execution
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaSetDevice(gpu1_id));
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaSetDevice(gpu2_id));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Launch kernels on both GPUs concurrently
    CUDA_CHECK(cudaSetDevice(gpu1_id));
    monte_carlo_kernel<<<num_blocks, block_size, 0, stream1>>>(N1, d_results1, seed);

    CUDA_CHECK(cudaSetDevice(gpu2_id));
    monte_carlo_kernel<<<num_blocks, block_size, 0, stream2>>>(N2, d_results2, seed + 1000000);

    // Wait for both kernels to complete
    CUDA_CHECK(cudaSetDevice(gpu1_id));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaSetDevice(gpu2_id));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Use P2P to combine results on GPU1
    CUDA_CHECK(cudaSetDevice(gpu1_id));
    combine_results_kernel<<<1, 1>>>(d_results1, d_results2, d_combined, num_threads, N1, N2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy final results back to host
    std::vector<double> h_combined(3);
    CUDA_CHECK(cudaMemcpy(h_combined.data(), d_combined, sizeof(double) * 3, cudaMemcpyDeviceToHost));

    // Calculate final statistics
    double total_sum = h_combined[0];
    double total_sum2 = h_combined[1];
    size_t total_N = (size_t)h_combined[2];

    mean = total_sum / total_N;
    stddev = std::sqrt((total_sum2 / total_N - mean * mean) / total_N);

    // Cleanup
    CUDA_CHECK(cudaSetDevice(gpu1_id));
    CUDA_CHECK(cudaFree(d_results1));
    CUDA_CHECK(cudaFree(d_combined));
    CUDA_CHECK(cudaStreamDestroy(stream1));

    CUDA_CHECK(cudaSetDevice(gpu2_id));
    CUDA_CHECK(cudaFree(d_results2));
    CUDA_CHECK(cudaStreamDestroy(stream2));
}

// Fallback dual GPU without P2P
void monte_carlo_dual_gpu_fallback(int gpu1_id, int gpu2_id, size_t N, double& mean, double& stddev, unsigned long long seed = DEFAULT_SEED) {
    size_t N1 = N / 2;
    size_t N2 = N - N1;

    double mean1, stddev1, mean2, stddev2;

    auto gpu1_task = std::async(std::launch::async, [&]() {
        monte_carlo_gpu(gpu1_id, N1, mean1, stddev1, seed);
    });

    auto gpu2_task = std::async(std::launch::async, [&]() {
        monte_carlo_gpu(gpu2_id, N2, mean2, stddev2, seed + 1000000);
    });

    gpu1_task.wait();
    gpu2_task.wait();

    double sum1 = mean1 * N1;
    double sum2 = mean2 * N2;
    double sum2_1 = (stddev1 * stddev1 * N1 + mean1 * mean1) * N1;
    double sum2_2 = (stddev2 * stddev2 * N2 + mean2 * mean2) * N2;

    mean = (sum1 + sum2) / N;
    stddev = std::sqrt(((sum2_1 + sum2_2) / N - mean * mean) / N);
}

// Smart dual GPU function that chooses P2P or fallback
void monte_carlo_dual_gpu_smart(int gpu1_id, int gpu2_id, size_t N, double& mean, double& stddev, unsigned long long seed = DEFAULT_SEED) {
    if (check_p2p_capability(gpu1_id, gpu2_id)) {
        monte_carlo_dual_gpu_p2p(gpu1_id, gpu2_id, N, mean, stddev, seed);
    } else {
        std::cout << "Using fallback method (no P2P support)\n";
        monte_carlo_dual_gpu_fallback(gpu1_id, gpu2_id, N, mean, stddev, seed);
    }
}

int get_gpu_count() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    return device_count;
}

void display_gpu_info() {
    int device_count = get_gpu_count();
    std::cout << "Available GPUs: " << device_count << "\n";

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "GPU " << i << ": " << prop.name
                  << " (Compute Capability: " << prop.major << "." << prop.minor << ")\n";
    }

    // Check P2P capabilities
    if (device_count >= 2) {
        std::cout << "\nP2P Capabilities:\n";
        for (int i = 0; i < device_count; ++i) {
            for (int j = i + 1; j < device_count; ++j) {
                bool p2p_capable = check_p2p_capability(i, j);
                std::cout << "GPU " << i << " <-> GPU " << j << ": "
                         << (p2p_capable ? "Supported" : "Not Supported") << "\n";
            }
        }
    }
    std::cout << "\n";
}

struct ResultRow {
    size_t N;
    double cpu_mean, cpu_stddev, cpu_time;
    double gpu1_mean, gpu1_stddev, gpu1_time;
    double gpu2_mean, gpu2_stddev, gpu2_time;
    double dual_gpu_mean, dual_gpu_stddev, dual_gpu_time;
    double dual_gpu_p2p_mean, dual_gpu_p2p_stddev, dual_gpu_p2p_time;
};

void run_comprehensive_benchmark() {
    int gpu_count = get_gpu_count();
    if (gpu_count < 2) {
        std::cerr << "Warning: Less than 2 GPUs available. Using GPU 0 for both tests.\n";
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

        // GPU 2
        auto t5 = std::chrono::high_resolution_clock::now();
        monte_carlo_gpu(gpu2_id, N, row.gpu2_mean, row.gpu2_stddev, DEFAULT_SEED + n + 100);
        auto t6 = std::chrono::high_resolution_clock::now();
        row.gpu2_time = std::chrono::duration<double>(t6 - t5).count();

        // Dual GPU (fallback method)
        auto t7 = std::chrono::high_resolution_clock::now();
        monte_carlo_dual_gpu_fallback(gpu1_id, gpu2_id, N, row.dual_gpu_mean, row.dual_gpu_stddev, DEFAULT_SEED + n + 200);
        auto t8 = std::chrono::high_resolution_clock::now();
        row.dual_gpu_time = std::chrono::duration<double>(t8 - t7).count();

        // Dual GPU with P2P optimization
        auto t9 = std::chrono::high_resolution_clock::now();
        monte_carlo_dual_gpu_smart(gpu1_id, gpu2_id, N, row.dual_gpu_p2p_mean, row.dual_gpu_p2p_stddev, DEFAULT_SEED + n + 300);
        auto t10 = std::chrono::high_resolution_clock::now();
        row.dual_gpu_p2p_time = std::chrono::duration<double>(t10 - t9).count();

        results.push_back(row);
        std::cout << " Done\n";
    }

    // Display results
    std::cout << "\n" << std::string(140, '=') << "\n";
    std::cout << "Monte Carlo 10D Integration - P2P Optimized Benchmark Results\n";
    std::cout << std::string(140, '=') << "\n\n";

    std::cout << std::setw(10) << "N"
              << std::setw(12) << "CPU Time(s)"
              << std::setw(12) << "GPU1 Time(s)"
              << std::setw(12) << "GPU2 Time(s)"
              << std::setw(12) << "Dual Time(s)"
              << std::setw(12) << "P2P Time(s)"
              << std::setw(12) << "P2P Speedup"
              << std::setw(12) << "P2P vs Dual"
              << "\n";
    std::cout << std::string(140, '-') << "\n";

    std::cout << std::fixed << std::setprecision(6);
    for (const auto& row : results) {
        double p2p_speedup = row.cpu_time / row.dual_gpu_p2p_time;
        double p2p_vs_dual = row.dual_gpu_time / row.dual_gpu_p2p_time;
        std::cout << std::setw(10) << row.N
                  << std::setw(12) << row.cpu_time
                  << std::setw(12) << row.gpu1_time
                  << std::setw(12) << row.gpu2_time
                  << std::setw(12) << row.dual_gpu_time
                  << std::setw(12) << row.dual_gpu_p2p_time
                  << std::setw(12) << p2p_speedup << "x"
                  << std::setw(12) << p2p_vs_dual << "x"
                  << "\n";
    }

    // Performance analysis
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "P2P Performance Analysis\n";
    std::cout << std::string(60, '=') << "\n";

    if (!results.empty()) {
        const auto& last_result = results.back();
        double p2p_improvement = (last_result.dual_gpu_time - last_result.dual_gpu_p2p_time) / last_result.dual_gpu_time * 100;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "For largest problem size (N = " << last_result.N << "):\n";
        std::cout << "P2P vs Standard Dual GPU improvement: " << p2p_improvement << "%\n";
        std::cout << "P2P communication overhead reduction: "
                  << (last_result.dual_gpu_time / last_result.dual_gpu_p2p_time) << "x faster\n";
    }
}

int main() {
    std::cout << "Monte Carlo 10D Integration - P2P Optimized Multi-GPU Benchmark\n";
    std::cout << std::string(70, '=') << "\n\n";

    try {
        run_comprehensive_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
