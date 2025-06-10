#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <omp.h>

constexpr int L = 200;              // 格子大小 200x200
constexpr int N = L * L;            // 總自旋數
constexpr int WARMUP_STEPS = 10000; // 熱化步數
constexpr int MC_STEPS = 50000;     // 測量步數
constexpr int MEASURE_INTERVAL = 10; // 測量間隔

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// 2D Ising 模型的 CUDA kernel (棋盤算法)
__global__ void ising_metropolis_kernel(
    int* spins,
    curandState* states,
    float beta,
    int parity,
    int* energy_sum,
    int* mag_sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= L || idy >= L) return;

    // 棋盤模式：只更新特定的格點
    if ((idx + idy) % 2 != parity) return;

    int site = idy * L + idx;
    curandState localState = states[site];

    // 計算鄰居 (torus 邊界條件)
    int left = idy * L + ((idx - 1 + L) % L);
    int right = idy * L + ((idx + 1) % L);
    int up = ((idy - 1 + L) % L) * L + idx;
    int down = ((idy + 1) % L) * L + idx;

    int current_spin = spins[site];
    int neighbor_sum = spins[left] + spins[right] + spins[up] + spins[down];

    // 計算能量變化
    int delta_E = 2 * current_spin * neighbor_sum;

    // Metropolis 接受準則
    if (delta_E <= 0 || curand_uniform(&localState) < expf(-beta * delta_E)) {
        spins[site] = -current_spin;

        // 原子操作更新總能量和磁化
        atomicAdd(energy_sum, delta_E);
        atomicAdd(mag_sum, -2 * current_spin);
    }

    states[site] = localState;
}

// 初始化隨機數生成器
__global__ void init_curand(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= L || idy >= L) return;

    int site = idy * L + idx;
    curand_init(seed, site, 0, &states[site]);
}

// 計算初始能量和磁化
__global__ void calculate_initial_observables(int* spins, int* energy, int* magnetization) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= L || idy >= L) return;

    int site = idy * L + idx;

    // 計算能量貢獻 (避免重複計算)
    if (idx < L-1) {
        int right = idy * L + idx + 1;
        atomicAdd(energy, -spins[site] * spins[right]);
    }
    if (idy < L-1) {
        int down = (idy + 1) * L + idx;
        atomicAdd(energy, -spins[site] * spins[down]);
    }
    // 邊界條件
    if (idx == L-1) {
        int right = idy * L;
        atomicAdd(energy, -spins[site] * spins[right]);
    }
    if (idy == L-1) {
        int down = idx;
        atomicAdd(energy, -spins[site] * spins[down]);
    }

    // 計算磁化
    atomicAdd(magnetization, spins[site]);
}

// CPU 版本的伊辛模型 (用於比較)
class IsingCPU {
private:
    std::vector<int> spins;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

public:
    IsingCPU(unsigned seed = 12345) : spins(N), rng(seed), dist(0.0f, 1.0f) {
        // 隨機初始化自旋
        for (int i = 0; i < N; ++i) {
            spins[i] = (rng() % 2) * 2 - 1;
        }
    }

    void metropolis_step(float beta) {
        for (int i = 0; i < N; ++i) {
            int x = i % L;
            int y = i / L;

            int left = y * L + ((x - 1 + L) % L);
            int right = y * L + ((x + 1) % L);
            int up = ((y - 1 + L) % L) * L + x;
            int down = ((y + 1) % L) * L + x;

            int current_spin = spins[i];
            int neighbor_sum = spins[left] + spins[right] + spins[up] + spins[down];
            int delta_E = 2 * current_spin * neighbor_sum;

            if (delta_E <= 0 || dist(rng) < std::exp(-beta * delta_E)) {
                spins[i] = -current_spin;
            }
        }
    }

    std::pair<double, double> get_observables() {
        int energy = 0, magnetization = 0;

        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                int site = y * L + x;
                magnetization += spins[site];

                int right = y * L + ((x + 1) % L);
                int down = ((y + 1) % L) * L + x;
                energy -= spins[site] * (spins[right] + spins[down]);
            }
        }

        return std::make_pair((double)energy / N, (double)magnetization / N);
    }
};

// GPU 伊辛模型類別
class IsingGPU {
private:
    int* d_spins;
    curandState* d_states;
    int* d_energy_sum;
    int* d_mag_sum;
    dim3 block_size;
    dim3 grid_size;

public:
    IsingGPU(dim3 block_sz) : block_size(block_sz) {
        grid_size = dim3((L + block_size.x - 1) / block_size.x,
                        (L + block_size.y - 1) / block_size.y);

        // 分配 GPU 記憶體
        CUDA_CHECK(cudaMalloc(&d_spins, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_states, N * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_energy_sum, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_mag_sum, sizeof(int)));

        // 初始化自旋 (隨機)
        std::vector<int> h_spins(N);
        std::random_device rd;
        std::mt19937 gen(rd());
        for (int i = 0; i < N; ++i) {
            h_spins[i] = (gen() % 2) * 2 - 1;
        }
        CUDA_CHECK(cudaMemcpy(d_spins, h_spins.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        // 初始化隨機數生成器
        init_curand<<<grid_size, block_size>>>(d_states, time(nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~IsingGPU() {
        cudaFree(d_spins);
        cudaFree(d_states);
        cudaFree(d_energy_sum);
        cudaFree(d_mag_sum);
    }

    void metropolis_step(float beta) {
        // 重置觀測量
        CUDA_CHECK(cudaMemset(d_energy_sum, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_mag_sum, 0, sizeof(int)));

        // 棋盤算法：兩個子格
        ising_metropolis_kernel<<<grid_size, block_size>>>(d_spins, d_states, beta, 0, d_energy_sum, d_mag_sum);
        CUDA_CHECK(cudaDeviceSynchronize());

        ising_metropolis_kernel<<<grid_size, block_size>>>(d_spins, d_states, beta, 1, d_energy_sum, d_mag_sum);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::pair<double, double> get_observables() {
        int h_energy = 0, h_magnetization = 0;

        CUDA_CHECK(cudaMemset(d_energy_sum, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_mag_sum, 0, sizeof(int)));

        calculate_initial_observables<<<grid_size, block_size>>>(d_spins, d_energy_sum, d_mag_sum);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_energy, d_energy_sum, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_magnetization, d_mag_sum, sizeof(int), cudaMemcpyDeviceToHost));

        return std::make_pair((double)h_energy / N, (double)h_magnetization / N);
    }

    double benchmark_performance(float beta, int steps = 1000) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; ++i) {
            metropolis_step(beta);
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};

// 多 GPU 伊辛模型類別
class MultiGPUIsingModel {
private:
    std::vector<IsingGPU*> gpus;
    int num_gpus;

public:
    MultiGPUIsingModel(int n_gpus, dim3 block_size) : num_gpus(n_gpus) {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpus.push_back(new IsingGPU(block_size));
        }
    }

    ~MultiGPUIsingModel() {
        for (auto gpu : gpus) {
            delete gpu;
        }
    }

    void metropolis_step(float beta) {
        omp_set_num_threads(num_gpus);

        #pragma omp parallel
        {
            int gpu_id = omp_get_thread_num();
            CUDA_CHECK(cudaSetDevice(gpu_id));
            gpus[gpu_id]->metropolis_step(beta);
        }
    }

    std::pair<double, double> get_observables() {
        double total_energy = 0.0, total_magnetization = 0.0;

        #pragma omp parallel for reduction(+:total_energy,total_magnetization)
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            std::pair<double, double> result = gpus[i]->get_observables();
            double energy = result.first;
            double mag = result.second;
            total_energy += energy;
            total_magnetization += mag;
        }

        return std::make_pair(total_energy / num_gpus, total_magnetization / num_gpus);
    }

    double benchmark_performance(float beta, int steps = 1000) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; ++i) {
            metropolis_step(beta);
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};

// 統計分析函數
struct Statistics {
    double mean, error;
};

Statistics calculate_statistics(const std::vector<double>& data) {
    double sum = 0.0, sum2 = 0.0;
    int n = data.size();

    for (double x : data) {
        sum += x;
        sum2 += x * x;
    }

    double mean = sum / n;
    double variance = (sum2 / n - mean * mean) / (n - 1);
    double error = std::sqrt(variance);

    return {mean, error};
}

// 問題 1: 尋找最佳 block size
void find_optimal_block_size() {
    std::cout << "=== Problem 1: Finding Optimal Block Size ===\n";

    std::vector<dim3> block_sizes = {
        {8, 8}, {16, 16}, {32, 32}, {8, 16}, {16, 8}, {32, 8}, {8, 32}
    };

    float beta = 1.0f / 2.269f; // 臨界溫度附近

    std::cout << "Block Size\t\tTime (s)\t\tPerformance (steps/s)\n";
    std::cout << std::string(60, '-') << "\n";

    dim3 best_block_size;
    double best_performance = 0.0;

    for (auto block_size : block_sizes) {
        try {
            IsingGPU ising(block_size);
            double time = ising.benchmark_performance(beta, 1000);
            double performance = 1000.0 / time;

            std::cout << block_size.x << "x" << block_size.y << "\t\t\t"
                      << std::fixed << std::setprecision(4) << time << "\t\t"
                      << std::setprecision(2) << performance << "\n";

            if (performance > best_performance) {
                best_performance = performance;
                best_block_size = block_size;
            }
        } catch (...) {
            std::cout << block_size.x << "x" << block_size.y << "\t\t\tFailed\n";
        }
    }

    std::cout << "\nOptimal block size: " << best_block_size.x << "x" << best_block_size.y << "\n";
    std::cout << "Best performance: " << best_performance << " steps/s\n\n";
}

// 問題 1 & 3: 溫度掃描
void temperature_scan(bool use_multi_gpu = false) {
    std::cout << "=== Temperature Scan (" << (use_multi_gpu ? "Multi-GPU" : "Single GPU") << ") ===\n";

    std::vector<double> temperatures = {2.0, 2.1, 2.2, 2.3, 2.4, 2.5};
    dim3 optimal_block_size(16, 16); // 使用找到的最佳 block size

    std::ofstream file(use_multi_gpu ? "multi_gpu_results.txt" : "single_gpu_results.txt");
    file << "T\t<E>\tδ<E>\t<M>\tδ<M>\n";

    std::cout << "T\t\t<E>\t\tδ<E>\t\t<M>\t\tδ<M>\n";
    std::cout << std::string(70, '-') << "\n";

    for (double T : temperatures) {
        float beta = 1.0f / T;

        std::vector<double> energies, magnetizations;

        if (use_multi_gpu) {
            int num_gpus = 2;
            MultiGPUIsingModel ising(num_gpus, optimal_block_size);

            // 熱化
            for (int i = 0; i < WARMUP_STEPS; ++i) {
                ising.metropolis_step(beta);
            }

            // 測量
            for (int i = 0; i < MC_STEPS; i += MEASURE_INTERVAL) {
                for (int j = 0; j < MEASURE_INTERVAL; ++j) {
                    ising.metropolis_step(beta);
                }
                std::pair<double, double> result = ising.get_observables();
                double energy = result.first;
                double mag = result.second;
                energies.push_back(energy);
                magnetizations.push_back(std::abs(mag));
            }
        } else {
            IsingGPU ising(optimal_block_size);

            // 熱化
            for (int i = 0; i < WARMUP_STEPS; ++i) {
                ising.metropolis_step(beta);
            }

            // 測量
            for (int i = 0; i < MC_STEPS; i += MEASURE_INTERVAL) {
                for (int j = 0; j < MEASURE_INTERVAL; ++j) {
                    ising.metropolis_step(beta);
                }
                std::pair<double, double> result = ising.get_observables();
                double energy = result.first;
                double mag = result.second;
                energies.push_back(energy);
                magnetizations.push_back(std::abs(mag));
            }
        }

        auto energy_stats = calculate_statistics(energies);
        auto mag_stats = calculate_statistics(magnetizations);

        std::cout << std::fixed << std::setprecision(1) << T << "\t\t"
                  << std::setprecision(4) << energy_stats.mean << "\t\t"
                  << energy_stats.error << "\t\t"
                  << mag_stats.mean << "\t\t"
                  << mag_stats.error << "\n";

        file << T << "\t" << energy_stats.mean << "\t" << energy_stats.error << "\t"
             << mag_stats.mean << "\t" << mag_stats.error << "\n";
    }

    file.close();
    std::cout << "\n";
}

// 問題 2: GPU vs CPU 比較
void compare_gpu_cpu() {
    std::cout << "=== Problem 2: GPU vs CPU Comparison ===\n";

    double T = 2.269; // 臨界溫度
    float beta = 1.0f / T;
    int test_steps = 1000;

    // CPU 測試
    std::cout << "Running CPU simulation...\n";
    IsingCPU cpu_ising;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_steps; ++i) {
        cpu_ising.metropolis_step(beta);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end - start).count();
    std::pair<double, double> cpu_result = cpu_ising.get_observables();
    double cpu_energy = cpu_result.first;
    double cpu_mag = cpu_result.second;

    // 單 GPU 測試
    std::cout << "Running single GPU simulation...\n";
    IsingGPU single_gpu(dim3(16, 16));
    double single_gpu_time = single_gpu.benchmark_performance(beta, test_steps);
    std::pair<double, double> single_result = single_gpu.get_observables();
    double single_energy = single_result.first;
    double single_mag = single_result.second;

    // 雙 GPU 測試
    std::cout << "Running dual GPU simulation...\n";
    MultiGPUIsingModel dual_gpu(2, dim3(16, 16));
    double dual_gpu_time = dual_gpu.benchmark_performance(beta, test_steps);
    std::pair<double, double> dual_result = dual_gpu.get_observables();
    double dual_energy = dual_result.first;
    double dual_mag = dual_result.second;

    // 結果比較
    std::cout << "\nPerformance Comparison:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Method\t\tTime (s)\tSpeedup\t\t<E>\t\t<M>\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU\t\t" << cpu_time << "\t1.00x\t\t" << cpu_energy << "\t" << cpu_mag << "\n";
    std::cout << "1 GPU\t\t" << single_gpu_time << "\t" << (cpu_time/single_gpu_time) << "x\t\t"
              << single_energy << "\t" << single_mag << "\n";
    std::cout << "2 GPUs\t\t" << dual_gpu_time << "\t" << (cpu_time/dual_gpu_time) << "x\t\t"
              << dual_energy << "\t" << dual_mag << "\n";
    std::cout << "\n";
}

int main() {
    std::cout << "2D Ising Model Monte Carlo Simulation on GPU\n";
    std::cout << "Lattice size: " << L << "x" << L << " = " << N << " spins\n";
    std::cout << std::string(70, '=') << "\n\n";

    try {
        // 檢查 GPU 數量
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        std::cout << "Available GPUs: " << device_count << "\n\n";

        // 問題 1: 尋找最佳 block size
        find_optimal_block_size();

        // 問題 1: 單 GPU 溫度掃描
        temperature_scan(false);

        // 問題 2: GPU vs CPU 比較
        compare_gpu_cpu();

        // 問題 3: 雙 GPU 溫度掃描
        if (device_count >= 2) {
            temperature_scan(true);
        } else {
            std::cout << "Skipping multi-GPU test (insufficient GPUs)\n";
        }

        std::cout << "All simulations completed successfully!\n";
        std::cout << "Results saved to single_gpu_results.txt and multi_gpu_results.txt\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
