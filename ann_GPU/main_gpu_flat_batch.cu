#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << static_cast<int>(err__)                    \
                      << " msg=" << cudaGetErrorString(err__) << std::endl;     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

template <typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);

    if (!fin.good()) {
        std::cerr << "cannot open " << data_path << std::endl;
        n = 0;
        d = 0;
        return nullptr;
    }

    uint32_t n32 = 0;
    uint32_t d32 = 0;

    fin.read(reinterpret_cast<char*>(&n32), 4);
    fin.read(reinterpret_cast<char*>(&d32), 4);

    n = static_cast<size_t>(n32);
    d = static_cast<size_t>(d32);

    T* data = new T[n * d];

    fin.read(reinterpret_cast<char*>(data),
             static_cast<std::streamsize>(n * d * sizeof(T)));

    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d
              << "  number: " << n
              << "  size_per_element: " << sizeof(T) << "\n";

    return data;
}

static inline std::string ensure_slash(std::string path) {
    if (!path.empty() && path.back() != '/' && path.back() != '\\') {
        path.push_back('/');
    }
    return path;
}

static inline double elapsed_us(std::chrono::steady_clock::time_point a,
                                std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

template <int TILE>
__global__ void flat_batch_matmul_kernel(const float* __restrict__ base,
                                         const float* __restrict__ query_batch,
                                         float* __restrict__ scores,
                                         int base_number,
                                         int vecdim,
                                         int batch_size) {
    __shared__ float s_base[TILE][TILE];
    __shared__ float s_query[TILE][TILE];

    int local_col = threadIdx.x;
    int local_row = threadIdx.y;

    int global_row = blockIdx.y * TILE + local_row;
    int global_col = blockIdx.x * TILE + local_col;

    float sum = 0.0f;

    for (int t = 0; t < vecdim; t += TILE) {
        int base_k = t + local_col;
        int query_k = t + local_row;

        if (global_row < base_number && base_k < vecdim) {
            s_base[local_row][local_col] =
                base[global_row * vecdim + base_k];
        } else {
            s_base[local_row][local_col] = 0.0f;
        }

        if (global_col < batch_size && query_k < vecdim) {
            s_query[local_row][local_col] =
                query_batch[global_col * vecdim + query_k];
        } else {
            s_query[local_row][local_col] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += s_base[local_row][k] * s_query[k][local_col];
        }

        __syncthreads();
    }

    if (global_row < base_number && global_col < batch_size) {
        scores[global_row * batch_size + global_col] = sum;
    }
}

struct BatchStats {
    double total_batch_us = 0.0;
    double h2d_us = 0.0;
    double kernel_us = 0.0;
    double d2h_us = 0.0;
    double cpu_topk_us = 0.0;
    double recall_sum = 0.0;
    size_t query_count = 0;
};

static void compute_topk_and_recall_cpu(const std::vector<float>& scores,
                                        int current_batch_size,
                                        size_t base_number,
                                        size_t k,
                                        const int* gt,
                                        size_t gt_d,
                                        size_t query_global_start,
                                        BatchStats& stats) {
    using Pair = std::pair<float, uint32_t>;

    for (int b = 0; b < current_batch_size; ++b) {
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;

        for (size_t base_id = 0; base_id < base_number; ++base_id) {
            float ip = scores[base_id * current_batch_size + b];

            Pair item{ip, static_cast<uint32_t>(base_id)};

            if (heap.size() < k) {
                heap.push(item);
            } else if (item.first > heap.top().first) {
                heap.pop();
                heap.push(item);
            }
        }

        std::set<uint32_t> gtset;
        size_t query_id = query_global_start + static_cast<size_t>(b);

        for (size_t j = 0; j < k; ++j) {
            gtset.insert(static_cast<uint32_t>(gt[query_id * gt_d + j]));
        }

        size_t hit = 0;

        while (!heap.empty()) {
            uint32_t id = heap.top().second;
            if (gtset.find(id) != gtset.end()) {
                ++hit;
            }
            heap.pop();
        }

        stats.recall_sum += static_cast<double>(hit) / static_cast<double>(k);
        stats.query_count += 1;
    }
}

int main(int argc, char* argv[]) {

    std::string data_path = argc >= 2 ? argv[1] : "./data/";
    data_path = ensure_slash(data_path);

    size_t test_limit =
        argc >= 3 ? static_cast<size_t>(std::stoul(argv[2])) : 1000;

    int batch_size =
        argc >= 4 ? std::stoi(argv[3]) : 32;

    size_t k =
        argc >= 5 ? static_cast<size_t>(std::stoul(argv[4])) : 10;

    if (batch_size <= 0) {
        std::cerr << "batch_size must be positive\n";
        return 1;
    }

    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cerr << "using GPU: " << prop.name
              << ", compute capability "
              << prop.major << "." << prop.minor << "\n";

    size_t query_number = 0;
    size_t query_dim = 0;

    size_t gt_number = 0;
    size_t gt_d = 0;

    size_t base_number = 0;
    size_t vecdim = 0;

    float* query =
        LoadData<float>(data_path + "DEEP100K.query.fbin",
                        query_number,
                        query_dim);

    int* gt =
        LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin",
                      gt_number,
                      gt_d);

    float* base =
        LoadData<float>(data_path + "DEEP100K.base.100k.fbin",
                        base_number,
                        vecdim);

    if (query == nullptr ||
        gt == nullptr ||
        base == nullptr ||
        query_dim != vecdim ||
        query_number == 0 ||
        base_number == 0 ||
        gt_number == 0 ||
        gt_d < k) {
        std::cerr << "data loading failed or dimension mismatch\n";
        delete[] query;
        delete[] gt;
        delete[] base;
        return 1;
    }

    size_t test_number = std::min(query_number, gt_number);
    test_number = std::min(test_number, test_limit);

    std::cerr << "config: test_number=" << test_number
              << ", base_number=" << base_number
              << ", vecdim=" << vecdim
              << ", batch_size=" << batch_size
              << ", k=" << k << "\n";

    size_t base_bytes = base_number * vecdim * sizeof(float);
    size_t query_batch_bytes =
        static_cast<size_t>(batch_size) * vecdim * sizeof(float);
    size_t score_bytes =
        base_number * static_cast<size_t>(batch_size) * sizeof(float);

    std::cerr << "GPU memory estimate:\n";
    std::cerr << "  base bytes: " << base_bytes / 1024.0 / 1024.0 << " MB\n";
    std::cerr << "  query batch bytes: " << query_batch_bytes / 1024.0 / 1024.0 << " MB\n";
    std::cerr << "  score bytes: " << score_bytes / 1024.0 / 1024.0 << " MB\n";

    float* d_base = nullptr;
    float* d_query_batch = nullptr;
    float* d_scores = nullptr;

    CUDA_CHECK(cudaMalloc(&d_base, base_bytes));
    CUDA_CHECK(cudaMalloc(&d_query_batch, query_batch_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, score_bytes));

    CUDA_CHECK(cudaMemcpy(d_base,
                          base,
                          base_bytes,
                          cudaMemcpyHostToDevice));

    std::vector<float> h_scores(
        base_number * static_cast<size_t>(batch_size)
    );

    cudaEvent_t h2d_start, h2d_stop;
    cudaEvent_t kernel_start, kernel_stop;
    cudaEvent_t d2h_start, d2h_stop;

    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);

    BatchStats total_stats;

    size_t processed = 0;
    size_t batch_id = 0;

    while (processed < test_number) {
        int current_batch_size = static_cast<int>(
            std::min(static_cast<size_t>(batch_size), test_number - processed)
        );

        auto batch_begin = std::chrono::steady_clock::now();

        size_t current_query_bytes =
            static_cast<size_t>(current_batch_size) * vecdim * sizeof(float);

        size_t current_score_bytes =
            base_number * static_cast<size_t>(current_batch_size) * sizeof(float);

        CUDA_CHECK(cudaEventRecord(h2d_start));

        CUDA_CHECK(cudaMemcpy(d_query_batch,
                              query + processed * vecdim,
                              current_query_bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(h2d_stop));
        CUDA_CHECK(cudaEventSynchronize(h2d_stop));

        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));

        dim3 grid(
            (current_batch_size + TILE - 1) / TILE,
            (static_cast<int>(base_number) + TILE - 1) / TILE
        );

        CUDA_CHECK(cudaEventRecord(kernel_start));

        flat_batch_matmul_kernel<TILE><<<grid, block>>>(
            d_base,
            d_query_batch,
            d_scores,
            static_cast<int>(base_number),
            static_cast<int>(vecdim),
            current_batch_size
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));

        CUDA_CHECK(cudaEventRecord(d2h_start));

        CUDA_CHECK(cudaMemcpy(h_scores.data(),
                              d_scores,
                              current_score_bytes,
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(d2h_stop));
        CUDA_CHECK(cudaEventSynchronize(d2h_stop));

        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

        auto topk_begin = std::chrono::steady_clock::now();

        compute_topk_and_recall_cpu(h_scores,
                                    current_batch_size,
                                    base_number,
                                    k,
                                    gt,
                                    gt_d,
                                    processed,
                                    total_stats);

        auto topk_end = std::chrono::steady_clock::now();

        auto batch_end = std::chrono::steady_clock::now();

        double cpu_topk_us = elapsed_us(topk_begin, topk_end);
        double batch_total_us = elapsed_us(batch_begin, batch_end);

        total_stats.h2d_us += h2d_ms * 1000.0;
        total_stats.kernel_us += kernel_ms * 1000.0;
        total_stats.d2h_us += d2h_ms * 1000.0;
        total_stats.cpu_topk_us += cpu_topk_us;
        total_stats.total_batch_us += batch_total_us;

        processed += static_cast<size_t>(current_batch_size);
        batch_id += 1;

        std::cerr << "batch " << batch_id
                  << " processed " << processed << "/" << test_number
                  << ", current_batch_size=" << current_batch_size
                  << "\r" << std::flush;
    }

    std::cerr << "\n";

    double avg_recall =
        total_stats.recall_sum / static_cast<double>(total_stats.query_count);

    double avg_latency_per_query_us =
        total_stats.total_batch_us / static_cast<double>(total_stats.query_count);

    double avg_h2d_per_query_us =
        total_stats.h2d_us / static_cast<double>(total_stats.query_count);

    double avg_kernel_per_query_us =
        total_stats.kernel_us / static_cast<double>(total_stats.query_count);

    double avg_d2h_per_query_us =
        total_stats.d2h_us / static_cast<double>(total_stats.query_count);

    double avg_topk_per_query_us =
        total_stats.cpu_topk_us / static_cast<double>(total_stats.query_count);

    double avg_batch_time_us =
        total_stats.total_batch_us / static_cast<double>(batch_id);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "average recall: "
              << avg_recall << "\n";

    std::cout << "average latency per query (us): "
              << avg_latency_per_query_us << "\n";

    std::cout << "average batch time (us): "
              << avg_batch_time_us << "\n";

    std::cout << "average H2D per query (us): "
              << avg_h2d_per_query_us << "\n";

    std::cout << "average kernel per query (us): "
              << avg_kernel_per_query_us << "\n";

    std::cout << "average D2H per query (us): "
              << avg_d2h_per_query_us << "\n";

    std::cout << "average CPU topk per query (us): "
              << avg_topk_per_query_us << "\n";

    std::cout << "total queries: "
              << total_stats.query_count << "\n";

    std::cout << "batch size: "
              << batch_size << "\n";

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_query_batch));
    CUDA_CHECK(cudaFree(d_scores));

    delete[] query;
    delete[] gt;
    delete[] base;

    return 0;
}