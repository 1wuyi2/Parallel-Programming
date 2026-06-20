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
#include <random>
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

static inline float inner_product_cpu(const float* x, const float* y, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

struct IVFIndex {
    size_t nlist = 0;
    std::vector<float> centroids;
    std::vector<std::vector<uint32_t>> inverted_lists;
};

static void assign_to_centroids_cpu(const float* base,
                                    size_t base_number,
                                    size_t vecdim,
                                    const std::vector<float>& centroids,
                                    size_t nlist,
                                    std::vector<uint32_t>& assignments) {
    assignments.resize(base_number);

    for (size_t i = 0; i < base_number; ++i) {
        const float* v = base + i * vecdim;

        float best_ip = -1e30f;
        uint32_t best_c = 0;

        for (size_t c = 0; c < nlist; ++c) {
            const float* center = centroids.data() + c * vecdim;
            float ip = inner_product_cpu(v, center, vecdim);

            if (ip > best_ip) {
                best_ip = ip;
                best_c = static_cast<uint32_t>(c);
            }
        }

        assignments[i] = best_c;
    }
}

static IVFIndex build_ivf_index_cpu(const float* base,
                                    size_t base_number,
                                    size_t vecdim,
                                    size_t nlist,
                                    int niter) {
    if (base == nullptr || base_number == 0 || vecdim == 0 || nlist == 0) {
        throw std::runtime_error("invalid arguments for build_ivf_index_cpu");
    }

    nlist = std::min(nlist, base_number);

    IVFIndex idx;
    idx.nlist = nlist;
    idx.centroids.resize(nlist * vecdim);
    idx.inverted_lists.assign(nlist, {});

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> uni(0, base_number - 1);

    for (size_t c = 0; c < nlist; ++c) {
        size_t ri = uni(rng);
        std::copy(base + ri * vecdim,
                  base + ri * vecdim + vecdim,
                  idx.centroids.begin() + c * vecdim);
    }

    std::vector<uint32_t> assignments(base_number, 0);
    std::vector<float> new_centroids(nlist * vecdim, 0.0f);
    std::vector<size_t> counts(nlist, 0);

    for (int iter = 0; iter < niter; ++iter) {
        std::cerr << "[build_ivf_index_cpu] kmeans iter "
                  << iter + 1 << "/" << niter << std::endl;

        assign_to_centroids_cpu(base,
                                base_number,
                                vecdim,
                                idx.centroids,
                                nlist,
                                assignments);

        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < base_number; ++i) {
            uint32_t c = assignments[i];
            const float* v = base + i * vecdim;
            float* dst = new_centroids.data() + static_cast<size_t>(c) * vecdim;

            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] += v[d];
            }

            ++counts[c];
        }

        for (size_t c = 0; c < nlist; ++c) {
            float* dst = new_centroids.data() + c * vecdim;

            if (counts[c] > 0) {
                float inv = 1.0f / static_cast<float>(counts[c]);
                for (size_t d = 0; d < vecdim; ++d) {
                    dst[d] *= inv;
                }
            } else {
                size_t ri = uni(rng);
                std::copy(base + ri * vecdim,
                          base + ri * vecdim + vecdim,
                          dst);
            }
        }

        idx.centroids.swap(new_centroids);
    }

    assign_to_centroids_cpu(base,
                            base_number,
                            vecdim,
                            idx.centroids,
                            nlist,
                            assignments);

    idx.inverted_lists.assign(nlist, {});

    for (size_t i = 0; i < base_number; ++i) {
        idx.inverted_lists[assignments[i]].push_back(static_cast<uint32_t>(i));
    }

    size_t non_empty = 0;
    size_t max_list_size = 0;

    for (const auto& list : idx.inverted_lists) {
        if (!list.empty()) {
            ++non_empty;
        }
        max_list_size = std::max(max_list_size, list.size());
    }

    std::cerr << "[build_ivf_index_cpu] finished. non_empty_lists="
              << non_empty << "/" << nlist
              << ", max_list_size=" << max_list_size << std::endl;

    return idx;
}

static std::vector<uint32_t> select_top_clusters_cpu(const float* query,
                                                     const IVFIndex& idx,
                                                     size_t vecdim,
                                                     size_t nprobe) {
    nprobe = std::min(nprobe, idx.nlist);

    std::vector<std::pair<float, uint32_t>> cluster_ips(idx.nlist);

    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = inner_product_cpu(query,
                                     idx.centroids.data() + c * vecdim,
                                     vecdim);
        cluster_ips[c] = {ip, static_cast<uint32_t>(c)};
    }

    if (nprobe < idx.nlist) {
        std::nth_element(cluster_ips.begin(),
                         cluster_ips.begin() + static_cast<std::ptrdiff_t>(nprobe),
                         cluster_ips.end(),
                         [](const auto& a, const auto& b) {
                             return a.first > b.first;
                         });
    }

    std::sort(cluster_ips.begin(),
              cluster_ips.begin() + static_cast<std::ptrdiff_t>(nprobe),
              [](const auto& a, const auto& b) {
                  return a.first > b.first;
              });

    std::vector<uint32_t> selected(nprobe);

    for (size_t i = 0; i < nprobe; ++i) {
        selected[i] = cluster_ips[i].second;
    }

    return selected;
}

/*
    IVF Batch Matrix Baseline

    candidate_ids:
        candidate_count candidates from the union of selected IVF clusters in one batch.

    query_batch:
        current_batch_size x vecdim.

    scores:
        candidate_count x current_batch_size.

    Each thread computes:
        scores[candidate_row, query_col]
        = dot(base[candidate_ids[candidate_row]], query_batch[query_col])
*/
template <int TILE>
__global__ void ivf_batch_matmul_kernel(const float* __restrict__ base,
                                        const float* __restrict__ query_batch,
                                        const uint32_t* __restrict__ candidate_ids,
                                        float* __restrict__ scores,
                                        int candidate_count,
                                        int vecdim,
                                        int current_batch_size) {
    __shared__ float s_base[TILE][TILE];
    __shared__ float s_query[TILE][TILE];

    int local_col = threadIdx.x;
    int local_row = threadIdx.y;

    int candidate_row = blockIdx.y * TILE + local_row;
    int query_col = blockIdx.x * TILE + local_col;

    float sum = 0.0f;

    for (int t = 0; t < vecdim; t += TILE) {
        int base_k = t + local_col;
        int query_k = t + local_row;

        if (candidate_row < candidate_count && base_k < vecdim) {
            uint32_t base_id = candidate_ids[candidate_row];
            s_base[local_row][local_col] =
                base[static_cast<size_t>(base_id) * vecdim + base_k];
        } else {
            s_base[local_row][local_col] = 0.0f;
        }

        if (query_col < current_batch_size && query_k < vecdim) {
            s_query[local_row][local_col] =
                query_batch[query_col * vecdim + query_k];
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

    if (candidate_row < candidate_count && query_col < current_batch_size) {
        scores[candidate_row * current_batch_size + query_col] = sum;
    }
}

struct BatchStats {
    double total_batch_us = 0.0;
    double cluster_select_us = 0.0;
    double h2d_us = 0.0;
    double kernel_us = 0.0;
    double d2h_us = 0.0;
    double cpu_topk_us = 0.0;

    double recall_sum = 0.0;
    size_t query_count = 0;

    double union_candidates_sum = 0.0;
    double valid_candidates_sum = 0.0;
};

static void compute_topk_and_recall_cpu(
    const std::vector<float>& scores,
    const std::vector<uint32_t>& candidate_ids,
    const std::vector<uint32_t>& candidate_cluster_ids,
    const std::vector<uint8_t>& query_cluster_selected,
    int current_batch_size,
    size_t nlist,
    size_t candidate_count,
    size_t k,
    const int* gt,
    size_t gt_d,
    size_t query_global_start,
    BatchStats& stats
) {
    using Pair = std::pair<float, uint32_t>;

    for (int b = 0; b < current_batch_size; ++b) {
        const uint8_t* selected =
            query_cluster_selected.data() + static_cast<size_t>(b) * nlist;

        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;

        for (size_t r = 0; r < candidate_count; ++r) {
            uint32_t cluster_id = candidate_cluster_ids[r];

            if (!selected[cluster_id]) {
                continue;
            }

            float ip = scores[r * current_batch_size + b];
            uint32_t base_id = candidate_ids[r];

            Pair item{ip, base_id};

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

static void reserve_gpu_buffers(float*& d_query_batch,
                                uint32_t*& d_candidate_ids,
                                float*& d_scores,
                                size_t& candidate_capacity,
                                size_t required_candidate_count,
                                int batch_size,
                                size_t vecdim) {
    if (d_query_batch == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_query_batch,
                              static_cast<size_t>(batch_size) * vecdim * sizeof(float)));
    }

    if (required_candidate_count <= candidate_capacity &&
        d_candidate_ids != nullptr &&
        d_scores != nullptr) {
        return;
    }

    size_t new_capacity = std::max<size_t>(candidate_capacity, 1);

    while (new_capacity < required_candidate_count) {
        new_capacity *= 2;
    }

    if (d_candidate_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_candidate_ids));
    }

    if (d_scores != nullptr) {
        CUDA_CHECK(cudaFree(d_scores));
    }

    CUDA_CHECK(cudaMalloc(&d_candidate_ids,
                          new_capacity * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d_scores,
                          new_capacity * static_cast<size_t>(batch_size) * sizeof(float)));

    candidate_capacity = new_capacity;
}

int main(int argc, char* argv[]) {
    /*
        Usage:
            main_gpu_ivf_batch.exe <data_path> <test_number> <batch_size> <nlist> <nprobe> <kmeans_iter> <k>

        Example:
            main_gpu_ivf_batch.exe ./data/ 100 8 1024 50 10 10
            main_gpu_ivf_batch.exe ./data/ 1000 8 1024 20 10 10
    */

    std::string data_path = argc >= 2 ? argv[1] : "./data/";
    data_path = ensure_slash(data_path);

    size_t test_limit =
        argc >= 3 ? static_cast<size_t>(std::stoul(argv[2])) : 1000;

    int batch_size =
        argc >= 4 ? std::stoi(argv[3]) : 8;

    size_t nlist =
        argc >= 5 ? static_cast<size_t>(std::stoul(argv[4])) : 1024;

    size_t nprobe =
        argc >= 6 ? static_cast<size_t>(std::stoul(argv[5])) : 50;

    int kmeans_iter =
        argc >= 7 ? std::stoi(argv[6]) : 10;

    size_t k =
        argc >= 8 ? static_cast<size_t>(std::stoul(argv[7])) : 10;

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

    nlist = std::min(nlist, base_number);
    nprobe = std::min(nprobe, nlist);

    std::cerr << "config: test_number=" << test_number
              << ", base_number=" << base_number
              << ", vecdim=" << vecdim
              << ", batch_size=" << batch_size
              << ", nlist=" << nlist
              << ", nprobe=" << nprobe
              << ", kmeans_iter=" << kmeans_iter
              << ", k=" << k << "\n";

    auto build_begin = std::chrono::steady_clock::now();

    IVFIndex ivf_idx =
        build_ivf_index_cpu(base,
                            base_number,
                            vecdim,
                            nlist,
                            kmeans_iter);

    auto build_end = std::chrono::steady_clock::now();

    std::cerr << "build ivf index time (s): "
              << elapsed_us(build_begin, build_end) / 1000000.0
              << "\n";

    size_t base_bytes = base_number * vecdim * sizeof(float);

    float* d_base = nullptr;
    float* d_query_batch = nullptr;
    uint32_t* d_candidate_ids = nullptr;
    float* d_scores = nullptr;

    size_t candidate_capacity = 0;

    CUDA_CHECK(cudaMalloc(&d_base, base_bytes));

    CUDA_CHECK(cudaMemcpy(d_base,
                          base,
                          base_bytes,
                          cudaMemcpyHostToDevice));

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

    std::vector<uint8_t> any_cluster_selected(nlist, 0);

    while (processed < test_number) {
        int current_batch_size = static_cast<int>(
            std::min(static_cast<size_t>(batch_size), test_number - processed)
        );

        auto batch_begin = std::chrono::steady_clock::now();

        auto select_begin = std::chrono::steady_clock::now();

        std::fill(any_cluster_selected.begin(), any_cluster_selected.end(), 0);

        std::vector<uint8_t> query_cluster_selected(
            static_cast<size_t>(current_batch_size) * nlist,
            0
        );

        double valid_candidate_count_this_batch = 0.0;

        for (int b = 0; b < current_batch_size; ++b) {
            const float* q = query + (processed + static_cast<size_t>(b)) * vecdim;

            std::vector<uint32_t> selected_clusters =
                select_top_clusters_cpu(q, ivf_idx, vecdim, nprobe);

            for (uint32_t cid : selected_clusters) {
                query_cluster_selected[static_cast<size_t>(b) * nlist + cid] = 1;
                any_cluster_selected[cid] = 1;
                valid_candidate_count_this_batch +=
                    static_cast<double>(ivf_idx.inverted_lists[cid].size());
            }
        }

        std::vector<uint32_t> candidate_ids;
        std::vector<uint32_t> candidate_cluster_ids;

        for (size_t c = 0; c < nlist; ++c) {
            if (!any_cluster_selected[c]) {
                continue;
            }

            const auto& list = ivf_idx.inverted_lists[c];

            for (uint32_t id : list) {
                candidate_ids.push_back(id);
                candidate_cluster_ids.push_back(static_cast<uint32_t>(c));
            }
        }

        auto select_end = std::chrono::steady_clock::now();

        size_t candidate_count = candidate_ids.size();

        total_stats.union_candidates_sum +=
            static_cast<double>(candidate_count);

        total_stats.valid_candidates_sum +=
            valid_candidate_count_this_batch / static_cast<double>(current_batch_size);

        if (candidate_count == 0) {
            processed += static_cast<size_t>(current_batch_size);
            batch_id += 1;
            continue;
        }

        reserve_gpu_buffers(d_query_batch,
                            d_candidate_ids,
                            d_scores,
                            candidate_capacity,
                            candidate_count,
                            batch_size,
                            vecdim);

        std::vector<float> h_scores(
            candidate_count * static_cast<size_t>(current_batch_size)
        );

        size_t current_query_bytes =
            static_cast<size_t>(current_batch_size) * vecdim * sizeof(float);

        size_t candidate_id_bytes =
            candidate_count * sizeof(uint32_t);

        size_t score_bytes =
            candidate_count * static_cast<size_t>(current_batch_size) * sizeof(float);

        CUDA_CHECK(cudaEventRecord(h2d_start));

        CUDA_CHECK(cudaMemcpy(d_query_batch,
                              query + processed * vecdim,
                              current_query_bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_candidate_ids,
                              candidate_ids.data(),
                              candidate_id_bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(h2d_stop));
        CUDA_CHECK(cudaEventSynchronize(h2d_stop));

        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));

        dim3 grid(
            (current_batch_size + TILE - 1) / TILE,
            (static_cast<int>(candidate_count) + TILE - 1) / TILE
        );

        CUDA_CHECK(cudaEventRecord(kernel_start));

        ivf_batch_matmul_kernel<TILE><<<grid, block>>>(
            d_base,
            d_query_batch,
            d_candidate_ids,
            d_scores,
            static_cast<int>(candidate_count),
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
                              score_bytes,
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(d2h_stop));
        CUDA_CHECK(cudaEventSynchronize(d2h_stop));

        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

        auto topk_begin = std::chrono::steady_clock::now();

        compute_topk_and_recall_cpu(h_scores,
                                    candidate_ids,
                                    candidate_cluster_ids,
                                    query_cluster_selected,
                                    current_batch_size,
                                    nlist,
                                    candidate_count,
                                    k,
                                    gt,
                                    gt_d,
                                    processed,
                                    total_stats);

        auto topk_end = std::chrono::steady_clock::now();

        auto batch_end = std::chrono::steady_clock::now();

        total_stats.cluster_select_us += elapsed_us(select_begin, select_end);
        total_stats.h2d_us += h2d_ms * 1000.0;
        total_stats.kernel_us += kernel_ms * 1000.0;
        total_stats.d2h_us += d2h_ms * 1000.0;
        total_stats.cpu_topk_us += elapsed_us(topk_begin, topk_end);
        total_stats.total_batch_us += elapsed_us(batch_begin, batch_end);

        processed += static_cast<size_t>(current_batch_size);
        batch_id += 1;

        std::cerr << "batch " << batch_id
                  << " processed " << processed << "/" << test_number
                  << ", current_batch_size=" << current_batch_size
                  << ", union_candidates=" << candidate_count
                  << "\r" << std::flush;
    }

    std::cerr << "\n";

    double query_count_d =
        static_cast<double>(std::max<size_t>(total_stats.query_count, 1));

    double batch_count_d =
        static_cast<double>(std::max<size_t>(batch_id, 1));

    double avg_recall =
        total_stats.recall_sum / query_count_d;

    double avg_latency_per_query_us =
        total_stats.total_batch_us / query_count_d;

    double avg_cluster_select_per_query_us =
        total_stats.cluster_select_us / query_count_d;

    double avg_h2d_per_query_us =
        total_stats.h2d_us / query_count_d;

    double avg_kernel_per_query_us =
        total_stats.kernel_us / query_count_d;

    double avg_d2h_per_query_us =
        total_stats.d2h_us / query_count_d;

    double avg_topk_per_query_us =
        total_stats.cpu_topk_us / query_count_d;

    double avg_batch_time_us =
        total_stats.total_batch_us / batch_count_d;

    double avg_union_candidates_per_batch =
        total_stats.union_candidates_sum / batch_count_d;

    double avg_valid_candidates_per_query =
        total_stats.valid_candidates_sum / batch_count_d;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "average recall: "
              << avg_recall << "\n";

    std::cout << "average latency per query (us): "
              << avg_latency_per_query_us << "\n";

    std::cout << "average batch time (us): "
              << avg_batch_time_us << "\n";

    std::cout << "average cluster select per query (us): "
              << avg_cluster_select_per_query_us << "\n";

    std::cout << "average H2D per query (us): "
              << avg_h2d_per_query_us << "\n";

    std::cout << "average kernel per query (us): "
              << avg_kernel_per_query_us << "\n";

    std::cout << "average D2H per query (us): "
              << avg_d2h_per_query_us << "\n";

    std::cout << "average CPU topk per query (us): "
              << avg_topk_per_query_us << "\n";

    std::cout << "average union candidates per batch: "
              << avg_union_candidates_per_batch << "\n";

    std::cout << "average valid candidates per query: "
              << avg_valid_candidates_per_query << "\n";

    std::cout << "total queries: "
              << total_stats.query_count << "\n";

    std::cout << "batch size: "
              << batch_size << "\n";

    std::cout << "nlist: "
              << nlist << "\n";

    std::cout << "nprobe: "
              << nprobe << "\n";

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    CUDA_CHECK(cudaFree(d_base));

    if (d_query_batch != nullptr) {
        CUDA_CHECK(cudaFree(d_query_batch));
    }

    if (d_candidate_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_candidate_ids));
    }

    if (d_scores != nullptr) {
        CUDA_CHECK(cudaFree(d_scores));
    }

    delete[] query;
    delete[] gt;
    delete[] base;

    return 0;
}