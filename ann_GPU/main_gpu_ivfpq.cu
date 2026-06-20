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
#include <stdexcept>
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
    std::vector<uint32_t> assignments;
};

struct PQIndex {
    int M = 8;
    int Ks = 256;
    int subdim = 12;

    // codebooks[(m * Ks + k) * subdim + j]
    std::vector<float> codebooks;

    // codes[base_id * M + m]
    std::vector<uint8_t> codes;
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

    idx.assignments = assignments;
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

static std::vector<uint32_t> select_top_clusters_from_scores_cpu(
    const float* scores,
    size_t nlist,
    size_t nprobe
) {
    nprobe = std::min(nprobe, nlist);

    std::vector<std::pair<float, uint32_t>> cluster_ips(nlist);

    for (size_t c = 0; c < nlist; ++c) {
        cluster_ips[c] = {scores[c], static_cast<uint32_t>(c)};
    }

    if (nprobe < nlist) {
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

static float residual_sub_l2(const float* base_vec,
                             const float* centroid,
                             const float* codeword,
                             int m,
                             int subdim) {
    float dist = 0.0f;
    int offset = m * subdim;

    for (int j = 0; j < subdim; ++j) {
        float residual = base_vec[offset + j] - centroid[offset + j];
        float diff = residual - codeword[j];
        dist += diff * diff;
    }

    return dist;
}

static PQIndex train_pq_cpu(const float* base,
                            size_t base_number,
                            size_t vecdim,
                            const IVFIndex& ivf,
                            int M,
                            int Ks,
                            int train_samples,
                            int pq_iter) {
    if (M <= 0 || Ks <= 0 || Ks > 256) {
        throw std::runtime_error("invalid PQ parameters");
    }

    if (static_cast<int>(vecdim) % M != 0) {
        throw std::runtime_error("vecdim must be divisible by M");
    }

    PQIndex pq;
    pq.M = M;
    pq.Ks = Ks;
    pq.subdim = static_cast<int>(vecdim) / M;
    pq.codebooks.resize(static_cast<size_t>(M) * Ks * pq.subdim);
    pq.codes.resize(base_number * static_cast<size_t>(M));

    train_samples = std::min<int>(train_samples, static_cast<int>(base_number));
    train_samples = std::max<int>(train_samples, Ks);

    std::vector<uint32_t> sample_ids(train_samples);

    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> uni(0, base_number - 1);

    for (int i = 0; i < train_samples; ++i) {
        sample_ids[i] = static_cast<uint32_t>(uni(rng));
    }

    std::cerr << "[train_pq_cpu] M=" << M
              << ", Ks=" << Ks
              << ", subdim=" << pq.subdim
              << ", train_samples=" << train_samples
              << ", pq_iter=" << pq_iter << "\n";

    for (int m = 0; m < M; ++m) {
        std::cerr << "[train_pq_cpu] train subquantizer "
                  << m + 1 << "/" << M << std::endl;

        float* cb_m = pq.codebooks.data()
                    + static_cast<size_t>(m) * Ks * pq.subdim;

        for (int k = 0; k < Ks; ++k) {
            uint32_t id = sample_ids[static_cast<size_t>(k) % sample_ids.size()];
            uint32_t cid = ivf.assignments[id];

            const float* x = base + static_cast<size_t>(id) * vecdim;
            const float* c = ivf.centroids.data() + static_cast<size_t>(cid) * vecdim;

            for (int j = 0; j < pq.subdim; ++j) {
                int offset = m * pq.subdim + j;
                cb_m[k * pq.subdim + j] = x[offset] - c[offset];
            }
        }

        std::vector<float> new_cb(static_cast<size_t>(Ks) * pq.subdim, 0.0f);
        std::vector<int> counts(Ks, 0);

        for (int iter = 0; iter < pq_iter; ++iter) {
            std::fill(new_cb.begin(), new_cb.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (int s = 0; s < train_samples; ++s) {
                uint32_t id = sample_ids[s];
                uint32_t cid = ivf.assignments[id];

                const float* x = base + static_cast<size_t>(id) * vecdim;
                const float* c = ivf.centroids.data() + static_cast<size_t>(cid) * vecdim;

                float best_dist = 1e30f;
                int best_k = 0;

                for (int k = 0; k < Ks; ++k) {
                    const float* cw = cb_m + k * pq.subdim;
                    float dist = residual_sub_l2(x, c, cw, m, pq.subdim);

                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                float* acc = new_cb.data() + best_k * pq.subdim;

                for (int j = 0; j < pq.subdim; ++j) {
                    int offset = m * pq.subdim + j;
                    acc[j] += x[offset] - c[offset];
                }

                counts[best_k]++;
            }

            for (int k = 0; k < Ks; ++k) {
                float* dst = cb_m + k * pq.subdim;
                float* src = new_cb.data() + k * pq.subdim;

                if (counts[k] > 0) {
                    float inv = 1.0f / static_cast<float>(counts[k]);

                    for (int j = 0; j < pq.subdim; ++j) {
                        dst[j] = src[j] * inv;
                    }
                }
            }
        }
    }

    std::cerr << "[train_pq_cpu] encode base vectors..." << std::endl;

    for (size_t i = 0; i < base_number; ++i) {
        uint32_t cid = ivf.assignments[i];

        const float* x = base + i * vecdim;
        const float* c = ivf.centroids.data() + static_cast<size_t>(cid) * vecdim;

        for (int m = 0; m < M; ++m) {
            const float* cb_m = pq.codebooks.data()
                              + static_cast<size_t>(m) * Ks * pq.subdim;

            float best_dist = 1e30f;
            int best_k = 0;

            for (int k = 0; k < Ks; ++k) {
                const float* cw = cb_m + k * pq.subdim;
                float dist = residual_sub_l2(x, c, cw, m, pq.subdim);

                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }

            pq.codes[i * static_cast<size_t>(M) + m] =
                static_cast<uint8_t>(best_k);
        }

        if ((i + 1) % 20000 == 0) {
            std::cerr << "[train_pq_cpu] encoded "
                      << i + 1 << "/" << base_number << "\r" << std::flush;
        }
    }

    std::cerr << "\n[train_pq_cpu] finished." << std::endl;

    return pq;
}

__global__ void build_pq_lut_kernel(const float* __restrict__ query_batch,
                                    const float* __restrict__ pq_codebooks,
                                    float* __restrict__ pq_lut,
                                    int current_batch_size,
                                    int M,
                                    int Ks,
                                    int subdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = current_batch_size * M * Ks;

    if (idx >= total) {
        return;
    }

    int code = idx % Ks;
    int tmp = idx / Ks;
    int m = tmp % M;
    int qid = tmp / M;

    const float* q = query_batch
                   + static_cast<size_t>(qid) * M * subdim
                   + static_cast<size_t>(m) * subdim;

    const float* cw = pq_codebooks
                    + (static_cast<size_t>(m) * Ks + code) * subdim;

    float sum = 0.0f;

    for (int j = 0; j < subdim; ++j) {
        sum += q[j] * cw[j];
    }

    pq_lut[idx] = sum;
}

__global__ void ivfpq_pair_score_kernel(const uint8_t* __restrict__ pq_codes,
                                        const uint32_t* __restrict__ pair_base_ids,
                                        const uint32_t* __restrict__ pair_query_ids,
                                        const uint32_t* __restrict__ pair_cluster_ids,
                                        const float* __restrict__ centroid_scores,
                                        const float* __restrict__ pq_lut,
                                        float* __restrict__ pair_scores,
                                        int pair_count,
                                        int nlist,
                                        int M,
                                        int Ks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= pair_count) {
        return;
    }

    uint32_t base_id = pair_base_ids[idx];
    uint32_t qid = pair_query_ids[idx];
    uint32_t cid = pair_cluster_ids[idx];

    float score =
        centroid_scores[static_cast<size_t>(qid) * nlist + cid];

    const uint8_t* codes =
        pq_codes + static_cast<size_t>(base_id) * M;

    for (int m = 0; m < M; ++m) {
        uint8_t code = codes[m];

        score += pq_lut[
            (static_cast<size_t>(qid) * M + m) * Ks + static_cast<int>(code)
        ];
    }

    pair_scores[idx] = score;
}

struct BatchStats {
    double total_batch_us = 0.0;
    double cluster_select_and_pair_build_us = 0.0;
    double h2d_us = 0.0;
    double lut_kernel_us = 0.0;
    double score_kernel_us = 0.0;
    double d2h_us = 0.0;
    double cpu_topk_us = 0.0;

    double recall_sum = 0.0;
    size_t query_count = 0;

    double pair_count_sum = 0.0;
    double touched_cluster_count_sum = 0.0;
};

static void reserve_pair_buffers(uint32_t*& d_pair_base_ids,
                                 uint32_t*& d_pair_query_ids,
                                 uint32_t*& d_pair_cluster_ids,
                                 float*& d_pair_scores,
                                 size_t& pair_capacity,
                                 size_t required_pair_count) {
    if (required_pair_count <= pair_capacity &&
        d_pair_base_ids != nullptr &&
        d_pair_query_ids != nullptr &&
        d_pair_cluster_ids != nullptr &&
        d_pair_scores != nullptr) {
        return;
    }

    size_t new_capacity = std::max<size_t>(pair_capacity, 1);

    while (new_capacity < required_pair_count) {
        new_capacity *= 2;
    }

    if (d_pair_base_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_pair_base_ids));
    }

    if (d_pair_query_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_pair_query_ids));
    }

    if (d_pair_cluster_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_pair_cluster_ids));
    }

    if (d_pair_scores != nullptr) {
        CUDA_CHECK(cudaFree(d_pair_scores));
    }

    CUDA_CHECK(cudaMalloc(&d_pair_base_ids,
                          new_capacity * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d_pair_query_ids,
                          new_capacity * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d_pair_cluster_ids,
                          new_capacity * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d_pair_scores,
                          new_capacity * sizeof(float)));

    pair_capacity = new_capacity;
}

static void compute_topk_and_recall_from_pairs_cpu(
    const std::vector<float>& pair_scores,
    const std::vector<uint32_t>& pair_base_ids,
    const std::vector<uint32_t>& pair_query_ids,
    int current_batch_size,
    size_t pair_count,
    size_t k,
    const int* gt,
    size_t gt_d,
    size_t query_global_start,
    BatchStats& stats
) {
    using Pair = std::pair<float, uint32_t>;

    std::vector<std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>> heaps(
        static_cast<size_t>(current_batch_size)
    );

    for (size_t i = 0; i < pair_count; ++i) {
        uint32_t qid = pair_query_ids[i];
        uint32_t bid = pair_base_ids[i];
        float ip = pair_scores[i];

        auto& heap = heaps[qid];
        Pair item{ip, bid};

        if (heap.size() < k) {
            heap.push(item);
        } else if (item.first > heap.top().first) {
            heap.pop();
            heap.push(item);
        }
    }

    for (int b = 0; b < current_batch_size; ++b) {
        std::set<uint32_t> gtset;
        size_t query_id = query_global_start + static_cast<size_t>(b);

        for (size_t j = 0; j < k; ++j) {
            gtset.insert(static_cast<uint32_t>(gt[query_id * gt_d + j]));
        }

        size_t hit = 0;
        auto& heap = heaps[static_cast<size_t>(b)];

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
    /*
        Usage:
            main_gpu_ivfpq.exe <data_path> <test_number> <batch_size>
                               <nlist> <nprobe> <ivf_iter>
                               <M> <Ks> <pq_train_samples> <pq_iter> <k>

        Fast debug example:
            main_gpu_ivfpq.exe ./data/ 100 16 1024 50 3 8 64 4000 4 10

        Formal example:
            main_gpu_ivfpq.exe ./data/ 1000 16 1024 50 10 8 256 8000 6 10
    */

    std::string data_path = argc >= 2 ? argv[1] : "./data/";
    data_path = ensure_slash(data_path);

    size_t test_limit =
        argc >= 3 ? static_cast<size_t>(std::stoul(argv[2])) : 1000;

    int batch_size =
        argc >= 4 ? std::stoi(argv[3]) : 16;

    size_t nlist =
        argc >= 5 ? static_cast<size_t>(std::stoul(argv[4])) : 1024;

    size_t nprobe =
        argc >= 6 ? static_cast<size_t>(std::stoul(argv[5])) : 50;

    int ivf_iter =
        argc >= 7 ? std::stoi(argv[6]) : 10;

    int M =
        argc >= 8 ? std::stoi(argv[7]) : 8;

    int Ks =
        argc >= 9 ? std::stoi(argv[8]) : 256;

    int pq_train_samples =
        argc >= 10 ? std::stoi(argv[9]) : 8000;

    int pq_iter =
        argc >= 11 ? std::stoi(argv[10]) : 6;

    size_t k =
        argc >= 12 ? static_cast<size_t>(std::stoul(argv[11])) : 10;

    if (batch_size <= 0) {
        std::cerr << "batch_size must be positive\n";
        return 1;
    }

    if (Ks > 256) {
        std::cerr << "Ks must be <= 256 because codes are stored as uint8_t\n";
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
        gt_d < k ||
        static_cast<int>(vecdim) % M != 0) {
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
              << ", ivf_iter=" << ivf_iter
              << ", M=" << M
              << ", Ks=" << Ks
              << ", pq_train_samples=" << pq_train_samples
              << ", pq_iter=" << pq_iter
              << ", k=" << k << "\n";

    auto ivf_build_begin = std::chrono::steady_clock::now();

    IVFIndex ivf_idx =
        build_ivf_index_cpu(base,
                            base_number,
                            vecdim,
                            nlist,
                            ivf_iter);

    auto ivf_build_end = std::chrono::steady_clock::now();

    std::cerr << "build ivf index time (s): "
              << elapsed_us(ivf_build_begin, ivf_build_end) / 1000000.0
              << "\n";

    auto pq_train_begin = std::chrono::steady_clock::now();

    PQIndex pq =
        train_pq_cpu(base,
                     base_number,
                     vecdim,
                     ivf_idx,
                     M,
                     Ks,
                     pq_train_samples,
                     pq_iter);

    auto pq_train_end = std::chrono::steady_clock::now();

    std::cerr << "train and encode pq time (s): "
              << elapsed_us(pq_train_begin, pq_train_end) / 1000000.0
              << "\n";

    float* d_query_batch = nullptr;
    float* d_pq_codebooks = nullptr;
    uint8_t* d_pq_codes = nullptr;
    float* d_centroid_scores = nullptr;
    float* d_pq_lut = nullptr;

    uint32_t* d_pair_base_ids = nullptr;
    uint32_t* d_pair_query_ids = nullptr;
    uint32_t* d_pair_cluster_ids = nullptr;
    float* d_pair_scores = nullptr;

    size_t pair_capacity = 0;

    CUDA_CHECK(cudaMalloc(&d_query_batch,
                          static_cast<size_t>(batch_size) * vecdim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_centroid_scores,
                          static_cast<size_t>(batch_size) * nlist * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_pq_lut,
                          static_cast<size_t>(batch_size) * pq.M * pq.Ks * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_pq_codebooks,
                          pq.codebooks.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pq_codebooks,
                          pq.codebooks.data(),
                          pq.codebooks.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pq_codes,
                          pq.codes.size() * sizeof(uint8_t)));

    CUDA_CHECK(cudaMemcpy(d_pq_codes,
                          pq.codes.data(),
                          pq.codes.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    cudaEvent_t h2d_start, h2d_stop;
    cudaEvent_t lut_start, lut_stop;
    cudaEvent_t score_start, score_stop;
    cudaEvent_t d2h_start, d2h_stop;

    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&lut_start));
    CUDA_CHECK(cudaEventCreate(&lut_stop));
    CUDA_CHECK(cudaEventCreate(&score_start));
    CUDA_CHECK(cudaEventCreate(&score_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    BatchStats total_stats;

    size_t processed = 0;
    size_t batch_id = 0;

    while (processed < test_number) {
        int current_batch_size = static_cast<int>(
            std::min(static_cast<size_t>(batch_size), test_number - processed)
        );

        auto batch_begin = std::chrono::steady_clock::now();
        auto pair_build_begin = std::chrono::steady_clock::now();

        std::vector<float> centroid_scores(
            static_cast<size_t>(current_batch_size) * nlist
        );

        std::vector<std::vector<uint32_t>> cluster_query_lists(nlist);
        std::vector<uint32_t> touched_clusters;
        std::vector<uint8_t> cluster_touched(nlist, 0);

        for (int b = 0; b < current_batch_size; ++b) {
            const float* q =
                query + (processed + static_cast<size_t>(b)) * vecdim;

            float* scores_b =
                centroid_scores.data() + static_cast<size_t>(b) * nlist;

            for (size_t c = 0; c < nlist; ++c) {
                scores_b[c] =
                    inner_product_cpu(q,
                                      ivf_idx.centroids.data() + c * vecdim,
                                      vecdim);
            }

            std::vector<uint32_t> selected_clusters =
                select_top_clusters_from_scores_cpu(scores_b, nlist, nprobe);

            for (uint32_t cid : selected_clusters) {
                if (!cluster_touched[cid]) {
                    cluster_touched[cid] = 1;
                    touched_clusters.push_back(cid);
                }

                cluster_query_lists[cid].push_back(static_cast<uint32_t>(b));
            }
        }

        std::vector<uint32_t> pair_base_ids;
        std::vector<uint32_t> pair_query_ids;
        std::vector<uint32_t> pair_cluster_ids;

        size_t estimated_pairs = 0;

        for (uint32_t cid : touched_clusters) {
            estimated_pairs +=
                ivf_idx.inverted_lists[cid].size() *
                cluster_query_lists[cid].size();
        }

        pair_base_ids.reserve(estimated_pairs);
        pair_query_ids.reserve(estimated_pairs);
        pair_cluster_ids.reserve(estimated_pairs);

        for (uint32_t cid : touched_clusters) {
            const auto& ids = ivf_idx.inverted_lists[cid];
            const auto& qids = cluster_query_lists[cid];

            for (uint32_t base_id : ids) {
                for (uint32_t qid : qids) {
                    pair_base_ids.push_back(base_id);
                    pair_query_ids.push_back(qid);
                    pair_cluster_ids.push_back(cid);
                }
            }
        }

        auto pair_build_end = std::chrono::steady_clock::now();

        size_t pair_count = pair_base_ids.size();

        total_stats.pair_count_sum += static_cast<double>(pair_count);
        total_stats.touched_cluster_count_sum +=
            static_cast<double>(touched_clusters.size());

        if (pair_count == 0) {
            processed += static_cast<size_t>(current_batch_size);
            batch_id += 1;
            continue;
        }

        reserve_pair_buffers(d_pair_base_ids,
                             d_pair_query_ids,
                             d_pair_cluster_ids,
                             d_pair_scores,
                             pair_capacity,
                             pair_count);

        std::vector<float> h_pair_scores(pair_count);

        size_t current_query_bytes =
            static_cast<size_t>(current_batch_size) * vecdim * sizeof(float);

        size_t current_centroid_score_bytes =
            static_cast<size_t>(current_batch_size) * nlist * sizeof(float);

        size_t pair_id_bytes =
            pair_count * sizeof(uint32_t);

        size_t pair_score_bytes =
            pair_count * sizeof(float);

        CUDA_CHECK(cudaEventRecord(h2d_start));

        CUDA_CHECK(cudaMemcpy(d_query_batch,
                              query + processed * vecdim,
                              current_query_bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_centroid_scores,
                              centroid_scores.data(),
                              current_centroid_score_bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_pair_base_ids,
                              pair_base_ids.data(),
                              pair_id_bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_pair_query_ids,
                              pair_query_ids.data(),
                              pair_id_bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_pair_cluster_ids,
                              pair_cluster_ids.data(),
                              pair_id_bytes,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(h2d_stop));
        CUDA_CHECK(cudaEventSynchronize(h2d_stop));

        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));

        int threads = 256;

        int lut_total = current_batch_size * pq.M * pq.Ks;
        int lut_blocks = (lut_total + threads - 1) / threads;

        CUDA_CHECK(cudaEventRecord(lut_start));

        build_pq_lut_kernel<<<lut_blocks, threads>>>(
            d_query_batch,
            d_pq_codebooks,
            d_pq_lut,
            current_batch_size,
            pq.M,
            pq.Ks,
            pq.subdim
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(lut_stop));
        CUDA_CHECK(cudaEventSynchronize(lut_stop));

        float lut_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&lut_ms, lut_start, lut_stop));

        int score_blocks =
            static_cast<int>((pair_count + threads - 1) / threads);

        CUDA_CHECK(cudaEventRecord(score_start));

        ivfpq_pair_score_kernel<<<score_blocks, threads>>>(
            d_pq_codes,
            d_pair_base_ids,
            d_pair_query_ids,
            d_pair_cluster_ids,
            d_centroid_scores,
            d_pq_lut,
            d_pair_scores,
            static_cast<int>(pair_count),
            static_cast<int>(nlist),
            pq.M,
            pq.Ks
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(score_stop));
        CUDA_CHECK(cudaEventSynchronize(score_stop));

        float score_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&score_ms, score_start, score_stop));

        CUDA_CHECK(cudaEventRecord(d2h_start));

        CUDA_CHECK(cudaMemcpy(h_pair_scores.data(),
                              d_pair_scores,
                              pair_score_bytes,
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(d2h_stop));
        CUDA_CHECK(cudaEventSynchronize(d2h_stop));

        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

        auto topk_begin = std::chrono::steady_clock::now();

        compute_topk_and_recall_from_pairs_cpu(
            h_pair_scores,
            pair_base_ids,
            pair_query_ids,
            current_batch_size,
            pair_count,
            k,
            gt,
            gt_d,
            processed,
            total_stats
        );

        auto topk_end = std::chrono::steady_clock::now();
        auto batch_end = std::chrono::steady_clock::now();

        total_stats.cluster_select_and_pair_build_us +=
            elapsed_us(pair_build_begin, pair_build_end);

        total_stats.h2d_us += h2d_ms * 1000.0;
        total_stats.lut_kernel_us += lut_ms * 1000.0;
        total_stats.score_kernel_us += score_ms * 1000.0;
        total_stats.d2h_us += d2h_ms * 1000.0;
        total_stats.cpu_topk_us += elapsed_us(topk_begin, topk_end);
        total_stats.total_batch_us += elapsed_us(batch_begin, batch_end);

        processed += static_cast<size_t>(current_batch_size);
        batch_id += 1;

        std::cerr << "batch " << batch_id
                  << " processed " << processed << "/" << test_number
                  << ", current_batch_size=" << current_batch_size
                  << ", touched_clusters=" << touched_clusters.size()
                  << ", pair_count=" << pair_count
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

    double avg_pair_build_per_query_us =
        total_stats.cluster_select_and_pair_build_us / query_count_d;

    double avg_h2d_per_query_us =
        total_stats.h2d_us / query_count_d;

    double avg_lut_kernel_per_query_us =
        total_stats.lut_kernel_us / query_count_d;

    double avg_score_kernel_per_query_us =
        total_stats.score_kernel_us / query_count_d;

    double avg_d2h_per_query_us =
        total_stats.d2h_us / query_count_d;

    double avg_topk_per_query_us =
        total_stats.cpu_topk_us / query_count_d;

    double avg_batch_time_us =
        total_stats.total_batch_us / batch_count_d;

    double avg_pairs_per_batch =
        total_stats.pair_count_sum / batch_count_d;

    double avg_pairs_per_query =
        total_stats.pair_count_sum / query_count_d;

    double avg_touched_clusters_per_batch =
        total_stats.touched_cluster_count_sum / batch_count_d;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "average recall: "
              << avg_recall << "\n";

    std::cout << "average latency per query (us): "
              << avg_latency_per_query_us << "\n";

    std::cout << "average batch time (us): "
              << avg_batch_time_us << "\n";

    std::cout << "average cluster select and pair build per query (us): "
              << avg_pair_build_per_query_us << "\n";

    std::cout << "average H2D per query (us): "
              << avg_h2d_per_query_us << "\n";

    std::cout << "average LUT kernel per query (us): "
              << avg_lut_kernel_per_query_us << "\n";

    std::cout << "average PQ score kernel per query (us): "
              << avg_score_kernel_per_query_us << "\n";

    std::cout << "average D2H per query (us): "
              << avg_d2h_per_query_us << "\n";

    std::cout << "average CPU topk per query (us): "
              << avg_topk_per_query_us << "\n";

    std::cout << "average pairs per batch: "
              << avg_pairs_per_batch << "\n";

    std::cout << "average pairs per query: "
              << avg_pairs_per_query << "\n";

    std::cout << "average touched clusters per batch: "
              << avg_touched_clusters_per_batch << "\n";

    std::cout << "total queries: "
              << total_stats.query_count << "\n";

    std::cout << "batch size: "
              << batch_size << "\n";

    std::cout << "nlist: "
              << nlist << "\n";

    std::cout << "nprobe: "
              << nprobe << "\n";

    std::cout << "PQ M: "
              << pq.M << "\n";

    std::cout << "PQ Ks: "
              << pq.Ks << "\n";

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(lut_start));
    CUDA_CHECK(cudaEventDestroy(lut_stop));
    CUDA_CHECK(cudaEventDestroy(score_start));
    CUDA_CHECK(cudaEventDestroy(score_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    if (d_query_batch != nullptr) {
        CUDA_CHECK(cudaFree(d_query_batch));
    }

    if (d_pq_codebooks != nullptr) {
        CUDA_CHECK(cudaFree(d_pq_codebooks));
    }

    if (d_pq_codes != nullptr) {
        CUDA_CHECK(cudaFree(d_pq_codes));
    }

    if (d_centroid_scores != nullptr) {
        CUDA_CHECK(cudaFree(d_centroid_scores));
    }

    if (d_pq_lut != nullptr) {
        CUDA_CHECK(cudaFree(d_pq_lut));
    }

    if (d_pair_base_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_pair_base_ids));
    }

    if (d_pair_query_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_pair_query_ids));
    }

    if (d_pair_cluster_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_pair_cluster_ids));
    }

    if (d_pair_scores != nullptr) {
        CUDA_CHECK(cudaFree(d_pair_scores));
    }

    delete[] query;
    delete[] gt;
    delete[] base;

    return 0;
}