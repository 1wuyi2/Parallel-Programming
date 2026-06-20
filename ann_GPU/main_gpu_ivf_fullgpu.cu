#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " : " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

static std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

static std::vector<float> read_fbin(const std::string& filename,
                                    uint32_t& n,
                                    uint32_t& d) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Cannot open fbin file: " + filename);
    }

    fin.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));

    if (!fin || n == 0 || d == 0) {
        throw std::runtime_error("Invalid fbin header: " + filename);
    }

    std::vector<float> data(static_cast<size_t>(n) * d);
    fin.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(float)));

    if (!fin) {
        throw std::runtime_error("Failed to read fbin data: " + filename);
    }

    return data;
}

static std::vector<uint32_t> read_ibin(const std::string& filename,
                                       uint32_t& n,
                                       uint32_t& d) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Cannot open ibin file: " + filename);
    }

    fin.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));

    if (!fin || n == 0 || d == 0) {
        throw std::runtime_error("Invalid ibin header: " + filename);
    }

    std::vector<uint32_t> data(static_cast<size_t>(n) * d);
    fin.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(uint32_t)));

    if (!fin) {
        throw std::runtime_error("Failed to read ibin data: " + filename);
    }

    return data;
}

static inline float l2_distance2(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

struct IVFIndexCPU {
    int nlist = 0;
    int vecdim = 0;
    std::vector<float> centroids;                  // nlist * vecdim
    std::vector<std::vector<uint32_t>> lists;      // inverted lists
};

static IVFIndexCPU build_ivf_index_cpu(const std::vector<float>& base,
                                       int base_number,
                                       int vecdim,
                                       int nlist,
                                       int kmeans_iter) {
    IVFIndexCPU index;
    index.nlist = nlist;
    index.vecdim = vecdim;
    index.centroids.resize(static_cast<size_t>(nlist) * vecdim);
    index.lists.resize(nlist);

    std::cout << "[build_ivf_index_cpu] init centroids.\n";

    for (int c = 0; c < nlist; ++c) {
        int id = static_cast<int>((static_cast<long long>(c) * 9973LL) % base_number);
        std::copy(base.data() + static_cast<size_t>(id) * vecdim,
                  base.data() + static_cast<size_t>(id + 1) * vecdim,
                  index.centroids.data() + static_cast<size_t>(c) * vecdim);
    }

    std::vector<int> assign(base_number, 0);
    std::vector<float> sums(static_cast<size_t>(nlist) * vecdim, 0.0f);
    std::vector<int> counts(nlist, 0);

    for (int iter = 0; iter < kmeans_iter; ++iter) {
        std::cout << "[build_ivf_index_cpu] kmeans iter "
                  << (iter + 1) << "/" << kmeans_iter << "\n";

        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (int i = 0; i < base_number; ++i) {
            const float* x = base.data() + static_cast<size_t>(i) * vecdim;

            int best_c = 0;
            float best_dist = std::numeric_limits<float>::max();

            for (int c = 0; c < nlist; ++c) {
                const float* cen = index.centroids.data() + static_cast<size_t>(c) * vecdim;
                float dist = l2_distance2(x, cen, vecdim);

                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }

            assign[i] = best_c;
            counts[best_c]++;

            float* sum_c = sums.data() + static_cast<size_t>(best_c) * vecdim;
            for (int d = 0; d < vecdim; ++d) {
                sum_c[d] += x[d];
            }
        }

        for (int c = 0; c < nlist; ++c) {
            float* cen = index.centroids.data() + static_cast<size_t>(c) * vecdim;

            if (counts[c] > 0) {
                const float inv = 1.0f / static_cast<float>(counts[c]);
                const float* sum_c = sums.data() + static_cast<size_t>(c) * vecdim;

                for (int d = 0; d < vecdim; ++d) {
                    cen[d] = sum_c[d] * inv;
                }
            } else {
                int id = static_cast<int>((static_cast<long long>(iter + 1) * 1315423911LL +
                                           static_cast<long long>(c) * 2654435761LL) %
                                          base_number);
                if (id < 0) id += base_number;

                std::copy(base.data() + static_cast<size_t>(id) * vecdim,
                          base.data() + static_cast<size_t>(id + 1) * vecdim,
                          cen);
            }
        }
    }

    std::cout << "[build_ivf_index_cpu] final assignment.\n";

    for (auto& list : index.lists) {
        list.clear();
    }

    for (int i = 0; i < base_number; ++i) {
        const float* x = base.data() + static_cast<size_t>(i) * vecdim;

        int best_c = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (int c = 0; c < nlist; ++c) {
            const float* cen = index.centroids.data() + static_cast<size_t>(c) * vecdim;
            float dist = l2_distance2(x, cen, vecdim);

            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }

        index.lists[best_c].push_back(static_cast<uint32_t>(i));
    }

    int non_empty = 0;
    size_t max_list_size = 0;
    for (const auto& list : index.lists) {
        if (!list.empty()) non_empty++;
        max_list_size = std::max(max_list_size, list.size());
    }

    std::cout << "[build_ivf_index_cpu] non_empty_lists="
              << non_empty << "/" << nlist
              << ", max_list_size=" << max_list_size << "\n";

    return index;
}

static void flatten_ivf_lists(const IVFIndexCPU& index,
                              std::vector<int>& offsets,
                              std::vector<uint32_t>& ids) {
    offsets.assign(index.nlist + 1, 0);
    ids.clear();

    for (int c = 0; c < index.nlist; ++c) {
        offsets[c] = static_cast<int>(ids.size());
        ids.insert(ids.end(), index.lists[c].begin(), index.lists[c].end());
    }

    offsets[index.nlist] = static_cast<int>(ids.size());
}

__global__ void centroid_score_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    float* __restrict__ centroid_scores,
    int batch_size,
    int nlist,
    int vecdim
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y;

    if (q >= batch_size || c >= nlist) return;

    const float* query = queries + static_cast<size_t>(q) * vecdim;
    const float* cen = centroids + static_cast<size_t>(c) * vecdim;

    float sum = 0.0f;
    for (int d = 0; d < vecdim; ++d) {
        sum += query[d] * cen[d];
    }

    centroid_scores[static_cast<size_t>(q) * nlist + c] = sum;
}

/*
    Simple GPU top-nprobe.
    One block handles one query.
    thread 0 does the selection.

    nlist is only 1024 in this experiment, so this serial selection on GPU
    is acceptable and avoids returning centroid scores to CPU.
*/
template <int MAX_NPROBE>
__global__ void select_topnprobe_kernel(
    const float* __restrict__ centroid_scores,
    int* __restrict__ selected_clusters,
    int batch_size,
    int nlist,
    int nprobe
) {
    int q = blockIdx.x;
    if (q >= batch_size || nprobe > MAX_NPROBE) return;

    if (threadIdx.x == 0) {
        float best_scores[MAX_NPROBE];
        int best_ids[MAX_NPROBE];

        for (int i = 0; i < MAX_NPROBE; ++i) {
            best_scores[i] = -1.0e30f;
            best_ids[i] = -1;
        }

        const float* scores = centroid_scores + static_cast<size_t>(q) * nlist;

        for (int c = 0; c < nlist; ++c) {
            float s = scores[c];

            int min_pos = 0;
            float min_val = best_scores[0];

            for (int i = 1; i < nprobe; ++i) {
                if (best_scores[i] < min_val) {
                    min_val = best_scores[i];
                    min_pos = i;
                }
            }

            if (s > min_val) {
                best_scores[min_pos] = s;
                best_ids[min_pos] = c;
            }
        }

        for (int i = 0; i < nprobe; ++i) {
            selected_clusters[q * nprobe + i] = best_ids[i];
        }
    }
}

/*
    Full-GPU online exact IVF search.

    One CUDA block handles one query.
    Each thread scans part of candidates in selected clusters.
    Each thread keeps local top-k.
    thread 0 merges all local top-k results.
*/
template <int MAX_K>
__global__ void ivf_exact_scan_topk_kernel(
    const float* __restrict__ base,
    const float* __restrict__ queries,
    const int* __restrict__ offsets,
    const uint32_t* __restrict__ inv_ids,
    const int* __restrict__ selected_clusters,
    uint32_t* __restrict__ topk_ids,
    float* __restrict__ topk_scores,
    int batch_size,
    int vecdim,
    int nprobe,
    int k
) {
    int q = blockIdx.x;
    int tid = threadIdx.x;

    if (q >= batch_size || k > MAX_K) return;

    const float* query = queries + static_cast<size_t>(q) * vecdim;

    float local_scores[MAX_K];
    uint32_t local_ids[MAX_K];

    for (int i = 0; i < MAX_K; ++i) {
        local_scores[i] = -1.0e30f;
        local_ids[i] = 0xffffffffu;
    }

    for (int pi = 0; pi < nprobe; ++pi) {
        int c = selected_clusters[q * nprobe + pi];
        if (c < 0) continue;

        int begin = offsets[c];
        int end = offsets[c + 1];

        for (int p = begin + tid; p < end; p += blockDim.x) {
            uint32_t base_id = inv_ids[p];
            const float* x = base + static_cast<size_t>(base_id) * vecdim;

            float score = 0.0f;
            for (int d = 0; d < vecdim; ++d) {
                score += query[d] * x[d];
            }

            int min_pos = 0;
            float min_val = local_scores[0];

            for (int i = 1; i < k; ++i) {
                if (local_scores[i] < min_val) {
                    min_val = local_scores[i];
                    min_pos = i;
                }
            }

            if (score > min_val) {
                local_scores[min_pos] = score;
                local_ids[min_pos] = base_id;
            }
        }
    }

    extern __shared__ unsigned char smem[];
    float* sh_scores = reinterpret_cast<float*>(smem);
    uint32_t* sh_ids = reinterpret_cast<uint32_t*>(sh_scores + blockDim.x * k);

    for (int i = 0; i < k; ++i) {
        sh_scores[tid * k + i] = local_scores[i];
        sh_ids[tid * k + i] = local_ids[i];
    }

    __syncthreads();

    if (tid == 0) {
        float final_scores[MAX_K];
        uint32_t final_ids[MAX_K];

        for (int i = 0; i < MAX_K; ++i) {
            final_scores[i] = -1.0e30f;
            final_ids[i] = 0xffffffffu;
        }

        for (int t = 0; t < blockDim.x; ++t) {
            for (int j = 0; j < k; ++j) {
                float s = sh_scores[t * k + j];
                uint32_t id = sh_ids[t * k + j];

                int min_pos = 0;
                float min_val = final_scores[0];

                for (int p = 1; p < k; ++p) {
                    if (final_scores[p] < min_val) {
                        min_val = final_scores[p];
                        min_pos = p;
                    }
                }

                if (s > min_val) {
                    final_scores[min_pos] = s;
                    final_ids[min_pos] = id;
                }
            }
        }

        for (int i = 0; i < k; ++i) {
            int best_pos = i;
            for (int j = i + 1; j < k; ++j) {
                if (final_scores[j] > final_scores[best_pos]) {
                    best_pos = j;
                }
            }

            if (best_pos != i) {
                float ts = final_scores[i];
                final_scores[i] = final_scores[best_pos];
                final_scores[best_pos] = ts;

                uint32_t ti = final_ids[i];
                final_ids[i] = final_ids[best_pos];
                final_ids[best_pos] = ti;
            }
        }

        for (int i = 0; i < k; ++i) {
            topk_scores[q * k + i] = final_scores[i];
            topk_ids[q * k + i] = final_ids[i];
        }
    }
}

static double compute_batch_recall(
    const std::vector<uint32_t>& topk_ids,
    const std::vector<uint32_t>& gt,
    int batch_start,
    int current_batch_size,
    int k,
    int gt_dim
) {
    double hit = 0.0;

    for (int q = 0; q < current_batch_size; ++q) {
        int global_qid = batch_start + q;

        std::unordered_set<uint32_t> gt_set;
        gt_set.reserve(k * 2);

        for (int j = 0; j < k; ++j) {
            gt_set.insert(gt[static_cast<size_t>(global_qid) * gt_dim + j]);
        }

        for (int j = 0; j < k; ++j) {
            uint32_t pred = topk_ids[static_cast<size_t>(q) * k + j];
            if (gt_set.find(pred) != gt_set.end()) {
                hit += 1.0;
            }
        }
    }

    return hit / static_cast<double>(current_batch_size * k);
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " <data_path> <test_number> <batch_size> <nlist> <nprobe> <kmeans_iter> <k>\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " ./data/ 1000 16 1024 50 10 10\n";
        return 1;
    }

    std::string data_path = argv[1];
    int test_number = std::atoi(argv[2]);
    int batch_size = std::atoi(argv[3]);
    int nlist = std::atoi(argv[4]);
    int nprobe = std::atoi(argv[5]);
    int kmeans_iter = std::atoi(argv[6]);
    int k = std::atoi(argv[7]);

    if (test_number <= 0 || batch_size <= 0 || nlist <= 0 ||
        nprobe <= 0 || kmeans_iter <= 0 || k <= 0) {
        std::cerr << "Invalid arguments.\n";
        return 1;
    }

    constexpr int MAX_K = 32;
    constexpr int MAX_NPROBE = 128;

    if (k > MAX_K) {
        std::cerr << "This implementation supports k <= "
                  << MAX_K << ". Current k=" << k << "\n";
        return 1;
    }

    if (nprobe > MAX_NPROBE) {
        std::cerr << "This implementation supports nprobe <= "
                  << MAX_NPROBE << ". Current nprobe=" << nprobe << "\n";
        return 1;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Using GPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";

    uint32_t base_n = 0, base_d = 0;
    uint32_t query_n = 0, query_d = 0;
    uint32_t gt_n = 0, gt_d = 0;

    std::string base_file = join_path(data_path, "DEEP100K.base.100k.fbin");
    std::string query_file = join_path(data_path, "DEEP100K.query.fbin");
    std::string gt_file = join_path(data_path, "DEEP100K.gt.query.100k.top100.bin");

    std::cout << "Reading base: " << base_file << "\n";
    std::vector<float> base = read_fbin(base_file, base_n, base_d);

    std::cout << "Reading query: " << query_file << "\n";
    std::vector<float> query = read_fbin(query_file, query_n, query_d);

    std::cout << "Reading gt: " << gt_file << "\n";
    std::vector<uint32_t> gt = read_ibin(gt_file, gt_n, gt_d);

    if (base_d != query_d) {
        std::cerr << "Dimension mismatch: base_d=" << base_d
                  << ", query_d=" << query_d << "\n";
        return 1;
    }

    if (gt_d < static_cast<uint32_t>(k)) {
        std::cerr << "GT dimension is smaller than k.\n";
        return 1;
    }

    if (test_number > static_cast<int>(query_n)) {
        test_number = static_cast<int>(query_n);
    }

    int base_number = static_cast<int>(base_n);
    int vecdim = static_cast<int>(base_d);

    std::cout << "base number: " << base_number << ", dim: " << vecdim << "\n";
    std::cout << "query number: " << query_n << ", dim: " << query_d << "\n";
    std::cout << "gt number: " << gt_n << ", gt dim: " << gt_d << "\n";
    std::cout << "config: test_number=" << test_number
              << ", batch_size=" << batch_size
              << ", nlist=" << nlist
              << ", nprobe=" << nprobe
              << ", kmeans_iter=" << kmeans_iter
              << ", k=" << k << "\n";

    cudaEvent_t ev1, ev2;
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2));

    std::cout << "\n========== Offline CPU IVF Build ==========\n";

    CUDA_CHECK(cudaEventRecord(ev1));
    IVFIndexCPU ivf = build_ivf_index_cpu(base, base_number, vecdim, nlist, kmeans_iter);
    CUDA_CHECK(cudaEventRecord(ev2));
    CUDA_CHECK(cudaEventSynchronize(ev2));

    float build_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&build_ms, ev1, ev2));

    std::vector<int> h_offsets;
    std::vector<uint32_t> h_inv_ids;
    flatten_ivf_lists(ivf, h_offsets, h_inv_ids);

    std::cout << "build ivf index time (s): " << build_ms / 1000.0f << "\n";
    std::cout << "total inverted ids: " << h_inv_ids.size() << "\n";
    std::cout << "==========================================\n\n";

    float* d_base = nullptr;
    float* d_queries = nullptr;
    float* d_centroids = nullptr;
    float* d_centroid_scores = nullptr;
    int* d_offsets = nullptr;
    uint32_t* d_inv_ids = nullptr;
    int* d_selected_clusters = nullptr;
    uint32_t* d_topk_ids = nullptr;
    float* d_topk_scores = nullptr;

    size_t base_bytes = static_cast<size_t>(base_number) * vecdim * sizeof(float);
    size_t query_batch_bytes = static_cast<size_t>(batch_size) * vecdim * sizeof(float);
    size_t centroid_bytes = static_cast<size_t>(nlist) * vecdim * sizeof(float);
    size_t centroid_score_bytes = static_cast<size_t>(batch_size) * nlist * sizeof(float);
    size_t offsets_bytes = static_cast<size_t>(nlist + 1) * sizeof(int);
    size_t inv_ids_bytes = h_inv_ids.size() * sizeof(uint32_t);
    size_t selected_bytes = static_cast<size_t>(batch_size) * nprobe * sizeof(int);
    size_t topk_ids_bytes = static_cast<size_t>(batch_size) * k * sizeof(uint32_t);
    size_t topk_scores_bytes = static_cast<size_t>(batch_size) * k * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_base, base_bytes));
    CUDA_CHECK(cudaMalloc(&d_queries, query_batch_bytes));
    CUDA_CHECK(cudaMalloc(&d_centroids, centroid_bytes));
    CUDA_CHECK(cudaMalloc(&d_centroid_scores, centroid_score_bytes));
    CUDA_CHECK(cudaMalloc(&d_offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc(&d_inv_ids, inv_ids_bytes));
    CUDA_CHECK(cudaMalloc(&d_selected_clusters, selected_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_ids, topk_ids_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_scores, topk_scores_bytes));

    CUDA_CHECK(cudaMemcpy(d_base, base.data(), base_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, ivf.centroids.data(), centroid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv_ids, h_inv_ids.data(), inv_ids_bytes, cudaMemcpyHostToDevice));

    std::vector<uint32_t> h_topk_ids(static_cast<size_t>(batch_size) * k);
    std::vector<float> h_topk_scores(static_cast<size_t>(batch_size) * k);

    double total_recall = 0.0;
    int processed_queries = 0;

    double total_h2d_us = 0.0;
    double total_centroid_kernel_us = 0.0;
    double total_select_kernel_us = 0.0;
    double total_scan_topk_kernel_us = 0.0;
    double total_d2h_us = 0.0;

    for (int batch_start = 0; batch_start < test_number; batch_start += batch_size) {
        int current_batch_size = std::min(batch_size, test_number - batch_start);

        const float* h_query_ptr =
            query.data() + static_cast<size_t>(batch_start) * vecdim;

        size_t current_query_bytes =
            static_cast<size_t>(current_batch_size) * vecdim * sizeof(float);

        size_t current_topk_ids_bytes =
            static_cast<size_t>(current_batch_size) * k * sizeof(uint32_t);

        size_t current_topk_scores_bytes =
            static_cast<size_t>(current_batch_size) * k * sizeof(float);

        float ms = 0.0f;

        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaMemcpy(d_queries,
                              h_query_ptr,
                              current_query_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev2));
        CUDA_CHECK(cudaEventSynchronize(ev2));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        total_h2d_us += ms * 1000.0;

        dim3 centroid_block(256);
        dim3 centroid_grid((nlist + 255) / 256, current_batch_size);

        CUDA_CHECK(cudaEventRecord(ev1));
        centroid_score_kernel<<<centroid_grid, centroid_block>>>(
            d_queries,
            d_centroids,
            d_centroid_scores,
            current_batch_size,
            nlist,
            vecdim
        );
        CUDA_CHECK(cudaEventRecord(ev2));
        CUDA_CHECK(cudaEventSynchronize(ev2));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        total_centroid_kernel_us += ms * 1000.0;

        CUDA_CHECK(cudaEventRecord(ev1));
        select_topnprobe_kernel<MAX_NPROBE><<<current_batch_size, 1>>>(
            d_centroid_scores,
            d_selected_clusters,
            current_batch_size,
            nlist,
            nprobe
        );
        CUDA_CHECK(cudaEventRecord(ev2));
        CUDA_CHECK(cudaEventSynchronize(ev2));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        total_select_kernel_us += ms * 1000.0;

        int scan_threads = 256;
        size_t scan_smem =
            static_cast<size_t>(scan_threads) * k * sizeof(float) +
            static_cast<size_t>(scan_threads) * k * sizeof(uint32_t);

        CUDA_CHECK(cudaEventRecord(ev1));
        ivf_exact_scan_topk_kernel<MAX_K><<<current_batch_size, scan_threads, scan_smem>>>(
            d_base,
            d_queries,
            d_offsets,
            d_inv_ids,
            d_selected_clusters,
            d_topk_ids,
            d_topk_scores,
            current_batch_size,
            vecdim,
            nprobe,
            k
        );
        CUDA_CHECK(cudaEventRecord(ev2));
        CUDA_CHECK(cudaEventSynchronize(ev2));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        total_scan_topk_kernel_us += ms * 1000.0;

        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaMemcpy(h_topk_ids.data(),
                              d_topk_ids,
                              current_topk_ids_bytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_topk_scores.data(),
                              d_topk_scores,
                              current_topk_scores_bytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev2));
        CUDA_CHECK(cudaEventSynchronize(ev2));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        total_d2h_us += ms * 1000.0;

        double batch_recall = compute_batch_recall(
            h_topk_ids,
            gt,
            batch_start,
            current_batch_size,
            k,
            static_cast<int>(gt_d)
        );

        total_recall += batch_recall * current_batch_size;
        processed_queries += current_batch_size;

        std::cout << "batch processed "
                  << processed_queries << "/" << test_number
                  << ", current_batch_size=" << current_batch_size
                  << ", batch recall=" << std::fixed << std::setprecision(6)
                  << batch_recall << "\n";
    }

    double avg_recall = total_recall / processed_queries;

    double avg_h2d_us = total_h2d_us / processed_queries;
    double avg_centroid_us = total_centroid_kernel_us / processed_queries;
    double avg_select_us = total_select_kernel_us / processed_queries;
    double avg_scan_topk_us = total_scan_topk_kernel_us / processed_queries;
    double avg_d2h_us = total_d2h_us / processed_queries;

    double avg_latency_us =
        avg_h2d_us + avg_centroid_us + avg_select_us + avg_scan_topk_us + avg_d2h_us;

    std::cout << "\n========== IVF Full-GPU Online Result ==========\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "average recall: " << avg_recall << "\n";
    std::cout << "average latency per query (us): " << avg_latency_us << "\n";
    std::cout << "average H2D query per query (us): " << avg_h2d_us << "\n";
    std::cout << "average centroid score kernel per query (us): " << avg_centroid_us << "\n";
    std::cout << "average GPU top-nprobe kernel per query (us): " << avg_select_us << "\n";
    std::cout << "average IVF scan + GPU top-k kernel per query (us): " << avg_scan_topk_us << "\n";
    std::cout << "average D2H top-k only per query (us): " << avg_d2h_us << "\n";
    std::cout << "total queries: " << processed_queries << "\n";
    std::cout << "batch size: " << batch_size << "\n";
    std::cout << "nlist: " << nlist << "\n";
    std::cout << "nprobe: " << nprobe << "\n";
    std::cout << "k: " << k << "\n";
    std::cout << "================================================\n";

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_centroid_scores));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_inv_ids));
    CUDA_CHECK(cudaFree(d_selected_clusters));
    CUDA_CHECK(cudaFree(d_topk_ids));
    CUDA_CHECK(cudaFree(d_topk_scores));

    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaEventDestroy(ev2));

    return 0;
}