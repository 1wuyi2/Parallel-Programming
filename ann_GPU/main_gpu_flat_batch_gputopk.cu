#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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
        }                                                                      \
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

/*
    Score layout:
        scores[query_id * base_number + base_id]

    Matrix form:
        Base:        base_number x vecdim
        QueryBatch:  batch_size  x vecdim
        Scores:      batch_size  x base_number

    Each CUDA block computes one TILE x TILE tile:
        x dimension -> base_id
        y dimension -> query_id
*/
template <int TILE>
__global__ void flat_batch_matmul_kernel(
    const float* __restrict__ base,
    const float* __restrict__ query_batch,
    float* __restrict__ scores,
    int base_number,
    int batch_size,
    int vecdim
) {
    __shared__ float s_base[TILE][TILE];   // [k][base]
    __shared__ float s_query[TILE][TILE];  // [query][k]

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_id = blockIdx.x * TILE + tx;
    int query_id = blockIdx.y * TILE + ty;

    float sum = 0.0f;

    for (int t = 0; t < vecdim; t += TILE) {
        int k_base = t + ty;
        int k_query = t + tx;

        if (base_id < base_number && k_base < vecdim) {
            s_base[ty][tx] = base[static_cast<size_t>(base_id) * vecdim + k_base];
        } else {
            s_base[ty][tx] = 0.0f;
        }

        if (query_id < batch_size && k_query < vecdim) {
            s_query[ty][tx] = query_batch[static_cast<size_t>(query_id) * vecdim + k_query];
        } else {
            s_query[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            sum += s_query[ty][kk] * s_base[kk][tx];
        }

        __syncthreads();
    }

    if (query_id < batch_size && base_id < base_number) {
        scores[static_cast<size_t>(query_id) * base_number + base_id] = sum;
    }
}

/*
    One CUDA block handles one query.

    For each query:
        1. Each thread scans part of base scores.
        2. Each thread keeps local top-k.
        3. Thread-local top-k results are stored in shared memory.
        4. thread 0 merges all thread-local top-k into final query top-k.

    This is simple and reliable for k <= 32.
*/
template <int MAX_K>
__global__ void topk_per_query_kernel(
    const float* __restrict__ scores,
    uint32_t* __restrict__ topk_ids,
    float* __restrict__ topk_scores,
    int base_number,
    int batch_size,
    int k
) {
    int qid = blockIdx.x;
    int tid = threadIdx.x;

    if (qid >= batch_size || k > MAX_K) return;

    float local_scores[MAX_K];
    uint32_t local_ids[MAX_K];

    for (int i = 0; i < MAX_K; ++i) {
        local_scores[i] = -1.0e30f;
        local_ids[i] = 0xffffffffu;
    }

    const float* q_scores = scores + static_cast<size_t>(qid) * base_number;

    for (int base_id = tid; base_id < base_number; base_id += blockDim.x) {
        float s = q_scores[base_id];

        int min_pos = 0;
        float min_val = local_scores[0];

        for (int i = 1; i < k; ++i) {
            if (local_scores[i] < min_val) {
                min_val = local_scores[i];
                min_pos = i;
            }
        }

        if (s > min_val) {
            local_scores[min_pos] = s;
            local_ids[min_pos] = static_cast<uint32_t>(base_id);
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

        // Sort final top-k in descending score order.
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
            topk_scores[qid * k + i] = final_scores[i];
            topk_ids[qid * k + i] = final_ids[i];
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
    if (argc < 5) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " <data_path> <test_number> <batch_size> <k>\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " ./data/ 1000 8 10\n";
        return 1;
    }

    std::string data_path = argv[1];
    int test_number = std::atoi(argv[2]);
    int batch_size = std::atoi(argv[3]);
    int k = std::atoi(argv[4]);

    if (test_number <= 0 || batch_size <= 0 || k <= 0) {
        std::cerr << "Invalid arguments.\n";
        return 1;
    }

    constexpr int MAX_K = 32;
    if (k > MAX_K) {
        std::cerr << "This simple GPU top-k kernel supports k <= "
                  << MAX_K << ". Current k = " << k << "\n";
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

    if (gt_n < query_n && gt_n < static_cast<uint32_t>(test_number)) {
        std::cerr << "GT query number is smaller than requested test_number.\n";
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
    std::cout << "test_number: " << test_number
              << ", batch_size: " << batch_size
              << ", k: " << k << "\n";

    float* d_base = nullptr;
    float* d_query_batch = nullptr;
    float* d_scores = nullptr;
    uint32_t* d_topk_ids = nullptr;
    float* d_topk_scores = nullptr;

    size_t base_bytes = static_cast<size_t>(base_number) * vecdim * sizeof(float);
    size_t query_batch_bytes = static_cast<size_t>(batch_size) * vecdim * sizeof(float);
    size_t scores_bytes = static_cast<size_t>(batch_size) * base_number * sizeof(float);
    size_t topk_ids_bytes = static_cast<size_t>(batch_size) * k * sizeof(uint32_t);
    size_t topk_scores_bytes = static_cast<size_t>(batch_size) * k * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_base, base_bytes));
    CUDA_CHECK(cudaMalloc(&d_query_batch, query_batch_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, scores_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_ids, topk_ids_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk_scores, topk_scores_bytes));

    CUDA_CHECK(cudaMemcpy(d_base, base.data(), base_bytes, cudaMemcpyHostToDevice));

    std::vector<uint32_t> h_topk_ids(static_cast<size_t>(batch_size) * k);
    std::vector<float> h_topk_scores(static_cast<size_t>(batch_size) * k);

    cudaEvent_t ev1, ev2;
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2));

    double total_recall = 0.0;
    int processed_queries = 0;

    double total_h2d_us = 0.0;
    double total_score_kernel_us = 0.0;
    double total_topk_kernel_us = 0.0;
    double total_d2h_us = 0.0;

    constexpr int TILE = 16;

    for (int batch_start = 0; batch_start < test_number; batch_start += batch_size) {
        int current_batch_size = std::min(batch_size, test_number - batch_start);

        const float* h_query_ptr = query.data() + static_cast<size_t>(batch_start) * vecdim;
        size_t current_query_bytes =
            static_cast<size_t>(current_batch_size) * vecdim * sizeof(float);

        size_t current_topk_ids_bytes =
            static_cast<size_t>(current_batch_size) * k * sizeof(uint32_t);

        size_t current_topk_scores_bytes =
            static_cast<size_t>(current_batch_size) * k * sizeof(float);

        // H2D query batch
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaMemcpy(d_query_batch,
                              h_query_ptr,
                              current_query_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev2));
        CUDA_CHECK(cudaEventSynchronize(ev2));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        total_h2d_us += ms * 1000.0;

        // Score matrix kernel
        dim3 block(TILE, TILE);
        dim3 grid((base_number + TILE - 1) / TILE,
                  (current_batch_size + TILE - 1) / TILE);

        CUDA_CHECK(cudaEventRecord(ev1));
        flat_batch_matmul_kernel<TILE><<<grid, block>>>(
            d_base,
            d_query_batch,
            d_scores,
            base_number,
            current_batch_size,
            vecdim
        );
        CUDA_CHECK(cudaEventRecord(ev2));
        CUDA_CHECK(cudaEventSynchronize(ev2));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        total_score_kernel_us += ms * 1000.0;

        // GPU top-k
        int topk_threads = 256;
        int topk_blocks = current_batch_size;
        size_t topk_smem =
            static_cast<size_t>(topk_threads) * k * sizeof(float) +
            static_cast<size_t>(topk_threads) * k * sizeof(uint32_t);

        CUDA_CHECK(cudaEventRecord(ev1));
        topk_per_query_kernel<MAX_K><<<topk_blocks, topk_threads, topk_smem>>>(
            d_scores,
            d_topk_ids,
            d_topk_scores,
            base_number,
            current_batch_size,
            k
        );
        CUDA_CHECK(cudaEventRecord(ev2));
        CUDA_CHECK(cudaEventSynchronize(ev2));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventElapsedTime(&ms, ev1, ev2));
        total_topk_kernel_us += ms * 1000.0;

        // D2H only final top-k
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
    double avg_score_kernel_us = total_score_kernel_us / processed_queries;
    double avg_topk_kernel_us = total_topk_kernel_us / processed_queries;
    double avg_d2h_us = total_d2h_us / processed_queries;

    double avg_latency_us =
        avg_h2d_us + avg_score_kernel_us + avg_topk_kernel_us + avg_d2h_us;

    std::cout << "\n========== GPU Flat Batch + GPU TopK Result ==========\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "average recall: " << avg_recall << "\n";
    std::cout << "average latency per query (us): " << avg_latency_us << "\n";
    std::cout << "average H2D per query (us): " << avg_h2d_us << "\n";
    std::cout << "average score kernel per query (us): " << avg_score_kernel_us << "\n";
    std::cout << "average GPU top-k kernel per query (us): " << avg_topk_kernel_us << "\n";
    std::cout << "average D2H top-k only per query (us): " << avg_d2h_us << "\n";
    std::cout << "=====================================================\n";

    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaEventDestroy(ev2));

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_query_batch));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_topk_ids));
    CUDA_CHECK(cudaFree(d_topk_scores));

    return 0;
}