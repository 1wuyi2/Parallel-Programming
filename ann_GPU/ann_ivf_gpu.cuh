#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <random>
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

struct IVFIndexGPU {
    size_t nlist = 0;
    std::vector<float> centroids;                    // nlist * vecdim
    std::vector<std::vector<uint32_t>> inverted_lists; // 每个 IVF 簇中的 base id
};

struct GpuSearchStats {
    size_t candidate_count = 0;
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
};

struct GpuIVFContext {
    float* d_base = nullptr;
    float* d_query = nullptr;
    uint32_t* d_cand_ids = nullptr;
    float* d_dists = nullptr;

    size_t base_number = 0;
    size_t vecdim = 0;
    size_t cand_capacity = 0;

    cudaEvent_t h2d_start{};
    cudaEvent_t h2d_stop{};
    cudaEvent_t kernel_start{};
    cudaEvent_t kernel_stop{};
    cudaEvent_t d2h_start{};
    cudaEvent_t d2h_stop{};
    bool events_created = false;
};

static inline float inner_product_cpu(const float* x, const float* y, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

inline IVFIndexGPU build_ivf_index_cpu(const float* base,
                                       size_t base_number,
                                       size_t vecdim,
                                       size_t nlist,
                                       int niter = 10) {
    if (base == nullptr || base_number == 0 || vecdim == 0 || nlist == 0) {
        throw std::runtime_error("invalid arguments for build_ivf_index_cpu");
    }
    nlist = std::min(nlist, base_number);

    IVFIndexGPU idx;
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
        std::cerr << "[build_ivf_index_cpu] kmeans iter " << (iter + 1)
                  << "/" << niter << std::endl;

        // 分配阶段：使用 inner product 最大的中心作为所属簇。
        for (size_t i = 0; i < base_number; ++i) {
            const float* v = base + i * vecdim;
            float best_ip = -1e30f;
            uint32_t best_c = 0;
            for (size_t c = 0; c < nlist; ++c) {
                float ip = inner_product_cpu(v, idx.centroids.data() + c * vecdim, vecdim);
                if (ip > best_ip) {
                    best_ip = ip;
                    best_c = static_cast<uint32_t>(c);
                }
            }
            assignments[i] = best_c;
        }

        // 更新阶段。
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < base_number; ++i) {
            size_t c = assignments[i];
            const float* v = base + i * vecdim;
            float* dst = new_centroids.data() + c * vecdim;
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
                // 空簇时重新随机初始化，避免中心全 0。
                size_t ri = uni(rng);
                std::copy(base + ri * vecdim,
                          base + ri * vecdim + vecdim,
                          dst);
            }
        }
        idx.centroids.swap(new_centroids);
    }

    idx.inverted_lists.assign(nlist, {});
    for (size_t i = 0; i < base_number; ++i) {
        idx.inverted_lists[assignments[i]].push_back(static_cast<uint32_t>(i));
    }

    size_t max_list = 0;
    size_t non_empty = 0;
    for (const auto& list : idx.inverted_lists) {
        max_list = std::max(max_list, list.size());
        if (!list.empty()) ++non_empty;
    }
    std::cerr << "[build_ivf_index_cpu] finished. non_empty_lists=" << non_empty
              << "/" << nlist << ", max_list_size=" << max_list << std::endl;

    return idx;
}

inline void gpu_ivf_init(GpuIVFContext& ctx,
                         const float* base,
                         size_t base_number,
                         size_t vecdim,
                         size_t initial_candidate_capacity = 1 << 20) {
    ctx.base_number = base_number;
    ctx.vecdim = vecdim;
    ctx.cand_capacity = std::max<size_t>(initial_candidate_capacity, 1);

    CUDA_CHECK(cudaMalloc(&ctx.d_base, base_number * vecdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_query, vecdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_cand_ids, ctx.cand_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_dists, ctx.cand_capacity * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(ctx.d_base,
                          base,
                          base_number * vecdim * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventCreate(&ctx.h2d_start));
    CUDA_CHECK(cudaEventCreate(&ctx.h2d_stop));
    CUDA_CHECK(cudaEventCreate(&ctx.kernel_start));
    CUDA_CHECK(cudaEventCreate(&ctx.kernel_stop));
    CUDA_CHECK(cudaEventCreate(&ctx.d2h_start));
    CUDA_CHECK(cudaEventCreate(&ctx.d2h_stop));
    ctx.events_created = true;
}

inline void gpu_ivf_reserve_candidates(GpuIVFContext& ctx, size_t required) {
    if (required <= ctx.cand_capacity) return;

    size_t new_capacity = ctx.cand_capacity;
    while (new_capacity < required) {
        new_capacity *= 2;
    }

    if (ctx.d_cand_ids != nullptr) CUDA_CHECK(cudaFree(ctx.d_cand_ids));
    if (ctx.d_dists != nullptr) CUDA_CHECK(cudaFree(ctx.d_dists));

    CUDA_CHECK(cudaMalloc(&ctx.d_cand_ids, new_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_dists, new_capacity * sizeof(float)));
    ctx.cand_capacity = new_capacity;
}

inline void gpu_ivf_free(GpuIVFContext& ctx) {
    if (ctx.d_base != nullptr) CUDA_CHECK(cudaFree(ctx.d_base));
    if (ctx.d_query != nullptr) CUDA_CHECK(cudaFree(ctx.d_query));
    if (ctx.d_cand_ids != nullptr) CUDA_CHECK(cudaFree(ctx.d_cand_ids));
    if (ctx.d_dists != nullptr) CUDA_CHECK(cudaFree(ctx.d_dists));

    ctx.d_base = nullptr;
    ctx.d_query = nullptr;
    ctx.d_cand_ids = nullptr;
    ctx.d_dists = nullptr;

    if (ctx.events_created) {
        CUDA_CHECK(cudaEventDestroy(ctx.h2d_start));
        CUDA_CHECK(cudaEventDestroy(ctx.h2d_stop));
        CUDA_CHECK(cudaEventDestroy(ctx.kernel_start));
        CUDA_CHECK(cudaEventDestroy(ctx.kernel_stop));
        CUDA_CHECK(cudaEventDestroy(ctx.d2h_start));
        CUDA_CHECK(cudaEventDestroy(ctx.d2h_stop));
        ctx.events_created = false;
    }
}

inline std::vector<uint32_t> collect_ivf_candidates_cpu(const float* query,
                                                        const IVFIndexGPU& idx,
                                                        size_t vecdim,
                                                        size_t nprobe_clusters) {
    nprobe_clusters = std::min(nprobe_clusters, idx.nlist);

    std::vector<std::pair<float, uint32_t>> cluster_ips(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = inner_product_cpu(query, idx.centroids.data() + c * vecdim, vecdim);
        cluster_ips[c] = {ip, static_cast<uint32_t>(c)};
    }

    if (nprobe_clusters < idx.nlist) {
        std::nth_element(cluster_ips.begin(),
                         cluster_ips.begin() + static_cast<long>(nprobe_clusters),
                         cluster_ips.end(),
                         [](const auto& a, const auto& b) {
                             return a.first > b.first;
                         });
    }
    std::sort(cluster_ips.begin(),
              cluster_ips.begin() + static_cast<long>(nprobe_clusters),
              [](const auto& a, const auto& b) {
                  return a.first > b.first;
              });

    std::vector<uint32_t> candidates;
    for (size_t i = 0; i < nprobe_clusters; ++i) {
        uint32_t cluster_id = cluster_ips[i].second;
        const auto& list = idx.inverted_lists[cluster_id];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }
    return candidates;
}

__global__ void ivf_exact_score_kernel(const float* __restrict__ base,
                                       const float* __restrict__ query,
                                       const uint32_t* __restrict__ cand_ids,
                                       float* __restrict__ dists,
                                       int cand_n,
                                       int vecdim) {
    int cand_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (cand_idx >= cand_n) return;

    uint32_t id = cand_ids[cand_idx];
    const float* vec = base + static_cast<size_t>(id) * vecdim;

    float local_sum = 0.0f;
    for (int d = tid; d < vecdim; d += blockDim.x) {
        local_sum += query[d] * vec[d];
    }

    extern __shared__ float shm[];
    shm[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dists[cand_idx] = 1.0f - shm[0];
    }
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_gpu(GpuIVFContext& ctx,
               const float* query,
               const IVFIndexGPU& idx,
               size_t k,
               size_t nprobe_clusters,
               GpuSearchStats* stats = nullptr,
               int block_threads = 128) {
    std::vector<uint32_t> candidates =
        collect_ivf_candidates_cpu(query, idx, ctx.vecdim, nprobe_clusters);

    std::priority_queue<std::pair<float, uint32_t>> final_pq;
    if (candidates.empty()) {
        if (stats) *stats = GpuSearchStats{};
        return final_pq;
    }

    gpu_ivf_reserve_candidates(ctx, candidates.size());

    GpuSearchStats local_stats;
    local_stats.candidate_count = candidates.size();

    CUDA_CHECK(cudaEventRecord(ctx.h2d_start));
    CUDA_CHECK(cudaMemcpy(ctx.d_query,
                          query,
                          ctx.vecdim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_cand_ids,
                          candidates.data(),
                          candidates.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ctx.h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(ctx.h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(&local_stats.h2d_ms,
                                    ctx.h2d_start,
                                    ctx.h2d_stop));

    int cand_n = static_cast<int>(candidates.size());
    int grid = cand_n;
    size_t shared_bytes = static_cast<size_t>(block_threads) * sizeof(float);

    CUDA_CHECK(cudaEventRecord(ctx.kernel_start));
    ivf_exact_score_kernel<<<grid, block_threads, shared_bytes>>>(
        ctx.d_base,
        ctx.d_query,
        ctx.d_cand_ids,
        ctx.d_dists,
        cand_n,
        static_cast<int>(ctx.vecdim));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ctx.kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(ctx.kernel_stop));
    CUDA_CHECK(cudaEventElapsedTime(&local_stats.kernel_ms,
                                    ctx.kernel_start,
                                    ctx.kernel_stop));

    std::vector<float> h_dists(candidates.size());
    CUDA_CHECK(cudaEventRecord(ctx.d2h_start));
    CUDA_CHECK(cudaMemcpy(h_dists.data(),
                          ctx.d_dists,
                          candidates.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ctx.d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(ctx.d2h_stop));
    CUDA_CHECK(cudaEventElapsedTime(&local_stats.d2h_ms,
                                    ctx.d2h_start,
                                    ctx.d2h_stop));

    for (size_t i = 0; i < candidates.size(); ++i) {
        std::pair<float, uint32_t> item{h_dists[i], candidates[i]};
        if (final_pq.size() < k) {
            final_pq.push(item);
        } else if (item.first < final_pq.top().first) {
            final_pq.push(item);
            final_pq.pop();
        }
    }

    if (stats) {
        *stats = local_stats;
    }
    return final_pq;
}
