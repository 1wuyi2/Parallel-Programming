// ivf_pq_utils.h
#pragma once
#include "pq_utils.h"
#include <vector>
#include <cstdint>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>
#include <arm_neon.h>
#include "ann_simd.h"

struct IVFPQIndex {
    size_t nlist;                // 簇数，例如 1024
    size_t nprobe_clusters;      // 查询时探测的簇数，默认 10
    std::vector<float> centroids;   // 簇中心，长度 nlist * vecdim
    std::vector<std::vector<uint32_t>> inverted_lists; // 倒排列表
    PQIndex pq;                  // 复用 PQ 索引
};

// 构建 IVF‑PQ 索引
inline IVFPQIndex build_ivf_pq_index(const float* base, size_t base_number, size_t vecdim,
                                     size_t nlist, size_t M, size_t Ks, size_t d_sub,
                                     int niter = 10) {
    IVFPQIndex idx;
    idx.nlist = nlist;
    idx.nprobe_clusters = 10;

    // 1. 随机初始化簇中心
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> uni(0, base_number - 1);
    idx.centroids.resize(nlist * vecdim);
    for (size_t c = 0; c < nlist; ++c) {
        size_t ri = uni(rng);
        std::copy(base + ri * vecdim, base + ri * vecdim + vecdim,
                  idx.centroids.begin() + c * vecdim);
    }

    // 2. K‑Means 迭代
    std::vector<uint32_t> assignments(base_number);
    std::vector<float> new_centroids(nlist * vecdim);
    std::vector<size_t> counts(nlist);

    for (int iter = 0; iter < niter; ++iter) {
        // 分配阶段
        for (size_t i = 0; i < base_number; ++i) {
            float best_ip = -1e30f;
            uint32_t best_c = 0;
            for (size_t c = 0; c < nlist; ++c) {
                float ip = inner_product_neon(base + i * vecdim,
                                              idx.centroids.data() + c * vecdim,
                                              vecdim);
                if (ip > best_ip) {
                    best_ip = ip;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        // 更新阶段
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < base_number; ++i) {
            size_t c = assignments[i];
            const float* v = base + i * vecdim;
            float* nc = new_centroids.data() + c * vecdim;
            for (size_t d = 0; d < vecdim; ++d)
                nc[d] += v[d];
            ++counts[c];
        }
        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] > 0) {
                float* nc = new_centroids.data() + c * vecdim;
                for (size_t d = 0; d < vecdim; ++d)
                    nc[d] /= counts[c];
            }
        }
        idx.centroids.swap(new_centroids);
    }

    // 3. 构建倒排列表
    idx.inverted_lists.resize(nlist);
    for (size_t i = 0; i < base_number; ++i)
        idx.inverted_lists[assignments[i]].push_back(i);

    // 4. 构建全局 PQ 索引
    idx.pq = build_pq_index(base, base_number, vecdim, M, Ks, d_sub);

    return idx;
}

// IVF‑PQ 查询函数
inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq_search(const float* base, const float* query,
              IVFPQIndex& idx, size_t base_number, size_t vecdim,
              size_t k, size_t nprobe, size_t nprobe_clusters) {
    // 1. 选择最近的 nprobe_clusters 个簇
    std::vector<std::pair<float, uint32_t>> cluster_ips(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = inner_product_neon(query,
                                      idx.centroids.data() + c * vecdim,
                                      vecdim);
        cluster_ips[c] = {ip, (uint32_t)c};
    }
    if (nprobe_clusters < idx.nlist) {
        std::nth_element(cluster_ips.begin(),
                         cluster_ips.begin() + nprobe_clusters - 1,
                         cluster_ips.end(),
                         [](const std::pair<float, uint32_t>& a,
                            const std::pair<float, uint32_t>& b) {
                             return a.first > b.first;
                         });
    }
    std::sort(cluster_ips.begin(), cluster_ips.begin() + nprobe_clusters,
              [](const std::pair<float, uint32_t>& a,
                 const std::pair<float, uint32_t>& b) {
                  return a.first > b.first;
              });

    // 2. 收集这 nprobe_clusters 个簇内的基向量索引
    std::vector<uint32_t> candidates;
    for (size_t c = 0; c < nprobe_clusters; ++c) {
        uint32_t cluster_id = cluster_ips[c].second;
        const auto& list = idx.inverted_lists[cluster_id];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    // 3. 构建 LUT 并计算近似分数
    std::vector<float> lut;
    build_lut(query, idx.pq, lut);

    std::vector<std::pair<float, uint32_t>> scores;
    scores.reserve(candidates.size());
    for (uint32_t id : candidates) {
        float sum = 0.0f;
        for (size_t m = 0; m < idx.pq.M; ++m) {
            uint8_t code = idx.pq.codes[id * idx.pq.M + m];
            sum += lut[m * idx.pq.Ks + code];
        }
        scores.push_back({sum, id});
    }

    // 4. 粗排选出 top‑nprobe
    size_t effective_nprobe = std::min(nprobe, scores.size());
    if (effective_nprobe < scores.size()) {
        std::nth_element(scores.begin(),
                         scores.begin() + effective_nprobe - 1,
                         scores.end(),
                         [](const std::pair<float, uint32_t>& a,
                            const std::pair<float, uint32_t>& b) {
                             return a.first > b.first;
                         });
    }
    std::sort(scores.begin(), scores.begin() + effective_nprobe,
              [](const std::pair<float, uint32_t>& a,
                 const std::pair<float, uint32_t>& b) {
                  return a.first > b.first;
              });

    // 5. 精排
    std::priority_queue<std::pair<float, uint32_t>> fine_pq;
    for (size_t i = 0; i < effective_nprobe; ++i) {
        uint32_t id = scores[i].second;
        float ip = inner_product_neon(query, base + id * vecdim, vecdim);
        float dist = 1.0f - ip;
        if (fine_pq.size() < k) {
            fine_pq.push({dist, id});
        } else if (dist < fine_pq.top().first) {
            fine_pq.push({dist, id});
            fine_pq.pop();
        }
    }
    return fine_pq;
}