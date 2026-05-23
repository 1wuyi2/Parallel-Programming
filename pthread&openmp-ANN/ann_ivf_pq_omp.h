// ann_ivf_pq_omp.h
#pragma once
#include <omp.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include "ivf_pq_utils.h"   
#include "ann_simd.h"       


inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq_search_omp(const float* base, const float* query,
                  IVFPQIndex& idx, size_t base_number, size_t vecdim,
                  size_t k, size_t nprobe_clusters, size_t nprobe,
                  int num_threads = 0, size_t local_p = 10,
                  const std::string& schedule_type = "static") {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (local_p < k) local_p = k;

    // 1. 粗排
    std::vector<std::pair<float, uint32_t>> cluster_ips(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = inner_product_neon(query, idx.centroids.data() + c * vecdim, vecdim);
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

    // 2. 收集候选向量的索引
    std::vector<uint32_t> candidates;
    for (size_t c = 0; c < nprobe_clusters; ++c) {
        uint32_t cluster_id = cluster_ips[c].second;
        const auto& list = idx.inverted_lists[cluster_id];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    // 3. 构建 LUT
    std::vector<float> lut;
    build_lut(query, idx.pq, lut);

    // 4. 多线程查表累加
    size_t total = candidates.size();
    std::vector<std::vector<std::pair<float, uint32_t>>> local_results(num_threads);

    // 小顶堆比较器
    auto cmp = [](const std::pair<float, uint32_t>& a,
                  const std::pair<float, uint32_t>& b) {
        return a.first > b.first;
    };

    if (schedule_type == "static") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(static) nowait
            for (size_t j = 0; j < total; ++j) {
                uint32_t id = candidates[j];
                float sum = 0.0f;
                for (size_t m = 0; m < idx.pq.M; ++m)
                    sum += lut[m * idx.pq.Ks + idx.pq.codes[id * idx.pq.M + m]];

                if (heap.size() < local_p) {
                    heap.push_back({sum, id});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (sum > heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {sum, id};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
            local_results[tid] = std::move(heap);
        }
    } else if (schedule_type == "dynamic") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(dynamic, 1) nowait
            for (size_t j = 0; j < total; ++j) {
                uint32_t id = candidates[j];
                float sum = 0.0f;
                for (size_t m = 0; m < idx.pq.M; ++m)
                    sum += lut[m * idx.pq.Ks + idx.pq.codes[id * idx.pq.M + m]];

                if (heap.size() < local_p) {
                    heap.push_back({sum, id});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (sum > heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {sum, id};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
            local_results[tid] = std::move(heap);
        }
    } else if (schedule_type == "guided") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(guided, 1) nowait
            for (size_t j = 0; j < total; ++j) {
                uint32_t id = candidates[j];
                float sum = 0.0f;
                for (size_t m = 0; m < idx.pq.M; ++m)
                    sum += lut[m * idx.pq.Ks + idx.pq.codes[id * idx.pq.M + m]];

                if (heap.size() < local_p) {
                    heap.push_back({sum, id});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (sum > heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {sum, id};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
            local_results[tid] = std::move(heap);
        }
    } else {
        // fallback to static
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(static) nowait
            for (size_t j = 0; j < total; ++j) {
                uint32_t id = candidates[j];
                float sum = 0.0f;
                for (size_t m = 0; m < idx.pq.M; ++m)
                    sum += lut[m * idx.pq.Ks + idx.pq.codes[id * idx.pq.M + m]];

                if (heap.size() < local_p) {
                    heap.push_back({sum, id});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (sum > heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {sum, id};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
            local_results[tid] = std::move(heap);
        }
    }

    // 5. 合并局部 top‑p，选出全局 top‑nprobe 候选
    std::vector<std::pair<float, uint32_t>> merged;
    for (auto& heap : local_results)
        merged.insert(merged.end(), heap.begin(), heap.end());

    if (nprobe < merged.size()) {
        std::nth_element(merged.begin(), merged.begin() + nprobe - 1,
                         merged.end(),
                         [](const std::pair<float, uint32_t>& a,
                            const std::pair<float, uint32_t>& b) {
                             return a.first > b.first;
                         });
        merged.resize(nprobe);
    }

    // 6. 精排
    std::priority_queue<std::pair<float, uint32_t>> fine_pq;
    for (auto& cand : merged) {
        uint32_t id = cand.second;
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