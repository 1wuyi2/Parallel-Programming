// ann_ivf_pq_pthread.h
#pragma once
#include <pthread.h>
#include <vector>
#include <queue>
#include <algorithm>
#include "ivf_pq_utils.h"
#include "ann_simd.h"

// 查表扫描线程参数
struct IVFPQScanArgs {
    int tid;
    size_t start, end;
    const std::vector<uint32_t>* candidates;
    const float* base;
    const float* query;
    size_t vecdim;
    IVFPQIndex* idx;
    const std::vector<float>* lut;
    size_t local_p;
    std::vector<std::pair<float, uint32_t>> local_heap;
};

void* ivf_pq_scan_worker(void* arg) {
    auto* a = static_cast<IVFPQScanArgs*>(arg);
    const auto& cand = *a->candidates;
    const auto& lut = *a->lut;
    const PQIndex& pq = a->idx->pq;
    size_t local_p = a->local_p;
    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(local_p + 1);

    // 小顶堆
    auto cmp = [](const std::pair<float, uint32_t>& x,
                  const std::pair<float, uint32_t>& y) {
        return x.first > y.first;
    };

    for (size_t j = a->start; j < a->end; ++j) {
        uint32_t id = cand[j];
        float sum = 0.0f;
        for (size_t m = 0; m < pq.M; ++m)
            sum += lut[m * pq.Ks + pq.codes[id * pq.M + m]];

        if (heap.size() < local_p) {
            heap.push_back({sum, id});
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (sum > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {sum, id};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    a->local_heap = std::move(heap);
    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq_search_pthread(const float* base, const float* query,
                      IVFPQIndex& idx, size_t base_number, size_t vecdim,
                      size_t k, size_t nprobe_clusters, size_t nprobe,
                      int num_threads = 4, size_t local_p = 10) {
    if (num_threads <= 0) num_threads = 4;
    if (local_p < k) local_p = k;

    // 1. 粗排（单线程）
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

    // 2. 收集候选索引
    std::vector<uint32_t> candidates;
    for (size_t c = 0; c < nprobe_clusters; ++c) {
        uint32_t cluster_id = cluster_ips[c].second;
        const auto& list = idx.inverted_lists[cluster_id];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    // 3. 构建 LUT（单线程）
    std::vector<float> lut;
    build_lut(query, idx.pq, lut);

    // 4. 多线程查表扫描
    size_t total = candidates.size();
    std::vector<pthread_t> threads(num_threads);
    std::vector<IVFPQScanArgs> args(num_threads);
    size_t chunk = (total + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].tid = t;
        args[t].start = t * chunk;
        args[t].end   = std::min((t + 1) * chunk, total);
        args[t].candidates = &candidates;
        args[t].base   = base;
        args[t].query  = query;
        args[t].vecdim = vecdim;
        args[t].idx    = &idx;
        args[t].lut    = &lut;
        args[t].local_p = local_p;
        pthread_create(&threads[t], nullptr, ivf_pq_scan_worker, &args[t]);
    }

    // 5. 收集并合并局部 top‑p
    std::vector<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
        auto& heap = args[t].local_heap;
        merged.insert(merged.end(), heap.begin(), heap.end());
    }

    if (nprobe < merged.size()) {
        std::nth_element(merged.begin(), merged.begin() + nprobe - 1,
                         merged.end(),
                         [](const std::pair<float, uint32_t>& a,
                            const std::pair<float, uint32_t>& b) {
                             return a.first > b.first;
                         });
        merged.resize(nprobe);
    }

    // 6. 精排（浮点内积）
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