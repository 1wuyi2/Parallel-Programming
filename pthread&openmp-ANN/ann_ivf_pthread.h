// ann_ivf_pthread.h
#pragma once
#include <pthread.h>
#include <vector>
#include <queue>
#include <algorithm>
#include "ivf_pq_utils.h"
#include "ann_simd.h"

struct IVFScanArgs {
    int tid;
    size_t start, end;
    const std::vector<uint32_t>* candidates;
    const float* base;
    const float* query;
    size_t vecdim;
    size_t local_p;
    std::vector<std::pair<float, uint32_t>> local_heap;
};

void* ivf_scan_worker(void* arg) {
    auto* a = static_cast<IVFScanArgs*>(arg);
    const auto& cand = *a->candidates;
    size_t local_p = a->local_p;
    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(local_p + 1);

    for (size_t j = a->start; j < a->end; ++j) {
        uint32_t id = cand[j];
        float ip = inner_product_neon(a->query, a->base + id * a->vecdim, a->vecdim);
        float dist = 1.0f - ip;

        if (heap.size() < local_p) {
            heap.push_back({dist, id});
            std::push_heap(heap.begin(), heap.end());
        } else if (dist < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {dist, id};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    a->local_heap = std::move(heap);
    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_pthread(const float* base, const float* query,
                   IVFPQIndex& idx, size_t base_number, size_t vecdim,
                   size_t k, size_t nprobe_clusters,
                   int num_threads = 4, size_t local_p = 10) {
    if (num_threads <= 0) num_threads = 4;
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
                            const std::pair<float, uint32_t>& b) { return a.first > b.first; });
    }
    std::sort(cluster_ips.begin(), cluster_ips.begin() + nprobe_clusters,
              [](const std::pair<float, uint32_t>& a,
                 const std::pair<float, uint32_t>& b) { return a.first > b.first; });

    // 2. 收集候选索引
    std::vector<uint32_t> candidates;
    for (size_t c = 0; c < nprobe_clusters; ++c) {
        uint32_t cluster_id = cluster_ips[c].second;
        const auto& list = idx.inverted_lists[cluster_id];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    // 3. 精排多线程
    size_t total = candidates.size();
    std::vector<pthread_t> threads(num_threads);
    std::vector<IVFScanArgs> args(num_threads);
    size_t chunk = (total + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].tid = t;
        args[t].start = t * chunk;
        args[t].end   = std::min((t + 1) * chunk, total);
        args[t].candidates = &candidates;
        args[t].base   = base;
        args[t].query  = query;
        args[t].vecdim = vecdim;
        args[t].local_p = local_p;
        pthread_create(&threads[t], nullptr, ivf_scan_worker, &args[t]);
    }

    // 4. 合并
    std::priority_queue<std::pair<float, uint32_t>> final_pq;
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
        for (auto& p : args[t].local_heap) {
            if (final_pq.size() < k) {
                final_pq.push(p);
            } else if (p.first < final_pq.top().first) {
                final_pq.push(p);
                final_pq.pop();
            }
        }
    }
    return final_pq;
}