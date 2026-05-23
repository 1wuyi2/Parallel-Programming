// ann_flat_pthread.h
#pragma once
#include <pthread.h>
#include <vector>
#include <queue>
#include <algorithm>
#include "ann_simd.h"

struct FlatThreadArgs {
    int tid;
    size_t start, end;
    const float* base;
    const float* query;
    size_t vecdim;
    size_t local_p;                // 每个线程保留的候选数
    std::vector<std::pair<float, uint32_t>> local_heap; // 输出
};

void* flat_worker(void* arg) {
    auto* args = static_cast<FlatThreadArgs*>(arg);
    size_t local_p = args->local_p;
    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(local_p + 1);

    for (size_t i = args->start; i < args->end; ++i) {
        float ip = inner_product_neon(args->query, args->base + i * args->vecdim, args->vecdim);
        float dist = 1.0f - ip;

        if (heap.size() < local_p) {
            heap.push_back({dist, i});
            std::push_heap(heap.begin(), heap.end());
        } else if (dist < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {dist, i};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    args->local_heap = std::move(heap);
    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pthread(const float* base, const float* query,
                    size_t base_number, size_t vecdim,
                    size_t k, int num_threads = 4, size_t local_p = 10)
{
    if (num_threads <= 0) num_threads = 4;
    if (local_p < k) local_p = k;

    std::vector<pthread_t> threads(num_threads);
    std::vector<FlatThreadArgs> args(num_threads);
    size_t chunk = (base_number + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].tid = t;
        args[t].start = t * chunk;
        args[t].end   = std::min((t + 1) * chunk, base_number);
        args[t].base  = base;
        args[t].query = query;
        args[t].vecdim = vecdim;
        args[t].local_p = local_p;
        pthread_create(&threads[t], nullptr, flat_worker, &args[t]);
    }

    // 合并局部结果
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