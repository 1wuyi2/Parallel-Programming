// ann_flat_omp.h
#pragma once
#include <omp.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include "ann_simd.h"

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_omp(const float* base, const float* query,
                size_t base_number, size_t vecdim,
                size_t k,
                int num_threads = 0,
                size_t local_p = 10,
                const std::string& schedule_type = "static",
                int chunk_size = 256)
{
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (local_p < k) local_p = k;

    std::vector<std::vector<std::pair<float, uint32_t>>> local_results(num_threads);

    // 根据 schedule_type 选择对应的调度策略
    if (schedule_type == "static") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < base_number; ++i) {
                float ip = inner_product_neon(query, base + i * vecdim, vecdim);
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
            local_results[tid] = std::move(heap);
        }
    }
    else if (schedule_type == "dynamic") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(dynamic, chunk_size) nowait
            for (size_t i = 0; i < base_number; ++i) {
                float ip = inner_product_neon(query, base + i * vecdim, vecdim);
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
            local_results[tid] = std::move(heap);
        }
    }
    else if (schedule_type == "guided") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(guided, chunk_size) nowait
            for (size_t i = 0; i < base_number; ++i) {
                float ip = inner_product_neon(query, base + i * vecdim, vecdim);
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
            local_results[tid] = std::move(heap);
        }
    }
    else {
        // 默认静态调度
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < base_number; ++i) {
                float ip = inner_product_neon(query, base + i * vecdim, vecdim);
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
            local_results[tid] = std::move(heap);
        }
    }

    // 合并局部结果
    std::priority_queue<std::pair<float, uint32_t>> final_pq;
    for (auto& heap : local_results) {
        for (auto& p : heap) {
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