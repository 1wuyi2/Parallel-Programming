// hnsw_search_omp.h
#pragma once
#include <omp.h>
#include <queue>
#include <random>
#include "hnswlib/hnswlib/hnswlib.h"

inline std::priority_queue<std::pair<float, hnswlib::labeltype>>
hnsw_search_omp(hnswlib::HierarchicalNSW<float>& appr_alg,
                const float* query,
                size_t k, int num_threads = 4) {
    using ResultPQ = std::priority_queue<std::pair<float, hnswlib::labeltype>>;

    // 每个线程的局部 top‑k
    std::vector<ResultPQ> local_results(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        // 注意：searchKnn 内部使用线程局部 visited list，多个线程同时读索引是安全的
        auto pq = appr_alg.searchKnn(query, k);
        local_results[tid] = std::move(pq);
    }

    // 合并所有线程的结果
    ResultPQ final_pq;
    for (auto& pq : local_results) {
        while (!pq.empty()) {
            auto p = pq.top();
            pq.pop();
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