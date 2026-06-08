// hnsw_search_pthread.h
#pragma once
#include <pthread.h>
#include <queue>
#include "hnswlib/hnswlib/hnswlib.h"

struct HNSWThreadArgs {
    int tid;
    hnswlib::HierarchicalNSW<float>* appr_alg;
    const float* query;
    size_t k;
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
};

void* hnsw_worker(void* arg) {
    auto* a = static_cast<HNSWThreadArgs*>(arg);
    a->result = a->appr_alg->searchKnn(a->query, a->k);
    return nullptr;
}

inline std::priority_queue<std::pair<float, hnswlib::labeltype>>
hnsw_search_pthread(hnswlib::HierarchicalNSW<float>& appr_alg,
                    const float* query,
                    size_t k, int num_threads = 4) {
    using ResultPQ = std::priority_queue<std::pair<float, hnswlib::labeltype>>;

    std::vector<pthread_t> threads(num_threads);
    std::vector<HNSWThreadArgs> args(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        args[t].tid = t;
        args[t].appr_alg = &appr_alg;
        args[t].query = query;
        args[t].k = k;
        pthread_create(&threads[t], nullptr, hnsw_worker, &args[t]);
    }

    ResultPQ final_pq;
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
        auto& pq = args[t].result;
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