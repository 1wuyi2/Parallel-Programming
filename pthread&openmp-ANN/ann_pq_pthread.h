// ann_pq_pthread.h
#pragma once
#include <pthread.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include "pq_utils.h"
#include "ann_simd.h"

// 编码线程参数
struct EncodeThreadArgs {
    int tid; size_t start, end; const float* base; size_t vecdim; PQIndex* idx;
};
void* encode_worker(void* arg) {
    auto* a = static_cast<EncodeThreadArgs*>(arg);
    const PQIndex& idx = *a->idx;
    for (size_t i = a->start; i < a->end; ++i) {
        const float* vec = a->base + i * a->vecdim;
        for (size_t m = 0; m < idx.M; ++m) {
            const float* sub = vec + m * idx.d_sub;
            float best_ip = -1e30f; uint8_t best_k = 0; size_t k = 0;
            for (; k + 3 < idx.Ks; k += 4) {
                const float* cent0 = idx.codebook.data() + (m * idx.Ks + k) * idx.d_sub;
                const float* cent1 = cent0 + idx.d_sub;
                const float* cent2 = cent1 + idx.d_sub;
                const float* cent3 = cent2 + idx.d_sub;
                float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f), sum3 = vdupq_n_f32(0.0f);
                for (size_t j = 0; j < idx.d_sub; j += 4) {
                    float32x4_t sub_v = vld1q_f32(sub + j);
                    sum0 = vfmaq_f32(sum0, sub_v, vld1q_f32(cent0 + j));
                    sum1 = vfmaq_f32(sum1, sub_v, vld1q_f32(cent1 + j));
                    sum2 = vfmaq_f32(sum2, sub_v, vld1q_f32(cent2 + j));
                    sum3 = vfmaq_f32(sum3, sub_v, vld1q_f32(cent3 + j));
                }
                float ip0 = vaddvq_f32(sum0), ip1 = vaddvq_f32(sum1);
                float ip2 = vaddvq_f32(sum2), ip3 = vaddvq_f32(sum3);
                if (ip0 > best_ip) { best_ip = ip0; best_k = (uint8_t)(k); }
                if (ip1 > best_ip) { best_ip = ip1; best_k = (uint8_t)(k+1); }
                if (ip2 > best_ip) { best_ip = ip2; best_k = (uint8_t)(k+2); }
                if (ip3 > best_ip) { best_ip = ip3; best_k = (uint8_t)(k+3); }
            }
            for (; k < idx.Ks; ++k) {
                const float* cent = idx.codebook.data() + (m * idx.Ks + k) * idx.d_sub;
                float ip = 0.0f;
                for (size_t j = 0; j < idx.d_sub; ++j) ip += sub[j] * cent[j];
                if (ip > best_ip) { best_ip = ip; best_k = (uint8_t)k; }
            }
            a->idx->codes[i * idx.M + m] = best_k;
        }
    }
    return nullptr;
}

// 多线程 PQ 索引构建
inline PQIndex build_pq_index_pthread(const float* base, size_t base_number, size_t vecdim,
                                      size_t M, size_t Ks, size_t d_sub, int num_threads = 4) {
    PQIndex idx;
    idx.M = M; idx.Ks = Ks; idx.d_sub = d_sub;
    idx.base_number = base_number; idx.vecdim = vecdim;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, base_number - 1);
    idx.codebook.resize(M * Ks * d_sub);
    for (size_t m = 0; m < M; ++m)
        for (size_t k = 0; k < Ks; ++k) {
            size_t rand_idx = dist(rng);
            const float* src = base + rand_idx * vecdim + m * d_sub;
            std::copy(src, src + d_sub, idx.codebook.data() + (m * Ks + k) * d_sub);
        }
    idx.codes.resize(base_number * M);
    if (num_threads <= 0) num_threads = 4;
    std::vector<pthread_t> threads(num_threads);
    std::vector<EncodeThreadArgs> args(num_threads);
    size_t chunk = (base_number + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        args[t].tid = t; args[t].start = t * chunk;
        args[t].end = std::min((t + 1) * chunk, base_number);
        args[t].base = base; args[t].vecdim = vecdim; args[t].idx = &idx;
        pthread_create(&threads[t], nullptr, encode_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(threads[t], nullptr);
    return idx;
}

// 查询阶段线程参数与函数
struct LutThreadArgs { int tid; const float* query; PQIndex* idx; std::vector<float>* lut; };
void* lut_worker(void* arg) {
    auto* a = static_cast<LutThreadArgs*>(arg);
    const PQIndex& idx = *a->idx;
    size_t m = a->tid;
    const float* q_sub = a->query + m * idx.d_sub;
    for (size_t k = 0; k < idx.Ks; ++k) {
        const float* cent = idx.codebook.data() + (m * idx.Ks + k) * idx.d_sub;
        (*a->lut)[m * idx.Ks + k] = inner_product_neon(q_sub, cent, idx.d_sub);
    }
    return nullptr;
}

struct ScanThreadArgs {
    int tid; size_t start, end; PQIndex* idx; const std::vector<float>* lut;
    size_t local_p; std::vector<std::pair<float, uint32_t>> local_heap;
};
void* scan_worker(void* arg) {
    auto* a = static_cast<ScanThreadArgs*>(arg);
    const PQIndex& idx = *a->idx; const auto& lut = *a->lut;
    size_t local_p = a->local_p;
    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(local_p + 1);

    // 小顶堆：堆顶是堆中最小内积，便于保留 top-p 最大的内积
    auto cmp = [](const std::pair<float, uint32_t>& x, const std::pair<float, uint32_t>& y) {
        return x.first > y.first;
    };

    for (size_t i = a->start; i < a->end; ++i) {
        float sum = 0.0f;
        for (size_t m = 0; m < idx.M; ++m)
            sum += lut[m * idx.Ks + idx.codes[i * idx.M + m]];

        if (heap.size() < local_p) {
            heap.push_back({sum, i});
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (sum > heap.front().first) {   // heap.front() 是最小内积
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {sum, i};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    a->local_heap = std::move(heap);
    return nullptr;
}

// 多线程 PQ 搜索
inline std::priority_queue<std::pair<float, uint32_t>>
pq_search_pthread(const float* base, const float* query,
                  PQIndex& idx, size_t base_number, size_t vecdim,
                  size_t k, size_t nprobe, int num_threads = 4, size_t local_p = 10) {
    if (num_threads <= 0) num_threads = 4;
    if (local_p < k) local_p = k;

    std::vector<float> lut(idx.M * idx.Ks);
    std::vector<pthread_t> lut_threads(idx.M);
    std::vector<LutThreadArgs> lut_args(idx.M);
    for (size_t m = 0; m < idx.M; ++m) {
        lut_args[m].tid = m; lut_args[m].query = query;
        lut_args[m].idx = &idx; lut_args[m].lut = &lut;
        pthread_create(&lut_threads[m], nullptr, lut_worker, &lut_args[m]);
    }
    for (auto& th : lut_threads) pthread_join(th, nullptr);

    std::vector<pthread_t> scan_threads(num_threads);
    std::vector<ScanThreadArgs> scan_args(num_threads);
    size_t chunk = (base_number + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        scan_args[t].tid = t; scan_args[t].start = t * chunk;
        scan_args[t].end = std::min((t + 1) * chunk, base_number);
        scan_args[t].idx = &idx; scan_args[t].lut = &lut;
        scan_args[t].local_p = local_p;
        pthread_create(&scan_threads[t], nullptr, scan_worker, &scan_args[t]);
    }

    std::vector<std::pair<float, uint32_t>> candidates;
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(scan_threads[t], nullptr);
        auto& heap = scan_args[t].local_heap;
        candidates.insert(candidates.end(), heap.begin(), heap.end());
    }
    if (nprobe < candidates.size()) {
        std::nth_element(candidates.begin(), candidates.begin() + nprobe - 1, candidates.end(),
            [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
                return a.first > b.first;
            });
        candidates.resize(nprobe);
    }

    std::priority_queue<std::pair<float, uint32_t>> fine_pq;
    for (auto& cand : candidates) {
        uint32_t id = cand.second;
        float ip = inner_product_neon(query, base + id * vecdim, vecdim);
        float dist = 1.0f - ip;
        if (fine_pq.size() < k) fine_pq.push({dist, id});
        else if (dist < fine_pq.top().first) { fine_pq.push({dist, id}); fine_pq.pop(); }
    }
    return fine_pq;
}