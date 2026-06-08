#pragma once
#include <omp.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <string>
#include "pq_utils.h"
#include "ann_simd.h"

// 多线程 PQ 索引构建
inline PQIndex build_pq_index_omp(const float* base, size_t base_number, size_t vecdim,
                                   size_t M, size_t Ks, size_t d_sub,
                                   int num_threads = 0) {
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
            float* dst = idx.codebook.data() + (m * Ks + k) * d_sub;
            std::copy(src, src + d_sub, dst);
        }

    idx.codes.resize(base_number * M);
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16)
    for (size_t i = 0; i < base_number; ++i) {
        const float* vec = base + i * vecdim;
        for (size_t m = 0; m < M; ++m) {
            const float* sub = vec + m * d_sub;
            float best_ip = -1e30f; uint8_t best_k = 0; size_t k = 0;
            for (; k + 3 < Ks; k += 4) {
                const float* cent0 = idx.codebook.data() + (m * Ks + k) * d_sub;
                const float* cent1 = cent0 + d_sub;
                const float* cent2 = cent1 + d_sub;
                const float* cent3 = cent2 + d_sub;
                float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f), sum3 = vdupq_n_f32(0.0f);
                for (size_t j = 0; j < d_sub; j += 4) {
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
            for (; k < Ks; ++k) {
                const float* cent = idx.codebook.data() + (m * Ks + k) * d_sub;
                float ip = 0.0f;
                for (size_t j = 0; j < d_sub; ++j) ip += sub[j] * cent[j];
                if (ip > best_ip) { best_ip = ip; best_k = (uint8_t)k; }
            }
            idx.codes[i * M + m] = best_k;
        }
    }
    return idx;
}

// 多线程 LUT 构建
inline void build_lut_omp(const float* query, const PQIndex& idx,
                          std::vector<float>& lut, int num_threads = 0) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    lut.resize(idx.M * idx.Ks);
    #pragma omp parallel for num_threads(num_threads)
    for (size_t m = 0; m < idx.M; ++m) {
        const float* q_sub = query + m * idx.d_sub;
        for (size_t k = 0; k < idx.Ks; ++k) {
            const float* cent = idx.codebook.data() + (m * idx.Ks + k) * idx.d_sub;
            lut[m * idx.Ks + k] = inner_product_neon(q_sub, cent, idx.d_sub);
        }
    }
}

// 多线程 PQ 搜索
inline std::priority_queue<std::pair<float, uint32_t>>
pq_search_omp(const float* base, const float* query,
              PQIndex& idx, size_t base_number, size_t vecdim,
              size_t k, size_t nprobe,
              int num_threads = 0, size_t local_p = 10,
              const std::string& schedule_type = "static", int chunk_size = 256) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (local_p < k) local_p = k;

    std::vector<float> lut;
    build_lut_omp(query, idx, lut, num_threads);

    std::vector<std::vector<std::pair<float, uint32_t>>> local_results(num_threads);
    auto cmp = [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
        return a.first > b.first;
    };

    if (schedule_type == "static") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < base_number; ++i) {
                float sum = 0.0f;
                for (size_t m = 0; m < idx.M; ++m)
                    sum += lut[m * idx.Ks + idx.codes[i * idx.M + m]];

                if (heap.size() < local_p) {
                    heap.push_back({sum, i});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (sum > heap.front().first) {   // heap.front() 是当前 top-p 中最小的内积
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {sum, i};
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

            #pragma omp for schedule(dynamic, chunk_size) nowait
            for (size_t i = 0; i < base_number; ++i) {
                float sum = 0.0f;
                for (size_t m = 0; m < idx.M; ++m)
                    sum += lut[m * idx.Ks + idx.codes[i * idx.M + m]];

                if (heap.size() < local_p) {
                    heap.push_back({sum, i});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (sum > heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {sum, i};
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

            #pragma omp for schedule(guided, chunk_size) nowait
            for (size_t i = 0; i < base_number; ++i) {
                float sum = 0.0f;
                for (size_t m = 0; m < idx.M; ++m)
                    sum += lut[m * idx.Ks + idx.codes[i * idx.M + m]];

                if (heap.size() < local_p) {
                    heap.push_back({sum, i});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (sum > heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {sum, i};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
            local_results[tid] = std::move(heap);
        }
    }

    // 合并局部 top‑p，选出全局 top‑nprobe 候选
    std::vector<std::pair<float, uint32_t>> candidates;
    for (auto& heap : local_results)
        candidates.insert(candidates.end(), heap.begin(), heap.end());

    if (nprobe < candidates.size()) {
        std::nth_element(candidates.begin(), candidates.begin() + nprobe - 1,
                         candidates.end(),
                         [](const std::pair<float, uint32_t>& a,
                            const std::pair<float, uint32_t>& b) {
                             return a.first > b.first;
                         });
        candidates.resize(nprobe);
    }

    // 精排
    std::priority_queue<std::pair<float, uint32_t>> fine_pq;
    for (auto& cand : candidates) {
        uint32_t idx_val = cand.second;
        float ip = inner_product_neon(query, base + idx_val * vecdim, vecdim);
        float dist = 1.0f - ip;
        if (fine_pq.size() < k) {
            fine_pq.push({dist, idx_val});
        } else if (dist < fine_pq.top().first) {
            fine_pq.push({dist, idx_val});
            fine_pq.pop();
        }
    }
    return fine_pq;
}