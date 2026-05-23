// pq_utils.h
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <arm_neon.h>
#include "ann_simd.h"

struct PQIndex {
    size_t M;
    size_t Ks;
    size_t d_sub;
    std::vector<float> codebook;
    std::vector<uint8_t> codes;
    size_t base_number;
    size_t vecdim;
};

inline PQIndex build_pq_index(const float* base, size_t base_number, size_t vecdim,
                               size_t M, size_t Ks, size_t d_sub) {
    PQIndex idx;
    idx.M = M;
    idx.Ks = Ks;
    idx.d_sub = d_sub;
    idx.base_number = base_number;
    idx.vecdim = vecdim;

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, base_number - 1);
    idx.codebook.resize(M * Ks * d_sub);
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < Ks; ++k) {
            size_t rand_idx = dist(rng);
            const float* src = base + rand_idx * vecdim + m * d_sub;
            float* dst = idx.codebook.data() + (m * Ks + k) * d_sub;
            std::copy(src, src + d_sub, dst);
        }
    }

    idx.codes.resize(base_number * M);
    for (size_t i = 0; i < base_number; ++i) {
        const float* vec = base + i * vecdim;
        for (size_t m = 0; m < M; ++m) {
            const float* sub = vec + m * d_sub;
            uint8_t best_k = 0;
            float best_ip = -1e30;
            size_t k = 0;
            // 每次处理 4 个中心
            for (; k + 3 < Ks; k += 4) {
                const float* cent0 = idx.codebook.data() + (m * Ks + k) * d_sub;
                const float* cent1 = cent0 + d_sub;
                const float* cent2 = cent1 + d_sub;
                const float* cent3 = cent2 + d_sub;

                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);

                for (size_t j = 0; j < d_sub; j += 4) {
                    float32x4_t sub_v = vld1q_f32(sub + j);
                    sum0 = vfmaq_f32(sum0, sub_v, vld1q_f32(cent0 + j));
                    sum1 = vfmaq_f32(sum1, sub_v, vld1q_f32(cent1 + j));
                    sum2 = vfmaq_f32(sum2, sub_v, vld1q_f32(cent2 + j));
                    sum3 = vfmaq_f32(sum3, sub_v, vld1q_f32(cent3 + j));
                }

                float ip0 = vaddvq_f32(sum0);
                float ip1 = vaddvq_f32(sum1);
                float ip2 = vaddvq_f32(sum2);
                float ip3 = vaddvq_f32(sum3);

                if (ip0 > best_ip) { best_ip = ip0; best_k = (uint8_t)(k); }
                if (ip1 > best_ip) { best_ip = ip1; best_k = (uint8_t)(k+1); }
                if (ip2 > best_ip) { best_ip = ip2; best_k = (uint8_t)(k+2); }
                if (ip3 > best_ip) { best_ip = ip3; best_k = (uint8_t)(k+3); }
            }
            for (; k < Ks; ++k) {
                const float* cent = idx.codebook.data() + (m * Ks + k) * d_sub;
                float ip = 0.0f;
                for (size_t j = 0; j < d_sub; ++j)
                    ip += sub[j] * cent[j];
                if (ip > best_ip) { best_ip = ip; best_k = (uint8_t)k; }
            }
            idx.codes[i * M + m] = best_k;
                    }
                }
                return idx;
            }

inline void build_lut(const float* query, const PQIndex& idx,
                      std::vector<float>& lut) {
    lut.resize(idx.M * idx.Ks);
    for (size_t m = 0; m < idx.M; ++m) {
        const float* q_sub = query + m * idx.d_sub;
        for (size_t k = 0; k < idx.Ks; ++k) {
            const float* cent = idx.codebook.data() + (m * idx.Ks + k) * idx.d_sub;
            lut[m * idx.Ks + k] = inner_product_neon(q_sub, cent, idx.d_sub);
        }
    }
}

inline void compute_approximate_scores(const PQIndex& idx,
                                       const std::vector<float>& lut,
                                       std::vector<std::pair<float, uint32_t>>& scores) {
    scores.resize(idx.base_number);
    const size_t M = idx.M;
    const size_t Ks = idx.Ks;
    size_t i = 0;

    // 每次处理 4 个基向量，循环展开减少开销
    for (; i + 3 < idx.base_number; i += 4) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        const uint8_t* c = idx.codes.data() + i * M;

        for (size_t m = 0; m < M; ++m) {
            s0 += lut[m*Ks + c[0*M + m]];
            s1 += lut[m*Ks + c[1*M + m]];
            s2 += lut[m*Ks + c[2*M + m]];
            s3 += lut[m*Ks + c[3*M + m]];
        }
        scores[i]   = {s0, i};
        scores[i+1] = {s1, i+1};
        scores[i+2] = {s2, i+2};
        scores[i+3] = {s3, i+3};
    }
    // 尾部
    for (; i < idx.base_number; ++i) {
        float sum = 0.0f;
        for (size_t m = 0; m < M; ++m) {
            uint8_t code = idx.codes[i * M + m];
            sum += lut[m * Ks + code];
        }
        scores[i] = {sum, i};
    }
}

inline std::priority_queue<std::pair<float, uint32_t>>
pq_search(const float* base, const float* query,
          const PQIndex& idx, size_t base_number, size_t vecdim,
          size_t k, size_t nprobe) {
    std::vector<float> lut;
    build_lut(query, idx, lut);

    std::vector<std::pair<float, uint32_t>> scores;
    compute_approximate_scores(idx, lut, scores);

    if (nprobe < base_number) {
        std::nth_element(scores.begin(),
                         scores.begin() + nprobe - 1,
                         scores.end(),
                         [](const std::pair<float, uint32_t>& a,
                            const std::pair<float, uint32_t>& b) {
                             return a.first > b.first;
                         });
    }

    std::priority_queue<std::pair<float, uint32_t>> fine_pq;
    size_t limit = std::min(nprobe, base_number);
    for (size_t i = 0; i < limit; ++i) {
        uint32_t idx_val = scores[i].second;
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