#pragma once
#include <arm_neon.h>
#include <cstddef>
#include <queue>

// 强制内联函数，减少函数调用开销
__attribute__((always_inline))
static inline float inner_product_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    // 对 96 维，每次处理 4 个，循环 24 次
    for (size_t i = 0; i < d; i += 4) {
        float32x4_t a = vld1q_f32(x + i);
        float32x4_t b = vld1q_f32(y + i);
        sum = vfmaq_f32(sum, a, b);   // FMA
    }
    // 水平归约
    float32x2_t low  = vget_low_f32(sum);
    float32x2_t high = vget_high_f32(sum);
    float32x2_t pair = vadd_f32(low, high);   // 64 位加法
    return vget_lane_f32(vpadd_f32(pair, pair), 0);
}

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_neon(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> pq;
    for (size_t i = 0; i < base_number; ++i) {
        float ip = inner_product_neon(query, base + i * vecdim, vecdim);
        float dist = 1.0f - ip;
        if (pq.size() < k) {
            pq.push({dist, i});
        } else if (dist < pq.top().first) {
            pq.push({dist, i});
            pq.pop();
        }
    }
    return pq;
}