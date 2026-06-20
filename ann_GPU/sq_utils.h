// sq_utils.h
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <arm_neon.h>
#include <queue>
#include "ann_simd.h"

struct SQIndex {
    float scale;               
    std::vector<int8_t> codes; 
    size_t base_number;
    size_t vecdim;
};

// 构建 SQ 索引
inline SQIndex build_sq_index(const float* base, size_t base_number, size_t vecdim) {
    SQIndex idx;
    idx.base_number = base_number;
    idx.vecdim = vecdim;

    // 1. 找出所有维度绝对值的最大值
    float max_abs = 0.0f;
    for (size_t i = 0; i < base_number * vecdim; ++i) {
        float v = std::fabs(base[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs == 0.0f) max_abs = 1.0f;      
    idx.scale = 127.0f / max_abs;

    // 2. 量化所有基向量
    idx.codes.resize(base_number * vecdim);
    for (size_t i = 0; i < base_number * vecdim; ++i) {
        float scaled = base[i] * idx.scale;          
        int val = static_cast<int>(std::round(scaled));
        if (val < -128) val = -128;
        if (val > 127)  val = 127;
        idx.codes[i] = static_cast<int8_t>(val);
    }
    return idx;
}

// 量化一个查询向量
inline std::vector<int8_t> quantize_query(const float* query,
                                          float scale,
                                          size_t d) {
    std::vector<int8_t> qcode(d);
    for (size_t i = 0; i < d; ++i) {
        float scaled = query[i] * scale;
        int val = static_cast<int>(std::round(scaled));
        if (val < -128) val = -128;
        if (val > 127)  val = 127;
        qcode[i] = static_cast<int8_t>(val);
    }
    return qcode;
}


__attribute__((always_inline))
static inline int32_t sq_inner_product_neon(const int8_t* x,
                                            const int8_t* y,
                                            size_t d) {
    int32x4_t sum = vdupq_n_s32(0);
    for (size_t i = 0; i < d; i += 16) {
        int8x16_t vx = vld1q_s8(x + i);
        int8x16_t vy = vld1q_s8(y + i);
        // 扩展为 16 位并乘累加到 32 位累加器
        int16x8_t vx_low  = vmovl_s8(vget_low_s8(vx));
        int16x8_t vy_low  = vmovl_s8(vget_low_s8(vy));
        sum = vmlal_s16(sum, vget_low_s16(vx_low), vget_low_s16(vy_low));
        sum = vmlal_high_s16(sum, vx_low, vy_low);

        int16x8_t vx_high = vmovl_s8(vget_high_s8(vx));
        int16x8_t vy_high = vmovl_s8(vget_high_s8(vy));
        sum = vmlal_s16(sum, vget_low_s16(vx_high), vget_low_s16(vy_high));
        sum = vmlal_high_s16(sum, vx_high, vy_high);
    }
    // 水平归约
    int32x2_t low  = vget_low_s32(sum);
    int32x2_t high = vget_high_s32(sum);
    int32x2_t pair = vadd_s32(low, high);
    int32x2_t acc  = vpadd_s32(pair, pair);
    return vget_lane_s32(acc, 0);
}

// SQ 搜索函数
inline std::priority_queue<std::pair<float, uint32_t>>
sq_search(const float* base, const float* query,
          const SQIndex& idx, size_t base_number, size_t vecdim,
          size_t k, size_t nprobe) {
    // 量化查询向量
    auto qcode = quantize_query(query, idx.scale, vecdim);

    // 粗排
    std::vector<std::pair<int32_t, uint32_t>> scores(base_number);
    for (size_t i = 0; i < base_number; ++i) {
        int32_t score = sq_inner_product_neon(qcode.data(),
                                              idx.codes.data() + i * vecdim,
                                              vecdim);
        scores[i] = {score, i};
    }

    if (nprobe < base_number) {
        std::nth_element(scores.begin(),
                         scores.begin() + nprobe - 1,
                         scores.end(),
                         [](const std::pair<int32_t, uint32_t>& a,
                            const std::pair<int32_t, uint32_t>& b) {
                             return a.first > b.first;   // 降序排列，大分在前
                         });
    }

    // 精排：对候选向量使用原始浮点内积
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