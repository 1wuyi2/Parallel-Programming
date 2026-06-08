#pragma once

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "ann_mpi_ivf_common.h"

inline void mpi_push_max_score_top_p(std::vector<std::pair<float, uint32_t>> &heap,
                                     size_t limit,
                                     float score,
                                     uint32_t id) {
    auto cmp = [](const std::pair<float, uint32_t> &a,
                  const std::pair<float, uint32_t> &b) {
        return a.first > b.first;
    };

    if (heap.size() < limit) {
        heap.push_back({score, id});
        std::push_heap(heap.begin(), heap.end(), cmp);
    } else if (score > heap.front().first) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.back() = {score, id};
        std::push_heap(heap.begin(), heap.end(), cmp);
    }
}

inline std::vector<std::pair<float, uint32_t>>
mpi_ivf_pq_local_search_omp(const float *base,
                            const float *query,
                            const IVFPQIndex &local_idx,
                            const std::vector<unsigned> &selected_clusters,
                            size_t vecdim,
                            size_t k,
                            size_t nprobe,
                            size_t local_p,
                            int num_threads,
                            const std::string &schedule_type,
                            MPIIVFSearchStats *stats = nullptr) {
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    if (local_p < k) {
        local_p = k;
    }
    if (nprobe < k) {
        nprobe = k;
    }

    std::vector<uint32_t> candidates;
    size_t reserve_n = 0;

    for (unsigned cid : selected_clusters) {
        if (cid < local_idx.inverted_lists.size()) {
            reserve_n += local_idx.inverted_lists[cid].size();
        }
    }

    candidates.reserve(reserve_n);
    for (unsigned cid : selected_clusters) {
        if (cid >= local_idx.inverted_lists.size()) {
            continue;
        }
        const auto &list = local_idx.inverted_lists[cid];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    if (stats != nullptr) {
        stats->local_candidates = candidates.size();
    }

    double t0 = MPI_Wtime();

    std::vector<float> lut;
    build_lut(query, local_idx.pq, lut);

    size_t total = candidates.size();
    std::vector<std::vector<std::pair<float, uint32_t>>> per_thread(std::max(1, num_threads));

    if (num_threads <= 1 || total <= local_p * 2) {
        std::vector<std::pair<float, uint32_t>> heap;
        heap.reserve(local_p + 1);

        for (uint32_t id : candidates) {
            float sum = 0.0f;
            for (size_t m = 0; m < local_idx.pq.M; ++m) {
                uint8_t code = local_idx.pq.codes[static_cast<size_t>(id) * local_idx.pq.M + m];
                sum += lut[m * local_idx.pq.Ks + code];
            }
            mpi_push_max_score_top_p(heap, local_p, sum, id);
        }

        per_thread[0] = std::move(heap);
    } else if (schedule_type == "dynamic") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(dynamic, 1) nowait
            for (size_t j = 0; j < total; ++j) {
                uint32_t id = candidates[j];
                float sum = 0.0f;
                for (size_t m = 0; m < local_idx.pq.M; ++m) {
                    uint8_t code = local_idx.pq.codes[static_cast<size_t>(id) * local_idx.pq.M + m];
                    sum += lut[m * local_idx.pq.Ks + code];
                }
                mpi_push_max_score_top_p(heap, local_p, sum, id);
            }

            per_thread[tid] = std::move(heap);
        }
    } else if (schedule_type == "guided") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(guided, 1) nowait
            for (size_t j = 0; j < total; ++j) {
                uint32_t id = candidates[j];
                float sum = 0.0f;
                for (size_t m = 0; m < local_idx.pq.M; ++m) {
                    uint8_t code = local_idx.pq.codes[static_cast<size_t>(id) * local_idx.pq.M + m];
                    sum += lut[m * local_idx.pq.Ks + code];
                }
                mpi_push_max_score_top_p(heap, local_p, sum, id);
            }

            per_thread[tid] = std::move(heap);
        }
    } else {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(static) nowait
            for (size_t j = 0; j < total; ++j) {
                uint32_t id = candidates[j];
                float sum = 0.0f;
                for (size_t m = 0; m < local_idx.pq.M; ++m) {
                    uint8_t code = local_idx.pq.codes[static_cast<size_t>(id) * local_idx.pq.M + m];
                    sum += lut[m * local_idx.pq.Ks + code];
                }
                mpi_push_max_score_top_p(heap, local_p, sum, id);
            }

            per_thread[tid] = std::move(heap);
        }
    }

    std::vector<std::pair<float, uint32_t>> merged_scores;
    for (auto &heap : per_thread) {
        merged_scores.insert(merged_scores.end(), heap.begin(), heap.end());
    }

    size_t effective_nprobe = std::min(nprobe, merged_scores.size());
    if (effective_nprobe < merged_scores.size()) {
        std::nth_element(merged_scores.begin(),
                         merged_scores.begin() + effective_nprobe - 1,
                         merged_scores.end(),
                         [](const std::pair<float, uint32_t> &a,
                            const std::pair<float, uint32_t> &b) {
                             return a.first > b.first;
                         });
        merged_scores.resize(effective_nprobe);
    }

    std::vector<std::pair<float, uint32_t>> fine_heap;
    fine_heap.reserve(local_p + 1);

    for (auto &cand : merged_scores) {
        uint32_t id = cand.second;
        float ip = inner_product_neon(query,
                                      base + static_cast<size_t>(id) * vecdim,
                                      vecdim);
        float dist = 1.0f - ip;
        mpi_push_min_dist_top_p(fine_heap, local_p, dist, id);
    }

    double t1 = MPI_Wtime();
    if (stats != nullptr) {
        stats->local_search_us = (t1 - t0) * 1000.0 * 1000.0;
    }

    return fine_heap;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq_search_mpi_omp(const float *base,
                      const float *query_on_root,
                      const IVFPQIndex &local_idx,
                      size_t base_number,
                      size_t vecdim,
                      size_t k,
                      size_t nprobe_clusters,
                      size_t nprobe,
                      int num_threads,
                      size_t local_p,
                      const std::string &schedule_type,
                      int root,
                      MPI_Comm comm,
                      MPIIVFSearchStats *stats = nullptr) {
    (void)base_number;

    if (local_p < k) {
        local_p = k;
    }
    if (nprobe < k) {
        nprobe = k;
    }

    std::vector<float> query;
    std::vector<unsigned> selected_clusters;

    mpi_prepare_query_and_clusters(query_on_root,
                                   local_idx,
                                   vecdim,
                                   nprobe_clusters,
                                   root,
                                   comm,
                                   query,
                                   selected_clusters,
                                   stats);

    auto local_heap = mpi_ivf_pq_local_search_omp(base,
                                                  query.data(),
                                                  local_idx,
                                                  selected_clusters,
                                                  vecdim,
                                                  k,
                                                  nprobe,
                                                  local_p,
                                                  num_threads,
                                                  schedule_type,
                                                  stats);

    return mpi_gather_merge_dist_topk(local_heap,
                                      local_p,
                                      k,
                                      root,
                                      comm,
                                      stats);
}
