#pragma once

#include <omp.h>

#include <cstdint>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "ann_mpi_ivf_common.h"

inline std::vector<std::pair<float, uint32_t>>
mpi_ivf_local_search_omp(const float *base,
                         const float *query,
                         const IVFPQIndex &local_idx,
                         const std::vector<unsigned> &selected_clusters,
                         size_t vecdim,
                         size_t local_p,
                         int num_threads,
                         const std::string &schedule_type,
                         MPIIVFSearchStats *stats = nullptr) {
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
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

    size_t total = candidates.size();
    double t0 = MPI_Wtime();

    if (num_threads <= 1 || total <= local_p * 2) {
        std::vector<std::pair<float, uint32_t>> heap;
        heap.reserve(local_p + 1);

        for (uint32_t id : candidates) {
            const float *v = base + static_cast<size_t>(id) * vecdim;
            float ip = inner_product_neon(query, v, vecdim);
            float dist = 1.0f - ip;
            mpi_push_min_dist_top_p(heap, local_p, dist, id);
        }

        double t1 = MPI_Wtime();
        if (stats != nullptr) {
            stats->local_search_us = (t1 - t0) * 1000.0 * 1000.0;
        }
        return heap;
    }

    std::vector<std::vector<std::pair<float, uint32_t>>> per_thread(num_threads);

    if (schedule_type == "dynamic") {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(local_p + 1);

            #pragma omp for schedule(dynamic, 1) nowait
            for (size_t j = 0; j < total; ++j) {
                uint32_t id = candidates[j];
                const float *v = base + static_cast<size_t>(id) * vecdim;
                float ip = inner_product_neon(query, v, vecdim);
                float dist = 1.0f - ip;
                mpi_push_min_dist_top_p(heap, local_p, dist, id);
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
                const float *v = base + static_cast<size_t>(id) * vecdim;
                float ip = inner_product_neon(query, v, vecdim);
                float dist = 1.0f - ip;
                mpi_push_min_dist_top_p(heap, local_p, dist, id);
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
                const float *v = base + static_cast<size_t>(id) * vecdim;
                float ip = inner_product_neon(query, v, vecdim);
                float dist = 1.0f - ip;
                mpi_push_min_dist_top_p(heap, local_p, dist, id);
            }

            per_thread[tid] = std::move(heap);
        }
    }

    std::vector<std::pair<float, uint32_t>> merged;
    merged.reserve(local_p + 1);

    for (auto &heap : per_thread) {
        for (auto &p : heap) {
            mpi_push_min_dist_top_p(merged, local_p, p.first, p.second);
        }
    }

    double t1 = MPI_Wtime();
    if (stats != nullptr) {
        stats->local_search_us = (t1 - t0) * 1000.0 * 1000.0;
    }

    return merged;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_mpi_omp(const float *base,
                   const float *query_on_root,
                   const IVFPQIndex &local_idx,
                   size_t base_number,
                   size_t vecdim,
                   size_t k,
                   size_t nprobe_clusters,
                   size_t local_p,
                   int num_threads,
                   const std::string &schedule_type,
                   int root,
                   MPI_Comm comm,
                   MPIIVFSearchStats *stats = nullptr) {
    (void)base_number;

    if (local_p < k) {
        local_p = k;
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

    auto local_heap = mpi_ivf_local_search_omp(base,
                                               query.data(),
                                               local_idx,
                                               selected_clusters,
                                               vecdim,
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
