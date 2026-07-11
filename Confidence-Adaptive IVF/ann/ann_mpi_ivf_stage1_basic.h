#pragma once

#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "ann_mpi_ivf_common.h"

inline std::vector<std::pair<float, uint32_t>>
mpi_ivf_local_search_basic(const float *base,
                           const float *query,
                           const IVFPQIndex &local_idx,
                           const std::vector<unsigned> &selected_clusters,
                           size_t vecdim,
                           size_t local_p,
                           MPIIVFSearchStats *stats = nullptr) {
    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(local_p + 1);

    size_t local_candidates = 0;
    double t0 = MPI_Wtime();

    for (unsigned cid : selected_clusters) {
        if (cid >= local_idx.inverted_lists.size()) {
            continue;
        }

        const auto &list = local_idx.inverted_lists[cid];
        for (uint32_t id : list) {
            ++local_candidates;
            const float *v = base + static_cast<size_t>(id) * vecdim;
            float ip = inner_product_neon(query, v, vecdim);
            float dist = 1.0f - ip;
            mpi_push_min_dist_top_p(heap, local_p, dist, id);
        }
    }

    double t1 = MPI_Wtime();
    if (stats != nullptr) {
        stats->local_search_us = (t1 - t0) * 1000.0 * 1000.0;
        stats->local_candidates = local_candidates;
    }

    return heap;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_mpi_basic(const float *base,
                     const float *query_on_root,
                     const IVFPQIndex &local_idx,
                     size_t base_number,
                     size_t vecdim,
                     size_t k,
                     size_t nprobe_clusters,
                     size_t local_p,
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

    auto local_heap = mpi_ivf_local_search_basic(base,
                                                 query.data(),
                                                 local_idx,
                                                 selected_clusters,
                                                 vecdim,
                                                 local_p,
                                                 stats);

    return mpi_gather_merge_dist_topk(local_heap,
                                      local_p,
                                      k,
                                      root,
                                      comm,
                                      stats);
}
