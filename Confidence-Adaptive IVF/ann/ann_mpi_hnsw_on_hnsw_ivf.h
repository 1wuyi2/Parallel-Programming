#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

#include "ann_mpi_graph_common.h"

struct MPIIVFHNSWOnHNSWIndex {
    size_t num_parts = 0;
    IVFPQIndex ivf_idx;
    hnswlib::HierarchicalNSW<float> *top_hnsw = nullptr;

    std::vector<hnswlib::HierarchicalNSW<float> *> bottom_hnsw;
    std::vector<unsigned char> owns_part;
    std::vector<size_t> part_counts;

    void clear() {
        delete top_hnsw;
        top_hnsw = nullptr;

        for (auto *p : bottom_hnsw) {
            delete p;
        }
        bottom_hnsw.clear();
        owns_part.clear();
        part_counts.clear();

        ivf_idx.centroids.clear();
        ivf_idx.inverted_lists.clear();
        num_parts = 0;
    }
};

inline hnswlib::HierarchicalNSW<float> *
mpi_ivf_h2h_build_top_hnsw(const IVFPQIndex &ivf_idx,
                           size_t vecdim,
                           hnswlib::InnerProductSpace &top_space,
                           int top_M,
                           int top_efConstruction,
                           int top_efSearch) {
    if (ivf_idx.nlist == 0 || ivf_idx.centroids.empty()) {
        return nullptr;
    }

    hnswlib::HierarchicalNSW<float> *top =
        new hnswlib::HierarchicalNSW<float>(&top_space,
                                            ivf_idx.nlist,
                                            top_M,
                                            top_efConstruction);

    for (size_t c = 0; c < ivf_idx.nlist; ++c) {
        top->addPoint(ivf_idx.centroids.data() + c * vecdim,
                      static_cast<hnswlib::labeltype>(c));
    }

    top->setEf(top_efSearch);
    return top;
}

inline MPIIVFHNSWOnHNSWIndex
mpi_ivf_hnsw_on_hnsw_build(const float *base,
                           size_t base_number,
                           size_t vecdim,
                           size_t num_parts,
                           int ivf_niter,
                           hnswlib::InnerProductSpace &bottom_space,
                           hnswlib::InnerProductSpace &top_space,
                           int bottom_M,
                           int bottom_efConstruction,
                           int bottom_efSearch,
                           int top_M,
                           int top_efConstruction,
                           int top_efSearch,
                           int root,
                           MPI_Comm comm) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    MPIIVFHNSWOnHNSWIndex out;
    out.num_parts = num_parts;
    if (rank == root) {
        out.ivf_idx = mpi_graph_build_ivf_index_only(base,
                                                     base_number,
                                                     vecdim,
                                                     num_parts,
                                                     ivf_niter);
    }

    mpi_graph_bcast_ivf_index(out.ivf_idx, root, comm);
    out.num_parts = out.ivf_idx.nlist;

    out.top_hnsw = mpi_ivf_h2h_build_top_hnsw(out.ivf_idx,
                                             vecdim,
                                             top_space,
                                             top_M,
                                             top_efConstruction,
                                             top_efSearch);

    out.bottom_hnsw.assign(out.num_parts, nullptr);
    out.owns_part.assign(out.num_parts, 0);
    out.part_counts.assign(out.num_parts, 0);

    size_t local_part_count = 0;
    size_t local_point_count = 0;

    for (size_t p = 0; p < out.num_parts; ++p) {
        out.part_counts[p] = out.ivf_idx.inverted_lists[p].size();

        int owner_rank = static_cast<int>(p % static_cast<size_t>(world_size));
        if (owner_rank != rank) {
            continue;
        }

        out.owns_part[p] = 1;
        const auto &ids = out.ivf_idx.inverted_lists[p];
        if (ids.empty()) {
            continue;
        }

        ++local_part_count;
        local_point_count += ids.size();

        out.bottom_hnsw[p] = mpi_graph_build_hnsw_for_ids(base,
                                                          vecdim,
                                                          ids,
                                                          bottom_space,
                                                          bottom_M,
                                                          bottom_efConstruction,
                                                          bottom_efSearch);
    }

    unsigned long long stat[2];
    stat[0] = static_cast<unsigned long long>(local_part_count);
    stat[1] = static_cast<unsigned long long>(local_point_count);

    std::vector<unsigned long long> all_stats;
    if (rank == root) {
        all_stats.resize(static_cast<size_t>(world_size) * 2);
    }

    MPI_Gather(stat,
               2,
               MPI_UNSIGNED_LONG_LONG,
               rank == root ? all_stats.data() : nullptr,
               2,
               MPI_UNSIGNED_LONG_LONG,
               root,
               comm);

    if (rank == root) {
        size_t non_empty = 0;
        size_t max_part = 0;
        size_t min_part = base_number;
        for (size_t p = 0; p < out.num_parts; ++p) {
            size_t sz = out.part_counts[p];
            if (sz > 0) {
                ++non_empty;
                max_part = std::max(max_part, sz);
                min_part = std::min(min_part, sz);
            }
        }

        std::cerr << "Optimized HNSW-on-HNSW: IVF/KMeans partitions\n";
        std::cerr << "num_parts: " << out.num_parts
                  << ", non_empty_parts: " << non_empty
                  << ", min_part_size: " << min_part
                  << ", max_part_size: " << max_part << "\n";
        std::cerr << "rank stats(part_count, point_count):";
        for (int r = 0; r < world_size; ++r) {
            std::cerr << " [" << all_stats[static_cast<size_t>(r) * 2]
                      << "," << all_stats[static_cast<size_t>(r) * 2 + 1]
                      << "]";
        }
        std::cerr << "\n";
    }

    return out;
}

inline std::vector<unsigned>
mpi_ivf_h2h_select_parts(MPIIVFHNSWOnHNSWIndex &index,
                         const float *query,
                         size_t top_probe_parts) {
    top_probe_parts = std::max<size_t>(1, std::min(top_probe_parts, index.num_parts));

    std::vector<unsigned> selected;
    selected.reserve(top_probe_parts);

    if (index.top_hnsw == nullptr) {
        return selected;
    }

    auto pq = index.top_hnsw->searchKnn(query, top_probe_parts);
    while (!pq.empty() && selected.size() < top_probe_parts) {
        auto p = pq.top();
        pq.pop();
        selected.push_back(static_cast<unsigned>(p.second));
    }

    return selected;
}

inline std::priority_queue<std::pair<float, uint32_t>>
mpi_ivf_hnsw_on_hnsw_search(MPIIVFHNSWOnHNSWIndex &index,
                            const float *query_on_root,
                            size_t vecdim,
                            size_t k,
                            size_t top_probe_parts,
                            size_t local_p,
                            MPIGraphLocalSearchMode local_search_mode,
                            int local_threads,
                            int root,
                            MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (local_p < k) {
        local_p = k;
    }

    std::vector<float> query(vecdim, 0.0f);
    if (rank == root && query_on_root != nullptr) {
        std::copy(query_on_root, query_on_root + vecdim, query.begin());
    }

    MPI_Bcast(query.data(),
              static_cast<int>(vecdim),
              MPI_FLOAT,
              root,
              comm);

    top_probe_parts = std::max<size_t>(1, std::min(top_probe_parts, index.num_parts));
    std::vector<unsigned> selected(top_probe_parts, 0);

    if (rank == root) {
        selected = mpi_ivf_h2h_select_parts(index,
                                            query.data(),
                                            top_probe_parts);
        if (selected.size() < top_probe_parts) {
            selected.resize(top_probe_parts, 0);
        }
    }

    MPI_Bcast(selected.data(),
              static_cast<int>(selected.size()),
              MPI_UNSIGNED,
              root,
              comm);

    std::vector<std::pair<float, uint32_t>> local_heap;
    local_heap.reserve(local_p + 1);

    for (unsigned part_u : selected) {
        size_t p = static_cast<size_t>(part_u);
        if (p >= index.bottom_hnsw.size()) {
            continue;
        }
        if (!index.owns_part[p]) {
            continue;
        }
        if (index.bottom_hnsw[p] == nullptr) {
            continue;
        }

        auto part_res = mpi_graph_search_one_hnsw(index.bottom_hnsw[p],
                                                  query.data(),
                                                  local_p,
                                                  local_search_mode,
                                                  local_threads);
        mpi_graph_merge_local_vector(local_heap, local_p, part_res);
    }

    return mpi_graph_gather_merge_topk(local_heap,
                                       local_p,
                                       k,
                                       root,
                                       comm);
}
