#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "ann_mpi_graph_common.h"


struct MPIHNSWOnHNSWIndex {
    size_t num_parts = 0;
    std::vector<float> part_centroids;
    std::vector<size_t> part_counts;
    hnswlib::HierarchicalNSW<float> *top_hnsw = nullptr;
    std::vector<hnswlib::HierarchicalNSW<float> *> bottom_hnsw;
    std::vector<unsigned char> owns_part;

    void clear() {
        delete top_hnsw;
        top_hnsw = nullptr;
        for (auto *p : bottom_hnsw) {
            delete p;
        }
        bottom_hnsw.clear();
        owns_part.clear();
        part_centroids.clear();
        part_counts.clear();
    }
};

inline int mpi_h2h_part_of_id(uint32_t id,
                              size_t base_number,
                              size_t num_parts,
                              MPIGraphPartitionMode mode) {
    return mpi_graph_owner_of_id(id,
                                 base_number,
                                 static_cast<int>(num_parts),
                                 mode);
}

inline void mpi_h2h_compute_part_centroids(const float *base,
                                           size_t base_number,
                                           size_t vecdim,
                                           size_t num_parts,
                                           MPIGraphPartitionMode part_mode,
                                           std::vector<float> &centroids,
                                           std::vector<size_t> &counts) {
    centroids.assign(num_parts * vecdim, 0.0f);
    counts.assign(num_parts, 0);

    for (uint32_t id = 0; id < static_cast<uint32_t>(base_number); ++id) {
        int p = mpi_h2h_part_of_id(id,
                                   base_number,
                                   num_parts,
                                   part_mode);
        float *dst = centroids.data() + static_cast<size_t>(p) * vecdim;
        const float *src = base + static_cast<size_t>(id) * vecdim;
        for (size_t d = 0; d < vecdim; ++d) {
            dst[d] += src[d];
        }
        ++counts[static_cast<size_t>(p)];
    }

    for (size_t p = 0; p < num_parts; ++p) {
        float *c = centroids.data() + p * vecdim;
        if (counts[p] > 0) {
            for (size_t d = 0; d < vecdim; ++d) {
                c[d] /= static_cast<float>(counts[p]);
            }
        } else if (base_number > 0) {
            // Keep a valid vector for empty partition.
            size_t id = p % base_number;
            std::copy(base + id * vecdim,
                      base + id * vecdim + vecdim,
                      c);
        }
    }
}

inline hnswlib::HierarchicalNSW<float> *
mpi_h2h_build_top_hnsw(const std::vector<float> &centroids,
                       size_t num_parts,
                       size_t vecdim,
                       hnswlib::InnerProductSpace &top_space,
                       int top_M,
                       int top_efConstruction,
                       int top_efSearch) {
    if (num_parts == 0) {
        return nullptr;
    }

    hnswlib::HierarchicalNSW<float> *top =
        new hnswlib::HierarchicalNSW<float>(&top_space,
                                            num_parts,
                                            top_M,
                                            top_efConstruction);

    for (size_t p = 0; p < num_parts; ++p) {
        top->addPoint(centroids.data() + p * vecdim,
                      static_cast<hnswlib::labeltype>(p));
    }
    top->setEf(top_efSearch);
    return top;
}

inline MPIHNSWOnHNSWIndex
mpi_hnsw_on_hnsw_build(const float *base,
                       size_t base_number,
                       size_t vecdim,
                       size_t num_parts,
                       MPIGraphPartitionMode part_mode,
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

    MPIHNSWOnHNSWIndex out;
    out.num_parts = num_parts;

    mpi_h2h_compute_part_centroids(base,
                                   base_number,
                                   vecdim,
                                   num_parts,
                                   part_mode,
                                   out.part_centroids,
                                   out.part_counts);

    out.top_hnsw = mpi_h2h_build_top_hnsw(out.part_centroids,
                                          num_parts,
                                          vecdim,
                                          top_space,
                                          top_M,
                                          top_efConstruction,
                                          top_efSearch);

    std::vector<std::vector<uint32_t>> part_ids(num_parts);
    for (uint32_t id = 0; id < static_cast<uint32_t>(base_number); ++id) {
        int p = mpi_h2h_part_of_id(id,
                                   base_number,
                                   num_parts,
                                   part_mode);
        int owner_rank = p % world_size;
        if (owner_rank == rank) {
            part_ids[static_cast<size_t>(p)].push_back(id);
        }
    }

    out.bottom_hnsw.assign(num_parts, nullptr);
    out.owns_part.assign(num_parts, 0);

    size_t local_part_count = 0;
    size_t local_point_count = 0;

    for (size_t p = 0; p < num_parts; ++p) {
        int owner_rank = static_cast<int>(p % static_cast<size_t>(world_size));
        if (owner_rank != rank) {
            continue;
        }

        out.owns_part[p] = 1;
        if (part_ids[p].empty()) {
            continue;
        }

        ++local_part_count;
        local_point_count += part_ids[p].size();

        out.bottom_hnsw[p] = mpi_graph_build_hnsw_for_ids(base,
                                                          vecdim,
                                                          part_ids[p],
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
        std::cerr << "HNSW-on-HNSW part mode: " << mpi_graph_partition_name(part_mode) << "\n";
        std::cerr << "num_parts: " << num_parts << "\n";
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
mpi_h2h_select_parts(MPIHNSWOnHNSWIndex &index,
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
mpi_hnsw_on_hnsw_search(MPIHNSWOnHNSWIndex &index,
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
        selected = mpi_h2h_select_parts(index,
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
