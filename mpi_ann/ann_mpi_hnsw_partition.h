#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "ann_mpi_graph_common.h"

struct MPIPartitionHNSWIndex {
    std::vector<uint32_t> local_ids;
    hnswlib::HierarchicalNSW<float> *local_hnsw = nullptr;

    void clear() {
        delete local_hnsw;
        local_hnsw = nullptr;
        local_ids.clear();
    }
};

inline MPIPartitionHNSWIndex
mpi_partition_hnsw_build(const float *base,
                         size_t base_number,
                         size_t vecdim,
                         hnswlib::InnerProductSpace &ipspace,
                         int hnsw_M,
                         int hnsw_efConstruction,
                         int hnsw_efSearch,
                         MPIGraphPartitionMode partition_mode,
                         int root,
                         MPI_Comm comm) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    MPIPartitionHNSWIndex out;
    out.local_ids.reserve(base_number / std::max(1, world_size) + 16);

    for (uint32_t id = 0; id < static_cast<uint32_t>(base_number); ++id) {
        int owner = mpi_graph_owner_of_id(id,
                                          base_number,
                                          world_size,
                                          partition_mode);
        if (owner == rank) {
            out.local_ids.push_back(id);
        }
    }

    out.local_hnsw = mpi_graph_build_hnsw_for_ids(base,
                                                  vecdim,
                                                  out.local_ids,
                                                  ipspace,
                                                  hnsw_M,
                                                  hnsw_efConstruction,
                                                  hnsw_efSearch);

    unsigned long long local_count = static_cast<unsigned long long>(out.local_ids.size());
    std::vector<unsigned long long> all_counts;
    if (rank == root) {
        all_counts.resize(world_size);
    }

    MPI_Gather(&local_count,
               1,
               MPI_UNSIGNED_LONG_LONG,
               rank == root ? all_counts.data() : nullptr,
               1,
               MPI_UNSIGNED_LONG_LONG,
               root,
               comm);

    if (rank == root) {
        std::cerr << "Partition-HNSW mode: " << mpi_graph_partition_name(partition_mode) << "\n";
        std::cerr << "local point counts:";
        for (auto x : all_counts) {
            std::cerr << " " << x;
        }
        std::cerr << "\n";
    }

    return out;
}

inline std::priority_queue<std::pair<float, uint32_t>>
mpi_partition_hnsw_search(MPIPartitionHNSWIndex &index,
                          const float *query_on_root,
                          size_t vecdim,
                          size_t k,
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

    auto local_res = mpi_graph_search_one_hnsw(index.local_hnsw,
                                               query.data(),
                                               local_p,
                                               local_search_mode,
                                               local_threads);

    std::vector<std::pair<float, uint32_t>> local_heap;
    local_heap.reserve(local_p + 1);
    mpi_graph_merge_local_vector(local_heap, local_p, local_res);

    return mpi_graph_gather_merge_topk(local_heap,
                                       local_p,
                                       k,
                                       root,
                                       comm);
}
