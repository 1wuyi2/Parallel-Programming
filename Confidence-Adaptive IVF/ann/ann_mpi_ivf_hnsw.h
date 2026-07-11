#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "ann_mpi_graph_common.h"

struct MPIIVFHNSWIndex {
    IVFPQIndex ivf;
    std::vector<hnswlib::HierarchicalNSW<float> *> cluster_hnsw;
    std::vector<unsigned char> owns_cluster;

    void clear() {
        for (auto *p : cluster_hnsw) {
            delete p;
        }
        cluster_hnsw.clear();
        owns_cluster.clear();
    }
};

inline int mpi_ivf_hnsw_cluster_owner(size_t cid,
                                      size_t nlist,
                                      int world_size,
                                      MPIGraphPartitionMode mode) {
    return mpi_graph_owner_of_id(static_cast<uint32_t>(cid), nlist, world_size, mode);
}

inline MPIIVFHNSWIndex
mpi_ivf_hnsw_build(const float *base,
                   size_t base_number,
                   size_t vecdim,
                   size_t nlist,
                   int ivf_niter,
                   hnswlib::InnerProductSpace &ipspace,
                   int hnsw_M,
                   int hnsw_efConstruction,
                   int hnsw_efSearch,
                   MPIGraphPartitionMode cluster_partition_mode,
                   int root,
                   MPI_Comm comm) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    MPIIVFHNSWIndex out;

    if (rank == root) {
        out.ivf = mpi_graph_build_ivf_index_only(base,
                                                base_number,
                                                vecdim,
                                                nlist,
                                                ivf_niter);
    }

    mpi_graph_bcast_ivf_index(out.ivf, root, comm);

    out.cluster_hnsw.assign(out.ivf.nlist, nullptr);
    out.owns_cluster.assign(out.ivf.nlist, 0);

    size_t local_cluster_count = 0;
    size_t local_point_count = 0;

    for (size_t cid = 0; cid < out.ivf.nlist; ++cid) {
        int owner = mpi_ivf_hnsw_cluster_owner(cid,
                                               out.ivf.nlist,
                                               world_size,
                                               cluster_partition_mode);
        if (owner != rank) {
            continue;
        }

        out.owns_cluster[cid] = 1;
        const auto &ids = out.ivf.inverted_lists[cid];
        if (ids.empty()) {
            continue;
        }

        ++local_cluster_count;
        local_point_count += ids.size();

        out.cluster_hnsw[cid] = mpi_graph_build_hnsw_for_ids(base,
                                                             vecdim,
                                                             ids,
                                                             ipspace,
                                                             hnsw_M,
                                                             hnsw_efConstruction,
                                                             hnsw_efSearch);
    }

    unsigned long long stat[2];
    stat[0] = static_cast<unsigned long long>(local_cluster_count);
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
        std::cerr << "IVF+HNSW cluster partition mode: "
                  << mpi_graph_partition_name(cluster_partition_mode) << "\n";
        std::cerr << "rank stats(cluster_count, point_count):";
        for (int r = 0; r < world_size; ++r) {
            std::cerr << " [" << all_stats[static_cast<size_t>(r) * 2]
                      << "," << all_stats[static_cast<size_t>(r) * 2 + 1]
                      << "]";
        }
        std::cerr << "\n";
    }

    return out;
}

inline std::priority_queue<std::pair<float, uint32_t>>
mpi_ivf_hnsw_search(MPIIVFHNSWIndex &index,
                    const float *query_on_root,
                    size_t vecdim,
                    size_t k,
                    size_t nprobe_clusters,
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

    nprobe_clusters = std::max<size_t>(1, std::min(nprobe_clusters, index.ivf.nlist));
    std::vector<unsigned> selected(nprobe_clusters, 0);

    if (rank == root) {
        selected = mpi_graph_select_ivf_clusters(index.ivf,
                                                 query.data(),
                                                 vecdim,
                                                 nprobe_clusters);
    }

    MPI_Bcast(selected.data(),
              static_cast<int>(selected.size()),
              MPI_UNSIGNED,
              root,
              comm);

    std::vector<std::pair<float, uint32_t>> local_heap;
    local_heap.reserve(local_p + 1);

    for (unsigned cid_u : selected) {
        size_t cid = static_cast<size_t>(cid_u);
        if (cid >= index.cluster_hnsw.size()) {
            continue;
        }
        if (!index.owns_cluster[cid]) {
            continue;
        }
        if (index.cluster_hnsw[cid] == nullptr) {
            continue;
        }

        auto part_res = mpi_graph_search_one_hnsw(index.cluster_hnsw[cid],
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
