#pragma once

#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "ivf_pq_utils.h"
#include "ann_simd.h"

enum class MPIIVFPartitionMode {
    CYCLIC,
    BLOCK
};

struct MPIIVFSearchStats {
    double query_bcast_us = 0.0;
    double cluster_bcast_us = 0.0;
    double local_search_us = 0.0;
    double gather_merge_us = 0.0;
    size_t local_candidates = 0;
};

inline MPIIVFPartitionMode parse_mpi_ivf_partition_mode(const std::string &s) {
    if (s == "block" || s == "BLOCK" || s == "Block") {
        return MPIIVFPartitionMode::BLOCK;
    }
    return MPIIVFPartitionMode::CYCLIC;
}

inline int mpi_ivf_owner_of_id(uint32_t id,
                               size_t base_number,
                               int world_size,
                               MPIIVFPartitionMode mode) {
    if (world_size <= 1) {
        return 0;
    }

    if (mode == MPIIVFPartitionMode::CYCLIC) {
        return static_cast<int>(id % static_cast<uint32_t>(world_size));
    }

    size_t q = base_number / static_cast<size_t>(world_size);
    size_t rem = base_number % static_cast<size_t>(world_size);
    size_t x = static_cast<size_t>(id);
    size_t first_big_part = (q + 1) * rem;

    if (x < first_big_part) {
        return static_cast<int>(x / (q + 1));
    }

    if (q == 0) {
        return static_cast<int>(rem);
    }

    return static_cast<int>(rem + (x - first_big_part) / q);
}

inline void mpi_bcast_ivfpq_index(IVFPQIndex &idx,
                                  int root,
                                  MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    unsigned long long meta[8];

    if (rank == root) {
        meta[0] = static_cast<unsigned long long>(idx.nlist);
        meta[1] = static_cast<unsigned long long>(idx.nprobe_clusters);
        meta[2] = static_cast<unsigned long long>(idx.centroids.size());
        meta[3] = static_cast<unsigned long long>(idx.pq.M);
        meta[4] = static_cast<unsigned long long>(idx.pq.Ks);
        meta[5] = static_cast<unsigned long long>(idx.pq.d_sub);
        meta[6] = static_cast<unsigned long long>(idx.pq.base_number);
        meta[7] = static_cast<unsigned long long>(idx.pq.vecdim);
    }

    MPI_Bcast(meta, 8, MPI_UNSIGNED_LONG_LONG, root, comm);

    if (rank != root) {
        idx.nlist = static_cast<size_t>(meta[0]);
        idx.nprobe_clusters = static_cast<size_t>(meta[1]);
        idx.centroids.resize(static_cast<size_t>(meta[2]));
        idx.inverted_lists.assign(idx.nlist, {});
        idx.pq.M = static_cast<size_t>(meta[3]);
        idx.pq.Ks = static_cast<size_t>(meta[4]);
        idx.pq.d_sub = static_cast<size_t>(meta[5]);
        idx.pq.base_number = static_cast<size_t>(meta[6]);
        idx.pq.vecdim = static_cast<size_t>(meta[7]);
    }

    if (!idx.centroids.empty()) {
        MPI_Bcast(idx.centroids.data(),
                  static_cast<int>(idx.centroids.size()),
                  MPI_FLOAT,
                  root,
                  comm);
    }

    std::vector<unsigned long long> list_sizes(idx.nlist, 0);
    unsigned long long total_ids = 0;
    std::vector<unsigned> flat_ids;

    if (rank == root) {
        for (size_t c = 0; c < idx.nlist; ++c) {
            list_sizes[c] = static_cast<unsigned long long>(idx.inverted_lists[c].size());
            total_ids += list_sizes[c];
        }

        flat_ids.reserve(static_cast<size_t>(total_ids));
        for (size_t c = 0; c < idx.nlist; ++c) {
            for (uint32_t id : idx.inverted_lists[c]) {
                flat_ids.push_back(static_cast<unsigned>(id));
            }
        }
    }

    MPI_Bcast(list_sizes.data(),
              static_cast<int>(idx.nlist),
              MPI_UNSIGNED_LONG_LONG,
              root,
              comm);

    if (rank != root) {
        total_ids = 0;
        for (auto x : list_sizes) {
            total_ids += x;
        }
        flat_ids.resize(static_cast<size_t>(total_ids));
    }

    if (total_ids > 0) {
        MPI_Bcast(flat_ids.data(),
                  static_cast<int>(total_ids),
                  MPI_UNSIGNED,
                  root,
                  comm);
    }

    if (rank != root) {
        size_t offset = 0;
        for (size_t c = 0; c < idx.nlist; ++c) {
            size_t len = static_cast<size_t>(list_sizes[c]);
            idx.inverted_lists[c].resize(len);
            for (size_t j = 0; j < len; ++j) {
                idx.inverted_lists[c][j] = static_cast<uint32_t>(flat_ids[offset + j]);
            }
            offset += len;
        }
    }

    unsigned long long codebook_size = 0;
    unsigned long long codes_size = 0;

    if (rank == root) {
        codebook_size = static_cast<unsigned long long>(idx.pq.codebook.size());
        codes_size = static_cast<unsigned long long>(idx.pq.codes.size());
    }

    MPI_Bcast(&codebook_size, 1, MPI_UNSIGNED_LONG_LONG, root, comm);
    MPI_Bcast(&codes_size, 1, MPI_UNSIGNED_LONG_LONG, root, comm);

    if (rank != root) {
        idx.pq.codebook.resize(static_cast<size_t>(codebook_size));
        idx.pq.codes.resize(static_cast<size_t>(codes_size));
    }

    if (codebook_size > 0) {
        MPI_Bcast(idx.pq.codebook.data(),
                  static_cast<int>(codebook_size),
                  MPI_FLOAT,
                  root,
                  comm);
    }

    if (codes_size > 0) {
        MPI_Bcast(idx.pq.codes.data(),
                  static_cast<int>(codes_size),
                  MPI_UNSIGNED_CHAR,
                  root,
                  comm);
    }
}

inline IVFPQIndex mpi_make_local_ivfpq_index(const IVFPQIndex &global_idx,
                                             size_t base_number,
                                             int rank,
                                             int world_size,
                                             MPIIVFPartitionMode mode) {
    IVFPQIndex local_idx;
    local_idx.nlist = global_idx.nlist;
    local_idx.nprobe_clusters = global_idx.nprobe_clusters;
    local_idx.centroids = global_idx.centroids;
    local_idx.pq = global_idx.pq;
    local_idx.inverted_lists.assign(global_idx.nlist, {});

    for (size_t c = 0; c < global_idx.nlist; ++c) {
        const auto &src = global_idx.inverted_lists[c];
        auto &dst = local_idx.inverted_lists[c];
        dst.reserve(src.size() / std::max(1, world_size) + 8);

        for (uint32_t id : src) {
            if (mpi_ivf_owner_of_id(id, base_number, world_size, mode) == rank) {
                dst.push_back(id);
            }
        }
    }

    return local_idx;
}

inline std::vector<unsigned> mpi_select_probe_clusters(const IVFPQIndex &idx,
                                                       const float *query,
                                                       size_t vecdim,
                                                       size_t nprobe_clusters) {
    nprobe_clusters = std::max<size_t>(1, std::min(nprobe_clusters, idx.nlist));
    std::vector<std::pair<float, unsigned>> cluster_ips(idx.nlist);

    for (size_t c = 0; c < idx.nlist; ++c) {
        const float *center = idx.centroids.data() + c * vecdim;
        float ip = inner_product_neon(query, center, vecdim);
        cluster_ips[c] = {ip, static_cast<unsigned>(c)};
    }

    if (nprobe_clusters < idx.nlist) {
        std::nth_element(cluster_ips.begin(),
                         cluster_ips.begin() + nprobe_clusters - 1,
                         cluster_ips.end(),
                         [](const std::pair<float, unsigned> &a,
                            const std::pair<float, unsigned> &b) {
                             return a.first > b.first;
                         });
    }

    std::sort(cluster_ips.begin(),
              cluster_ips.begin() + nprobe_clusters,
              [](const std::pair<float, unsigned> &a,
                 const std::pair<float, unsigned> &b) {
                  return a.first > b.first;
              });

    std::vector<unsigned> selected(nprobe_clusters);
    for (size_t i = 0; i < nprobe_clusters; ++i) {
        selected[i] = cluster_ips[i].second;
    }
    return selected;
}

inline void mpi_prepare_query_and_clusters(const float *query_on_root,
                                           const IVFPQIndex &local_idx,
                                           size_t vecdim,
                                           size_t nprobe_clusters,
                                           int root,
                                           MPI_Comm comm,
                                           std::vector<float> &query,
                                           std::vector<unsigned> &selected,
                                           MPIIVFSearchStats *stats = nullptr) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    query.assign(vecdim, 0.0f);
    if (rank == root && query_on_root != nullptr) {
        std::copy(query_on_root, query_on_root + vecdim, query.begin());
    }

    double t0 = MPI_Wtime();
    MPI_Bcast(query.data(), static_cast<int>(vecdim), MPI_FLOAT, root, comm);
    double t1 = MPI_Wtime();

    nprobe_clusters = std::max<size_t>(1, std::min(nprobe_clusters, local_idx.nlist));
    selected.assign(nprobe_clusters, 0);

    if (rank == root) {
        selected = mpi_select_probe_clusters(local_idx, query.data(), vecdim, nprobe_clusters);
    }

    double t2 = MPI_Wtime();
    MPI_Bcast(selected.data(), static_cast<int>(selected.size()), MPI_UNSIGNED, root, comm);
    double t3 = MPI_Wtime();

    if (stats != nullptr) {
        stats->query_bcast_us = (t1 - t0) * 1000.0 * 1000.0;
        stats->cluster_bcast_us = (t3 - t2) * 1000.0 * 1000.0;
    }
}

inline void mpi_push_min_dist_top_p(std::vector<std::pair<float, uint32_t>> &heap,
                                    size_t limit,
                                    float dist,
                                    uint32_t id) {
    if (heap.size() < limit) {
        heap.push_back({dist, id});
        std::push_heap(heap.begin(), heap.end());
    } else if (dist < heap.front().first) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = {dist, id};
        std::push_heap(heap.begin(), heap.end());
    }
}

inline std::priority_queue<std::pair<float, uint32_t>>
mpi_gather_merge_dist_topk(const std::vector<std::pair<float, uint32_t>> &local_heap,
                           size_t local_p,
                           size_t k,
                           int root,
                           MPI_Comm comm,
                           MPIIVFSearchStats *stats = nullptr) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    if (local_p < k) {
        local_p = k;
    }

    std::vector<float> send_dist(local_p, std::numeric_limits<float>::infinity());
    std::vector<unsigned> send_id(local_p, std::numeric_limits<unsigned>::max());

    for (size_t i = 0; i < local_heap.size() && i < local_p; ++i) {
        send_dist[i] = local_heap[i].first;
        send_id[i] = static_cast<unsigned>(local_heap[i].second);
    }

    std::vector<float> recv_dist;
    std::vector<unsigned> recv_id;

    if (rank == root) {
        recv_dist.resize(static_cast<size_t>(world_size) * local_p);
        recv_id.resize(static_cast<size_t>(world_size) * local_p);
    }

    double t0 = MPI_Wtime();
    MPI_Gather(send_dist.data(),
               static_cast<int>(local_p),
               MPI_FLOAT,
               rank == root ? recv_dist.data() : nullptr,
               static_cast<int>(local_p),
               MPI_FLOAT,
               root,
               comm);

    MPI_Gather(send_id.data(),
               static_cast<int>(local_p),
               MPI_UNSIGNED,
               rank == root ? recv_id.data() : nullptr,
               static_cast<int>(local_p),
               MPI_UNSIGNED,
               root,
               comm);
    double t1 = MPI_Wtime();

    if (stats != nullptr) {
        stats->gather_merge_us = (t1 - t0) * 1000.0 * 1000.0;
    }

    std::priority_queue<std::pair<float, uint32_t>> final_pq;

    if (rank == root) {
        const unsigned invalid_id = std::numeric_limits<unsigned>::max();
        const size_t total = static_cast<size_t>(world_size) * local_p;

        for (size_t i = 0; i < total; ++i) {
            if (recv_id[i] == invalid_id) {
                continue;
            }

            float dist = recv_dist[i];
            uint32_t id = static_cast<uint32_t>(recv_id[i]);

            if (final_pq.size() < k) {
                final_pq.push({dist, id});
            } else if (dist < final_pq.top().first) {
                final_pq.push({dist, id});
                final_pq.pop();
            }
        }
    }

    return final_pq;
}
