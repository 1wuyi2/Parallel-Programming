#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib/hnswlib.h"
#include "ann_simd.h"
#include "ivf_pq_utils.h"
#include "hnsw_search_omp.h"
#include "hnsw_search_pthread.h"

enum class MPIGraphPartitionMode {
    BLOCK,
    CYCLIC,
    HASH
};

enum class MPIGraphLocalSearchMode {
    SINGLE,
    OPENMP,
    PTHREAD
};

inline const char *mpi_graph_partition_name(MPIGraphPartitionMode mode) {
    if (mode == MPIGraphPartitionMode::BLOCK) {
        return "BLOCK";
    }
    if (mode == MPIGraphPartitionMode::CYCLIC) {
        return "CYCLIC";
    }
    return "HASH";
}

inline const char *mpi_graph_local_search_name(MPIGraphLocalSearchMode mode) {
    if (mode == MPIGraphLocalSearchMode::OPENMP) {
        return "OPENMP";
    }
    if (mode == MPIGraphLocalSearchMode::PTHREAD) {
        return "PTHREAD";
    }
    return "SINGLE";
}

inline int mpi_graph_block_owner(size_t x, size_t n, int parts) {
    if (parts <= 1) {
        return 0;
    }
    size_t p = static_cast<size_t>(parts);
    size_t q = n / p;
    size_t rem = n % p;
    size_t first_big_part = (q + 1) * rem;

    if (x < first_big_part) {
        return static_cast<int>(x / (q + 1));
    }
    if (q == 0) {
        return static_cast<int>(rem);
    }
    return static_cast<int>(rem + (x - first_big_part) / q);
}

inline uint64_t mpi_graph_hash64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

inline int mpi_graph_owner_of_id(uint32_t id,
                                 size_t base_number,
                                 int parts,
                                 MPIGraphPartitionMode mode) {
    if (parts <= 1) {
        return 0;
    }
    if (mode == MPIGraphPartitionMode::CYCLIC) {
        return static_cast<int>(id % static_cast<uint32_t>(parts));
    }
    if (mode == MPIGraphPartitionMode::HASH) {
        return static_cast<int>(mpi_graph_hash64(id) % static_cast<uint64_t>(parts));
    }
    return mpi_graph_block_owner(static_cast<size_t>(id), base_number, parts);
}

// dist smaller is better. std::push_heap default gives largest dist at front.
inline void mpi_graph_push_top_p(std::vector<std::pair<float, uint32_t>> &heap,
                                 size_t limit,
                                 float dist,
                                 uint32_t id) {
    if (limit == 0) {
        return;
    }

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
mpi_graph_gather_merge_topk(const std::vector<std::pair<float, uint32_t>> &local_results,
                            size_t local_p,
                            size_t k,
                            int root,
                            MPI_Comm comm) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    if (local_p < k) {
        local_p = k;
    }

    std::vector<float> send_dist(local_p, std::numeric_limits<float>::infinity());
    std::vector<unsigned> send_id(local_p, std::numeric_limits<unsigned>::max());

    for (size_t i = 0; i < local_results.size() && i < local_p; ++i) {
        send_dist[i] = local_results[i].first;
        send_id[i] = static_cast<unsigned>(local_results[i].second);
    }

    std::vector<float> recv_dist;
    std::vector<unsigned> recv_id;
    if (rank == root) {
        recv_dist.resize(static_cast<size_t>(world_size) * local_p);
        recv_id.resize(static_cast<size_t>(world_size) * local_p);
    }

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

    std::priority_queue<std::pair<float, uint32_t>> final_pq;

    if (rank == root) {
        const unsigned invalid = std::numeric_limits<unsigned>::max();
        size_t total = static_cast<size_t>(world_size) * local_p;

        for (size_t i = 0; i < total; ++i) {
            if (recv_id[i] == invalid) {
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

inline std::vector<std::pair<float, uint32_t>>
mpi_graph_search_one_hnsw(hnswlib::HierarchicalNSW<float> *index,
                          const float *query,
                          size_t local_p,
                          MPIGraphLocalSearchMode mode,
                          int local_threads) {
    std::vector<std::pair<float, uint32_t>> out;
    if (index == nullptr || local_p == 0) {
        return out;
    }

    if (local_threads <= 1 || mode == MPIGraphLocalSearchMode::SINGLE) {
        auto pq = index->searchKnn(query, local_p);
        out.reserve(pq.size());
        while (!pq.empty()) {
            auto p = pq.top();
            pq.pop();
            out.push_back({p.first, static_cast<uint32_t>(p.second)});
        }
        return out;
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> pq;
    if (mode == MPIGraphLocalSearchMode::OPENMP) {
        pq = hnsw_search_omp(*index, query, local_p, local_threads);
    } else {
        pq = hnsw_search_pthread(*index, query, local_p, local_threads);
    }

    out.reserve(pq.size());
    while (!pq.empty()) {
        auto p = pq.top();
        pq.pop();
        out.push_back({p.first, static_cast<uint32_t>(p.second)});
    }
    return out;
}

inline void mpi_graph_merge_local_vector(std::vector<std::pair<float, uint32_t>> &dst_heap,
                                         size_t local_p,
                                         const std::vector<std::pair<float, uint32_t>> &src) {
    for (const auto &p : src) {
        mpi_graph_push_top_p(dst_heap, local_p, p.first, p.second);
    }
}

inline hnswlib::HierarchicalNSW<float> *
mpi_graph_build_hnsw_for_ids(const float *base,
                             size_t vecdim,
                             const std::vector<uint32_t> &ids,
                             hnswlib::InnerProductSpace &ipspace,
                             int M,
                             int efConstruction,
                             int efSearch) {
    if (ids.empty()) {
        return nullptr;
    }

    hnswlib::HierarchicalNSW<float> *index =
        new hnswlib::HierarchicalNSW<float>(&ipspace,
                                            ids.size(),
                                            M,
                                            efConstruction);

    for (uint32_t id : ids) {
        index->addPoint(base + static_cast<size_t>(id) * vecdim,
                        static_cast<hnswlib::labeltype>(id));
    }

    index->setEf(efSearch);
    return index;
}

// Lightweight IVF-only builder. It reuses IVFPQIndex as a container but does not build PQ.
inline IVFPQIndex mpi_graph_build_ivf_index_only(const float *base,
                                                 size_t base_number,
                                                 size_t vecdim,
                                                 size_t nlist,
                                                 int niter) {
    IVFPQIndex idx;
    idx.nlist = nlist;
    idx.nprobe_clusters = 10;
    idx.centroids.assign(nlist * vecdim, 0.0f);
    idx.inverted_lists.assign(nlist, {});

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> uni(0, base_number - 1);

    for (size_t c = 0; c < nlist; ++c) {
        size_t ri = uni(rng);
        std::copy(base + ri * vecdim,
                  base + ri * vecdim + vecdim,
                  idx.centroids.begin() + c * vecdim);
    }

    std::vector<uint32_t> assignments(base_number, 0);
    std::vector<float> new_centroids(nlist * vecdim, 0.0f);
    std::vector<size_t> counts(nlist, 0);

    for (int iter = 0; iter < niter; ++iter) {
        for (size_t i = 0; i < base_number; ++i) {
            float best_ip = -1e30f;
            uint32_t best_c = 0;
            for (size_t c = 0; c < nlist; ++c) {
                float ip = inner_product_neon(base + i * vecdim,
                                              idx.centroids.data() + c * vecdim,
                                              vecdim);
                if (ip > best_ip) {
                    best_ip = ip;
                    best_c = static_cast<uint32_t>(c);
                }
            }
            assignments[i] = best_c;
        }

        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < base_number; ++i) {
            size_t c = assignments[i];
            const float *v = base + i * vecdim;
            float *nc = new_centroids.data() + c * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                nc[d] += v[d];
            }
            ++counts[c];
        }

        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] > 0) {
                float *nc = new_centroids.data() + c * vecdim;
                for (size_t d = 0; d < vecdim; ++d) {
                    nc[d] /= static_cast<float>(counts[c]);
                }
            } else {
                // Keep a valid centroid for empty clusters.
                size_t ri = uni(rng);
                std::copy(base + ri * vecdim,
                          base + ri * vecdim + vecdim,
                          new_centroids.begin() + c * vecdim);
            }
        }
        idx.centroids.swap(new_centroids);
    }

    for (size_t i = 0; i < base_number; ++i) {
        idx.inverted_lists[assignments[i]].push_back(static_cast<uint32_t>(i));
    }

    return idx;
}

inline void mpi_graph_bcast_ivf_index(IVFPQIndex &idx,
                                      int root,
                                      MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    unsigned long long meta[3];
    if (rank == root) {
        meta[0] = static_cast<unsigned long long>(idx.nlist);
        meta[1] = static_cast<unsigned long long>(idx.nprobe_clusters);
        meta[2] = static_cast<unsigned long long>(idx.centroids.size());
    }

    MPI_Bcast(meta, 3, MPI_UNSIGNED_LONG_LONG, root, comm);

    if (rank != root) {
        idx.nlist = static_cast<size_t>(meta[0]);
        idx.nprobe_clusters = static_cast<size_t>(meta[1]);
        idx.centroids.resize(static_cast<size_t>(meta[2]));
        idx.inverted_lists.assign(idx.nlist, {});
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
}

inline std::vector<unsigned>
mpi_graph_select_ivf_clusters(const IVFPQIndex &idx,
                              const float *query,
                              size_t vecdim,
                              size_t nprobe_clusters) {
    nprobe_clusters = std::max<size_t>(1, std::min(nprobe_clusters, idx.nlist));

    std::vector<std::pair<float, unsigned>> cluster_ips(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = inner_product_neon(query,
                                      idx.centroids.data() + c * vecdim,
                                      vecdim);
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
