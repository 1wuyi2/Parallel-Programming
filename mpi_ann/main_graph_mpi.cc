#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <queue>
#include <algorithm>
#include <cstdint>

#include <omp.h>
#include <mpi.h>

#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"

// Existing project headers.
#include "ann_simd.h"
#include "sq_utils.h"
#include "pq_utils.h"
#include "ivf_pq_utils.h"
#include "ann_flat_omp.h"
#include "ann_flat_pthread.h"
#include "ann_pq_omp.h"
#include "ann_pq_pthread.h"
#include "ann_ivf_omp.h"
#include "ann_ivf_pthread.h"
#include "ann_ivf_pq_omp.h"
#include "ann_ivf_pq_pthread.h"
#include "hnsw_search_pthread.h"
#include "hnsw_search_omp.h"

// Graph-index MPI methods.
#include "ann_mpi_ivf_hnsw.h"
#include "ann_mpi_hnsw_partition.h"
#include "ann_mpi_hnsw_on_hnsw.h"
#include "ann_mpi_hnsw_on_hnsw_ivf.h"

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);

    if (!fin.good()) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0) {
            std::cerr << "cannot open " << data_path << "\n";
        }

        n = 0;
        d = 0;
        return nullptr;
    }

    uint32_t n32 = 0;
    uint32_t d32 = 0;

    fin.read(reinterpret_cast<char*>(&n32), 4);
    fin.read(reinterpret_cast<char*>(&d32), 4);

    n = static_cast<size_t>(n32);
    d = static_cast<size_t>(d32);

    T* data = new T[n * d];
    int sz = sizeof(T);

    for(size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char*>(data + i * d), d * sz);
    }

    fin.close();

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cerr << "load data " << data_path << "\n";
        std::cerr << "dimension: " << d
                  << "  number:" << n
                  << "  size_per_element:" << sizeof(T) << "\n";
    }

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // us
};

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); 

    // 1: IVF + HNSW
    // 2: Partition-HNSW
    // 3: HNSW-on-HNSW
    // 4: Optimized HNSW-on-HNSW with IVF/KMeans partitions
    const int GRAPH_METHOD = 2;

    const size_t TEST_NUMBER_LIMIT = 2000;
    const size_t k = 10;

    // Each rank returns local top-p, rank 0 merges global top-k.
    const size_t LOCAL_P = 20;

    // Local HNSW parameters.
    const int HNSW_M = 16;
    const int EF_CONSTRUCTION = 150;
    const int EF_SEARCH = 20;

    // Local HNSW query mode.
    // SINGLE: one searchKnn per local graph.
    // OPENMP/PTHREAD: reuse your hnsw_search_omp / hnsw_search_pthread.
    const MPIGraphLocalSearchMode LOCAL_SEARCH_MODE = MPIGraphLocalSearchMode::SINGLE;
    const int HNSW_LOCAL_THREADS = 1;

    // Partition modes: BLOCK / CYCLIC / HASH.
    const MPIGraphPartitionMode PARTITION_MODE = MPIGraphPartitionMode::BLOCK;

    // Method 1: IVF + HNSW parameters.
    // For HNSW experiments, nlist=256 is usually lighter than 1024.
    const size_t IVF_HNSW_NLIST = 256;
    const size_t IVF_HNSW_NPROBE_CLUSTERS = 20;
    const int IVF_HNSW_NITER = 10;

    // Method 3: HNSW-on-HNSW parameters.
    const size_t H2H_NUM_PARTS = 32;
    const size_t H2H_TOP_PROBE_PARTS = 8;
    const int H2H_TOP_M = 8;
    const int H2H_TOP_EF_CONSTRUCTION = 80;
    const int H2H_TOP_EF_SEARCH = 32;

    omp_set_num_threads(std::max(1, HNSW_LOCAL_THREADS));

     
    // Data loading.
     

    size_t test_number = 0;
    size_t base_number = 0;
    size_t test_gt_d = 0;
    size_t vecdim = 0;
    size_t query_dim = 0;
    size_t gt_number = 0;

    std::string data_path = "/anndata/";

    auto test_query =
        LoadData<float>(data_path + "DEEP100K.query.fbin",
                        test_number,
                        query_dim);

    auto test_gt =
        LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin",
                      gt_number,
                      test_gt_d);

    auto base =
        LoadData<float>(data_path + "DEEP100K.base.100k.fbin",
                        base_number,
                        vecdim);

    if (test_query == nullptr ||
        test_gt == nullptr ||
        base == nullptr ||
        query_dim != vecdim ||
        test_gt_d == 0) {

        if (rank == 0) {
            std::cerr << "data loading failed or dimension mismatch\n";
        }

        delete[] test_query;
        delete[] test_gt;
        delete[] base;
        MPI_Finalize();
        return 1;
    }

    test_number = std::min(TEST_NUMBER_LIMIT, test_number);
    test_number = std::min(test_number, gt_number);

    std::vector<SearchResult> results(test_number);

     
    // Build selected graph index.
    // Build time is printed to stderr, but not counted in per-query latency.
     

    InnerProductSpace ipspace(vecdim);
    InnerProductSpace top_ipspace(vecdim);

    MPIIVFHNSWIndex ivf_hnsw_index;
    MPIPartitionHNSWIndex partition_hnsw_index;
    MPIHNSWOnHNSWIndex hnsw_on_hnsw_index;
    MPIIVFHNSWOnHNSWIndex ivf_hnsw_on_hnsw_index;

    double build_t0 = MPI_Wtime();

    if (GRAPH_METHOD == 1) {
        if (rank == 0) {
            std::cerr << "GRAPH_METHOD=1: IVF + HNSW\n";
        }
        ivf_hnsw_index = mpi_ivf_hnsw_build(base,
                                            base_number,
                                            vecdim,
                                            IVF_HNSW_NLIST,
                                            IVF_HNSW_NITER,
                                            ipspace,
                                            HNSW_M,
                                            EF_CONSTRUCTION,
                                            EF_SEARCH,
                                            PARTITION_MODE,
                                            0,
                                            MPI_COMM_WORLD);
    } else if (GRAPH_METHOD == 2) {
        if (rank == 0) {
            std::cerr << "GRAPH_METHOD=2: Partition-HNSW\n";
        }
        partition_hnsw_index = mpi_partition_hnsw_build(base,
                                                        base_number,
                                                        vecdim,
                                                        ipspace,
                                                        HNSW_M,
                                                        EF_CONSTRUCTION,
                                                        EF_SEARCH,
                                                        PARTITION_MODE,
                                                        0,
                                                        MPI_COMM_WORLD);
    } else if (GRAPH_METHOD == 3) {
        if (rank == 0) {
            std::cerr << "GRAPH_METHOD=3: HNSW-on-HNSW\n";
        }
        hnsw_on_hnsw_index = mpi_hnsw_on_hnsw_build(base,
                                                    base_number,
                                                    vecdim,
                                                    H2H_NUM_PARTS,
                                                    PARTITION_MODE,
                                                    ipspace,
                                                    top_ipspace,
                                                    HNSW_M,
                                                    EF_CONSTRUCTION,
                                                    EF_SEARCH,
                                                    H2H_TOP_M,
                                                    H2H_TOP_EF_CONSTRUCTION,
                                                    H2H_TOP_EF_SEARCH,
                                                    0,
                                                    MPI_COMM_WORLD);
    } else {
        if (rank == 0) {
            std::cerr << "GRAPH_METHOD=4: Optimized HNSW-on-HNSW with IVF/KMeans partitions\n";
        }
        ivf_hnsw_on_hnsw_index = mpi_ivf_hnsw_on_hnsw_build(base,
                                                            base_number,
                                                            vecdim,
                                                            H2H_NUM_PARTS,
                                                            IVF_HNSW_NITER,
                                                            ipspace,
                                                            top_ipspace,
                                                            HNSW_M,
                                                            EF_CONSTRUCTION,
                                                            EF_SEARCH,
                                                            H2H_TOP_M,
                                                            H2H_TOP_EF_CONSTRUCTION,
                                                            H2H_TOP_EF_SEARCH,
                                                            0,
                                                            MPI_COMM_WORLD);
    }

    double build_t1 = MPI_Wtime();
    double local_build_time = build_t1 - build_t0;
    double max_build_time = 0.0;
    MPI_Reduce(&local_build_time,
               &max_build_time,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "max build time(s): " << max_build_time << "\n";
        std::cerr << "local search mode: "
                  << mpi_graph_local_search_name(LOCAL_SEARCH_MODE)
                  << ", local_threads=" << HNSW_LOCAL_THREADS << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

     
    // Query loop.
     

    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            gettimeofday(&val, NULL);
        }

        std::priority_queue<std::pair<float, uint32_t>> res;

        if (GRAPH_METHOD == 1) {
            res = mpi_ivf_hnsw_search(ivf_hnsw_index,
                                      rank == 0 ? test_query + i * vecdim : nullptr,
                                      vecdim,
                                      k,
                                      IVF_HNSW_NPROBE_CLUSTERS,
                                      LOCAL_P,
                                      LOCAL_SEARCH_MODE,
                                      HNSW_LOCAL_THREADS,
                                      0,
                                      MPI_COMM_WORLD);
        } else if (GRAPH_METHOD == 2) {
            res = mpi_partition_hnsw_search(partition_hnsw_index,
                                            rank == 0 ? test_query + i * vecdim : nullptr,
                                            vecdim,
                                            k,
                                            LOCAL_P,
                                            LOCAL_SEARCH_MODE,
                                            HNSW_LOCAL_THREADS,
                                            0,
                                            MPI_COMM_WORLD);
        } else if (GRAPH_METHOD == 3) {
            res = mpi_hnsw_on_hnsw_search(hnsw_on_hnsw_index,
                                          rank == 0 ? test_query + i * vecdim : nullptr,
                                          vecdim,
                                          k,
                                          H2H_TOP_PROBE_PARTS,
                                          LOCAL_P,
                                          LOCAL_SEARCH_MODE,
                                          HNSW_LOCAL_THREADS,
                                          0,
                                          MPI_COMM_WORLD);
        } else {
            res = mpi_ivf_hnsw_on_hnsw_search(ivf_hnsw_on_hnsw_index,
                                              rank == 0 ? test_query + i * vecdim : nullptr,
                                              vecdim,
                                              k,
                                              H2H_TOP_PROBE_PARTS,
                                              LOCAL_P,
                                              LOCAL_SEARCH_MODE,
                                              HNSW_LOCAL_THREADS,
                                              0,
                                              MPI_COMM_WORLD);
        }

        if (rank == 0) {
            struct timeval newVal;
            gettimeofday(&newVal, NULL);

            int64_t diff =
                (newVal.tv_sec * Converter + newVal.tv_usec) -
                (val.tv_sec * Converter + val.tv_usec);

            std::set<uint32_t> gtset;
            for(int j = 0; j < static_cast<int>(k); ++j) {
                int t = test_gt[j + i * test_gt_d];
                gtset.insert(static_cast<uint32_t>(t));
            }

            size_t acc = 0;
            while (res.size()) {
                int x = static_cast<int>(res.top().second);
                if(gtset.find(static_cast<uint32_t>(x)) != gtset.end()) {
                    ++acc;
                }
                res.pop();
            }

            float recall = static_cast<float>(acc) / static_cast<float>(k);
            results[i] = {recall, diff};
        }
    }

     
    // Output.
     

    if (rank == 0) {
        float avg_recall = 0.0f;
        float avg_latency = 0.0f;

        for(int i = 0; i < static_cast<int>(test_number); ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }

        std::cout << "average recall: " << avg_recall / test_number << "\n";
        std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    }

    ivf_hnsw_index.clear();
    partition_hnsw_index.clear();
    hnsw_on_hnsw_index.clear();
    ivf_hnsw_on_hnsw_index.clear();

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    MPI_Finalize();
    return 0;
}
