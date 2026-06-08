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
#include <omp.h>
#include <mpi.h>

#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"

// 原有实验头文件
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

// MPI-IVF 分阶段实现
#include "ann_mpi_ivf_stage1_basic.h"
#include "ann_mpi_ivf_stage2_omp.h"
#include "ann_mpi_ivf_stage3_pq.h"

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

    // MPI 多进程下只让 rank 0 打印，避免重复刷屏
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
    int64_t latency; // 单位 us
};

// 保留原 HNSW 构建函数，当前 4.1 IVF-MPI 不使用。
// 后续如果做 4.2 图索引，可以继续扩展。
void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150;
    const int M = 16;

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);

    #pragma omp parallel for
    for(int i = 1; i < static_cast<int>(base_number); ++i) {
        appr_alg->addPoint(base + 1ll * vecdim * i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);

    delete appr_alg;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   
    const int MPI_METHOD = 1;

    const size_t TEST_NUMBER_LIMIT = 2000;

    const size_t k = 10;

    const size_t NLIST =1024;
    const size_t NPROBE_CLUSTERS = 50;
    const size_t LOCAL_P = 20;
    const int MPI_OMP_THREADS = 1;
    const std::string SCHEDULE_TYPE = "static";
    const int NITER = 10;
    const MPIIVFPartitionMode PARTITION_MODE = MPIIVFPartitionMode::BLOCK;

    // ============================================================
    // 数据加载
    // ============================================================

    omp_set_num_threads(MPI_OMP_THREADS);

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

    // 只测试前若干条查询
    test_number = std::min(TEST_NUMBER_LIMIT, test_number);
    test_number = std::min(test_number, gt_number);

    std::vector<SearchResult> results;
    results.resize(test_number);

    // ============================================================
    // IVF 索引构建与 MPI 划分
    // ============================================================

    // 只让 rank 0 构建一次 IVFPQIndex，然后广播到所有 MPI 进程。
    // 这里仍然复用你原来的 build_ivf_pq_index，因此是基于原 IVF-SIMD / IVF-PQ 基础迁移。
    IVFPQIndex global_ivf_idx;

    if (rank == 0) {
        global_ivf_idx = build_ivf_pq_index(base,
                                            base_number,
                                            vecdim,
                                            NLIST,
                                            4,
                                            256,
                                            24,
                                            NITER);
    }

    // 广播完整 IVFPQIndex。
    // 阶段 1 / 阶段 2 只使用 IVF 部分；
    // 阶段 3 会使用 PQ 部分。
    mpi_bcast_ivfpq_index(global_ivf_idx, 0, MPI_COMM_WORLD);

    // 每个 MPI 进程只保留自己负责的一部分 inverted list。
    IVFPQIndex local_ivf_idx =
        mpi_make_local_ivfpq_index(global_ivf_idx,
                                   base_number,
                                   rank,
                                   world_size,
                                   PARTITION_MODE);

    // 释放完整 inverted list 和 centroid，保留 local_ivf_idx。
    // 注意：local_ivf_idx 中已经复制了本进程需要的数据。
    std::vector<std::vector<uint32_t>>().swap(global_ivf_idx.inverted_lists);
    std::vector<float>().swap(global_ivf_idx.centroids);

    // ============================================================
    // 查询测试代码
    // ============================================================

    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;

        // 对齐所有进程，保证每个 query 的 MPI 通信顺序一致
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            gettimeofday(&val, NULL);
        }

        MPIIVFSearchStats stats;

        // 这里保持之前实验风格：
        // 想测哪个方法，就修改 MPI_METHOD 常量。
        std::priority_queue<std::pair<float, uint32_t>> res;

        if (MPI_METHOD == 1) {
            // 阶段 1：基础 MPI-IVF-SIMD
            // 对应基础要求：
            // base data / inverted list 划分，query 广播，local top-p，rank 0 merge top-k
            res = ivf_search_mpi_basic(base,
                                       rank == 0 ? test_query + i * vecdim : nullptr,
                                       local_ivf_idx,
                                       base_number,
                                       vecdim,
                                       k,
                                       NPROBE_CLUSTERS,
                                       LOCAL_P,
                                       0,
                                       MPI_COMM_WORLD,
                                       &stats);
        } else if (MPI_METHOD == 2) {
            // 阶段 2：MPI + OpenMP + SIMD
            // 进程间 MPI 并行，进程内 OpenMP 并行扫描 local candidates，
            // 距离计算继续调用 inner_product_neon。
            res = ivf_search_mpi_omp(base,
                                     rank == 0 ? test_query + i * vecdim : nullptr,
                                     local_ivf_idx,
                                     base_number,
                                     vecdim,
                                     k,
                                     NPROBE_CLUSTERS,
                                     LOCAL_P,
                                     MPI_OMP_THREADS,
                                     SCHEDULE_TYPE,
                                     0,
                                     MPI_COMM_WORLD,
                                     &stats);
        } else {
            // 阶段 3：MPI + IVF-PQ + OpenMP + SIMD
            // 迁移原 ivf_pq_search_omp 思路。
            // 这里 LOCAL_P 同时作为 PQ 粗排候选数 nprobe 和 local top-p。
            res = ivf_pq_search_mpi_omp(base,
                                        rank == 0 ? test_query + i * vecdim : nullptr,
                                        local_ivf_idx,
                                        base_number,
                                        vecdim,
                                        k,
                                        NPROBE_CLUSTERS,
                                        LOCAL_P,
                                        MPI_OMP_THREADS,
                                        LOCAL_P,
                                        SCHEDULE_TYPE,
                                        0,
                                        MPI_COMM_WORLD,
                                        &stats);
        }

        // 只有 rank 0 得到全局 top-k，因此只在 rank 0 计算 recall / latency。
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

    // ============================================================
    // 输出结果：保持原实验风格，只输出两行
    // ============================================================

    if (rank == 0) {
        float avg_recall = 0.0f;
        float avg_latency = 0.0f;

        for(int i = 0; i < static_cast<int>(test_number); ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }

        // 浮点误差可能导致一些精确算法平均 recall 不是 1
        std::cout << "average recall: " << avg_recall / test_number << "\n";
        std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    }

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    MPI_Finalize();
    return 0;
}