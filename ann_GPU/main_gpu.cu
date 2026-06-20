#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_ivf_gpu.cuh"

template<typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin.good()) {
        std::cerr << "cannot open " << data_path << std::endl;
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
    fin.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(n * d * sizeof(T)));
    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d
              << "  number: " << n
              << "  size_per_element: " << sizeof(T) << "\n";
    return data;
}

struct SearchResult {
    float recall = 0.0f;
    double latency_us = 0.0;
    size_t candidate_count = 0;
    double h2d_us = 0.0;
    double kernel_us = 0.0;
    double d2h_us = 0.0;
};

static inline std::string ensure_slash(std::string path) {
    if (!path.empty() && path.back() != '/' && path.back() != '\\') {
        path.push_back('/');
    }
    return path;
}

static inline double elapsed_us(std::chrono::steady_clock::time_point a,
                                std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

int main(int argc, char* argv[]) {
    std::string data_path = argc >= 2 ? argv[1] : "./";
    data_path = ensure_slash(data_path);

    size_t test_limit = argc >= 3 ? static_cast<size_t>(std::stoul(argv[2])) : 2000;
    size_t nlist = argc >= 4 ? static_cast<size_t>(std::stoul(argv[3])) : 1024;
    size_t nprobe_clusters = argc >= 5 ? static_cast<size_t>(std::stoul(argv[4])) : 50;
    int niter = argc >= 6 ? std::stoi(argv[5]) : 10;

    const size_t k = 10;

    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cerr << "using GPU: " << prop.name
              << ", compute capability " << prop.major << "." << prop.minor
              << "\n";

    size_t test_number = 0;
    size_t query_dim = 0;
    size_t gt_number = 0;
    size_t test_gt_d = 0;
    size_t base_number = 0;
    size_t vecdim = 0;

    float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, query_dim);
    int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", gt_number, test_gt_d);
    float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    if (test_query == nullptr || test_gt == nullptr || base == nullptr || query_dim != vecdim) {
        std::cerr << "data loading failed or dimension mismatch\n";
        delete[] test_query;
        delete[] test_gt;
        delete[] base;
        return 1;
    }

    test_number = std::min(test_number, gt_number);
    test_number = std::min(test_number, test_limit);

    std::cerr << "config: test_number=" << test_number
              << ", base_number=" << base_number
              << ", vecdim=" << vecdim
              << ", k=" << k
              << ", nlist=" << nlist
              << ", nprobe_clusters=" << nprobe_clusters
              << ", niter=" << niter << "\n";

    // 1. CPU 离线构建 IVF 索引。GPU 实验重点放在线上查询阶段。
    auto build_begin = std::chrono::steady_clock::now();
    IVFIndexGPU ivf_idx = build_ivf_index_cpu(base, base_number, vecdim, nlist, niter);
    auto build_end = std::chrono::steady_clock::now();
    std::cerr << "build ivf index time (s): "
              << elapsed_us(build_begin, build_end) / 1000000.0 << "\n";

    // 2. base 常驻显存。每个 query 只传 query 和 candidate ids。
    GpuIVFContext gpu_ctx;
    gpu_ivf_init(gpu_ctx, base, base_number, vecdim);

    std::vector<SearchResult> results(test_number);

    // 3. 查询测试。
    for (size_t i = 0; i < test_number; ++i) {
        const float* query = test_query + i * vecdim;

        auto t0 = std::chrono::steady_clock::now();
        GpuSearchStats stats;
        auto res = ivf_search_gpu(gpu_ctx, query, ivf_idx, k, nprobe_clusters, &stats, 128);
        auto t1 = std::chrono::steady_clock::now();

        std::set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j) {
            gtset.insert(static_cast<uint32_t>(test_gt[j + i * test_gt_d]));
        }

        size_t acc = 0;
        while (!res.empty()) {
            uint32_t id = res.top().second;
            if (gtset.find(id) != gtset.end()) {
                ++acc;
            }
            res.pop();
        }

        results[i].recall = static_cast<float>(acc) / static_cast<float>(k);
        results[i].latency_us = elapsed_us(t0, t1);
        results[i].candidate_count = stats.candidate_count;
        results[i].h2d_us = stats.h2d_ms * 1000.0;
        results[i].kernel_us = stats.kernel_ms * 1000.0;
        results[i].d2h_us = stats.d2h_ms * 1000.0;

        if ((i + 1) % 100 == 0) {
            std::cerr << "finished query " << (i + 1) << "/" << test_number << "\r" << std::flush;
        }
    }
    std::cerr << "\n";

    double avg_recall = 0.0;
    double avg_latency = 0.0;
    double avg_candidates = 0.0;
    double avg_h2d = 0.0;
    double avg_kernel = 0.0;
    double avg_d2h = 0.0;

    for (const auto& r : results) {
        avg_recall += r.recall;
        avg_latency += r.latency_us;
        avg_candidates += static_cast<double>(r.candidate_count);
        avg_h2d += r.h2d_us;
        avg_kernel += r.kernel_us;
        avg_d2h += r.d2h_us;
    }

    avg_recall /= static_cast<double>(test_number);
    avg_latency /= static_cast<double>(test_number);
    avg_candidates /= static_cast<double>(test_number);
    avg_h2d /= static_cast<double>(test_number);
    avg_kernel /= static_cast<double>(test_number);
    avg_d2h /= static_cast<double>(test_number);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "average recall: " << avg_recall << "\n";
    std::cout << "average latency (us): " << avg_latency << "\n";
    std::cout << "average candidates: " << avg_candidates << "\n";
    std::cout << "average H2D memcpy (us): " << avg_h2d << "\n";
    std::cout << "average kernel (us): " << avg_kernel << "\n";
    std::cout << "average D2H memcpy (us): " << avg_d2h << "\n";

    gpu_ivf_free(gpu_ctx);
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}
