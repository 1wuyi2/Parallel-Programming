#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ann_ivf_adaptive_omp.h"
#include "ivf_pq_utils.h"

template <typename T>
T *LoadData(const std::string &data_path, size_t &n, size_t &d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);

    if (!fin.good()) {
        std::cerr << "[ERROR] cannot open data file: " << data_path << "\n";
        std::exit(1);
    }

    fin.read(reinterpret_cast<char *>(&n), 4);
    fin.read(reinterpret_cast<char *>(&d), 4);

    T *data = new T[n * d];
    const int sz = sizeof(T);

    for (size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char *>(data + i * d), d * sz);
    }

    fin.close();
    return data;
}

struct EvalSummary {
    std::string method;
    std::string config;

    double avg_recall = 0.0;
    double avg_latency_us = 0.0;

    double avg_nprobe = 0.0;
    double avg_candidates = 0.0;

    double avg_select_us = 0.0;
    double avg_collect_us = 0.0;
    double avg_search_us = 0.0;
    double avg_merge_us = 0.0;
};

inline double MainNowUs() {
    using Clock = std::chrono::high_resolution_clock;
    auto now = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::micro>(now).count();
}

float ComputeRecall(std::priority_queue<std::pair<float, uint32_t>> pq,
                    const int *gt, size_t gt_d, size_t query_id, size_t k) {
    std::set<uint32_t> gtset;

    for (size_t j = 0; j < k; ++j) {
        gtset.insert(static_cast<uint32_t>(gt[j + query_id * gt_d]));
    }

    size_t acc = 0;

    while (!pq.empty()) {
        uint32_t id = pq.top().second;
        pq.pop();

        if (gtset.find(id) != gtset.end()) {
            ++acc;
        }
    }

    return static_cast<float>(acc) / static_cast<float>(k);
}

void PrintCompactHeader() {
    std::cout << "method,config,recall,latency_us,avg_nprobe,avg_candidates,"
              << "select_us,collect_us,search_us,merge_us\n";
}

void PrintCompactResult(const EvalSummary &s) {
    std::cout << s.method << ","
              << s.config << ","
              << std::fixed << std::setprecision(6)
              << s.avg_recall << ","
              << std::setprecision(3)
              << s.avg_latency_us << ","
              << s.avg_nprobe << ","
              << s.avg_candidates << ","
              << s.avg_select_us << ","
              << s.avg_collect_us << ","
              << s.avg_search_us << ","
              << s.avg_merge_us
              << "\n";
}

EvalSummary EvaluateFixedIVF(const float *base,
                             const float *queries,
                             const int *gt,
                             IVFPQIndex &ivf_idx,
                             size_t test_number,
                             size_t gt_d,
                             size_t base_number,
                             size_t vecdim,
                             size_t k,
                             size_t nprobe,
                             int num_threads,
                             size_t local_p,
                             const std::string &schedule_type) {
    EvalSummary summary;
    summary.method = "Fixed-IVF";

    {
        std::ostringstream oss;
        oss << "nprobe=" << nprobe;
        summary.config = oss.str();
    }

    for (size_t i = 0; i < test_number; ++i) {
        const float *query = queries + i * vecdim;

        double t0 = MainNowUs();

        auto output = ivf_search_fixed_omp_with_stats(base, query,
                                                      ivf_idx, base_number,
                                                      vecdim, k, nprobe,
                                                      num_threads, local_p,
                                                      schedule_type);

        double t1 = MainNowUs();

        float recall = ComputeRecall(output.topk, gt, gt_d, i, k);

        summary.avg_recall += recall;
        summary.avg_latency_us += (t1 - t0);

        summary.avg_nprobe += output.stats.chosen_nprobe;
        summary.avg_candidates += output.stats.candidate_count;

        summary.avg_select_us += output.stats.select_us;
        summary.avg_collect_us += output.stats.collect_us;
        summary.avg_search_us += output.stats.search_us;
        summary.avg_merge_us += output.stats.merge_us;
    }

    summary.avg_recall /= test_number;
    summary.avg_latency_us /= test_number;

    summary.avg_nprobe /= test_number;
    summary.avg_candidates /= test_number;

    summary.avg_select_us /= test_number;
    summary.avg_collect_us /= test_number;
    summary.avg_search_us /= test_number;
    summary.avg_merge_us /= test_number;

    return summary;
}

EvalSummary EvaluateAdaptiveIVF(const float *base,
                                const float *queries,
                                const int *gt,
                                IVFPQIndex &ivf_idx,
                                size_t test_number,
                                size_t gt_d,
                                size_t base_number,
                                size_t vecdim,
                                size_t k,
                                size_t min_nprobe,
                                size_t max_nprobe,
                                double tau,
                                double coverage,
                                int num_threads,
                                size_t local_p,
                                const std::string &schedule_type) {
    EvalSummary summary;
    summary.method = "Adaptive-IVF";

    {
        std::ostringstream oss;
        oss << "min=" << min_nprobe
            << "_max=" << max_nprobe
            << "_tau=" << tau
            << "_cov=" << coverage;
        summary.config = oss.str();
    }

    for (size_t i = 0; i < test_number; ++i) {
        const float *query = queries + i * vecdim;

        double t0 = MainNowUs();

        auto output = ivf_search_adaptive_omp_with_stats(base, query,
                                                         ivf_idx, base_number,
                                                         vecdim, k,
                                                         min_nprobe,
                                                         max_nprobe,
                                                         tau, coverage,
                                                         num_threads, local_p,
                                                         schedule_type);

        double t1 = MainNowUs();

        float recall = ComputeRecall(output.topk, gt, gt_d, i, k);

        summary.avg_recall += recall;
        summary.avg_latency_us += (t1 - t0);

        summary.avg_nprobe += output.stats.chosen_nprobe;
        summary.avg_candidates += output.stats.candidate_count;

        summary.avg_select_us += output.stats.select_us;
        summary.avg_collect_us += output.stats.collect_us;
        summary.avg_search_us += output.stats.search_us;
        summary.avg_merge_us += output.stats.merge_us;
    }

    summary.avg_recall /= test_number;
    summary.avg_latency_us /= test_number;

    summary.avg_nprobe /= test_number;
    summary.avg_candidates /= test_number;

    summary.avg_select_us /= test_number;
    summary.avg_collect_us /= test_number;
    summary.avg_search_us /= test_number;
    summary.avg_merge_us /= test_number;

    return summary;
}

int main(int argc, char *argv[]) {
    std::string data_path = "/anndata/";
    size_t test_number_limit = 2000;
    int num_threads = 8;

    if (argc >= 2) {
        data_path = argv[1];
    }

    if (!data_path.empty() && data_path.back() != '/') {
        data_path.push_back('/');
    }

    if (argc >= 3) {
        test_number_limit = static_cast<size_t>(std::stoul(argv[2]));
    }

    if (argc >= 4) {
        num_threads = std::stoi(argv[3]);
    }

    size_t test_number = 0;
    size_t base_number = 0;
    size_t test_gt_d = 0;
    size_t vecdim = 0;

    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin",
                                      test_number, vecdim);

    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin",
                                 test_number, test_gt_d);

    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",
                                base_number, vecdim);

    test_number = std::min(test_number, test_number_limit);

    const size_t k = 10;
    const size_t local_p = 20;
    const std::string schedule_type = "static";

    std::cout << "[META]"
              << " data_path=" << data_path
              << " test_number=" << test_number
              << " base_number=" << base_number
              << " vecdim=" << vecdim
              << " k=" << k
              << " threads=" << num_threads
              << " local_p=" << local_p
              << " schedule=" << schedule_type
              << "\n";

    double build_t0 = MainNowUs();

    IVFPQIndex ivf_idx = build_ivf_pq_index(base,
                                            base_number,
                                            vecdim,
                                            1024,
                                            4,
                                            256,
                                            24,
                                            10);

    double build_t1 = MainNowUs();

    std::cout << "[BUILD]"
              << " nlist=" << ivf_idx.nlist
              << " build_time_ms=" << std::fixed << std::setprecision(3)
              << (build_t1 - build_t0) / 1000.0
              << "\n";

    PrintCompactHeader();

    std::vector<size_t> fixed_nprobes = {5, 10, 20, 50, 100};

    for (size_t nprobe : fixed_nprobes) {
        auto summary = EvaluateFixedIVF(base,
                                        test_query,
                                        test_gt,
                                        ivf_idx,
                                        test_number,
                                        test_gt_d,
                                        base_number,
                                        vecdim,
                                        k,
                                        nprobe,
                                        num_threads,
                                        local_p,
                                        schedule_type);

        PrintCompactResult(summary);
    }

    std::vector<double> taus = {0.015, 0.030, 0.060};
    std::vector<double> coverages = {0.80, 0.90, 0.95};

    const size_t min_nprobe = 5;
    const size_t max_nprobe = 100;

    for (double tau : taus) {
        for (double coverage : coverages) {
            auto summary = EvaluateAdaptiveIVF(base,
                                               test_query,
                                               test_gt,
                                               ivf_idx,
                                               test_number,
                                               test_gt_d,
                                               base_number,
                                               vecdim,
                                               k,
                                               min_nprobe,
                                               max_nprobe,
                                               tau,
                                               coverage,
                                               num_threads,
                                               local_p,
                                               schedule_type);

            PrintCompactResult(summary);
        }
    }

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}