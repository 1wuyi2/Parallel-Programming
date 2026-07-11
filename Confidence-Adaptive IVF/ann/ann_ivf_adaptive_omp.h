#pragma once

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "ann_simd.h"
#include "ivf_pq_utils.h"

struct AdaptiveIVFSearchStats {
    size_t chosen_nprobe = 0;
    size_t candidate_count = 0;

    double select_us = 0.0;
    double collect_us = 0.0;
    double search_us = 0.0;
    double merge_us = 0.0;
};

struct AdaptiveIVFSearchOutput {
    std::priority_queue<std::pair<float, uint32_t>> topk;
    AdaptiveIVFSearchStats stats;
};

inline double AdaptiveIVFNowUs() {
    using Clock = std::chrono::high_resolution_clock;
    auto now = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::micro>(now).count();
}

inline void AdaptiveIVFSetSchedule(const std::string &schedule_type) {
    if (schedule_type == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, 1);
    } else if (schedule_type == "guided") {
        omp_set_schedule(omp_sched_guided, 1);
    } else {
        omp_set_schedule(omp_sched_static, 0);
    }
}

inline void AdaptiveIVFPushTopP(std::vector<std::pair<float, uint32_t>> &heap,
                                size_t local_p, float dist, uint32_t id) {
    if (heap.size() < local_p) {
        heap.push_back({dist, id});
        std::push_heap(heap.begin(), heap.end());
    } else if (dist < heap.front().first) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = {dist, id};
        std::push_heap(heap.begin(), heap.end());
    }
}

inline std::vector<std::pair<float, uint32_t>>
AdaptiveIVFComputeSortedCentroids(const float *query, IVFPQIndex &idx, size_t vecdim,
                                  size_t top_limit) {
    top_limit = std::min(top_limit, idx.nlist);
    if (top_limit == 0) {
        top_limit = 1;
    }

    std::vector<std::pair<float, uint32_t>> cluster_ips(idx.nlist);

    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = inner_product_neon(query, idx.centroids.data() + c * vecdim, vecdim);
        cluster_ips[c] = {ip, static_cast<uint32_t>(c)};
    }

    if (top_limit < idx.nlist) {
        std::nth_element(
            cluster_ips.begin(),
            cluster_ips.begin() + top_limit - 1,
            cluster_ips.end(),
            [](const std::pair<float, uint32_t> &a,
               const std::pair<float, uint32_t> &b) {
                return a.first > b.first;
            });
    }

    std::sort(
        cluster_ips.begin(),
        cluster_ips.begin() + top_limit,
        [](const std::pair<float, uint32_t> &a,
           const std::pair<float, uint32_t> &b) {
            return a.first > b.first;
        });

    cluster_ips.resize(top_limit);
    return cluster_ips;
}

inline std::vector<uint32_t>
AdaptiveIVFSelectFixedClusters(const float *query, IVFPQIndex &idx, size_t vecdim,
                               size_t nprobe_clusters,
                               AdaptiveIVFSearchStats *stats = nullptr) {
    double t0 = AdaptiveIVFNowUs();

    nprobe_clusters = std::min(nprobe_clusters, idx.nlist);
    if (nprobe_clusters == 0) {
        nprobe_clusters = 1;
    }

    auto cluster_ips = AdaptiveIVFComputeSortedCentroids(query, idx, vecdim, nprobe_clusters);

    std::vector<uint32_t> selected;
    selected.reserve(nprobe_clusters);

    for (size_t i = 0; i < nprobe_clusters; ++i) {
        selected.push_back(cluster_ips[i].second);
    }

    double t1 = AdaptiveIVFNowUs();

    if (stats != nullptr) {
        stats->chosen_nprobe = selected.size();
        stats->select_us = t1 - t0;
    }

    return selected;
}

inline std::vector<uint32_t>
AdaptiveIVFSelectAdaptiveClusters(const float *query, IVFPQIndex &idx, size_t vecdim,
                                  size_t min_nprobe, size_t max_nprobe,
                                  double tau, double coverage,
                                  AdaptiveIVFSearchStats *stats = nullptr) {
    double t0 = AdaptiveIVFNowUs();

    min_nprobe = std::max<size_t>(1, min_nprobe);
    max_nprobe = std::max(min_nprobe, max_nprobe);
    max_nprobe = std::min(max_nprobe, idx.nlist);

    tau = std::max(tau, 1e-8);
    coverage = std::max(0.0, std::min(coverage, 1.0));

    auto cluster_ips = AdaptiveIVFComputeSortedCentroids(query, idx, vecdim, max_nprobe);

    const float best_score = cluster_ips[0].first;
    std::vector<double> weights(cluster_ips.size());
    double weight_sum = 0.0;

    for (size_t i = 0; i < cluster_ips.size(); ++i) {
        double v = std::exp((static_cast<double>(cluster_ips[i].first) -
                             static_cast<double>(best_score)) / tau);
        weights[i] = v;
        weight_sum += v;
    }

    size_t chosen = max_nprobe;
    double cumulative = 0.0;

    for (size_t i = 0; i < cluster_ips.size(); ++i) {
        cumulative += weights[i] / weight_sum;

        if (i + 1 >= min_nprobe && cumulative >= coverage) {
            chosen = i + 1;
            break;
        }
    }

    chosen = std::max(chosen, min_nprobe);
    chosen = std::min(chosen, max_nprobe);

    std::vector<uint32_t> selected;
    selected.reserve(chosen);

    for (size_t i = 0; i < chosen; ++i) {
        selected.push_back(cluster_ips[i].second);
    }

    double t1 = AdaptiveIVFNowUs();

    if (stats != nullptr) {
        stats->chosen_nprobe = selected.size();
        stats->select_us = t1 - t0;
    }

    return selected;
}

inline AdaptiveIVFSearchOutput
AdaptiveIVFSearchGivenClustersOMP(const float *base, const float *query,
                                  IVFPQIndex &idx, size_t vecdim, size_t k,
                                  const std::vector<uint32_t> &selected_clusters,
                                  int num_threads = 8,
                                  size_t local_p = 20,
                                  const std::string &schedule_type = "static") {
    AdaptiveIVFSearchOutput output;

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    if (local_p < k) {
        local_p = k;
    }

    double t_collect0 = AdaptiveIVFNowUs();

    std::vector<uint32_t> candidates;
    for (uint32_t cluster_id : selected_clusters) {
        const auto &list = idx.inverted_lists[cluster_id];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    double t_collect1 = AdaptiveIVFNowUs();

    output.stats.chosen_nprobe = selected_clusters.size();
    output.stats.candidate_count = candidates.size();
    output.stats.collect_us = t_collect1 - t_collect0;

    std::vector<std::vector<std::pair<float, uint32_t>>> local_results(num_threads);

    omp_sched_t old_kind;
    int old_modifier;
    omp_get_schedule(&old_kind, &old_modifier);
    AdaptiveIVFSetSchedule(schedule_type);

    double t_search0 = AdaptiveIVFNowUs();

#pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        std::vector<std::pair<float, uint32_t>> heap;
        heap.reserve(local_p + 1);

#pragma omp for schedule(runtime) nowait
        for (size_t j = 0; j < candidates.size(); ++j) {
            uint32_t id = candidates[j];

            float ip = inner_product_neon(query, base + id * vecdim, vecdim);
            float dist = 1.0f - ip;

            AdaptiveIVFPushTopP(heap, local_p, dist, id);
        }

        local_results[tid] = std::move(heap);
    }

    double t_search1 = AdaptiveIVFNowUs();

    omp_set_schedule(old_kind, old_modifier);

    output.stats.search_us = t_search1 - t_search0;

    double t_merge0 = AdaptiveIVFNowUs();

    std::priority_queue<std::pair<float, uint32_t>> final_pq;

    for (auto &heap : local_results) {
        for (auto &p : heap) {
            if (final_pq.size() < k) {
                final_pq.push(p);
            } else if (p.first < final_pq.top().first) {
                final_pq.push(p);
                final_pq.pop();
            }
        }
    }

    double t_merge1 = AdaptiveIVFNowUs();

    output.stats.merge_us = t_merge1 - t_merge0;
    output.topk = std::move(final_pq);

    return output;
}

inline AdaptiveIVFSearchOutput
ivf_search_fixed_omp_with_stats(const float *base, const float *query,
                                IVFPQIndex &idx, size_t base_number, size_t vecdim,
                                size_t k, size_t nprobe_clusters,
                                int num_threads = 8, size_t local_p = 20,
                                const std::string &schedule_type = "static") {
    (void)base_number;

    AdaptiveIVFSearchStats select_stats;
    auto selected = AdaptiveIVFSelectFixedClusters(query, idx, vecdim,
                                                   nprobe_clusters,
                                                   &select_stats);

    auto output = AdaptiveIVFSearchGivenClustersOMP(base, query, idx, vecdim,
                                                    k, selected, num_threads,
                                                    local_p, schedule_type);

    output.stats.select_us = select_stats.select_us;
    output.stats.chosen_nprobe = select_stats.chosen_nprobe;

    return output;
}

inline AdaptiveIVFSearchOutput
ivf_search_adaptive_omp_with_stats(const float *base, const float *query,
                                   IVFPQIndex &idx, size_t base_number, size_t vecdim,
                                   size_t k, size_t min_nprobe, size_t max_nprobe,
                                   double tau, double coverage,
                                   int num_threads = 8, size_t local_p = 20,
                                   const std::string &schedule_type = "static") {
    (void)base_number;

    AdaptiveIVFSearchStats select_stats;
    auto selected = AdaptiveIVFSelectAdaptiveClusters(query, idx, vecdim,
                                                      min_nprobe, max_nprobe,
                                                      tau, coverage,
                                                      &select_stats);

    auto output = AdaptiveIVFSearchGivenClustersOMP(base, query, idx, vecdim,
                                                    k, selected, num_threads,
                                                    local_p, schedule_type);

    output.stats.select_us = select_stats.select_us;
    output.stats.chosen_nprobe = select_stats.chosen_nprobe;

    return output;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_adaptive_omp(const float *base, const float *query,
                        IVFPQIndex &idx, size_t base_number, size_t vecdim,
                        size_t k, size_t min_nprobe, size_t max_nprobe,
                        double tau, double coverage,
                        int num_threads = 8, size_t local_p = 20,
                        const std::string &schedule_type = "static") {
    auto output = ivf_search_adaptive_omp_with_stats(base, query, idx,
                                                     base_number, vecdim, k,
                                                     min_nprobe, max_nprobe,
                                                     tau, coverage,
                                                     num_threads, local_p,
                                                     schedule_type);
    return output.topk;
}