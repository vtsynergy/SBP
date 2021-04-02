/// Blockmodel entropy calculations
#ifndef SBP_ENTROPY_HPP
#define SBP_ENTROPY_HPP

#include "blockmodel/blockmodel.hpp"
#include "graph.hpp"
#include "spence.hpp"

#include <math.h>

namespace entropy {

/// Calculates the overall entropy of the degree-corrected blockmodel.
double blockmodel_entropy(Blockmodel &blockmodel, const Graph &graph);

double delta_entropy(int vertex, int current_block, int proposed_block, Blockmodel &blockmodel, const Graph &graph,
                     EntryMap &deltas);

double delta_entropy(int vertex, int current_block, int proposed_block, Blockmodel &blockmodel, const Graph &graph,
                     SparseEdgeCountUpdates &edge_count_delta);

double delta_entropy(int current_block, int proposed_block, Blockmodel &blockmodel, const Graph &graph,
                     SparseEdgeCountUpdates &edge_count_delta);

double delta_entropy(int current_block, int proposed_block, Blockmodel &blockmodel, const Graph &graph,
                     EntryMap &deltas);

// TODO: declare as [[gnu::const]] or [[gnu::always_inline]]
inline double lbinom(double N, double k) {
    if (N == 0 || k == 0 || k >= N)
        return 0;
    assert(N > 0);
    assert(k > 0);
    return ((std::lgamma(N + 1) - std::lgamma(k + 1)) - std::lgamma(N - k + 1));
}

inline double get_v(double u, double epsilon=1e-8) {
    double v = u;
    double delta = 1;
    while (delta > epsilon) {
        // spence(exp(v)) = -spence(exp(-v)) - (v*v)/2
        double n_v = u * std::sqrt(spence(std::exp(-v)));
        delta = std::abs(n_v - v);
        v = n_v;
    }
    return v;
}

inline double log_q_approx_small(size_t n, size_t k) {
    return lbinom(n - 1, k - 1) - std::lgamma(k + 1);
}

inline double log_q_approx(size_t n, size_t k) {
    if (k < std::pow(n, 1/4.))
        return log_q_approx_small(n, k);
    double u = k / std::sqrt(n);
    double v = get_v(u);
    double lf = std::log(v) - std::log1p(- std::exp(-v) * (1 + u * u/2)) / 2 - std::log(2) * 3 / 2.
        - std::log(u) - std::log(M_PI);
    double g = 2 * v / u - u * std::log1p(-std::exp(-v));
    return lf - std::log(n) + std::sqrt(n) * g;
}

// TODO: declare as [[gnu::const]] and [[gnu::hot]] or [[gnu::always_inline]]
inline double log_q(double n, double k) {
    if (n <= 0 || k < 1)
        return 0;
    if (k > n)
        k = n;
    // TODO: maybe add a cache for faster access?
    // if (size_t(n) < __q_cache.shape()[0])
    //     return __q_cache[n][k];
    return log_q_approx(n, k);
}

inline double get_Se(int delta, int kin, int kout, int block, Blockmodel &blockmodel) {
    double S = 0.0;
    S += log_q(blockmodel.getBlock_degrees_in()[block] + kin, blockmodel.block_size(block) + delta);
    S += log_q(blockmodel.getBlock_degrees_out()[block] + kout, blockmodel.block_size(block) + delta);
    return S;
}

inline double get_Sr(int delta, int block, Blockmodel &blockmodel) {
    return std::lgamma(blockmodel.block_size(block) + delta + 1);
}

inline double get_Sk(const std::pair<int, int> &degrees, int delta, int block, Blockmodel &blockmodel) {
    int nd = 0;
    nd = blockmodel.degree_histogram(block)[degrees];
    return std::lgamma(nd + delta + 1);
}

/// change in minimum description length due to change in degree distribution for vertex moves 
inline double get_delta_deg_dl_dist_change(int kin, int kout, int block, int diff, Blockmodel &blockmodel) {
    /* get_delta_deg_dl_dist_change(r, dop, -1) */

    // double S_b = 0, S_a = 0;
    // int tkin = 0, tkout = 0, n = 0;
    // dop([&](size_t kin, size_t kout, int nk)
    //     {
    //         tkin += kin * nk;
    //         tkout += kout * nk;
    //         n += nk;

    //         auto deg = make_pair(kin, kout);
    //         S_b += get_Sk(deg,         0);
    //         S_a += get_Sk(deg, diff * nk);
    //     });
    double S_b = 0.0, S_a = 0.0;
    const auto degrees = std::make_pair(kin, kout);
    S_b += get_Sk(degrees, 0, block, blockmodel);
    S_a += get_Sk(degrees, diff, block, blockmodel);

    S_b += get_Se(0, 0, 0, block, blockmodel);
    S_a += get_Se(diff, diff * kin, diff * kout, block, blockmodel);

    S_b += get_Sr(0, block, blockmodel);
    S_a += get_Sr(diff, block, blockmodel);

    return S_a - S_b;
    /* get_delta_deg_dl_dist_change() */
}

inline double get_delta_deg_dl_dist_change(int kin, int kout, int block_weight, int block, int diff, const DegreeHistogram &histogram, Blockmodel &blockmodel) {
    /* get_delta_deg_dl_dist_change(r, dop, -1) */

    // double S_b = 0, S_a = 0;
    // int tkin = 0, tkout = 0, n = 0;
    // dop([&](size_t kin, size_t kout, int nk)
    //     {
    //         tkin += kin * nk;
    //         tkout += kout * nk;
    //         n += nk;

    //         auto deg = make_pair(kin, kout);
    //         S_b += get_Sk(deg,         0);
    //         S_a += get_Sk(deg, diff * nk);
    //     });
    double S_b = 0.0, S_a = 0.0;
    for (const std::pair<std::pair<int, int>, int> &degrees : histogram) {
        S_b += std::lgamma(degrees.second + 0 + 1);
        S_a += std::lgamma(degrees.second + diff * degrees.second + 1);
    }
    if (std::isinf(S_b) || std::isinf(S_a)) {
        std::cout << "part 1) S_b: " << S_b << " S_a: " << S_a << std::endl;
    }

    S_b += get_Se(0, 0, 0, block, blockmodel);
    S_a += get_Se(diff, diff * kin, diff * kout, block, blockmodel);
    if (std::isinf(S_b) || std::isinf(S_a)) {
        std::cout << "part 2) S_b: " << S_b << " S_a: " << S_a << std::endl;
    }

    S_b += get_Sr(0, block, blockmodel);
    S_a += get_Sr(diff, block, blockmodel);
    if (std::isinf(S_b) || std::isinf(S_a)) {
        std::cout << "part 3) S_b: " << S_b << " S_a: " << S_a << std::endl;
    }

    return S_a - S_b;
    /* get_delta_deg_dl_dist_change() */
}

}  // entropy

#endif  // SBP_ENTROPY_HPP
