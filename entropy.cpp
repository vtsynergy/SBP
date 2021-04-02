#include "entropy.hpp"

namespace entropy {

// TODO: add a thread-safe version of lgamma
// TODO: add a faster (cached) version of lgamma. Both are available in graph-tool
// TODO: add support for weighted graphs
double blockmodel_entropy(Blockmodel &blockmodel, const Graph &graph) {
//     double S = 0, S_dl = 0;
    double entropy = 0.0, entropy_dl = 0.0;
    // /* sparse_entropy() */
    // for (auto e : edges_range(_bg))
    //     S += eterm_exact(source(e, _bg), target(e, _bg), _mrs[e], _bg);
    for (const int &edge_weight : blockmodel.getBlockmodel().values()) {
        entropy += -std::lgamma(edge_weight + 1);
    }
    // for (auto v : vertices_range(_bg))
    //     S += vterm_exact(_mrp[v], _mrm[v], _wr[v], _deg_corr, _bg);
    for (int community = 0; community < blockmodel.getNum_blocks(); ++community) {
        entropy += std::lgamma(blockmodel.getBlock_degrees_out()[community] + 1);
        entropy += std::lgamma(blockmodel.getBlock_degrees_in()[community] + 1);
    }
    // for (auto v : vertices_range(_g))
    //     S += get_deg_entropy(v, _degs);
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        /* get_deg_entropy */
        // auto kin = in_degreeS()(v, _g, _eweight);
        // auto kout = out_degreeS()(v, _g, _eweight);
        // double S = -lgamma_fast(kin + 1) - lgamma_fast(kout + 1);
        // return S * _vweight[v];
        /* get_deg_entropy */
        int kin = graph.in_neighbors[vertex].size();
        int kout = graph.out_neighbors[vertex].size();
        entropy += -std::lgamma(kin + 1) - std::lgamma(kout + 1);
    }

    // TODO: If we want to support multigraphs, have to add the code for parallel_entropy();

    // /* get_partition_dl(); */
    // double S = 0;
    // S += lbinom(_N - 1, _actual_B - 1);  TODO: actual_num_blocks?
    entropy_dl += lbinom(graph.num_vertices - 1, blockmodel.getNum_blocks() - 1);
    // S += lgamma_fast(_N + 1);
    entropy_dl += std::lgamma(graph.num_vertices + 1);
    // for (auto nr : _total)
    //     S -= lgamma_fast(nr + 1);
    for (const int &size : blockmodel.block_sizes()) {
        entropy_dl -= std::lgamma(size + 1);
    }
    // S += safelog_fast(_N);
    entropy_dl += std::log(graph.num_vertices);
    // return S;
    // /* get_partition_dl(); */
    // std::cout << "c) entropy_dl: " << entropy_dl << std::endl;

    // /* get_deg_dl(ea.degree_dl_kind); */
    // TODO: figure out what _ep[r] && _total[r] && _em[r] && _hist[r] && ks stand for
    // rs : list of blocks
    // for (auto r : rs) {
    //     r = get_r(r);
    //     S += log_q(_ep[r], _total[r]);
    //     S += log_q(_em[r], _total[r]);
    //     size_t total = 0;
        // for (auto& k_c : _hist[r]) {  // I think _hist[r].second is the size of the block
        //     S -= lgamma_fast(k_c.second + 1);
        //     total += k_c.second;  // TODO: check if this happens only once
        // }
    //     S += lgamma_fast(total + 1);
    // }
    for (int community = 0; community < blockmodel.getNum_blocks(); ++community) {
        entropy_dl += log_q(blockmodel.getBlock_degrees_out()[community], blockmodel.block_size(community));
        entropy_dl += log_q(blockmodel.getBlock_degrees_in()[community], blockmodel.block_size(community));
        // std::cout << "b) after log_q: " << entropy_dl << std::endl;
        int total = 0;
        for (const auto &pair : blockmodel.degree_histogram(community)) {
            entropy_dl -= std::lgamma(pair.second + 1);
            total += pair.second;
        }
        entropy_dl += std::lgamma(total + 1);
        // std::cout << "b) after the lgamma 2: " << entropy_dl << std::endl;
    }
    // std::cout << "b) entropy_dl: " << entropy_dl << " big total: " << big_total << std::endl;
    // /* get_deg_dl */

//     /* S_dl += get_edges_dl(actual_B, _partition_stats.front().get_E(), _g); */
        // size_t BB = (graph_tool::is_directed(g)) ? B * B : (B * (B + 1)) / 2;
        // return lbinom(BB + _E - 1, _E);
    double B = blockmodel.getNum_blocks() * blockmodel.getNum_blocks();
    entropy_dl += lbinom(B + graph.num_edges - 1, graph.num_edges);
       /* get_edges_dl */
    // std::cout << "a) entropy_dl: " << entropy_dl << std::endl;

//     for (auto v : vertices_range(_g)) {
//         auto& f = _bfield[v];  // pretty sure bfield[v] is always 0
//         if (f.empty())
//             continue;
//         size_t r = _b[v];
//         S_dl -= (r < f.size()) ? f[r] : f.back();
//     }

//     std::cout << "S: " << S << " S_dl: " << S_dl << std::endl;
//     return S + S_dl * ea.beta_dl;  // beta_dl == 1.0 ?
    double result = entropy + entropy_dl * 1.0;  // 1.0 == beta_dl, add as user parameter if wanted.
    std::cout << "entropy: " << entropy << " entropy_dl: " << entropy_dl << std::endl;
    std::cout << "blockmodel entropy = " << result << std::endl;
    return result;
}

double delta_entropy(int vertex, int current_block, int proposed_block, Blockmodel &blockmodel, const Graph &graph,
                     EntryMap &deltas) {
    if (current_block == proposed_block)
        return 0.0;

    double delta_entropy = 0.0, delta_entropy_dl = 0.0;

    EntryMap entries = blockmodel.entries2(current_block, proposed_block);
    for (const std::pair<std::pair<int, int>, int> &delta : deltas) {
        int entry = entries[delta.first];  // If row,col isn't in entries, a 0 will be returned
        delta_entropy += -std::lgamma(entry + delta.second + 1) - -std::lgamma(entry + 1);
    }

    int kout = graph.out_neighbors[vertex].size();
    int kin = graph.in_neighbors[vertex].size();

    int current_block_degrees_out = blockmodel.getBlock_degrees_out()[current_block];
    int current_block_degrees_in = blockmodel.getBlock_degrees_in()[current_block];
    delta_entropy += std::lgamma(current_block_degrees_out - kout + 1) + std::lgamma(current_block_degrees_in - kin + 1);
    delta_entropy -= std::lgamma(current_block_degrees_out + 1) + std::lgamma(current_block_degrees_in + 1);

    int proposed_block_degrees_out = blockmodel.getBlock_degrees_out()[proposed_block];
    int proposed_block_degrees_in = blockmodel.getBlock_degrees_in()[proposed_block];
    delta_entropy += std::lgamma(proposed_block_degrees_out + kout + 1) + std::lgamma(proposed_block_degrees_in + kin + 1);
    delta_entropy -= std::lgamma(proposed_block_degrees_out + 1) + std::lgamma(proposed_block_degrees_in + 1);

    double S_b = 0, S_a = 0;
    S_b += -std::lgamma(blockmodel.block_size(current_block) + 1);
    S_a += -std::lgamma(blockmodel.block_size(current_block));  // -1 + 1
    S_b += -std::lgamma(blockmodel.block_size(proposed_block) + 1);
    S_a += -std::lgamma(blockmodel.block_size(proposed_block) + 2);  // +1 + 1
    delta_entropy_dl += S_a - S_b;

    delta_entropy_dl += get_delta_deg_dl_dist_change(kin, kout, current_block, -1, blockmodel);
    delta_entropy_dl += get_delta_deg_dl_dist_change(kin, kout, proposed_block, +1, blockmodel);

    return delta_entropy + 1.0 * delta_entropy_dl;
}

double delta_entropy(int vertex, int current_block, int proposed_block, Blockmodel &blockmodel, const Graph &graph,
                     SparseEdgeCountUpdates &edge_count_delta) {

    // get_move_entries(v, r, nr, m_entries, [](auto) constexpr { return false; });

    if (current_block == proposed_block)
        return 0.0;

    double delta_entropy = 0.0, delta_entropy_dl = 0.0;

    // double dS = 0;  // change in blockmodel term from total community edges & size
    /* dS = virtual_move_sparse<true>(v, r, nr, m_entries); */
    // double dS = entries_dS<exact>(m_entries, _mrs, _emat, _bg);
    // double dS = 0;
    // const auto& entries = m_entries.get_entries();
    /// Entries
    // Note: double counting is prevented because edge_count_delta only has a delta for `current_block` in the
    // rows, not in the columns.
    MapVector<int> current_block_row = blockmodel.getBlockmodel().getrow_sparse(current_block);  // out_edges
    MapVector<int> current_block_col = blockmodel.getBlockmodel().getcol_sparse(current_block);  // in_edges
    MapVector<int> proposed_block_row = blockmodel.getBlockmodel().getrow_sparse(proposed_block);
    MapVector<int> proposed_block_col = blockmodel.getBlockmodel().getcol_sparse(proposed_block);
    // const auto& delta = m_entries.get_delta();  == edge_count_delta
    // auto& mes = m_entries.get_mes(emat);  == MapVector[index]
    for (const std::pair<int, int> &delta : edge_count_delta.block_row) {
        int weight = current_block_row[delta.first];
        delta_entropy += -std::lgamma(weight + delta.second + 1) - -std::lgamma(weight + 1);
        // if (std::isnan(delta_entropy)) {
        //     std::cout << "NAN! delta_entropy a! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
        // if (std::isinf(delta_entropy)) {
        //     std::cout << "INF! delta_entropy a! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
    }
    // std::cout << "delta_entropy a: " << delta_entropy;
    for (const std::pair<int, int> &delta : edge_count_delta.block_col) {
        int weight = current_block_col[delta.first];
        delta_entropy += -std::lgamma(weight + delta.second + 1) - -std::lgamma(weight + 1);
        // if (std::isnan(delta_entropy)) {
        //     std::cout << "NAN! delta_entropy b! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
        // if (std::isinf(delta_entropy)) {
        //     std::cout << "INF! delta_entropy b! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
    }
    // std::cout << "delta_entropy b: " << delta_entropy;
    for (const std::pair<int, int> &delta : edge_count_delta.proposal_row) {
        int weight = proposed_block_row[delta.first];
        delta_entropy += -std::lgamma(weight + delta.second + 1) - -std::lgamma(weight + 1);
        // if (std::isnan(delta_entropy)) {
        //     std::cout << "NAN! delta_entropy c! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
        // if (std::isinf(delta_entropy)) {
        //     std::cout << "INF! delta_entropy c! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
    }
    // std::cout << "delta_entropy c: " << delta_entropy;
    for (const std::pair<int, int> &delta : edge_count_delta.proposal_col) {
        int weight = proposed_block_col[delta.first];
        delta_entropy += -std::lgamma(weight + delta.second + 1) - -std::lgamma(weight + 1);
        // if (std::isnan(delta_entropy)) {
        //     std::cout << "NAN! delta_entropy d! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
        // if (std::isinf(delta_entropy)) {
        //     std::cout << "INF! delta_entropy d! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
    }
    // std::cout << "delta_entropy d: " << delta_entropy;

    // std::cout << "delta_entropy: " << delta_entropy << " ";
    // for (size_t i = 0; i < entries.size(); ++i)
    // {
    //     auto& entry = entries[i];
    //     auto er = entry.first;
    //     auto es = entry.second;
    //     size_t ers = 0;
    //     if (mes[i] != emat.get_null_edge()) {
    //         // mrs[me] == 1 or 2?
    //         ers = mrs[mes[i]];
    //         // me = boost::detail::adj_edge_descriptor<long unsigned int>. Those fuckers didn't give it a printable representation even though it very clearly is an int
    //     //    std::cout << "mrs[me] = " << ers << std::endl;
    //     }
    //     assert(int(ers) + delta[i] >= 0);
    //     dS += eterm_exact(er, es, ers + delta[i], bg) - eterm_exact(er, es, ers, bg);
    //     // This is unfortunately just a guess: entries here = all the changes in blockmodel (row, col x r, nr)
    //     // delta here = teh differences between those particular blockmodel edge weights
    //     // eterm_exact === -lgamma(ers + 1);  only because it's directed though!
    // }
    // return dS;
    int kout = graph.out_neighbors[vertex].size();
    int kin = graph.in_neighbors[vertex].size();

    // int dwr = _vweight[v];
    // int dwnr = dwr; == 1

    // auto vt = [&](auto mrp, auto mrm, auto nr)
    //     {
    //         assert(mrp >= 0 && mrm >=0 && nr >= 0);
    //         return vterm_exact(mrp, mrm, nr, _deg_corr, _bg); == lgamma(mrp+1) + lgamma(mrm+1)
    //     };

    // auto mrp_r = _mrp[r];
    // auto mrm_r = _mrm[r];
    // auto wr_r = _wr[r];  --> only needed if not degree corrected
    // dS += vt(mrp_r - kout, mrm_r - kin, wr_r - dwr);  // add the Entropy of r without v (proposed)
    // dS -= vt(mrp_r       , mrm_r      , wr_r      );  // subtract Entropy of r with v (current)
    int current_block_degrees_out = blockmodel.getBlock_degrees_out()[current_block];
    int current_block_degrees_in = blockmodel.getBlock_degrees_in()[current_block];
    delta_entropy += std::lgamma(current_block_degrees_out - kout + 1) + std::lgamma(current_block_degrees_in - kin + 1);
    delta_entropy -= std::lgamma(current_block_degrees_out + 1) + std::lgamma(current_block_degrees_in + 1);

    // auto mrp_nr = _mrp[nr];
    // auto mrm_nr = _mrm[nr];
    // auto wr_nr = _wr[nr];
    // dS += vt(mrp_nr + kout, mrm_nr + kin, wr_nr + dwnr);  // add entropy of nr with v (proposed)
    // dS -= vt(mrp_nr       , mrm_nr      , wr_nr       );  // subtract entropy of nr without v (current)
    int proposed_block_degrees_out = blockmodel.getBlock_degrees_out()[proposed_block];
    int proposed_block_degrees_in = blockmodel.getBlock_degrees_in()[proposed_block];
    delta_entropy += std::lgamma(proposed_block_degrees_out + kout + 1) + std::lgamma(proposed_block_degrees_in + kin + 1);
    delta_entropy -= std::lgamma(proposed_block_degrees_out + 1) + std::lgamma(proposed_block_degrees_in + 1);
    // std::cout << "delta_entropy: " << delta_entropy << " ";

    // return dS;
    /* virtual_move_sparse */

    /* dS_dl += get_delta_partition_dl(v, r, nr, ea); */
    // int n = vweight[v]; == 1

    double S_b = 0, S_a = 0;
    //     S_b += -lgamma_fast(_total[r] + 1);
    //     S_a += -lgamma_fast(_total[r] - n + 1);
    S_b += -std::lgamma(blockmodel.block_size(current_block) + 1);
    S_a += -std::lgamma(blockmodel.block_size(current_block));  // -1 + 1

    //     S_b += -lgamma_fast(_total[nr] + 1);
    //     S_a += -lgamma_fast(_total[nr] + n + 1);
    S_b += -std::lgamma(blockmodel.block_size(proposed_block) + 1);
    S_a += -std::lgamma(blockmodel.block_size(proposed_block) + 2);  // +1 + 1

    delta_entropy_dl += S_a - S_b;
    // std::cout << "delta_entropy_dl: " << delta_entropy_dl << " ";
    // return S_a - S_b;

    /* get_delta_partition_dl */
    // double dS = 0;
    // dS += get_delta_deg_dl_dist_change(r, -1);
    // dS += get_delta_deg_dl_dist_change(ne, +1);
    delta_entropy_dl += get_delta_deg_dl_dist_change(kin, kout, current_block, -1, blockmodel);
    delta_entropy_dl += get_delta_deg_dl_dist_change(kin, kout, proposed_block, +1, blockmodel);
    // std::cout << "delta_entropy_dl: " << delta_entropy_dl << std::endl;
    /* get_delta_partition_dl */

    return delta_entropy + 1.0 * delta_entropy_dl;
}

double delta_entropy(int current_block, int proposed_block, Blockmodel &blockmodel, const Graph &graph,
                     EntryMap &deltas) {
    if (current_block == proposed_block)
        return 0.0;

    double delta_entropy = 0.0, delta_entropy_dl = 0.0;

    EntryMap entries = blockmodel.entries2(current_block, proposed_block);
    for (const std::pair<std::pair<int, int>, int> &delta : deltas) {
        int entry = entries[delta.first];
        delta_entropy += -std::lgamma(entry + delta.second + 1) - -std::lgamma(entry + 1);
        if (std::isinf(delta_entropy)) {
            std::cout << "INF! delta_entropy a! row: " << delta.first.first << " col: " << delta.first.second << " weight << " << delta.second << std::endl;
        }
    }

    int kout = blockmodel.getBlock_degrees_out()[current_block];
    int kin = blockmodel.getBlock_degrees_in()[current_block];

    int current_block_degrees_out = blockmodel.getBlock_degrees_out()[current_block];
    int current_block_degrees_in = blockmodel.getBlock_degrees_in()[current_block];
    // TODO: replace lgamma(degrees -kout/in + 1) with lgamma(1) or just 0
    delta_entropy += std::lgamma(current_block_degrees_out - kout + 1) + std::lgamma(current_block_degrees_in - kin + 1);
    delta_entropy -= std::lgamma(current_block_degrees_out + 1) + std::lgamma(current_block_degrees_in + 1);

    int proposed_block_degrees_out = blockmodel.getBlock_degrees_out()[proposed_block];
    int proposed_block_degrees_in = blockmodel.getBlock_degrees_in()[proposed_block];
    delta_entropy += std::lgamma(proposed_block_degrees_out + kout + 1) + std::lgamma(proposed_block_degrees_in + kin + 1);
    delta_entropy -= std::lgamma(proposed_block_degrees_out + 1) + std::lgamma(proposed_block_degrees_in + 1);

    // return dS;
    /* virtual_move_sparse */

    /* dS_dl += get_delta_partition_dl(v, r, nr, ea); */
    // int n = vweight[v]; == 1

    double S_b = 0, S_a = 0;
    int block_weight = blockmodel.block_size(current_block);
    S_b += -std::lgamma(blockmodel.block_size(current_block) + 1);
    S_a += -std::lgamma(blockmodel.block_size(current_block) - block_weight + 1);  // -1 + 1
    S_b += -std::lgamma(blockmodel.block_size(proposed_block) + 1);
    S_a += -std::lgamma(blockmodel.block_size(proposed_block) + block_weight + 1);  // +1 + 1

    int change_in_num_blocks = -1;  // The merge will delete one block
    S_b += lbinom(graph.num_vertices - 1, blockmodel.getNum_blocks() - 1);
    S_a += lbinom(graph.num_vertices - 1, blockmodel.getNum_blocks() + change_in_num_blocks - 1);
    delta_entropy_dl += S_a - S_b;
    // std::cout << "dS_dl a) " << delta_entropy_dl;

    const DegreeHistogram &histogram = blockmodel.degree_histogram(current_block);
    delta_entropy_dl += get_delta_deg_dl_dist_change(kin, kout, block_weight, current_block, -1, histogram, blockmodel);
    delta_entropy_dl += get_delta_deg_dl_dist_change(kin, kout, block_weight, proposed_block, +1, histogram, blockmodel);
    // std::cout << " | dS_dl b) " << delta_entropy_dl;

    /* get_delta_edges_dl */
    S_b = 0.0, S_a = 0.0;
    int B = blockmodel.getNum_blocks();
    S_b += lbinom((B * B) + graph.num_edges - 1, graph.num_edges);
    S_a += lbinom(((B - 1) * (B - 1)) + graph.num_edges - 1, graph.num_edges);
    delta_entropy_dl += S_a - S_b;
    // std::cout << " | dS_dl c) " << delta_entropy_dl << std::endl;

    if (std::isnan(delta_entropy) || std::isinf(delta_entropy) || std::isnan(delta_entropy_dl) || std::isinf(delta_entropy_dl)) {
        std::cout << "delta_entropy: " << delta_entropy << " delta_entropy_dl: " << delta_entropy_dl << std::endl;
        exit(-15);
    }
    return delta_entropy + 1.0 * delta_entropy_dl;
}

double delta_entropy(int current_block, int proposed_block, Blockmodel &blockmodel, const Graph &graph,
                     SparseEdgeCountUpdates &edge_count_delta) {
    if (current_block == proposed_block)
        return 0.0;

    double delta_entropy = 0.0, delta_entropy_dl = 0.0;

    /// Entries
    MapVector<int> current_block_row = blockmodel.getBlockmodel().getrow_sparse(current_block);  // out_edges
    MapVector<int> current_block_col = blockmodel.getBlockmodel().getcol_sparse(current_block);  // in_edges
    MapVector<int> proposed_block_row = blockmodel.getBlockmodel().getrow_sparse(proposed_block);
    MapVector<int> proposed_block_col = blockmodel.getBlockmodel().getcol_sparse(proposed_block);

    // TODO: replace std::lgamma(weight + delta.second + 1) with either 0, or std::lgamma(1);
    for (const std::pair<int, int> &delta : edge_count_delta.block_row) {
        int weight = current_block_row[delta.first];
        delta_entropy += -std::lgamma(weight + delta.second + 1) - -std::lgamma(weight + 1);
        // if (std::isnan(delta_entropy)) {
        //     std::cout << "NAN! delta_entropy a! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
        if (std::isinf(delta_entropy)) {
            std::cout << "INF! delta_entropy a! weight: " << weight << " delta: " << delta.second << std::endl;
        }
    }
    // std::cout << "delta_entropy a: " << delta_entropy;
    for (const std::pair<int, int> &delta : edge_count_delta.block_col) {
        int weight = current_block_col[delta.first];
        delta_entropy += -std::lgamma(weight + delta.second + 1) - -std::lgamma(weight + 1);
        // if (std::isnan(delta_entropy)) {
        //     std::cout << "NAN! delta_entropy b! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
        if (std::isinf(delta_entropy)) {
            std::cout << "INF! delta_entropy b! weight: " << weight << " delta: " << delta.second << std::endl;
        }
    }
    // std::cout << "delta_entropy b: " << delta_entropy;
    for (const std::pair<int, int> &delta : edge_count_delta.proposal_row) {
        int weight = proposed_block_row[delta.first];
        delta_entropy += -std::lgamma(weight + delta.second + 1) - -std::lgamma(weight + 1);
        // if (std::isnan(delta_entropy)) {
        //     std::cout << "NAN! delta_entropy c! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
        if (std::isinf(delta_entropy)) {
            std::cout << "INF! delta_entropy c! weight: " << weight << " delta: " << delta.second << std::endl;
        }
    }
    // std::cout << "delta_entropy c: " << delta_entropy;
    for (const std::pair<int, int> &delta : edge_count_delta.proposal_col) {
        int weight = proposed_block_col[delta.first];
        delta_entropy += -std::lgamma(weight + delta.second + 1) - -std::lgamma(weight + 1);
        // if (std::isnan(delta_entropy)) {
        //     std::cout << "NAN! delta_entropy d! weight: " << weight << " delta: " << delta.second << std::endl;
        // }
        if (std::isinf(delta_entropy)) {
            std::cout << "INF! delta_entropy d! blockmodel[" << delta.first << "," << proposed_block << "] += " << delta.second << " = " << weight + delta.second << std::endl;
        }
    }
    // std::cout << "delta_entropy d: " << delta_entropy;

    // return dS;
    int kout = blockmodel.getBlock_degrees_out()[current_block];
    int kin = blockmodel.getBlock_degrees_in()[current_block];
    // int kout = graph.out_neighbors[vertex].size();
    // int kin = graph.in_neighbors[vertex].size();

    // int dwr = _vweight[v];
    // int dwnr = dwr; == 1

    // auto vt = [&](auto mrp, auto mrm, auto nr)
    //     {
    //         assert(mrp >= 0 && mrm >=0 && nr >= 0);
    //         return vterm_exact(mrp, mrm, nr, _deg_corr, _bg); == lgamma(mrp+1) + lgamma(mrm+1)
    //     };

    // auto mrp_r = _mrp[r];
    // auto mrm_r = _mrm[r];
    // auto wr_r = _wr[r];  --> only needed if not degree corrected
    // dS += vt(mrp_r - kout, mrm_r - kin, wr_r - dwr);  // add the Entropy of r without v (proposed)
    // dS -= vt(mrp_r       , mrm_r      , wr_r      );  // subtract Entropy of r with v (current)
    int current_block_degrees_out = blockmodel.getBlock_degrees_out()[current_block];
    int current_block_degrees_in = blockmodel.getBlock_degrees_in()[current_block];
    // TODO: replace lgamma(degrees -kout/in + 1) with lgamma(1) or just 0
    delta_entropy += std::lgamma(current_block_degrees_out - kout + 1) + std::lgamma(current_block_degrees_in - kin + 1);
    delta_entropy -= std::lgamma(current_block_degrees_out + 1) + std::lgamma(current_block_degrees_in + 1);

    // auto mrp_nr = _mrp[nr];
    // auto mrm_nr = _mrm[nr];
    // auto wr_nr = _wr[nr];
    // dS += vt(mrp_nr + kout, mrm_nr + kin, wr_nr + dwnr);  // add entropy of nr with v (proposed)
    // dS -= vt(mrp_nr       , mrm_nr      , wr_nr       );  // subtract entropy of nr without v (current)
    int proposed_block_degrees_out = blockmodel.getBlock_degrees_out()[proposed_block];
    int proposed_block_degrees_in = blockmodel.getBlock_degrees_in()[proposed_block];
    delta_entropy += std::lgamma(proposed_block_degrees_out + kout + 1) + std::lgamma(proposed_block_degrees_in + kin + 1);
    delta_entropy -= std::lgamma(proposed_block_degrees_out + 1) + std::lgamma(proposed_block_degrees_in + 1);
    // std::cout << "delta_entropy: " << delta_entropy << " ";

    // return dS;
    /* virtual_move_sparse */

    /* dS_dl += get_delta_partition_dl(v, r, nr, ea); */
    // int n = vweight[v]; == 1

    double S_b = 0, S_a = 0;
    int block_weight = blockmodel.block_size(current_block);
    //     S_b += -lgamma_fast(_total[r] + 1);
    //     S_a += -lgamma_fast(_total[r] - n + 1);
    S_b += -std::lgamma(blockmodel.block_size(current_block) + 1);
    S_a += -std::lgamma(blockmodel.block_size(current_block) - block_weight + 1);  // -1 + 1

    //     S_b += -lgamma_fast(_total[nr] + 1);
    //     S_a += -lgamma_fast(_total[nr] + n + 1);
    S_b += -std::lgamma(blockmodel.block_size(proposed_block) + 1);
    S_a += -std::lgamma(blockmodel.block_size(proposed_block) + block_weight + 1);  // +1 + 1

    int change_in_num_blocks = -1;  // The merge will delete one block
    S_b += lbinom(graph.num_vertices - 1, blockmodel.getNum_blocks() - 1);
    S_a += lbinom(graph.num_vertices - 1, blockmodel.getNum_blocks() + change_in_num_blocks - 1);

    delta_entropy_dl += S_a - S_b;
    std::cout << "dS_dl a) " << delta_entropy_dl;
    // std::cout << "delta_entropy_dl: " << delta_entropy_dl << " ";
    // return S_a - S_b;

    /* get_delta_deg_dl */
    // double dS = 0;
    // dS += get_delta_deg_dl_dist_change(r, -1);
    // dS += get_delta_deg_dl_dist_change(ne, +1);
    const DegreeHistogram &histogram = blockmodel.degree_histogram(current_block);
    delta_entropy_dl += get_delta_deg_dl_dist_change(kin, kout, block_weight, current_block, -1, histogram, blockmodel);
    // std::cout << "\n A: " << delta_entropy_dl;
    delta_entropy_dl += get_delta_deg_dl_dist_change(kin, kout, block_weight, proposed_block, +1, histogram, blockmodel);
    // std::cout << " B: " << delta_entropy_dl << std::endl;
    std::cout << " | dS_dl b) " << delta_entropy_dl;
    // std::cout << "delta_entropy_dl: " << delta_entropy_dl << std::endl;
    /* get_delta_deg_dl */

    /* get_delta_edges_dl */
    S_b = 0.0, S_a = 0.0;
    int B = blockmodel.getNum_blocks();
    S_b += lbinom((B * B) + graph.num_edges - 1, graph.num_edges);
    B--;
    S_a += lbinom((B * B) + graph.num_edges - 1, graph.num_edges);
    delta_entropy_dl += S_a - S_b;
    std::cout << " | dS_dl c) " << delta_entropy_dl << std::endl;;
    /* get_delta_edges_dl */
    std::cout << "delta_entropy: " << delta_entropy << " delta_entropy_dl: " << delta_entropy_dl << std::endl;
    if (std::isnan(delta_entropy) || std::isinf(delta_entropy) || std::isnan(delta_entropy_dl) || std::isinf(delta_entropy_dl)) {
        exit(-15);
    }
    return delta_entropy + 1.0 * delta_entropy_dl;
}

}  // entropy
