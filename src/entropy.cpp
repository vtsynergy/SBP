#include "entropy.hpp"

#include "cmath"

namespace entropy {

double delta_mdl(int current_block, int proposal, const Blockmodel &blockmodel, int num_edges,
                 EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<int> old_block_row = blockmodel.blockmatrix()->getrow(current_block); // M_r_t1
    std::vector<int> old_proposal_row = blockmodel.blockmatrix()->getrow(proposal);   // M_s_t1
    std::vector<int> old_block_col = blockmodel.blockmatrix()->getcol(current_block); // M_t2_r
    std::vector<int> old_proposal_col = blockmodel.blockmatrix()->getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    std::vector<int> new_block_col = common::exclude_indices(updates.block_col, current_block, proposal); // added
    std::vector<int> new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    std::vector<int> new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block,
                                                                     proposal);
    std::vector<int> old_block_degrees_out = common::exclude_indices(blockmodel.degrees_out(), current_block, proposal);

    // Remove 0 indices
    std::vector<int> new_block_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in,
                                                                      updates.block_row); // added
    std::vector<int> new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in,
                                                                         updates.proposal_row);
    std::vector<int> new_block_row = common::nonzeros(updates.block_row); // added
    std::vector<int> new_proposal_row = common::nonzeros(updates.proposal_row);
    std::vector<int> new_block_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_block_col); // added
    std::vector<int> new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_block_col = common::nonzeros(new_block_col); // added
    new_proposal_col = common::nonzeros(new_proposal_col);

    std::vector<int> old_block_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(), old_block_row);
    std::vector<int> old_proposal_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(), old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    std::vector<int> old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    std::vector<int> old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = common::nonzeros(old_block_col);
    old_proposal_col = common::nonzeros(old_proposal_col);

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(new_block_row, new_block_row_degrees_in,
                                                block_degrees.block_degrees_out[current_block], num_edges); // added
    delta_entropy -= common::delta_entropy_temp(new_proposal_row, new_proposal_row_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    delta_entropy -= common::delta_entropy_temp(new_block_col, new_block_col_degrees_out,
                                                block_degrees.block_degrees_in[current_block], num_edges); // added
    delta_entropy -= common::delta_entropy_temp(new_proposal_col, new_proposal_col_degrees_out,
                                                block_degrees.block_degrees_in[proposal], num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_row, old_block_row_degrees_in,
                                                blockmodel.degrees_out(current_block), num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, old_proposal_row_degrees_in,
                                                blockmodel.degrees_out(proposal), num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_col, old_block_col_degrees_out,
                                                blockmodel.degrees_in(current_block), num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, old_proposal_col_degrees_out,
                                                blockmodel.degrees_in(proposal), num_edges);
    if (std::isnan(delta_entropy)) {
        std::cout << "Error: Dense delta entropy is NaN" << std::endl;
        exit(-142321);
    }
    return delta_entropy;
}

double delta_mdl(int current_block, int proposal, const Blockmodel &blockmodel, int num_edges,
                 SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    const MapVector<int> &old_block_row = matrix->getrow_sparse(current_block); // M_r_t1
    const MapVector<int> &old_proposal_row = matrix->getrow_sparse(proposal);   // M_s_t1
    const MapVector<int> &old_block_col = matrix->getcol_sparse(current_block); // M_t2_r
    const MapVector<int> &old_proposal_col = matrix->getcol_sparse(proposal);   // M_t2_s

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(updates.block_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[current_block], num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.proposal_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.block_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[current_block], current_block, proposal,
                                                num_edges);
    if (std::isnan(delta_entropy)) {
        std::cout << "block_col: ";
        utils::print<int>(updates.block_col);
        std::cout << "_block_degrees_out: ";
        utils::print<int>(block_degrees.block_degrees_out);
        std::cout << "block_degree in: " << block_degrees.block_degrees_in[current_block] << std::endl;
    }
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.proposal_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[proposal], current_block, proposal,
                                                num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_block_row, blockmodel.degrees_in(),
                                                blockmodel.degrees_out(current_block), num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_proposal_row, blockmodel.degrees_in(),
                                                blockmodel.degrees_out(proposal), num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_block_col, blockmodel.degrees_out(),
                                                blockmodel.degrees_in(current_block), current_block,
                                                proposal, num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_proposal_col, blockmodel.degrees_out(),
                                                blockmodel.degrees_in(proposal), current_block, proposal,
                                                num_edges);
    assert(!std::isnan(delta_entropy));
    if (std::isnan(delta_entropy)) {
        std::cerr << "Error: Sparse delta entropy is NaN" << std::endl;
        exit(-142321);
    }
    return delta_entropy;
}

double delta_mdl(const Blockmodel &blockmodel, const Delta &delta, const utils::ProposalAndEdgeCounts &proposal) {
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    double delta_entropy = 0.0;
    int current_block = delta.current_block();
    int proposed_block = delta.proposed_block();
    auto get_deg_in = [&blockmodel, &proposal, current_block, proposed_block](int index) -> float {
        int value = blockmodel.degrees_in(index);
        if (index == current_block)
            value -= proposal.num_in_neighbor_edges;
        else if (index == proposed_block)
            value += proposal.num_in_neighbor_edges;
        return float(value);
    };
    auto get_deg_out = [&blockmodel, &proposal, current_block, proposed_block](int index) -> float {
        int value = blockmodel.degrees_out(index);
        if (index == current_block)
            value -= proposal.num_out_neighbor_edges;
        else if (index == proposed_block)
            value += proposal.num_out_neighbor_edges;
        return float(value);
    };
    for (const std::tuple<int, int, int> &entry: delta.entries()) {
        int row = std::get<0>(entry);
        int col = std::get<1>(entry);
        int change = std::get<2>(entry);
        delta_entropy += common::cell_entropy((float) matrix->get(row, col), (float) blockmodel.degrees_in(col),
                                              (float) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(float(matrix->get(row, col) + change), get_deg_in(col),
                                              get_deg_out(row));
    }
    // Compute change in entropy for cells with no delta
    for (const std::pair<const int, int> &entry: blockmodel.blockmatrix()->getrow_sparse(current_block)) {
        int row = current_block;
        int col = entry.first;
        int value = entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy((float) value, (float) blockmodel.degrees_in(col),
                                              (float) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy((float) value, get_deg_in(col), get_deg_out(row));
    }
    for (const std::pair<const int, int> &entry: blockmodel.blockmatrix()->getrow_sparse(proposed_block)) {
        int row = proposed_block;
        int col = entry.first;
        int value = entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy((float) value, (float) blockmodel.degrees_in(col),
                                              (float) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy((float) value, get_deg_in(col), get_deg_out(row));
    }
    for (const std::pair<const int, int> &entry: blockmodel.blockmatrix()->getcol_sparse(current_block)) {
        int row = entry.first;
        int col = current_block;
        int value = entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy((float) value, (float) blockmodel.degrees_in(col),
                                              (float) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy((float) value, get_deg_in(col), get_deg_out(row));
    }
    for (const std::pair<const int, int> &entry: blockmodel.blockmatrix()->getcol_sparse(proposed_block)) {
        int row = entry.first;
        int col = proposed_block;
        int value = entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy((float) value, (float) blockmodel.degrees_in(col),
                                              (float) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy((float) value, get_deg_in(col), get_deg_out(row));
    }
    return delta_entropy;
}

double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees) {
    if (proposal.num_neighbor_edges == 0) {
        return 1.0;
    }
    // Compute block weights
    std::map<int, int> block_counts;
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int block = out_blocks.indices[i];
        int weight = out_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int block = in_blocks.indices[i];
        int weight = in_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    // Create Arrays using unique blocks
    size_t num_unique_blocks = block_counts.size();
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
    std::vector<int> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<int> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    int index = 0;
    int num_blocks = blockmodel.getNum_blocks();
    const std::vector<int> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry : block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = proposal_row[entry.first] + proposal_col[entry.first] + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = updates.block_row[entry.first] + updates.block_col[entry.first] + 1.0;
        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + num_blocks;
        index++;
    }
    // Compute p_forward and p_backward
    auto p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
    auto p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
    return p_backward / p_forward;
}

double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, SparseEdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees) {
    if (proposal.num_neighbor_edges == 0) {
        return 1.0;
    }
    // Compute block weights
    std::map<int, int> block_counts;
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int block = out_blocks.indices[i];
        int weight = out_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int block = in_blocks.indices[i];
        int weight = in_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    // Create Arrays using unique blocks
    size_t num_unique_blocks = block_counts.size();
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
    std::vector<int> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<int> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    int index = 0;
    int num_blocks = blockmodel.getNum_blocks();
    const std::vector<int> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry : block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = proposal_row[entry.first] + proposal_col[entry.first] + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = updates.block_row[entry.first] + updates.block_col[entry.first] + 1.0;
        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + num_blocks;
        index++;
    }
    // Compute p_forward and p_backward
    auto p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
    auto p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
    return p_backward / p_forward;
}

double hastings_correction(int vertex, const Graph &graph, const Blockmodel &blockmodel, const Delta &delta,
                           int current_block, const utils::ProposalAndEdgeCounts &proposal) {
    if (proposal.num_neighbor_edges == 0) {
        return 1.0;
    }
    // Compute block weights
    MapVector<int> block_counts;
    for (const int neighbor : graph.out_neighbors(vertex)) {
        int neighbor_block = blockmodel.block_assignment(neighbor);
        block_counts[neighbor_block] += 1;
    }
    for (const int neighbor : graph.in_neighbors(vertex)) {
        if (neighbor == vertex) continue;
        int neighbor_block = blockmodel.block_assignment(neighbor);
        block_counts[neighbor_block] += 1;
    }
    // Create Arrays using unique blocks
    size_t num_unique_blocks = block_counts.size();
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
    std::vector<int> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<int> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    int index = 0;
    int num_blocks = blockmodel.getNum_blocks();
    const std::vector<int> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry : block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = proposal_row[entry.first] + proposal_col[entry.first] + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = blockmodel.blockmatrix()->get(current_block, entry.first) +
                               delta.get(current_block, entry.first) +
                               //                get(delta, std::make_pair(current_block, entry.first)) +
                               blockmodel.blockmatrix()->get(entry.first, current_block) +
                               delta.get(entry.first, current_block) + 1.0;
//                get(delta, std::make_pair(entry.first, current_block)) + 1.0;
        int new_block_degree = blockmodel.degrees(entry.first);
        if (entry.first == current_block) {
            int current_block_self_edges = blockmodel.blockmatrix()->get(current_block, current_block)
                                           + delta.get(current_block, current_block);
            int degree_out = blockmodel.degrees_out(current_block) - proposal.num_out_neighbor_edges;
            int degree_in = blockmodel.degrees_in(current_block) - proposal.num_in_neighbor_edges;
            new_block_degree = degree_out + degree_in - current_block_self_edges;
        } else if (entry.first == proposal.proposal) {
            int proposed_block_self_edges = blockmodel.blockmatrix()->get(proposal.proposal, proposal.proposal)
                                            + delta.get(proposal.proposal, proposal.proposal);
            int degree_out = blockmodel.degrees_out(proposal.proposal) + proposal.num_out_neighbor_edges;
            int degree_in = blockmodel.degrees_in(proposal.proposal) + proposal.num_in_neighbor_edges;
            new_block_degree = degree_out + degree_in - proposed_block_self_edges;
        }
//        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + num_blocks;
        proposal_degrees[index] = new_block_degree + num_blocks;
        index++;
    }
    // Compute p_forward and p_backward
    auto p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
    auto p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
    return p_backward / p_forward;
}

double mdl(const Blockmodel &blockmodel, int num_vertices, int num_edges) {
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

}