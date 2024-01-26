#include "entropy.hpp"
#include "fastlgamma.hpp"
#include "spence.hpp"

#include "cmath"

namespace entropy {

double block_merge_delta_mdl(long current_block, long proposal, long num_edges, const Blockmodel &blockmodel,
                             EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<long> old_block_row = blockmodel.blockmatrix()->getrow(current_block); // M_r_t1
    std::vector<long> old_proposal_row = blockmodel.blockmatrix()->getrow(proposal);   // M_s_t1
    std::vector<long> old_block_col = blockmodel.blockmatrix()->getcol(current_block); // M_t2_r
    std::vector<long> old_proposal_col = blockmodel.blockmatrix()->getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    std::vector<long> new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    std::vector<long> new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block,
                                                                     proposal);
    std::vector<long> old_block_degrees_out = common::exclude_indices(blockmodel.degrees_out(),
                                                                     current_block, proposal);

    // Remove 0 indices
    std::vector<long> new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in,
                                                                         updates.proposal_row);
    std::vector<long> new_proposal_row = common::nonzeros(updates.proposal_row);
    std::vector<long> new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_proposal_col = common::nonzeros(new_proposal_col);

    std::vector<long> old_block_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(),
                                                                      old_block_row);
    std::vector<long> old_proposal_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(),
                                                                         old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    std::vector<long> old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    std::vector<long> old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = common::nonzeros(old_block_col);
    old_proposal_col = common::nonzeros(old_proposal_col);

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(new_proposal_row, new_proposal_row_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
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
    return delta_entropy;
}

double block_merge_delta_mdl(long current_block, long proposal, long num_edges, const Blockmodel &blockmodel,
                             SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    const MapVector<long> &old_block_row = matrix->getrow_sparse(current_block); // M_r_t1
    const MapVector<long> &old_proposal_row = matrix->getrow_sparse(proposal);   // M_s_t1
    const MapVector<long> &old_block_col = matrix->getcol_sparse(current_block); // M_t2_r
    const MapVector<long> &old_proposal_col = matrix->getcol_sparse(proposal);   // M_t2_s

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(updates.proposal_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    delta_entropy -= common::delta_entropy_temp(updates.proposal_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[proposal], current_block, proposal,
                                                num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_row, blockmodel.degrees_in(),
                                                blockmodel.degrees_out(current_block), num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, blockmodel.degrees_in(),
                                                blockmodel.degrees_out(proposal), num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_col, blockmodel.degrees_out(),
                                                blockmodel.degrees_in(current_block), current_block,
                                                proposal, num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, blockmodel.degrees_out(),
                                                blockmodel.degrees_in(proposal), current_block, proposal,
                                                num_edges);
    return delta_entropy;
}

double block_merge_delta_mdl(long current_block, const Blockmodel &blockmodel, const Delta &delta,
                             common::NewBlockDegrees &block_degrees) {
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    double delta_entropy = 0.0;
    long proposed_block = delta.proposed_block();
    for (const std::tuple<long, long, long> &entry: delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        long change = std::get<2>(entry);
        // delta += + E(old) - E(new)
        delta_entropy += common::cell_entropy(matrix->get(row, col), blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        if (row == current_block || col == current_block) continue;  // the "new" cell entropy == 0;
        delta_entropy -= common::cell_entropy(matrix->get(row, col) + change, block_degrees.block_degrees_in[col],
                                              block_degrees.block_degrees_out[row]);
    }
    for (const std::pair<const long, long> &entry: blockmodel.blockmatrix()->getrow_sparse(proposed_block)) {
        long row = proposed_block;
        long col = entry.first;
        long value = entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy((double) value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy((double) value, (double) block_degrees.block_degrees_in[col],
                                              (double) block_degrees.block_degrees_out[row]);
    }
    for (const std::pair<const long, long> &entry: blockmodel.blockmatrix()->getcol_sparse(proposed_block)) {
        long row = entry.first;
        long col = proposed_block;
        long value = entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy((double) value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy((double) value, (double) block_degrees.block_degrees_in[col],
                                              (double) block_degrees.block_degrees_out[row]);
    }
    return delta_entropy;
}

double block_merge_delta_mdl(long current_block, utils::ProposalAndEdgeCounts proposal, const Blockmodel &blockmodel,
                             const Delta &delta) {
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    double delta_entropy = 0.0;
    long proposed_block = delta.proposed_block();
    auto get_deg_in = [&blockmodel, &proposal, current_block, proposed_block](long index) -> double {
        long value = blockmodel.degrees_in(index);
        if (index == current_block)
            value -= proposal.num_in_neighbor_edges;
        else if (index == proposed_block)
            value += proposal.num_in_neighbor_edges;
        return double(value);
    };
    auto get_deg_out = [&blockmodel, &proposal, current_block, proposed_block](long index) -> double {
        long value = blockmodel.degrees_out(index);
        if (index == current_block)
            value -= proposal.num_out_neighbor_edges;
        else if (index == proposed_block)
            value += proposal.num_out_neighbor_edges;
        return double(value);
    };
    for (const std::tuple<long, long, long> &entry: delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        auto change = (double) std::get<2>(entry);
        // delta += + E(old) - E(new)
        auto value = (double) matrix->get(row, col);
        delta_entropy += common::cell_entropy(value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        if (row == current_block || col == current_block) continue;  // the "new" cell entropy == 0;
        delta_entropy -= common::cell_entropy(value + change, get_deg_in(col), get_deg_out(row));
    }
    for (const std::pair<long, long> &entry: blockmodel.blockmatrix()->getrow_sparse(proposed_block)) {
        long row = proposed_block;
        long col = entry.first;
        auto value = (double) entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy((double) value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
    }
    for (const std::pair<long, long> &entry: blockmodel.blockmatrix()->getcol_sparse(proposed_block)) {
        long row = entry.first;
        long col = proposed_block;
        auto value = (double) entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy(value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
    }
    return delta_entropy;
}

double delta_mdl(long current_block, long proposal, const Blockmodel &blockmodel, long num_edges,
                 EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<long> old_block_row = blockmodel.blockmatrix()->getrow(current_block); // M_r_t1
    std::vector<long> old_proposal_row = blockmodel.blockmatrix()->getrow(proposal);   // M_s_t1
    std::vector<long> old_block_col = blockmodel.blockmatrix()->getcol(current_block); // M_t2_r
    std::vector<long> old_proposal_col = blockmodel.blockmatrix()->getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    std::vector<long> new_block_col = common::exclude_indices(updates.block_col, current_block, proposal); // added
    std::vector<long> new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    std::vector<long> new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block,
                                                                     proposal);
    std::vector<long> old_block_degrees_out = common::exclude_indices(blockmodel.degrees_out(), current_block, proposal);

    // Remove 0 indices
    std::vector<long> new_block_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in,
                                                                      updates.block_row); // added
    std::vector<long> new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in,
                                                                         updates.proposal_row);
    std::vector<long> new_block_row = common::nonzeros(updates.block_row); // added
    std::vector<long> new_proposal_row = common::nonzeros(updates.proposal_row);
    std::vector<long> new_block_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_block_col); // added
    std::vector<long> new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_block_col = common::nonzeros(new_block_col); // added
    new_proposal_col = common::nonzeros(new_proposal_col);

    std::vector<long> old_block_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(), old_block_row);
    std::vector<long> old_proposal_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(), old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    std::vector<long> old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    std::vector<long> old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
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

double delta_mdl(long current_block, long proposal, const Blockmodel &blockmodel, long num_edges,
                 SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    const MapVector<long> &old_block_row = matrix->getrow_sparseref(current_block); // M_r_t1
    const MapVector<long> &old_proposal_row = matrix->getrow_sparseref(proposal);   // M_s_t1
    const MapVector<long> &old_block_col = matrix->getcol_sparseref(current_block); // M_t2_r
    const MapVector<long> &old_proposal_col = matrix->getcol_sparseref(proposal);   // M_t2_s

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
        utils::print<long>(updates.block_col);
        std::cout << "_block_degrees_out: ";
        utils::print<long>(block_degrees.block_degrees_out);
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
        std::cerr << "ERROR " << "Error: Sparse delta entropy is NaN" << std::endl;
        exit(-142321);
    }
    return delta_entropy;
}

double delta_mdl(const Blockmodel &blockmodel, const Delta &delta, const utils::ProposalAndEdgeCounts &proposal) {
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    double delta_entropy = 0.0;
    long current_block = delta.current_block();
    long proposed_block = delta.proposed_block();
    auto get_deg_in = [&blockmodel, &proposal, &delta, current_block, proposed_block](long index) -> size_t {
        long value = blockmodel.degrees_in(index);
        if (index == current_block)
            value -= (proposal.num_in_neighbor_edges + delta.self_edge_weight());
        else if (index == proposed_block)
            value += (proposal.num_in_neighbor_edges + delta.self_edge_weight());
        return value;
    };
    auto get_deg_out = [&blockmodel, &proposal, current_block, proposed_block](long index) -> size_t {
        long value = blockmodel.degrees_out(index);
        if (index == current_block)
            value -= proposal.num_out_neighbor_edges;
        else if (index == proposed_block)
            value += proposal.num_out_neighbor_edges;
        return value;
    };
    for (const std::tuple<long, long, long> &entry: delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        long change = std::get<2>(entry);
        delta_entropy += common::cell_entropy(matrix->get(row, col), blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << matrix->get(row, col) << " delta: " << change << std::endl;
            utils::print<long>(blockmodel.blockmatrix()->getrow_sparse(row));
            utils::print<long>(blockmodel.blockmatrix()->getcol_sparse(col));
            std::cout << "d_out[row]: " << blockmodel.degrees_out(row) << " d_in[col]: " << blockmodel.degrees_in(row) << std::endl;
            throw std::invalid_argument("nan/inf in bm delta for old bm when delta != 0");
        }
        delta_entropy -= common::cell_entropy(matrix->get(row, col) + change, get_deg_in(col),
                                              get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << matrix->get(row, col) << " delta: " << change;
            std::cout << " current: " << current_block << " proposed: " << proposed_block << std::endl;
            utils::print<long>(blockmodel.blockmatrix()->getrow_sparse(row));
            utils::print<long>(blockmodel.blockmatrix()->getcol_sparse(col));
            std::cout << "d_out[row]: " << blockmodel.degrees_out(row) << " d_in[col]: " << blockmodel.degrees_in(col) << std::endl;
            std::cout << "new d_out[row]: " << get_deg_out(row) << " d_in[col]: " << get_deg_in(col) << std::endl;
            std::cout << "v_out: " << proposal.num_out_neighbor_edges << " v_in: " << proposal.num_in_neighbor_edges << " v_total: " << proposal.num_neighbor_edges << std::endl;
            throw std::invalid_argument("nan/inf in bm delta for new bm when delta != 0");
        }
    }
    // Compute change in entropy for cells with no delta
    for (const auto &entry: blockmodel.blockmatrix()->getrow_sparseref(current_block)) {
        long row = current_block;
        long col = entry.first;
        long value = entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy(value, blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << value << " delta: 0" << std::endl;
            throw std::invalid_argument("nan/inf in bm delta when delta = 0 and row = current block");
        }
    }
    for (const auto &entry: blockmodel.blockmatrix()->getrow_sparseref(proposed_block)) {
        long row = proposed_block;
        long col = entry.first;
        long value = entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy(value, blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << value << " delta: 0" << std::endl;
            throw std::invalid_argument("nan/inf in bm delta when delta = 0 and row = proposed block");
        }
    }
    for (const auto &entry: blockmodel.blockmatrix()->getcol_sparseref(current_block)) {
        long row = entry.first;
        long col = current_block;
        long value = entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy(value, blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << value << " delta: 0" << std::endl;
            throw std::invalid_argument("nan/inf in bm delta when delta = 0 and col = current block");
        }
    }
    for (const auto &entry: blockmodel.blockmatrix()->getcol_sparseref(proposed_block)) {
        long row = entry.first;
        long col = proposed_block;
        long value = entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy(value, blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << value << " delta: 0" << std::endl;
            throw std::invalid_argument("nan/inf in bm delta when delta = 0 and col = proposed block");
        }
    }
    return delta_entropy;
}

double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees) {
    if (proposal.num_neighbor_edges == 0 || args.greedy) {
        return 1.0;
    }
    // Compute block weights
    std::map<long, long> block_counts;
    for (ulong i = 0; i < out_blocks.indices.size(); ++i) {
        long block = out_blocks.indices[i];
        long weight = out_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    for (ulong i = 0; i < in_blocks.indices.size(); ++i) {
        long block = in_blocks.indices[i];
        long weight = in_blocks.values[i];
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
    std::vector<long> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<long> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    long index = 0;
    long num_blocks = blockmodel.getNum_blocks();
    const std::vector<long> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry: block_counts) {
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
    if (proposal.num_neighbor_edges == 0 || args.greedy) {
        return 1.0;
    }
    // Compute block weights
    std::map<long, long> block_counts;
    for (ulong i = 0; i < out_blocks.indices.size(); ++i) {
        long block = out_blocks.indices[i];
        long weight = out_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    for (ulong i = 0; i < in_blocks.indices.size(); ++i) {
        long block = in_blocks.indices[i];
        long weight = in_blocks.values[i];
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
    std::vector<long> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<long> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    long index = 0;
    long num_blocks = blockmodel.getNum_blocks();
    const std::vector<long> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry: block_counts) {
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

double hastings_correction(long vertex, const Graph &graph, const Blockmodel &blockmodel, const Delta &delta,
                           long current_block, const utils::ProposalAndEdgeCounts &proposal) {
    if (proposal.num_neighbor_edges == 0 || args.greedy) {  // No correction needed with greedy proposals
        return 1.0;
    }
    // Compute block weights
    MapVector<long> block_counts;
    for (const long neighbor: graph.out_neighbors(vertex)) {
        long neighbor_block = blockmodel.block_assignment(neighbor);
        block_counts[neighbor_block] += 1;
    }
    for (const long neighbor: graph.in_neighbors(vertex)) {
        if (neighbor == vertex) continue;
        long neighbor_block = blockmodel.block_assignment(neighbor);
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
//    std::vector<long> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
//    std::vector<long> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    const MapVector<long> &proposal_row = blockmodel.blockmatrix()->getrow_sparseref(proposal.proposal);
    const MapVector<long> &proposal_col = blockmodel.blockmatrix()->getcol_sparseref(proposal.proposal);
    // Fill Arrays
    long index = 0;
    long num_blocks = blockmodel.getNum_blocks();
    const std::vector<long> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry: block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = map_vector::get(proposal_row, entry.first) + map_vector::get(proposal_col, entry.first) + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = blockmodel.blockmatrix()->get(current_block, entry.first) +
                               delta.get(current_block, entry.first) +
                               //                get(delta, std::make_pair(current_block, entry.first)) +
                               blockmodel.blockmatrix()->get(entry.first, current_block) +
                               delta.get(entry.first, current_block) + 1.0;
//                get(delta, std::make_pair(entry.first, current_block)) + 1.0;
        long new_block_degree = blockmodel.degrees(entry.first);
        if (entry.first == current_block) {
            long current_block_self_edges = blockmodel.blockmatrix()->get(current_block, current_block)
                                           + delta.get(current_block, current_block);
            long degree_out = blockmodel.degrees_out(current_block) - proposal.num_out_neighbor_edges;
            long degree_in = blockmodel.degrees_in(current_block) - proposal.num_in_neighbor_edges;
            new_block_degree = degree_out + degree_in - current_block_self_edges;
        } else if (entry.first == proposal.proposal) {
            long proposed_block_self_edges = blockmodel.blockmatrix()->get(proposal.proposal, proposal.proposal)
                                            + delta.get(proposal.proposal, proposal.proposal);
            long degree_out = blockmodel.degrees_out(proposal.proposal) + proposal.num_out_neighbor_edges;
            long degree_in = blockmodel.degrees_in(proposal.proposal) + proposal.num_in_neighbor_edges;
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

double normalize_mdl_v1(double mdl, long num_edges) {
    return mdl / null_mdl_v1(num_edges);
}

double normalize_mdl_v2(double mdl, long num_vertices, long num_edges) {
    return mdl / null_mdl_v2(num_vertices, num_edges);
}

double null_mdl_v1(long num_edges) {
    // TODO: not sure how this works in nonparametric version
    double log_posterior_p = num_edges * log(1.0 / num_edges);
    double x = 1.0 / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    std::cout << "log posterior = " << log_posterior_p << " blockmodel = " << (num_edges * h) << std::endl;
    return (num_edges * h) - log_posterior_p;
}

double null_mdl_v2(long num_vertices, long num_edges) {
    // TODO: not sure how this works in nonparametric version
    double log_posterior_p = num_edges * log(1.0 / num_edges);
    // done calculating log_posterior_probability
    double x = pow(num_vertices, 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    std::cout << "log posterior = " << log_posterior_p << " blockmodel = " << (num_edges * h) + (num_vertices * log(num_vertices)) << std::endl;
    return (num_edges * h) + (num_vertices * log(num_vertices)) - log_posterior_p;
}

double mdl(const Blockmodel &blockmodel, const Graph &graph) {
    if (args.nonparametric)
        return nonparametric::mdl(blockmodel, graph);
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / graph.num_edges();
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (graph.num_edges() * h) + (graph.num_vertices() * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

namespace dist {

double mdl(const TwoHopBlockmodel &blockmodel, long num_vertices, long num_edges) {
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

}  // namespace dist

namespace nonparametric {

//inline double eterm_exact(long source, long destination, long weight) {
//    double val = fastlgamma(weight + 1);
//
//    if (args.undirected && source == destination) {
//        double log_2 = log(2);
//        return -val - weight * log_2;
//    } else {
//        return -val;
//    }
//}
//
//inline double vterm_exact(long out_degree, long in_degree) { // out_degree, in_degree, wr=size of community, true? meh?
////    if (deg_corr)
////    {
////    if constexpr (is_directed_::apply<Graph>::type::value)
////        return fastlgamma(out_degree + 1) + fastlgamma(in_degree + 1);
//    if (args.undirected)
//        return fastlgamma(out_degree + 1);
//    return fastlgamma(out_degree + 1) + fastlgamma(in_degree + 1);
////    }
////    else
////    {
////        if constexpr (is_directed_::apply<Graph>::type::value)
////            return (out_degree + in_degree) * safelog_fast(wr);
////        else
////            return out_degree * safelog_fast(wr);
////    }
//}

double get_deg_entropy(const Graph &graph, long vertex) {  // , const simple_degs_t&) {
    long k_in = (long) graph.in_neighbors(vertex).size();
    long k_out = (long) graph.out_neighbors(vertex).size();
//    auto kin = in_degreeS()(v, _g, _eweight);
//    auto kout = out_degreeS()(v, _g, _eweight);
    // vertices are unweighted, so no need to return S * weight(vertex);
    return -fastlgamma(k_in + 1) - fastlgamma(k_out + 1);
//    double S = -lgamma_fast(kin + 1) - lgamma_fast(kout + 1);
//    return S * _vweight[v];
}

double sparse_entropy(const Blockmodel &blockmodel, const Graph &graph) {
    double S = 0;

//    for (auto e : edges_range(_bg))
//        S += eterm_exact(source(e, _bg), target(e, _bg), _mrs[e], _bg);
//    for (auto v : vertices_range(_bg))
//        S += vterm_exact(_mrp[v], _mrm[v], _wr[v], _deg_corr, _bg);

    for (const auto &edge : blockmodel.blockmatrix()->entries()) {
        long source = std::get<0>(edge);
        long destination = std::get<1>(edge);
        long weight = std::get<2>(edge);
        S += eterm_exact(source, destination, weight);
    }

    for (long block = 0; block < blockmodel.getNum_blocks(); ++block) {
        S += vterm_exact(blockmodel.degrees_out(block), blockmodel.degrees_in(block));
    }

    // In distributed case, we would only compute these for vertices we're responsible for. Since it's a simple addition, we can do an allreduce.
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        S += get_deg_entropy(graph, vertex);
    }
//    for (auto v : vertices_range(_g))
//        S += get_deg_entropy(v, _degs);

//    if (multigraph)
//        S += get_parallel_entropy();

    return S;
}

//inline double fastlbinom(long N, long k) {
//    if (N == 0 || k == 0 || k > N)
//        return 0;
//    return ((fastlgamma(N + 1) - fastlgamma(k + 1)) - fastlgamma(N - k + 1));
//}

double get_partition_dl(long N, const Blockmodel &blockmodel) { // _N = number of vertices, _actual_B = nonzero blocks, _total = vector of block sizes
    double S = 0;
    S += fastlbinom(N - 1, blockmodel.num_nonempty_blocks() - 1);
    S += fastlgamma(N + 1);
    for (const long &block_size : blockmodel.block_sizes())
        S -= fastlgamma(block_size + 1);
    S += fastlog(N);
    return S;
}

//double get_deg_dl(const Blockmodel &blockmodel) {  // kind = dist
//    double S = 0;
//    for (int block = 0; block < blockmodel.getNum_blocks(); ++block) {
//
//    }
////    double S = 0;
////    for (auto& ps : _partition_stats)
////        S += ps
//    return get_deg_dl_dist(boost::counting_range(size_t(0), _total_B), std::array<std::pair<size_t,size_t>,0>());
////    return S;
//}

//double get_deg_dl(int kind) {  // intermediate call
//    return get_deg_dl_dist(kind, boost::counting_range(size_t(0), _total_B),
//                           std::array<std::pair<size_t,size_t>,0>());
//}

/// No idea what this function does. See int_part.cc in https://git.skewed.de/count0/graph-tool
double get_v(double u, double epsilon) {
    double v = u;
    double delta = 1;
    while (delta > epsilon) {
        // spence(exp(v)) = -spence(exp(-v)) - (v*v)/2
        double n_v = u * sqrt(spence(exp(-v)));
        delta = abs(n_v - v);
        v = n_v;
    }
    return v;
}

double log_q_approx_small(size_t n, size_t k) {
    return fastlbinom(n - 1, k - 1) - fastlgamma(k + 1);
}

/// Computes the number of restricted of integer n into at most m parts. This is part of teh prior for the
/// degree-corrected SBM.
/// TO-DO: the current function contains only the approximation of log_q. If it becomes a bottleneck, you'll want to
/// compute a cache of log_q(n, m) for ~20k n and maybe a few hundred m? I feel like for larger graphs, the cache
/// will be a waste of time.
/// See int_part.cc in https://git.skewed.de/count0/graph-tool
double log_q(size_t n, size_t k) {
    if (k < pow(n, 1/4.))
        return log_q_approx_small(n, k);
    double u = k / sqrt(n);
    double v = get_v(u);
    double lf = log(v) - log1p(- exp(-v) * (1 + u * u/2)) / 2 - log(2) * 3 / 2.
                - log(u) - log(M_PI);
    double g = 2 * v / u - u * log1p(-exp(-v));
    return lf - log(n) + sqrt(n) * g;
}

double get_deg_dl_dist(const Blockmodel &blockmodel) { // Rs&& rs, Ks&& ks) {  // RS: range from 0 to B, KS is an empty array of pairs?
    double S = 0;
//    for (auto r : rs) {
    for (int block = 0; block < blockmodel.getNum_blocks(); ++block) {
//        r = get_r(r);
//        S += log_q(_ep[r], _total[r]);  // _ep[r] = in/out degree of r, _total[r] = block size of r (?)
//        S += log_q(_em[r], _total[r]);
        S += log_q(blockmodel.degrees_out(block), blockmodel.block_size(block));
        S += log_q(blockmodel.degrees_in(block), blockmodel.block_size(block));

        size_t total = 0;
        if (!args.undirected) {
            for (const std::pair<long, long> &entry : blockmodel.in_degree_histogram(block)) {
                S -= fastlgamma(entry.second + 1);
            }
        }
        for (const std::pair<long, long> &entry : blockmodel.out_degree_histogram(block)) {
            S -= fastlgamma(entry.second + 1);
            total += entry.second;
        }

        if (args.undirected) {
            S += fastlgamma(total + 1);
        } else {
            S += 2 * fastlgamma(total + 1);
        }
//        if (ks.empty()) {
//            if (_directed) {
//                for (auto& k_c : get_hist<false, false>(r))
//                    S -= lgamma_fast(k_c.second + 1);
//            }
//
//            for (auto& k_c : get_hist<true, false>(r)) {
//                S -= lgamma_fast(k_c.second + 1);
//                total += k_c.second;
//            }
//        } else {
//            auto& h_out = get_hist<true, false>(r);
//            auto& h_in = (_directed) ? get_hist<false, false>(r) : h_out;
//
//            for (auto& k : ks) {
//                if (_directed) {
//                    auto iter = h_in.find(get<0>(k));
//                    auto k_c = (iter != h_in.end()) ? iter->second : 0;
//                    S -= lgamma_fast(k_c + 1);
//                }
//
//                auto iter = h_out.find(get<1>(k));
//                auto k_c = (iter != h_out.end()) ? iter->second : 0;
//                S -= lgamma_fast(k_c + 1);
//            }
//            total = _total[r];
//        }

//        if (_directed)
//            S += 2 * lgamma_fast(total + 1);
//        else
//            S += lgamma_fast(total + 1);
    }
    return S;
}

double get_edges_dl(size_t B, size_t E) {
    size_t NB = !args.undirected ? B * B : (B * (B + 1)) / 2;
    return fastlbinom(NB + E - 1, E);
}

double mdl(const Blockmodel &blockmodel, const Graph &graph) {
    double S = 0, S_dl = 0;

    S = sparse_entropy(blockmodel, graph);

//    if (ea.partition_dl)
    S_dl += get_partition_dl(graph.num_vertices(), blockmodel);

//    if (_deg_corr && ea.degree_dl)
    S_dl += get_deg_dl_dist(blockmodel);  // (ea.degree_dl_kind);

//    if (ea.edges_dl)
//    {
//        size_t actual_B = 0;
//        for (auto& ps : _partition_stats)  // looks like ps.get_actual_B() is the number of nonempty blocks
//            actual_B += ps.get_actual_B();
    S_dl += get_edges_dl(blockmodel.num_nonempty_blocks(), graph.num_edges());
//    }

//    if (ea.recs)  // recs is for weighted graphs, so it looks like we can ignore this. Yay!
//    {
//        auto rdS = rec_entropy(*this, ea);
//        S += get<0>(rdS);
//        S_dl += get<1>(rdS);
//    }

    return S + S_dl * BETA_DL;
}

}

}  // namespace entropy
