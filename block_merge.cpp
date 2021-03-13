#include "block_merge.hpp"

namespace block_merge {

void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
                                    const std::vector<int> &best_merge_for_each_block, const Graph &graph) {
    // The following code is modeled after the `merge_sweep` function in
    // https://git.skewed.de/count0/graph-tool/-/blob/master/src/graph/inference/loops/merge_loop.hh
    typedef std::tuple<int, int, double> merge_t;
    auto cmp_fxn = [](merge_t left, merge_t right) { return std::get<2>(left) > std::get<2>(right); };
    std::priority_queue<merge_t, std::vector<merge_t>, decltype(cmp_fxn)> queue(cmp_fxn);
    for (int i = 0; i < delta_entropy_for_each_block.size(); ++i)
        queue.push(std::make_tuple(i, best_merge_for_each_block[i], delta_entropy_for_each_block[i]));
    double delta_entropy = 0.0;
    int num_merged = 0;
    // Block map is here so that, if you move merge block A to block C, and then block B to block A, all three blocks
    // end up with the same block assignment (C)
    std::vector<int> block_map = utils::range<int>(0, blockmodel.getNum_blocks());
    while (num_merged < blockmodel.getNum_blocks_to_merge() && !queue.empty()) {
        merge_t merge = queue.top();
        queue.pop();
        int merge_from = std::get<0>(merge);
        int merge_to = block_map[std::get<1>(merge)];
        int delta_entropy_hint = std::get<2>(merge);
        if (merge_from != merge_to) {
            // Calculate the delta entropy given the current block assignment
            EdgeWeights out_blocks = blockmodel.getBlockmodel().outgoing_edges(merge_from);
            EdgeWeights in_blocks = blockmodel.getBlockmodel().incoming_edges(merge_from);
            int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
            int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
            int k = k_out + k_in;
            common::ProposalAndEdgeCounts proposal { merge_to, k_out, k_in, k };
            SparseEdgeCountUpdates updates;
            edge_count_updates_sparse(blockmodel.getBlockmodel(), merge_from, proposal.proposal, out_blocks, in_blocks,
                                      updates);
            common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(merge_from, blockmodel,
                                                                                          proposal);
            double delta_entropy_actual =
                compute_delta_entropy_sparse(merge_from, proposal.proposal, blockmodel, updates, new_block_degrees);
            // If the actual change in entropy is more positive (greater) than anticipated, put it back in queue
            if (!queue.empty() && delta_entropy_actual > std::get<2>(queue.top())) {
                std::get<2>(merge) = delta_entropy_actual;
                queue.push(merge);
                continue;
            }
            // Perform the merge
            // 1. Update the assignment
            for (int i = 0; i < block_map.size(); ++i) {
                int block = block_map[i];
                if (block == merge_from) {
                    block_map[i] = merge_to;
                }
            }
            blockmodel.merge_blocks(merge_from, merge_to, graph);
            // 2. Update the matrix
            // NOTE: if getting funky results, try replacing the following with a matrix rebuild
            blockmodel.getBlockmodel().clearrow(merge_from);
            blockmodel.getBlockmodel().clearcol(merge_from);
            blockmodel.getBlockmodel().setrow(merge_to, updates.proposal_row);
            blockmodel.getBlockmodel().setcol(merge_to, updates.proposal_col);
            blockmodel.setBlock_degrees_out(new_block_degrees.block_degrees_out);
            blockmodel.setBlock_degrees_in(new_block_degrees.block_degrees_in);
            blockmodel.setBlock_degrees(new_block_degrees.block_degrees);
            num_merged++;
        }
    }
    std::vector<int> mapping = blockmodel.build_mapping(blockmodel.getBlock_assignment());
    for (int i = 0; i < blockmodel.getBlock_assignment().size(); ++i) {
        int block = blockmodel.getBlock_assignment()[i];
        int new_block = mapping[block];
        blockmodel.getBlock_assignment()[i] = new_block;
    }
    blockmodel.setNum_blocks(blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge());
}

Blockmodel &merge_blocks(Blockmodel &blockmodel, const Graph &graph, Args &args) {
    // TODO: add block merge timings to evaluation
    int num_blocks = blockmodel.getNum_blocks();
    std::vector<int> best_merge_for_each_block = utils::constant<int>(num_blocks, -1);
    std::vector<double> delta_entropy_for_each_block =
        utils::constant<double>(num_blocks, std::numeric_limits<double>::max());
    std::vector<int> block_assignment = utils::range<int>(0, num_blocks);
    // TODO: keep track of already proposed merges, do not re-process those
    int num_avoided = 0;  // number of avoided/skipped calculations
    #pragma omp parallel for schedule(dynamic) reduction( + : num_avoided)
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        std::unordered_map<int, bool> past_proposals;
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            ProposalEvaluation proposal = propose_merge_sparse(current_block, blockmodel, block_assignment,
                                                               past_proposals);
            if (proposal.delta_entropy == std::numeric_limits<double>::max()) num_avoided++;
            if (proposal.delta_entropy < delta_entropy_for_each_block[current_block]) {
                best_merge_for_each_block[current_block] = proposal.proposed_block;
                delta_entropy_for_each_block[current_block] = proposal.delta_entropy;
            }
        }
    }
    std::cout << "Avoided " << num_avoided << " / " << NUM_AGG_PROPOSALS_PER_BLOCK * num_blocks << " comparisons." << std::endl;
    if (args.approximate)
        blockmodel.carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block, graph);
    else
        carry_out_best_merges_advanced(blockmodel, delta_entropy_for_each_block, best_merge_for_each_block, graph);
    blockmodel.initialize_edge_counts(graph);
    return blockmodel;
}

ProposalEvaluation propose_merge(int current_block, Blockmodel &blockmodel, std::vector<int> &block_blockmodel) {
    EdgeWeights out_blocks = blockmodel.getBlockmodel().outgoing_edges(current_block);
    EdgeWeights in_blocks = blockmodel.getBlockmodel().incoming_edges(current_block);
    common::ProposalAndEdgeCounts proposal =
        common::propose_new_block(current_block, out_blocks, in_blocks, block_blockmodel, blockmodel, true);
    EdgeCountUpdates updates =
        edge_count_updates(blockmodel.getBlockmodel(), current_block, proposal.proposal, out_blocks, in_blocks);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(current_block, blockmodel, proposal);
    double delta_entropy =
        compute_delta_entropy(current_block, proposal.proposal, blockmodel, updates, new_block_degrees);
    // std::cout << "dE: " << delta_entropy << std::endl;
    return ProposalEvaluation{proposal.proposal, delta_entropy};
}

ProposalEvaluation propose_merge_sparse(int current_block, Blockmodel &blockmodel, std::vector<int> &block_blockmodel,
                                        std::unordered_map<int, bool> &past_proposals) {
    EdgeWeights out_blocks = blockmodel.getBlockmodel().outgoing_edges(current_block);
    EdgeWeights in_blocks = blockmodel.getBlockmodel().incoming_edges(current_block);
    common::ProposalAndEdgeCounts proposal =
        common::propose_new_block(current_block, out_blocks, in_blocks, block_blockmodel, blockmodel, true);
    if (past_proposals[proposal.proposal] == true)
        return ProposalEvaluation{ proposal.proposal, std::numeric_limits<double>::max() };
    SparseEdgeCountUpdates updates;
    edge_count_updates_sparse(blockmodel.getBlockmodel(), current_block, proposal.proposal, out_blocks, in_blocks,
                              updates);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(current_block, blockmodel, proposal);
    double delta_entropy =
        compute_delta_entropy_sparse(current_block, proposal.proposal, blockmodel, updates, new_block_degrees);
    past_proposals[proposal.proposal] = true;
    return ProposalEvaluation{proposal.proposal, delta_entropy};
}

double compute_delta_entropy(int current_block, int proposal, Blockmodel &blockmodel, EdgeCountUpdates &updates,
                             common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<int> old_block_row = blockmodel.getBlockmodel().getrow(current_block); // M_r_t1
    std::vector<int> old_proposal_row = blockmodel.getBlockmodel().getrow(proposal);   // M_s_t1
    std::vector<int> old_block_col = blockmodel.getBlockmodel().getcol(current_block); // M_t2_r
    std::vector<int> old_proposal_col = blockmodel.getBlockmodel().getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    std::vector<int> new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    std::vector<int> new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block, proposal);
    std::vector<int> old_block_degrees_out = common::exclude_indices(blockmodel.getBlock_degrees_out(), current_block, proposal);

    // Remove 0 indices
    std::vector<int> new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in, updates.proposal_row);
    std::vector<int> new_proposal_row = common::nonzeros(updates.proposal_row);
    std::vector<int> new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_proposal_col = common::nonzeros(new_proposal_col);

    std::vector<int> old_block_row_degrees_in = common::index_nonzero(blockmodel.getBlock_degrees_in(), old_block_row);
    std::vector<int> old_proposal_row_degrees_in = common::index_nonzero(blockmodel.getBlock_degrees_in(), old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    std::vector<int> old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    std::vector<int> old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = common::nonzeros(old_block_col);
    old_proposal_col = common::nonzeros(old_proposal_col);

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(new_proposal_row, new_proposal_row_degrees_in,
                                                block_degrees.block_degrees_out[proposal]);
    delta_entropy -= common::delta_entropy_temp(new_proposal_col, new_proposal_col_degrees_out,
                                                block_degrees.block_degrees_in[proposal]);
    delta_entropy += common::delta_entropy_temp(old_block_row, old_block_row_degrees_in,
                                                blockmodel.getBlock_degrees_out()[current_block]);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, old_proposal_row_degrees_in,
                                                blockmodel.getBlock_degrees_out()[proposal]);
    delta_entropy += common::delta_entropy_temp(old_block_col, old_block_col_degrees_out,
                                                blockmodel.getBlock_degrees_in()[current_block]);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, old_proposal_col_degrees_out,
                                                blockmodel.getBlock_degrees_in()[proposal]);
    return delta_entropy;
}

double compute_delta_entropy_sparse(int current_block, int proposal, Blockmodel &blockmodel,
                                    SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    const DictTransposeMatrix &matrix = blockmodel.getBlockmodel();
    const MapVector<int> &old_block_row = matrix.getrow_sparse(current_block); // M_r_t1
    const MapVector<int> &old_proposal_row = matrix.getrow_sparse(proposal);   // M_s_t1
    const MapVector<int> &old_block_col = matrix.getcol_sparse(current_block); // M_t2_r
    const MapVector<int> &old_proposal_col = matrix.getcol_sparse(proposal);   // M_t2_s

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(updates.proposal_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[proposal]);
    delta_entropy -= common::delta_entropy_temp(updates.proposal_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[proposal], current_block, proposal);
    delta_entropy += common::delta_entropy_temp(old_block_row, blockmodel.getBlock_degrees_in(),
                                                blockmodel.getBlock_degrees_out()[current_block]);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, blockmodel.getBlock_degrees_in(),
                                                blockmodel.getBlock_degrees_out()[proposal]);
    delta_entropy += common::delta_entropy_temp(old_block_col, blockmodel.getBlock_degrees_out(),
                                                blockmodel.getBlock_degrees_in()[current_block], current_block, proposal);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, blockmodel.getBlock_degrees_out(),
                                                blockmodel.getBlock_degrees_in()[proposal], current_block, proposal);
    return delta_entropy;
}

EdgeCountUpdates edge_count_updates(DictTransposeMatrix &blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks) {
    // TODO: these are copy constructors, can we safely get rid of them?
    std::vector<int> proposal_row = blockmodel.getrow(proposed_block);
    std::vector<int> proposal_col = blockmodel.getcol(proposed_block);
    int count_self = blockmodel.get(current_block, current_block);
    int count_in = count_self, count_out = count_self;
    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int index = in_blocks.indices[i];
        int value = in_blocks.values[i];
        if (index == proposed_block) {
            count_in += value;
        }
        proposal_col[index] += value;
    }
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int index = out_blocks.indices[i];
        int value = out_blocks.values[i];
        if (index == proposed_block) {
            count_out += value;
        }
        proposal_row[index] += value;
    }
    proposal_row[current_block] -= count_in;
    proposal_row[proposed_block] += count_in;
    proposal_col[current_block] -= count_out;
    proposal_col[proposed_block] += count_out;
    return EdgeCountUpdates{std::vector<int>(), proposal_row, std::vector<int>(), proposal_col};
}

void edge_count_updates_sparse(DictTransposeMatrix &blockmodel, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates) {
    // TODO: these are copy constructors, can we safely get rid of them?
    updates.proposal_row = blockmodel.getrow_sparse(proposed_block);
    updates.proposal_col = blockmodel.getcol_sparse(proposed_block);
    int count_self = blockmodel.get(current_block, current_block);
    int count_in = count_self, count_out = count_self;
    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int index = in_blocks.indices[i];
        int value = in_blocks.values[i];
        if (index == proposed_block) {
            count_in += value;
        }
        updates.proposal_col[index] += value;
    }
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int index = out_blocks.indices[i];
        int value = out_blocks.values[i];
        if (index == proposed_block) {
            count_out += value;
        }
        updates.proposal_row[index] += value;
    }
    updates.proposal_row[current_block] -= count_in;
    updates.proposal_row[proposed_block] += count_in;
    updates.proposal_col[current_block] -= count_out;
    updates.proposal_col[proposed_block] += count_out;
}

}  // namespace block_merge
