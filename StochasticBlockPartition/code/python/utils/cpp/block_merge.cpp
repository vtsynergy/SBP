#include "block_merge.hpp"

Partition &block_merge::merge_blocks(Partition &partition, std::vector<Matrix2Column> &out_neighbors) {
    // add block merge timings to evaluation
    int num_blocks = partition.getNum_blocks();
    Vector best_merge_for_each_block = Vector::Constant(num_blocks, -1);
    Eigen::VectorXd delta_entropy_for_each_block =
        Eigen::VectorXd::Constant(num_blocks, std::numeric_limits<double>::max());
    Vector block_partition = Vector::LinSpaced(num_blocks, 0, num_blocks - 1);

    #pragma omp parallel for
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        // if (current_block == 0) {
        //     std::cout << "Total number of threads = "<< omp_get_num_threads() << std::endl;
        // }
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            ProposalEvaluation proposal = propose_merge(current_block, partition, block_partition);
            if (proposal.delta_entropy < delta_entropy_for_each_block[current_block]) {
                best_merge_for_each_block[current_block] = proposal.proposed_block;
                delta_entropy_for_each_block[current_block] = proposal.delta_entropy;
            }
        }
    }

    partition.carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block);
    partition.initialize_edge_counts(out_neighbors);

    return partition;
}

block_merge::ProposalEvaluation block_merge::propose_merge(int current_block, Partition &partition,
                                                           Vector &block_partition) {
    EdgeWeights out_blocks = partition.getBlockmodel().outgoing_edges(current_block);
    EdgeWeights in_blocks = partition.getBlockmodel().incoming_edges(current_block);
    common::ProposalAndEdgeCounts proposal =
        common::propose_new_block(current_block, out_blocks, in_blocks, block_partition, partition, true);
    EdgeCountUpdates updates =
        edge_count_updates(partition.getBlockmodel(), current_block, proposal.proposal, out_blocks, in_blocks);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(current_block, partition, proposal);
    double delta_entropy =
        compute_delta_entropy(current_block, proposal.proposal, partition, updates, new_block_degrees);
    return ProposalEvaluation{proposal.proposal, delta_entropy};
}

double block_merge::compute_delta_entropy(int current_block, int proposal, Partition &partition,
                                          EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    Vector old_block_row = partition.getBlockmodel().getrow(current_block); // M_r_t1
    Vector old_proposal_row = partition.getBlockmodel().getrow(proposal);   // M_s_t1
    Vector old_block_col = partition.getBlockmodel().getcol(current_block); // M_t2_r
    Vector old_proposal_col = partition.getBlockmodel().getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    Vector new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    Vector new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block, proposal);
    Vector old_block_degrees_out = common::exclude_indices(partition.getBlock_degrees_out(), current_block, proposal);

    // Remove 0 indices
    Vector new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in, updates.proposal_row);
    Vector new_proposal_row = common::nonzeros(updates.proposal_row);
    Vector new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_proposal_col = common::nonzeros(new_proposal_col);

    Vector old_block_row_degrees_in = common::index_nonzero(partition.getBlock_degrees_in(), old_block_row);
    Vector old_proposal_row_degrees_in = common::index_nonzero(partition.getBlock_degrees_in(), old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    Vector old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    Vector old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = common::nonzeros(old_block_col);
    old_proposal_col = common::nonzeros(old_proposal_col);

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(new_proposal_row, new_proposal_row_degrees_in,
                                                block_degrees.block_degrees_out[proposal]);
    delta_entropy -= common::delta_entropy_temp(new_proposal_col, new_proposal_col_degrees_out,
                                                block_degrees.block_degrees_in[proposal]);
    delta_entropy += common::delta_entropy_temp(old_block_row, old_block_row_degrees_in,
                                                partition.getBlock_degrees_out()[current_block]);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, old_proposal_row_degrees_in,
                                                partition.getBlock_degrees_out()[proposal]);
    delta_entropy += common::delta_entropy_temp(old_block_col, old_block_col_degrees_out,
                                                partition.getBlock_degrees_in()[current_block]);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, old_proposal_col_degrees_out,
                                                partition.getBlock_degrees_in()[proposal]);
    return delta_entropy;
}

EdgeCountUpdates block_merge::edge_count_updates(BoostMappedMatrix &blockmodel, int current_block, int proposed_block,
                                                 EdgeWeights &out_blocks, EdgeWeights &in_blocks) {
    Vector proposal_row = blockmodel.getrow(proposed_block);
    Vector proposal_col = blockmodel.getcol(proposed_block);
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
    return EdgeCountUpdates{Vector::Zero(0), proposal_row, Vector::Zero(0), proposal_col};
}
