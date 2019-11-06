#include "sbp.hpp"
// #include "boost_mapped_matrix.hpp"

Partition sbp::merge_blocks(Partition partition, int num_agg_proposals_per_block, std::vector<Matrix2Column> out_neighbors) {
    // add block merge timings to evaluation
    int num_blocks = partition.getNum_blocks();
    Vector best_merge_for_each_block = Vector::Constant(num_blocks, -1);
    Eigen::VectorXd delta_entropy_for_each_block = Eigen::VectorXd::Constant(num_blocks, std::numeric_limits<double>::max());
    Vector block_partition = Vector::LinSpaced(1, 0, num_blocks - 1);

    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        for (int i = 0; i < num_agg_proposals_per_block; ++i) {
            // propose_merge
            // if delta_entropy < recorded delta_entropy
            //   best_merge_for_each_block[current_block] = current_block
            //   delta_entropy_for_each_block[current_block] = delta_entropy
        }
    }

    // carry_out_best_merges
    // initialize edge counts

    return partition;
}

sbp::ProposalEvaluation sbp::propose_merge(int current_block, Partition partition, Vector block_partition) {
    std::pair<std::vector<int>, std::vector<int>> out_blocks = partition.blockmodel.outgoing_edges();
    std::pair<std::vector<int>, std::vector<int>> in_blocks = partition.blockmodel.incoming_edges();
    // propose new block
    // block merge edge count updates
    // compute_new_block_degrees
    // compute_delta_entropy
    // return ProposalEvaluation { proposal, delta_entropy };
}

sbp::ProposalAndEdgeCounts sbp::propose_new_block(int current_block, EdgeWeights out_blocks, EdgeWeights in_blocks, Vector block_partition, Partition partition) {
    std::vector<int> neighbor_indices = util::concatenate<int>(out_blocks.first, in_blocks.first);
    std::vector<int> neighbor_weights = util::concatenate<int>(out_blocks.second, in_blocks.second);
    int k_out = std::accumulate(out_blocks.second.begin(), out_blocks.second.end(), 0);
    int k_in = std::accumulate(in_blocks.second.begin(), in_blocks.second.end(), 0);
    int k = k_out + k_in;
    if (k == 0) {
        int proposal = propose_random_block(current_block, partition.num_blocks);
        return ProposalAndEdgeCounts {proposal, k_out, k_in, k};
    }
}

int sbp::propose_random_block(int current_block, int num_blocks) {
    int proposed = std::rand() % (num_blocks - 1);
    if (proposed >= current_block) {
        proposed++;
    }
    return proposed;
}
