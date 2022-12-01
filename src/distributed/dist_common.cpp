#include "distributed/dist_common.hpp"

#include "common.hpp"

namespace common::dist {

// TODO: get rid of block_assignment, just use blockmodel?
utils::ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<int> &block_assignment,
                                               const TwoHopBlockmodel &blockmodel, bool block_merge) {
    std::vector<int> neighbor_indices = utils::concatenate<int>(out_blocks.indices, in_blocks.indices);
    std::vector<int> neighbor_weights = utils::concatenate<int>(out_blocks.values, in_blocks.values);
    int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    int k = k_out + k_in;
    int num_blocks = blockmodel.getNum_blocks();

    if (k == 0) { // If the current block has no neighbors, propose merge with random block
        std::vector<int> blocks = utils::range<int>(0, blockmodel.getNum_blocks());
        std::vector<int> weights = utils::to_int<bool>(blockmodel.in_two_hop_radius());
        int proposal = choose_neighbor(blocks, weights);
        // int proposal = propose_random_block(current_block, num_blocks);  // TODO: only propose blocks in 2 hop radius
        assert(blockmodel.stores(proposal));
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    int neighbor_block;
    if (block_merge)
        neighbor_block = choose_neighbor(neighbor_indices, neighbor_weights);
    else {
        int neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
        neighbor_block = block_assignment[neighbor];
    }
    assert(blockmodel.stores(neighbor_block));

    // With a probability inversely proportional to block degree, propose a random block merge
    if (std::rand() <= (num_blocks / ((float) blockmodel.degrees(neighbor_block) + num_blocks))) {
        std::vector<int> blocks = utils::range<int>(0, blockmodel.getNum_blocks());
        std::vector<int> weights = utils::to_int<bool>(blockmodel.in_two_hop_radius());
        int proposal = choose_neighbor(blocks, weights);
        // int proposal = propose_random_block(current_block, num_blocks);
        assert(blockmodel.stores(proposal));
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }

    // Build multinomial distribution
    double total_edges = 0.0;
    MapVector<int> edges = blockmodel.blockmatrix()->neighbors_weights(neighbor_block);
    if (block_merge) {  // Make sure proposal != current_block
        edges[current_block] = 0;
        total_edges = utils::sum<double, int>(edges);
        if (total_edges == 0.0) { // Neighbor block has no neighbors, so propose a random block
            int proposal = propose_random_block(current_block, num_blocks);
            assert(blockmodel.stores(proposal));
            return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
    } else {
        total_edges = utils::sum<double, int>(edges);
    }
    if (edges.empty()) {
        std::cerr << "ERROR: NO EDGES! k = " << blockmodel.degrees(neighbor_block) << " "
        << blockmodel.degrees_out(neighbor_block) << " " << blockmodel.degrees_in(neighbor_block)
        << std::endl;
        utils::print<int>(blockmodel.blockmatrix()->getrow_sparse(neighbor_block));
        utils::print<int>(blockmodel.blockmatrix()->getcol_sparse(neighbor_block));
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    SparseVector<double> multinomial_distribution;
    utils::div(edges, total_edges, multinomial_distribution);
    int proposal = choose_neighbor(multinomial_distribution);
    assert(blockmodel.stores(proposal));
    return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

}  // namespace common::dist