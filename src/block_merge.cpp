#include "block_merge.hpp"

#include <cmath>

#include "args.hpp"
#include "entropy.hpp"
#include "mpi_data.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

namespace block_merge {

double BlockMerge_time = 0.0;
double BlockMerge_loop_time = 0.0;

Delta blockmodel_delta(int current_block, int proposed_block, const Blockmodel &blockmodel) {
    Delta delta(current_block, proposed_block, blockmodel.degrees(current_block));
    for (const std::pair<const int, int> &entry: blockmodel.blockmatrix()->getrow_sparse(current_block)) {
        int col = entry.first;  // row = current_block
        int value = entry.second;
        if (col == current_block || col == proposed_block) {  // entry = current_block, current_block
            delta.add(proposed_block, proposed_block, value);
        } else {
            delta.add(proposed_block, col, value);
        }
        delta.sub(current_block, col, value);
    }
    for (const std::pair<const int, int> &entry: blockmodel.blockmatrix()->getcol_sparse(current_block)) {
        int row = entry.first;  // col = current_block
        if (row == current_block) continue;  // already handled above
        int value = entry.second;
        if (row == proposed_block) {  // entry = current_block, current_block
            delta.add(proposed_block, proposed_block, value);
        } else {
            delta.add(row, proposed_block, value);
        }
        delta.sub(row, current_block, value);
    }
    return delta;
}

void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
                                    const std::vector<int> &best_merge_for_each_block, int num_edges) {
    // The following code is modeled after the `merge_sweep` function in
    // https://git.skewed.de/count0/graph-tool/-/blob/master/src/graph/inference/loops/merge_loop.hh
    typedef std::tuple<int, int, double> merge_t;
    auto cmp_fxn = [](merge_t left, merge_t right) { return std::get<2>(left) > std::get<2>(right); };
    std::priority_queue<merge_t, std::vector<merge_t>, decltype(cmp_fxn)> queue(cmp_fxn);
    for (int i = 0; i < (int) delta_entropy_for_each_block.size(); ++i)
        queue.push(std::make_tuple(i, best_merge_for_each_block[i], delta_entropy_for_each_block[i]));
//     double delta_entropy = 0.0;
    int num_merged = 0;
    // Block map is here so that, if you move merge block A to block C, and then block B to block A, all three blocks
    // end up with the same block assignment (C)
    std::vector<int> block_map = utils::range<int>(0, blockmodel.getNum_blocks());
    while (num_merged < blockmodel.getNum_blocks_to_merge() && !queue.empty()) {
        merge_t merge = queue.top();
        queue.pop();
        int merge_from = std::get<0>(merge);
        int merge_to = block_map[std::get<1>(merge)];
//         double delta_entropy_hint = std::get<2>(merge);
        if (merge_from != merge_to) {
            // Calculate the delta entropy given the current block assignment
            EdgeWeights out_blocks = blockmodel.blockmatrix()->outgoing_edges(merge_from);
            EdgeWeights in_blocks = blockmodel.blockmatrix()->incoming_edges(merge_from);
            int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
            int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
            int k = k_out + k_in;
            utils::ProposalAndEdgeCounts proposal{merge_to, k_out, k_in, k};
            Delta delta = blockmodel_delta(merge_from, proposal.proposal, blockmodel);
//             common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
//                     merge_from, blockmodel, blockmodel.blockmatrix()->get(merge_from, merge_from), proposal);


//            int current_block_self_edges = blockmodel.blockmatrix()->get(merge_from, merge_from)
//                                           + delta.get(merge_from, merge_from);
            int proposed_block_self_edges = blockmodel.blockmatrix()->get(merge_to, merge_to)
                                            + delta.get(merge_to, merge_to);
//            common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
//                    merge_from, blockmodel, current_block_self_edges, proposed_block_self_edges, proposal);
//            double delta_entropy_actual = entropy::block_merge_delta_mdl(merge_from, blockmodel, delta,
//                                                                       new_block_degrees);
//            if (std::isnan(delta_entropy_actual)) {
//                std::cout << merge_from << " --> " << merge_to << " : " << delta_entropy_actual << std::endl;
//                std::cout << "proposal --> k_out: " << proposal.num_out_neighbor_edges << " k_in: "
//                          << proposal.num_in_neighbor_edges << " k: " << proposal.num_neighbor_edges << std::endl;
//                std::cout << "new block degrees out: ";
//                utils::print<int>(new_block_degrees._block_degrees_out);
//                std::cout << "new block degrees in: ";
//                utils::print<int>(new_block_degrees._block_degrees_in);
//                std::cout << "new block degrees: ";
//                utils::print<int>(new_block_degrees._block_degrees);
//                exit(-100);
//            }


            double delta_entropy_actual = entropy::block_merge_delta_mdl(merge_from, proposal, blockmodel, delta);
            // If the actual change in entropy is more positive (greater) than anticipated, put it back in queue
            if (!queue.empty() && delta_entropy_actual > std::get<2>(queue.top())) {
                std::get<2>(merge) = delta_entropy_actual;
                queue.push(merge);
                continue;
            }
            // Perform the merge
            // 1. Update the assignment
            for (size_t i = 0; i < block_map.size(); ++i) {
                int block = block_map[i];
                if (block == merge_from) {
                    block_map[i] = merge_to;
                }
            }
            blockmodel.update_block_assignment(merge_from, merge_to);
            // 2. Update the matrix
            blockmodel.blockmatrix()->update_edge_counts(delta);
//            blockmodel.degrees_out(new_block_degrees._block_degrees_out);
//            blockmodel.degrees_in(new_block_degrees._block_degrees_in);
//            blockmodel.degrees(new_block_degrees._block_degrees);
            blockmodel.degrees_out(merge_from, 0);
            blockmodel.degrees_out(merge_to, blockmodel.degrees_out(merge_to) + proposal.num_out_neighbor_edges);
            blockmodel.degrees_in(merge_from, 0);
            blockmodel.degrees_in(merge_to, blockmodel.degrees_in(merge_to) + proposal.num_in_neighbor_edges);
            blockmodel.degrees(merge_from, 0);
            blockmodel.degrees(merge_to, blockmodel.degrees_out(merge_to) + blockmodel.degrees_in(merge_to)
                               - proposed_block_self_edges);
            num_merged++;
        }
    }
    std::vector<int> mapping = Blockmodel::build_mapping(blockmodel.block_assignment());
    for (int i = 0; i < (int) blockmodel.block_assignment().size(); ++i) {
        int block = blockmodel.block_assignment(i);
        int new_block_index = mapping[block];
        blockmodel.set_block_assignment(i, new_block_index);
        // blockmodel.getBlock_assignment()[i] = new_block;
    }
    blockmodel.setNum_blocks(blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge());
}

EdgeCountUpdates edge_count_updates(std::shared_ptr<ISparseMatrix> blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks) {
    // TODO: these are copy constructors, can we safely get rid of them?
    std::vector<int> proposal_row = blockmodel->getrow(proposed_block);
    std::vector<int> proposal_col = blockmodel->getcol(proposed_block);
    int count_self = blockmodel->get(current_block, current_block);
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

void edge_count_updates_sparse(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates) {
    // TODO: these are copy constructors, can we safely get rid of them?
    updates.proposal_row = blockmodel->getrow_sparse(proposed_block);
    updates.proposal_col = blockmodel->getcol_sparse(proposed_block);
    int count_self = blockmodel->get(current_block, current_block);
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

Blockmodel &merge_blocks(Blockmodel &blockmodel, const Graph &graph, int num_edges) {
    // TODO: add block merge timings to evaluation
    int num_blocks = blockmodel.getNum_blocks();
    std::vector<int> best_merge_for_each_block = utils::constant<int>(num_blocks, -1);
    std::vector<double> delta_entropy_for_each_block =
            utils::constant<double>(num_blocks, std::numeric_limits<double>::max());
    std::vector<int> block_assignment = utils::range<int>(0, num_blocks);
    // TODO: keep track of already proposed merges, do not re-process those
    int num_avoided = 0;  // number of avoided/skipped calculations
    double start_t = MPI_Wtime();
    #pragma omp parallel for schedule(dynamic) reduction( + : num_avoided) default(none) \
    shared(num_blocks, num_edges, blockmodel, block_assignment, delta_entropy_for_each_block, best_merge_for_each_block)
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        std::unordered_map<int, bool> past_proposals;
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            ProposalEvaluation proposal = propose_merge_sparse(current_block, num_edges, blockmodel, past_proposals);
            if (proposal.delta_entropy == std::numeric_limits<double>::max()) num_avoided++;
            if (proposal.delta_entropy < delta_entropy_for_each_block[current_block]) {
                best_merge_for_each_block[current_block] = proposal.proposed_block;
                delta_entropy_for_each_block[current_block] = proposal.delta_entropy;
            }
        }
    }
    BlockMerge_loop_time += MPI_Wtime() - start_t;
    std::cout << "Avoided " << num_avoided << " / " << NUM_AGG_PROPOSALS_PER_BLOCK * num_blocks << " comparisons."
              << std::endl;
    if (args.approximate)
        blockmodel.carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block);
    else
        carry_out_best_merges_advanced(blockmodel, delta_entropy_for_each_block, best_merge_for_each_block,
                                       num_edges);
    blockmodel.initialize_edge_counts(graph);
    return blockmodel;
}

// TODO: get rid of block_assignment (block_assignment), just use blockmodel
ProposalEvaluation propose_merge(int current_block, int num_edges, Blockmodel &blockmodel,
                                 std::vector<int> &block_assignment) {
    EdgeWeights out_blocks = blockmodel.blockmatrix()->outgoing_edges(current_block);
    EdgeWeights in_blocks = blockmodel.blockmatrix()->incoming_edges(current_block);
    utils::ProposalAndEdgeCounts proposal =
            common::propose_new_block(current_block, out_blocks, in_blocks, block_assignment, blockmodel, true);
    EdgeCountUpdates updates =
            edge_count_updates(blockmodel.blockmatrix(), current_block, proposal.proposal, out_blocks, in_blocks);
    int current_block_self_edges = blockmodel.blockmatrix()->get(current_block, current_block)
                                   + updates.block_row[current_block];
    int proposed_block_self_edges = blockmodel.blockmatrix()->get(proposal.proposal, proposal.proposal)
                                    + updates.proposal_row[proposal.proposal];
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, blockmodel, current_block_self_edges, proposed_block_self_edges, proposal);
    double delta_entropy =
            entropy::block_merge_delta_mdl(current_block, proposal.proposal, num_edges, blockmodel, updates,
                                  new_block_degrees);
    return ProposalEvaluation{proposal.proposal, delta_entropy};
}

// TODO: get rid of block_assignment (block_assignment), just use blockmodel
ProposalEvaluation
propose_merge_sparse(int current_block, int num_edges, const Blockmodel &blockmodel,
                                        std::unordered_map<int, bool> &past_proposals) {
    EdgeWeights out_blocks = blockmodel.blockmatrix()->outgoing_edges(current_block);
    EdgeWeights in_blocks = blockmodel.blockmatrix()->incoming_edges(current_block);
    utils::ProposalAndEdgeCounts proposal =
            common::propose_new_block(current_block, out_blocks, in_blocks, blockmodel.block_assignment(), blockmodel, true);
    if (past_proposals[proposal.proposal])
        return ProposalEvaluation{proposal.proposal, std::numeric_limits<double>::max()};
    Delta delta = blockmodel_delta(current_block, proposal.proposal, blockmodel);
    //==========NEW==============
    double delta_entropy = entropy::block_merge_delta_mdl(current_block, proposal, blockmodel, delta);
    //==========OLD==============
//     SparseEdgeCountUpdates updates;
//        // edge_count_updates_sparse(blockmodel.blockmatrix(), current_block, proposal.proposal, out_blocks, in_blocks,
//        //                           updates);
//        // Note: if it becomes an issue, figuring out degrees on the fly could save some RAM. The
//        // only ones that change are the degrees for current_block and proposal anyway...
//        int current_block_self_edges = blockmodel.blockmatrix()->get(current_block, current_block)
//                                       + delta.get(current_block, current_block);
//        int proposed_block_self_edges = blockmodel.blockmatrix()->get(proposal.proposal, proposal.proposal)
//                                        + delta.get(proposal.proposal, proposal.proposal);
//    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
//            current_block, blockmodel, current_block_self_edges, proposed_block_self_edges, proposal);
////    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
////            current_block, blockmodel, blockmodel.blockmatrix()->get(current_block, current_block), proposal);
//    double delta_entropy = entropy::block_merge_delta_mdl(current_block, blockmodel, delta, new_block_degrees);
//    // double delta_entropy =
//    //     entropy::block_merge_delta_mdl(current_block, proposal.proposal, num_edges, blockmodel, updates,
//    //                                  new_block_degrees);
    //=========OLD==============
    past_proposals[proposal.proposal] = true;
    return ProposalEvaluation{proposal.proposal, delta_entropy};
}

}  // namespace block_merge