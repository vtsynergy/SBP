#include "block_merge.hpp"

#include <cmath>

#include "args.hpp"
#include "mpi_data.hpp"

namespace block_merge {

// TODO: test this
PairIndexVector blockmodel_delta(int current_block, int proposed_block, const Blockmodel &blockmodel) {
    PairIndexVector delta;
    for (const std::pair<const int, int> &entry : blockmodel.blockmatrix()->getrow_sparse(current_block)) {
        int col = entry.first;  // row = current_block
        int value = entry.second;
        if (col == current_block || col == proposed_block) {  // entry = current_block, current_block
            delta[std::make_pair(proposed_block, proposed_block)] += value;
        } else {
            delta[std::make_pair(proposed_block, col)] += value;
        }
        delta[std::make_pair(current_block, col)] -= value;
    }
    for (const std::pair<const int, int> &entry : blockmodel.blockmatrix()->getcol_sparse(current_block)) {
        int row = entry.first;  // col = current_block
        if (row == current_block) continue;  // already handled above
        int value = entry.second;
        if (row == proposed_block) {  // entry = current_block, current_block
            delta[std::make_pair(proposed_block, proposed_block)] += value;
        } else {
            delta[std::make_pair(row, proposed_block)] += value;
        }
        delta[std::make_pair(row, current_block)] -= value;
    }
    return delta;
}

//void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
//                                    const std::vector<int> &best_merge_for_each_block, int num_edges) {
//    // The following code is modeled after the `merge_sweep` function in
//    // https://git.skewed.de/count0/graph-tool/-/blob/master/src/graph/inference/loops/merge_loop.hh
//    typedef std::tuple<int, int, double> merge_t;
//    auto cmp_fxn = [](merge_t left, merge_t right) { return std::get<2>(left) > std::get<2>(right); };
//    std::priority_queue<merge_t, std::vector<merge_t>, decltype(cmp_fxn)> queue(cmp_fxn);
//    for (int i = 0; i < delta_entropy_for_each_block.size(); ++i)
//        queue.push(std::make_tuple(i, best_merge_for_each_block[i], delta_entropy_for_each_block[i]));
//    double delta_entropy = 0.0;
//    int num_merged = 0;
//    // Block map is here so that, if you move merge block A to block C, and then block B to block A, all three blocks
//    // end up with the same block assignment (C)
//    std::vector<int> block_map = utils::range<int>(0, blockmodel.getNum_blocks());
//    while (num_merged < blockmodel.getNum_blocks_to_merge() && !queue.empty()) {
//        merge_t merge = queue.top();
//        queue.pop();
//        int merge_from = std::get<0>(merge);
//        int merge_to = block_map[std::get<1>(merge)];
//        int delta_entropy_hint = std::get<2>(merge);
//        if (merge_from != merge_to) {
//            // Calculate the delta entropy given the current block assignment
//            EdgeWeights out_blocks = blockmodel.blockmatrix()->outgoing_edges(merge_from);
//            EdgeWeights in_blocks = blockmodel.blockmatrix()->incoming_edges(merge_from);
//            int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
//            int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
//            int k = k_out + k_in;
//            common::ProposalAndEdgeCounts proposal { merge_to, k_out, k_in, k };
//            SparseEdgeCountUpdates updates;
//            edge_count_updates_sparse(blockmodel.blockmatrix(), merge_from, proposal.proposal, out_blocks, in_blocks,
//                                      updates);
//            common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
//                      merge_from, blockmodel, blockmodel.blockmatrix()->get(merge_from, merge_from), proposal);
//            double delta_entropy_actual =
//                compute_delta_entropy_sparse(merge_from, proposal.proposal, num_edges, blockmodel, updates,
//                                             new_block_degrees);
//            if (std::isnan(delta_entropy_actual)) {
//                std::cout << merge_from << " --> " << merge_to << " : " << delta_entropy_actual << std::endl;
//                std::cout << "proposal --> k_out: " << proposal.num_out_neighbor_edges << " k_in: " << proposal.num_in_neighbor_edges << " k: " << proposal.num_neighbor_edges << std::endl;
//                std::cout << "new block degrees out: ";
//                utils::print<int>(new_block_degrees.block_degrees_out);
//                std::cout << "new block degrees in: ";
//                utils::print<int>(new_block_degrees.block_degrees_in);
//                std::cout << "new block degrees: ";
//                utils::print<int>(new_block_degrees.block_degrees);
//                exit(-100);
//            }
//            // If the actual change in entropy is more positive (greater) than anticipated, put it back in queue
//            if (!queue.empty() && delta_entropy_actual > std::get<2>(queue.top())) {
//                std::get<2>(merge) = delta_entropy_actual;
//                queue.push(merge);
//                continue;
//            }
//            // Perform the merge
//            // 1. Update the assignment
//            for (int i = 0; i < block_map.size(); ++i) {
//                int block = block_map[i];
//                if (block == merge_from) {
//                    block_map[i] = merge_to;
//                }
//            }
//            blockmodel.update_block_assignment(merge_from, merge_to);
//            // 2. Update the matrix
//            // NOTE: if getting funky results, try replacing the following with a matrix rebuild
//            blockmodel.blockmatrix()->clearrow(merge_from);
//            blockmodel.blockmatrix()->clearcol(merge_from);
//            blockmodel.blockmatrix()->setrow(merge_to, updates.proposal_row);
//            blockmodel.blockmatrix()->setcol(merge_to, updates.proposal_col);
//            blockmodel.setBlock_degrees_out(new_block_degrees.block_degrees_out);
//            blockmodel.setBlock_degrees_in(new_block_degrees.block_degrees_in);
//            blockmodel.setBlock_degrees(new_block_degrees.block_degrees);
//            num_merged++;
//        }
//    }
//    std::vector<int> mapping = blockmodel.build_mapping(blockmodel.block_assignment());
//    for (int i = 0; i < blockmodel.block_assignment().size(); ++i) {
//        int block = blockmodel.block_assignment(i);
//        int new_block = mapping[block];
//        blockmodel.set_block_assignment(i, new_block);
//        // blockmodel.getBlock_assignment()[i] = new_block;
//    }
//    blockmodel.setNum_blocks(blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge());
//}

 void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
                                     const std::vector<int> &best_merge_for_each_block, int num_edges) {
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
         double delta_entropy_hint = std::get<2>(merge);
         if (merge_from != merge_to) {
             // Calculate the delta entropy given the current block assignment
             EdgeWeights out_blocks = blockmodel.blockmatrix()->outgoing_edges(merge_from);
             EdgeWeights in_blocks = blockmodel.blockmatrix()->incoming_edges(merge_from);
             int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
             int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
             int k = k_out + k_in;
             common::ProposalAndEdgeCounts proposal { merge_to, k_out, k_in, k };
             PairIndexVector delta = blockmodel_delta(merge_from, proposal.proposal, blockmodel);
//             common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
//                     merge_from, blockmodel, blockmodel.blockmatrix()->get(merge_from, merge_from), proposal);
             int current_block_self_edges = blockmodel.blockmatrix()->get(merge_from, merge_from)
                     + get(delta, std::make_pair(merge_from, merge_from));
             int proposed_block_self_edges = blockmodel.blockmatrix()->get(merge_to, merge_to)
                     + get(delta, std::make_pair(merge_to, merge_to));
             common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
                     merge_from, blockmodel, current_block_self_edges, proposed_block_self_edges, proposal);
             double delta_entropy_actual = compute_delta_entropy_sparse(merge_from, blockmodel, delta,
                                                                        new_block_degrees);
             if (std::isnan(delta_entropy_actual)) {
                 std::cout << merge_from << " --> " << merge_to << " : " << delta_entropy_actual << std::endl;
                 std::cout << "proposal --> k_out: " << proposal.num_out_neighbor_edges << " k_in: " << proposal.num_in_neighbor_edges << " k: " << proposal.num_neighbor_edges << std::endl;
                 std::cout << "new block degrees out: ";
                 utils::print<int>(new_block_degrees.block_degrees_out);
                 std::cout << "new block degrees in: ";
                 utils::print<int>(new_block_degrees.block_degrees_in);
                 std::cout << "new block degrees: ";
                 utils::print<int>(new_block_degrees.block_degrees);
                 exit(-100);
             }
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
             blockmodel.update_block_assignment(merge_from, merge_to);
             // 2. Update the matrix
             blockmodel.blockmatrix()->update_edge_counts(delta);
             blockmodel.setBlock_degrees_out(new_block_degrees.block_degrees_out);
             blockmodel.setBlock_degrees_in(new_block_degrees.block_degrees_in);
             blockmodel.setBlock_degrees(new_block_degrees.block_degrees);
             num_merged++;
         }
     }
     std::vector<int> mapping = blockmodel.build_mapping(blockmodel.block_assignment());
     for (int i = 0; i < blockmodel.block_assignment().size(); ++i) {
         int block = blockmodel.block_assignment(i);
         int new_block_index = mapping[block];
         blockmodel.set_block_assignment(i, new_block_index);
         // blockmodel.getBlock_assignment()[i] = new_block;
     }
     blockmodel.setNum_blocks(blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge());
 }

double compute_delta_entropy(int current_block, int proposal, int num_edges, const Blockmodel &blockmodel, 
                             EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<int> old_block_row = blockmodel.blockmatrix()->getrow(current_block); // M_r_t1
    std::vector<int> old_proposal_row = blockmodel.blockmatrix()->getrow(proposal);   // M_s_t1
    std::vector<int> old_block_col = blockmodel.blockmatrix()->getcol(current_block); // M_t2_r
    std::vector<int> old_proposal_col = blockmodel.blockmatrix()->getcol(proposal);   // M_t2_s

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
                                                block_degrees.block_degrees_out[proposal], num_edges);
    delta_entropy -= common::delta_entropy_temp(new_proposal_col, new_proposal_col_degrees_out,
                                                block_degrees.block_degrees_in[proposal], num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_row, old_block_row_degrees_in,
                                                blockmodel.getBlock_degrees_out()[current_block], num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, old_proposal_row_degrees_in,
                                                blockmodel.getBlock_degrees_out()[proposal], num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_col, old_block_col_degrees_out,
                                                blockmodel.getBlock_degrees_in()[current_block], num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, old_proposal_col_degrees_out,
                                                blockmodel.getBlock_degrees_in()[proposal], num_edges);
    return delta_entropy;
}

double compute_delta_entropy_sparse(int current_block, int proposal, int num_edges, const Blockmodel &blockmodel,
                                    SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    const ISparseMatrix *matrix = blockmodel.blockmatrix();
    const MapVector<int> &old_block_row = matrix->getrow_sparse(current_block); // M_r_t1
    const MapVector<int> &old_proposal_row = matrix->getrow_sparse(proposal);   // M_s_t1
    const MapVector<int> &old_block_col = matrix->getcol_sparse(current_block); // M_t2_r
    const MapVector<int> &old_proposal_col = matrix->getcol_sparse(proposal);   // M_t2_s

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(updates.proposal_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    delta_entropy -= common::delta_entropy_temp(updates.proposal_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[proposal], current_block, proposal,
                                                num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_row, blockmodel.getBlock_degrees_in(),
                                                blockmodel.getBlock_degrees_out()[current_block], num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, blockmodel.getBlock_degrees_in(),
                                                blockmodel.getBlock_degrees_out()[proposal], num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_col, blockmodel.getBlock_degrees_out(),
                                                blockmodel.getBlock_degrees_in()[current_block], current_block,
                                                proposal, num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, blockmodel.getBlock_degrees_out(),
                                                blockmodel.getBlock_degrees_in()[proposal], current_block, proposal,
                                                num_edges);
    return delta_entropy;
}

double compute_delta_entropy_sparse(int current_block, const Blockmodel &blockmodel, const PairIndexVector &delta,
                                    common::NewBlockDegrees &block_degrees) {
    const ISparseMatrix *matrix = blockmodel.blockmatrix();
    double delta_entropy = 0.0;
    for (const std::pair<const std::pair<int, int>, int> &cell_delta : delta) {
        int row = cell_delta.first.first;
        int col = cell_delta.first.second;
        int change = cell_delta.second;
        // delta += + E(old) - E(new)
        delta_entropy += common::cell_entropy(matrix->get(row, col), blockmodel.getBlock_degrees_in()[col], blockmodel.getBlock_degrees_out()[row]);
        if (row == current_block || col == current_block) continue;  // the "new" cell entropy == 0;
        delta_entropy -= common::cell_entropy(matrix->get(row, col) + change, block_degrees.block_degrees_in[col], block_degrees.block_degrees_out[row]);
    }
    return delta_entropy;
}

EdgeCountUpdates edge_count_updates(ISparseMatrix *blockmodel, int current_block, int proposed_block,
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

Blockmodel &merge_blocks(Blockmodel &blockmodel, const NeighborList &out_neighbors, int num_edges) {
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
            ProposalEvaluation proposal = propose_merge_sparse(current_block, num_edges, blockmodel, block_assignment,
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
        blockmodel.carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block);
    else
        carry_out_best_merges_advanced(blockmodel, delta_entropy_for_each_block, best_merge_for_each_block, num_edges);
    blockmodel.initialize_edge_counts(out_neighbors);
    return blockmodel;
}

// TODO: get rid of block_assignment (block_assignment), just use blockmodel
ProposalEvaluation propose_merge(int current_block, int num_edges, Blockmodel &blockmodel,
                                 std::vector<int> &block_assignment) {
    EdgeWeights out_blocks = blockmodel.blockmatrix()->outgoing_edges(current_block);
    EdgeWeights in_blocks = blockmodel.blockmatrix()->incoming_edges(current_block);
    common::ProposalAndEdgeCounts proposal =
        common::propose_new_block(current_block, out_blocks, in_blocks, block_assignment, blockmodel, true);
    EdgeCountUpdates updates =
        edge_count_updates(blockmodel.blockmatrix(), current_block, proposal.proposal, out_blocks, in_blocks);
    int current_block_self_edges = blockmodel.blockmatrix()->get(current_block, current_block)
            + updates.block_row[current_block];
    int proposed_block_self_edges = blockmodel.blockmatrix()->get(proposal.proposal, proposal.proposal)
            + updates.proposal_row[proposal.proposal];
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, blockmodel, current_block_self_edges, proposed_block_self_edges, proposal);
//    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
//            current_block, blockmodel, blockmodel.blockmatrix()->get(current_block, current_block), proposal);
    double delta_entropy =
        compute_delta_entropy(current_block, proposal.proposal, num_edges, blockmodel, updates, new_block_degrees);
    // std::cout << "dE: " << delta_entropy << std::endl;
    return ProposalEvaluation{proposal.proposal, delta_entropy};
}

// TODO: get rid of block_assignment (block_assignment), just use blockmodel
ProposalEvaluation propose_merge_sparse(int current_block, int num_edges, Blockmodel &blockmodel,
                                        std::vector<int> &block_assignment,
                                        std::unordered_map<int, bool> &past_proposals) {
    EdgeWeights out_blocks = blockmodel.blockmatrix()->outgoing_edges(current_block);
    EdgeWeights in_blocks = blockmodel.blockmatrix()->incoming_edges(current_block);
    common::ProposalAndEdgeCounts proposal =
        common::propose_new_block(current_block, out_blocks, in_blocks, block_assignment, blockmodel, true);
    if (past_proposals[proposal.proposal])
        return ProposalEvaluation{ proposal.proposal, std::numeric_limits<double>::max() };
    PairIndexVector delta = blockmodel_delta(current_block, proposal.proposal, blockmodel);
    // SparseEdgeCountUpdates updates;
    // edge_count_updates_sparse(blockmodel.blockmatrix(), current_block, proposal.proposal, out_blocks, in_blocks,
    //                           updates);
    // Note: if it becomes an issue, figuring out degrees on the fly could save some RAM. The
    // only ones that change are the degrees for current_block and proposal anyway...
    int current_block_self_edges = blockmodel.blockmatrix()->get(current_block, current_block)
                                   + get(delta, std::make_pair(current_block, current_block));
    int proposed_block_self_edges = blockmodel.blockmatrix()->get(proposal.proposal, proposal.proposal)
                                    + get(delta, std::make_pair(proposal.proposal, proposal.proposal));
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, blockmodel, current_block_self_edges, proposed_block_self_edges, proposal);
//    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
//            current_block, blockmodel, blockmodel.blockmatrix()->get(current_block, current_block), proposal);
    double delta_entropy = compute_delta_entropy_sparse(current_block, blockmodel, delta, new_block_degrees);
    // double delta_entropy =
    //     compute_delta_entropy_sparse(current_block, proposal.proposal, num_edges, blockmodel, updates,
    //                                  new_block_degrees);
    past_proposals[proposal.proposal] = true;
    return ProposalEvaluation{proposal.proposal, delta_entropy};
}

namespace dist {

TwoHopBlockmodel &merge_blocks(TwoHopBlockmodel &blockmodel, const NeighborList &out_neighbors, int num_edges) {
    // MPI Datatype init
    MPI_Datatype Merge_t;
    int merge_blocklengths[3] = { 1, 1, 1 };
    MPI_Aint merge_displacements[3] = { 0, sizeof(int), sizeof(int) + sizeof(int) };
    MPI_Datatype merge_types[3] = { MPI_INT, MPI_INT, MPI_DOUBLE };
    MPI_Type_create_struct(3, merge_blocklengths, merge_displacements, merge_types, &Merge_t);
    MPI_Type_commit(&Merge_t);
    // MPI Datatype init
    int num_blocks = blockmodel.getNum_blocks();
    std::vector<int> block_assignment = utils::range<int>(0, num_blocks);
    // int my_blocks = ceil(((double) num_blocks - (double) mpi.rank) / (double) mpi.num_processes);
    // merge_buffer stores best Merges as if all blocks are owned by this MPI rank. Used to avoid locks
    std::vector<Merge> merge_buffer(num_blocks);
    // int num_avoided = 0;  // number of avoided/skipped calculations
    // int index = 0;
    int my_blocks = 0;
    #pragma omp parallel for schedule(dynamic)  // reduction( + : num_avoided)
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
    // for (int current_block = mpi.rank; current_block < num_blocks; current_block += mpi.num_processes) {
        if (blockmodel.owns_block(current_block) == false) continue;
        #pragma omp atomic update
        my_blocks++;
        std::unordered_map<int, bool> past_proposals;
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            ProposalEvaluation proposal = propose_merge_sparse(current_block, num_edges, blockmodel, block_assignment,
                                                               past_proposals);
            // std::cout << "proposal = " << proposal.proposed_block << " with DE " << proposal.delta_entropy << std::endl;
            // TODO: find a way to do this without having a large merge buffer. Maybe store list of my blocks in
            // TwoHopBlockmodel?
            if (proposal.delta_entropy < merge_buffer[current_block].delta_entropy) {
                merge_buffer[current_block] = Merge { current_block, proposal.proposed_block, proposal.delta_entropy };
            }
        }
    }
    // Get list of best merges owned by this MPI rank. Used in Allgatherv.
    std::vector<Merge> best_merges(my_blocks);
    int index = 0;
    // std::cout << "size of merge buffer: " << merge_buffer.size() << std::endl;
    for (const Merge & merge : merge_buffer) {
        if (merge.block == -1) continue;
        // std::cout << "Merge : block " << merge.block << " proposal " << merge.proposal << " dE " << merge.delta_entropy << std::endl;
        best_merges[index] = merge;
        index++;
    }
    // std::cout << "best merges size: " << best_merges.size() << " index = " << index << std::endl;
    // MPI COMMUNICATION
    int numblocks[mpi.num_processes];
    MPI_Allgather(&(my_blocks), 1, MPI_INT, &numblocks, 1, MPI_INT, MPI_COMM_WORLD);
    int offsets[mpi.num_processes];
    offsets[0] = 0;
    for (int i = 1; i < mpi.num_processes; ++i) {
        offsets[i] = offsets[i-1] + numblocks[i-1];
    }
    int total_blocks = offsets[mpi.num_processes-1] + numblocks[mpi.num_processes-1];
    // TODO: change the size of this to total_blocks? Otherwise when there is overlapping computation there may be a segfault
    std::vector<Merge> all_best_merges(total_blocks);
    std::cout << mpi.rank << " best merges size: " << best_merges.size() << std::endl;
    std::cout << mpi.rank << " my blocks number: " << my_blocks << std::endl;
    std::cout << mpi.rank << " all best merges size: " << all_best_merges.size() << std::endl;
    std::cout << mpi.rank << " numblocks: ";
    utils::print<int>(numblocks);
    std::cout << mpi.rank << " offsets: ";
    utils::print<int>(offsets);
    std::cout << "strategy == " << args.distribute << std::endl;
    MPI_Allgatherv(best_merges.data(), my_blocks, Merge_t, all_best_merges.data(), numblocks, offsets,
                   Merge_t, MPI_COMM_WORLD);
    // END MPI COMMUNICATION
    std::vector<int> best_merge_for_each_block = utils::constant<int>(num_blocks, -1);
    std::vector<double> delta_entropy_for_each_block = utils::constant<double>(num_blocks, -1);
    // TODO: use a more intelligent way to assign these when there is overlap?
    for (const Merge& m : all_best_merges) {
        // std::cout << "block: " << m.block << " proposal: " << m.proposal << " dE: " << m.delta_entropy << std::endl;
        best_merge_for_each_block[m.block] = m.proposal;
        delta_entropy_for_each_block[m.block] = m.delta_entropy;
    }
    // std::cout << mpi.rank << " best merges";
    // utils::print<int>(best_merge_for_each_block);
    blockmodel.carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block);
    // else
        // carry_out_best_merges_advanced(blockmodel, delta_entropy_for_each_block, best_merge_for_each_block, num_edges);
    blockmodel.distribute(out_neighbors);
    blockmodel.initialize_edge_counts(out_neighbors);
    MPI_Type_free(&Merge_t);
    return blockmodel;
}

}  // namespace dist

}  // namespace block_merge