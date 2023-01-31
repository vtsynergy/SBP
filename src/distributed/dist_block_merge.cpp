#include "distributed/dist_block_merge.hpp"

namespace block_merge::dist {

MPI_Datatype Merge_t;

std::vector<Merge> mpi_get_best_merges(std::vector<Merge> &merge_buffer, int my_blocks) {
    // Get list of best merges owned by this MPI rank. Used in Allgatherv.
    std::vector<Merge> best_merges(my_blocks);
    int index = 0;
    for (const Merge &merge: merge_buffer) {
        if (merge.block == -1) continue;
        best_merges[index] = merge;
        index++;
    }
    int numblocks[mpi.num_processes];
    MPI_Allgather(&(my_blocks), 1, MPI_INT, &numblocks, 1, MPI_INT, MPI_COMM_WORLD);
    int offsets[mpi.num_processes];
    offsets[0] = 0;
    for (int i = 1; i < mpi.num_processes; ++i) {
        offsets[i] = offsets[i - 1] + numblocks[i - 1];
    }
    int total_blocks = offsets[mpi.num_processes - 1] + numblocks[mpi.num_processes - 1];
    // TODO: change the size of this to total_blocks? Otherwise when there is overlapping computation there may be a segfault
    std::vector<Merge> all_best_merges(total_blocks);
    MPI_Allgatherv(best_merges.data(), my_blocks, Merge_t, all_best_merges.data(), numblocks, offsets,
                   Merge_t, MPI_COMM_WORLD);
    return all_best_merges;
}

TwoHopBlockmodel &merge_blocks(TwoHopBlockmodel &blockmodel, const Graph &graph) {
    // MPI Datatype init
    int merge_blocklengths[3] = {1, 1, 1};
    MPI_Aint merge_displacements[3] = {0, sizeof(int), sizeof(int) + sizeof(int)};
    MPI_Datatype merge_types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
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
    #pragma omp parallel for schedule(dynamic) default(none) \
    shared(num_blocks, blockmodel, my_blocks, graph, block_assignment, merge_buffer) // reduction( + : num_avoided)
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        // for (int current_block = mpi.rank; current_block < num_blocks; current_block += mpi.num_processes) {
        if (!blockmodel.owns_block(current_block)) continue;
        #pragma omp atomic update
        my_blocks++;
        std::unordered_map<int, bool> past_proposals;
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            ProposalEvaluation proposal = propose_merge_sparse(current_block, graph.num_edges(), blockmodel,
                                                               block_assignment, past_proposals);
            // std::cout << "proposal = " << proposal.proposed_block << " with DE " << proposal.delta_entropy << std::endl;
            // TODO: find a way to do this without having a large merge buffer. Maybe store list of my blocks in
            // TwoHopBlockmodel?
            if (proposal.delta_entropy < merge_buffer[current_block].delta_entropy) {
                merge_buffer[current_block] = Merge{current_block, proposal.proposed_block,
                                                    proposal.delta_entropy};
            }
        }
    }
    // MPI COMMUNICATION
    std::vector<Merge> all_best_merges = mpi_get_best_merges(merge_buffer, my_blocks);
    // END MPI COMMUNICATION
    std::vector<int> best_merge_for_each_block = utils::constant<int>(num_blocks, -1);
    std::vector<double> delta_entropy_for_each_block = utils::constant<double>(num_blocks, -1);
    // TODO: use a more intelligent way to assign these when there is overlap?
    for (const Merge &m: all_best_merges) {
//        std::cout << "block: " << m.block << " proposal: " << m.proposal << " dE: " << m.delta_entropy << std::endl;
        best_merge_for_each_block[m.block] = m.proposal;
        delta_entropy_for_each_block[m.block] = m.delta_entropy;
    }
    // std::cout << mpi.rank << " best merges";
    // utils::print<int>(best_merge_for_each_block);
    std::cout << "Carrying out best merges" << std::endl;
    blockmodel.carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block);
    // else
    // carry_out_best_merges_advanced(blockmodel, delta_entropy_for_each_block, best_merge_for_each_block, num_edges);
    blockmodel.distribute(graph.out_neighbors());
    blockmodel.initialize_edge_counts(graph);
    MPI_Type_free(&Merge_t);
    return blockmodel;
}

} // namespace block_merge::dist