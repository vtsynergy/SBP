#include "dist_divisive_sbp.hpp"

#include "common.hpp"
#include "distributed/dist_block_merge.hpp"
#include "distributed/dist_divisive_sbp.hpp"
#include "distributed/dist_finetune.hpp"
#include "distributed/dist_sbp.hpp"
#include "distributed/two_hop_blockmodel.hpp"
#include "entropy.hpp"

namespace divisive::dist {

void apply_best_splits(Blockmodel &blockmodel, const std::vector<double> &split_entropy,
                       const std::vector<long> &comm_assignment, int target_num_communities) {
    // Sort entropies in descending order; best split ends up in the last spot
    std::vector<long> sorted_splits = utils::argsort<double>(split_entropy);
    if (mpi.rank == 0) {
        std::cout << "the argsort result = " << std::endl;
        utils::print<long>(sorted_splits);
    }
    long num_blocks = blockmodel.getNum_blocks();
    std::vector<long> translator = utils::range<long>(0, num_blocks);
    // Build translator, increasing new block id until reaching target
    for (int index = (int) blockmodel.getNum_blocks() - 1; index > -1; --index) {
        long block = sorted_splits[index];
        double dE = split_entropy[block];
        // If split was invalid for some reason, skip. Technically, this could be a break to save some cycles, but
        // this is a tad safer
        if (dE == std::numeric_limits<double>::max()) continue;
        translator[block] = num_blocks;
        num_blocks++;
        if (num_blocks >= target_num_communities) break;
    }
    // Re-assign vertices using translator
    for (int vertex = 0; vertex < (int) comm_assignment.size(); ++vertex) {
        long current_block = blockmodel.block_assignment(vertex);
        long proposed_block = translator[current_block];
        if (current_block != proposed_block && comm_assignment[vertex] == 1) {
            blockmodel.set_block_assignment(vertex, proposed_block);
        }
    }
    // Update block counts
    blockmodel.setNum_blocks(num_blocks);
}

TwoHopBlockmodel continue_agglomerative(Graph &graph, DistDivisiveBlockmodelTriplet &blockmodel_triplet,
                                        float iteration) {
    TwoHopBlockmodel blockmodel;
    DistBlockmodelTriplet gr_blockmodel_triplet = DistBlockmodelTriplet();
    blockmodel = gr_blockmodel_triplet.get_next_blockmodel(blockmodel_triplet.get(0));
    blockmodel = gr_blockmodel_triplet.get_next_blockmodel(blockmodel_triplet.get(1));
    blockmodel = gr_blockmodel_triplet.get_next_blockmodel(blockmodel_triplet.get(2));
    while (!sbp::dist::done_blockmodeling(blockmodel, gr_blockmodel_triplet)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        double start_bm = MPI_Wtime();
        blockmodel = block_merge::dist::merge_blocks(blockmodel, graph);
        timers::BlockMerge_time += MPI_Wtime() - start_bm;
        std::cout << "Starting MCMC vertex moves" << std::endl;
        double start_mcmc = MPI_Wtime();
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
        if (args.algorithm == "async_gibbs" && iteration < double(args.asynciterations))
            blockmodel = finetune::dist::asynchronous_gibbs(blockmodel, graph, false);
        else if (args.algorithm == "hybrid_mcmc")
            blockmodel = finetune::dist::hybrid_mcmc(blockmodel, graph, false);
        else // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::dist::metropolis_hastings(blockmodel, graph, false);
        timers::MCMC_time += MPI_Wtime() - start_mcmc;
        timers::total_time += MPI_Wtime() - start_bm;
        double mdl = blockmodel.getOverall_entropy();
        utils::save_partial_profile(++iteration, -1, mdl, entropy::normalize_mdl_v1(mdl, graph));
        blockmodel = gr_blockmodel_triplet.get_next_blockmodel(blockmodel);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
    }
    return blockmodel;
}

bool end_condition_not_reached(TwoHopBlockmodel &blockmodel, DistDivisiveBlockmodelTriplet &triplet) {
    if (args.mix) {
        return triplet.golden_ratio_not_reached();
    }
    return !sbp::dist::done_blockmodeling(blockmodel, triplet);
}

void mpi_get_best_splits(std::vector<double> &delta_entropy_for_each_block, std::vector<long> &comm_assignment) {
    MPI_Allreduce(MPI_IN_PLACE, delta_entropy_for_each_block.data(), (int) delta_entropy_for_each_block.size(),
                  MPI_DOUBLE, MPI_MIN, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, comm_assignment.data(), (int) comm_assignment.size(), MPI_LONG, MPI_MAX, mpi.comm);
}

Blockmodel run(Graph &graph) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    std::vector<long> initial_memberships = utils::constant<long>(graph.num_vertices(), 0);
    TwoHopBlockmodel blockmodel(1, graph, 1.5, initial_memberships);
    common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
    double initial_mdl = entropy::nonparametric::mdl(blockmodel, graph);
    utils::save_partial_profile(0, -1, initial_mdl, entropy::normalize_mdl_v1(initial_mdl, graph));
    DistDivisiveBlockmodelTriplet blockmodel_triplet = DistDivisiveBlockmodelTriplet();
    blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    float iteration = 0;
    while (dist::end_condition_not_reached(blockmodel, blockmodel_triplet)) {
        std::cout << "============= Block sizes ============" << std::endl;
        utils::print<long>(blockmodel.block_sizes());
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Splitting blocks up from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = dist::split_communities(blockmodel, graph, blockmodel.getNum_blocks_to_merge());
        std::cout << "============== Block sizes after split" << std::endl;
        utils::print<long>(blockmodel.block_sizes());
        std::cout << "============== num blocks after split = " << blockmodel.getNum_blocks() << std::endl;
        if (iteration < 1) {
            double mdl = entropy::nonparametric::mdl(blockmodel, graph);
            utils::save_partial_profile(0.5, -1, mdl, entropy::normalize_mdl_v1(mdl, graph));
        }
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
        std::cout << "Starting MCMC vertex moves" << std::endl;
        double start = MPI_Wtime();
        if (args.algorithm == "async_gibbs" && iteration < float(args.asynciterations))
            blockmodel = finetune::dist::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        else if (args.algorithm == "hybrid_mcmc")
            blockmodel = finetune::dist::hybrid_mcmc(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        else // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::dist::metropolis_hastings(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        timers::MCMC_time += MPI_Wtime() - start;
        double mdl = blockmodel.getOverall_entropy();
        utils::save_partial_profile(++iteration, -1, mdl, entropy::normalize_mdl_v1(mdl, graph));
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
    }
    if (args.mix)
        blockmodel = dist::continue_agglomerative(graph, blockmodel_triplet, iteration);
    blockmodel.prune(graph);
    return blockmodel;
}

TwoHopBlockmodel split_communities(TwoHopBlockmodel &blockmodel, const Graph &graph, int target_num_communities) {
    // TODO: figure out how to communicate best splits across nodes AND selectively process blocks
    auto num_blocks = (int) blockmodel.getNum_blocks();
    std::vector<double> delta_entropy_for_each_block =
            utils::constant<double>(num_blocks, std::numeric_limits<double>::max());
    std::vector<omp_lock_t> locks(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        omp_init_lock(&locks[i]);
    }
    // The assignment to be communicated to other nodes
    std::vector<long> comm_assignment = utils::constant<long>(graph.num_vertices(), -1);
    // for communication, can do an all_reduce (MIN) on dE for each block and an all_reduce (MAX) on comm_assignment
    #pragma omp parallel for schedule(dynamic) collapse(2) default(none) \
    shared(num_blocks, NUM_AGG_PROPOSALS_PER_BLOCK, blockmodel, graph, comm_assignment, delta_entropy_for_each_block, locks, std::cout)
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            if (!blockmodel.owns_block(current_block)) continue;
            // Do not attempt to split small clusters
            if ((double) blockmodel.block_size(current_block) < 0.005 * (double) graph.num_vertices()) {
                omp_set_lock(&locks[current_block]);
                delta_entropy_for_each_block[current_block] = std::numeric_limits<double>::max();
                omp_unset_lock(&locks[current_block]);
                continue;
            }
            Split split = propose_split(current_block, graph, blockmodel);
            // TODO: currently computing delta entropy for the split ONLY. Can we compute dE for entire blockmodel?
            double new_entropy = entropy::nonparametric::mdl(*(split.blockmodel),
                                                             split.subgraph);  // split.num_vertices, split.num_edges);
            double old_entropy = entropy::null_mdl_v1(split.subgraph);
            double delta_entropy = new_entropy - old_entropy;
            omp_set_lock(&locks[current_block]);
            if (delta_entropy < delta_entropy_for_each_block[current_block]) {
                delta_entropy_for_each_block[current_block] = delta_entropy;
                for (const LongEntry &entry : split.translator) {
                    long graph_vertex = entry.first;
                    long subgraph_vertex = entry.second;
                    comm_assignment[graph_vertex] = split.blockmodel->block_assignment(subgraph_vertex);
                }
            }
            omp_unset_lock(&locks[current_block]);
        }
    }
    for (int i = 0; i < num_blocks; ++i) {
        omp_destroy_lock(&locks[i]);
    }
    mpi_get_best_splits(delta_entropy_for_each_block, comm_assignment);
    if (mpi.rank == 0) {
        utils::print<double>(delta_entropy_for_each_block);
        utils::print<long>(blockmodel.block_sizes());
    }
    dist::apply_best_splits(blockmodel, delta_entropy_for_each_block, comm_assignment, target_num_communities);
    blockmodel.distribute(graph);
    blockmodel.initialize_edge_counts(graph);
    return blockmodel;
}

}