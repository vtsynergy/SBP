#include "sbp.hpp"

#include "block_merge.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "fs.hpp"
#include "globals.hpp"
#include "mpi_data.hpp"

#include "assert.h"
#include <fenv.h>
#include <sstream>

namespace sbp {

Blockmodel stochastic_block_partition(Graph &graph, Args &args, bool divide_and_conquer) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph, double(BLOCK_REDUCTION_RATE));
    common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
//    Blockmodel_first_build_time = BLOCKMODEL_BUILD_TIME;
    timers::BLOCKMODEL_BUILD_TIME = 0.0;
    double initial_mdl = entropy::mdl(blockmodel, graph);
    utils::save_partial_profile(0, -1, initial_mdl, entropy::normalize_mdl_v1(initial_mdl, graph));
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    double iteration = 0;
    while (!done_blockmodeling(blockmodel, blockmodel_triplet)) {
        if (divide_and_conquer) {
            if (!blockmodel_triplet.golden_ratio_not_reached() ||
                (blockmodel_triplet.get(0).getNum_blocks() > 1 && blockmodel_triplet.get(1).getNum_blocks() <= 1)) {
                blockmodel_triplet.status();
                blockmodel = blockmodel_triplet.get(0).copy();
                break;
            }
        }
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        double start_bm = MPI_Wtime();
        blockmodel = block_merge::merge_blocks(blockmodel, graph, graph.num_edges());
        timers::BlockMerge_time += MPI_Wtime() - start_bm;
        if (iteration < 1) {
            double mdl = entropy::mdl(blockmodel, graph);
            utils::save_partial_profile(0.5, -1, mdl, entropy::normalize_mdl_v1(mdl, graph));
        }
        std::cout << "Starting MCMC vertex moves" << std::endl;
        double start_mcmc = MPI_Wtime();
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
        if (args.algorithm == "async_gibbs" && iteration < double(args.asynciterations))
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        else if (args.algorithm == "hybrid_mcmc")
            blockmodel = finetune::hybrid_mcmc(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        else if (args.algorithm == "hybrid_mcmc_load_balanced")
            blockmodel = finetune::hybrid_mcmc_load_balanced(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        else // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
        timers::MCMC_time += MPI_Wtime() - start_mcmc;
        timers::total_time += MPI_Wtime() - start_bm;
        double mdl = blockmodel.getOverall_entropy();
        utils::save_partial_profile(++iteration, -1, mdl, entropy::normalize_mdl_v1(mdl, graph));
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
    }
    // only last iteration result will calculate expensive modularity
    double modularity = -1;
    if (args.modularity)
        modularity = graph.modularity(blockmodel.block_assignment());
    double mdl = blockmodel.getOverall_entropy();
    utils::save_partial_profile(-1, modularity, mdl, entropy::normalize_mdl_v1(mdl, graph));
    return blockmodel;
}

bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, long min_num_blocks) {
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || !blockmodel_triplet.get(2).empty) {
            return true;
        }
    }
    if (blockmodel_triplet.optimal_num_blocks_found) {
        blockmodel_triplet.status();
        std::cout << "Optimal number of blocks was found" << std::endl;
        return true;
    }
    return false;
}

}  // namespace sbp
