#include "sbp.hpp"

#include "blockmodel/dist_blockmodel.hpp"

#include "assert.h"

namespace sbp {

Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
              << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    while (!done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::merge_blocks(blockmodel, graph.out_neighbors(), graph.num_edges());
        std::cout << "Starting MCMC vertex moves" << std::endl;
        blockmodel.print_blockmodel();
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else  // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    }
    return blockmodel;
}

bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks) {
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || (blockmodel_triplet.get(2).empty == false)) {
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

namespace dist {

// Blockmodel stochastic_block_partition(Graph &graph, MPI &mpi, Args &args) {
Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    // DistBlockmodel blockmodel(graph, args, mpi);
    TwoHopBlockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    // Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    if (mpi.rank == 0)
        std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
                  << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    DistBlockmodelTriplet blockmodel_triplet = DistBlockmodelTriplet();
    while (!dist::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (mpi.rank == 0 && blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::dist::merge_blocks(blockmodel, graph.out_neighbors(), graph.num_edges());
        if (mpi.rank == 0) std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::dist::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else
            blockmodel = finetune::dist::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    }
    std::cout << "Total MCMC iterations: " << finetune::num_iterations << std::endl;
    return blockmodel;
}

bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet, int min_num_blocks) {
    if (mpi.rank == 0) std::cout << "distributed done_blockmodeling" << std::endl;
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || (blockmodel_triplet.get(2).empty == false)) {
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

}  // namespace dist

}  // namespace sbp