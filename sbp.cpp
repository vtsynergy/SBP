#include "sbp.hpp"

namespace sbp {

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

Blockmodel hierarchical_iteration(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodel_triplet,
                                  Args &args) {
    int target_num_blocks = blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge();
    while (blockmodel.getNum_blocks() > target_num_blocks) {
        int num_blocks_to_merge = next_num_blocks_to_merge(blockmodel, target_num_blocks);
        blockmodel.setNum_blocks_to_merge(num_blocks_to_merge);
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << blockmodel.getNum_blocks() << " --> "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::merge_blocks(blockmodel, graph.out_neighbors, args);
        std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet, args);
        else  // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet, args);
    }
    return blockmodel;
}

int next_num_blocks_to_merge(Blockmodel &blockmodel, int target_num_blocks) {
    int next_target = blockmodel.getNum_blocks() / 1.3;
    if (next_target > blockmodel.getNum_blocks() - 1)
        next_target = blockmodel.getNum_blocks() - 1;
    if (next_target < target_num_blocks)
        next_target = target_num_blocks;
    return blockmodel.getNum_blocks() - next_target;
}

Blockmodel flat_iteration(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodel_triplet, Args &args) {
    blockmodel = block_merge::merge_blocks(blockmodel, graph.out_neighbors, args);
    std::cout << "Starting MCMC vertex moves" << std::endl;
    if (args.algorithm == "async_gibbs")
        blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet, args);
    else  // args.algorithm == "metropolis_hastings"
        blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet, args);
    return blockmodel;
}

Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices, graph.out_neighbors, BLOCK_REDUCTION_RATE);
    std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices << " vertices "
              << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();  // TODO: Start with max & min blockmodel. done_blockmodeling can be a member of blockmodel_triplet.
    blockmodel.setNum_blocks_to_merge(blockmodel_triplet.get_mid(1, graph.num_vertices));
    while (!done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        if (args.flat)
            blockmodel = flat_iteration(blockmodel, graph, blockmodel_triplet, args);
        else
            blockmodel = hierarchical_iteration(blockmodel, graph, blockmodel_triplet, args);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    }
    return blockmodel;
}

}
