#include "sbp.hpp"

namespace sbp {

namespace parallel {

Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices, graph.out_neighbors, BLOCK_REDUCTION_RATE);
    std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices << " vertices "
              << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    while (!done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::parallel::merge_blocks(blockmodel, graph.out_neighbors, args);
        std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet, args);
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

} // namespace parallel

namespace naive_distributed {

bool done_blockmodeling(partition::BlockmodelPartition &blockmodel, BlockmodelTriplet &blockmodel_triplet,
                        int min_num_blocks) {
    if (min_num_blocks > 0) {
        if ((blockmodel.blockmodel().getNum_blocks() <= min_num_blocks) || (blockmodel_triplet.get(2).empty == false)) {
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

partition::GraphPartition initialize(utils::mpi::Info &mpi, Args &args) {
    mpi.initialize_datatypes();
    // Partition the graph based on arguments. NOTE: as of now, only round robin partitioning is supported
    // Maybe TODO: random graph partitioning
    // TODO: change the graph storage format so that we no longer have to read in the entire graph before distributing it
    Graph graph = Graph::load(args);
    // TODO: partition method that does not mess with the vertex indices
    partition::GraphPartition partition = partition::distribute(graph, mpi, args);
}

Blockmodel stochastic_block_partition(partition::GraphPartition &partition, utils::mpi::Info &mpi, Args &args) {
    // if (args.threads > 0)
    //     omp_set_num_threads(args.threads);
    // else
    //     omp_set_num_threads(omp_get_num_procs());
    // For now, set num threads to 1
    omp_set_num_threads(1);
    // Initially, the blocks/communities are equivalent to the vertices
    partition::BlockmodelPartition local_blockmodel(partition.vertices(), partition.global_num_vertices(),
                                                    partition.graph().out_neighbors, BLOCK_REDUCTION_RATE);
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    // Maybe TODO: figure out if done_blockmodeling needs to change for the distributed case
    while (!done_blockmodeling(local_blockmodel, blockmodel_triplet, 0)) {
        // Maybe TODO: send a signal to all the processes to start this
        if (local_blockmodel.blockmodel().getNum_blocks_to_merge() != 0) {
            Blockmodel &blockmodel = local_blockmodel.blockmodel();
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        local_blockmodel = block_merge::naive_distributed::merge_blocks(
            local_blockmodel, partition.graph().out_neighbors, mpi, args);
    }
    // std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    // Blockmodel blockmodel(graph.num_vertices, graph.out_neighbors, BLOCK_REDUCTION_RATE);
    // std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices << " vertices "
    //           << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    // BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    // while (!parallel::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
    //     if (blockmodel.getNum_blocks_to_merge() != 0) {
    //         std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
    //                   << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
    //     }
    //     blockmodel = block_merge::parallel::merge_blocks(blockmodel, graph.out_neighbors, args);
    //     std::cout << "Starting MCMC vertex moves" << std::endl;
    //     if (args.algorithm == "async_gibbs")
    //         blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet, args);
    //     else  // args.algorithm == "metropolis_hastings"
    //         blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
    //     blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    // }
    // return blockmodel;
}

} // namespace naive_distributed

} // namespace sbp
