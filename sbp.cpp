#include "sbp.hpp"

Partition sbp::stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Partition partition(graph.num_vertices, graph.out_neighbors, BLOCK_REDUCTION_RATE);
    std::cout << "Performing stochastic block partitioning on graph with " << graph.num_vertices << " vertices "
              << " and " << partition.getNum_blocks() << " blocks." << std::endl;
    PartitionTriplet partition_triplet = PartitionTriplet();
    while (!done_partitioning(partition, partition_triplet, 0)) {
        if (partition.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << partition.getNum_blocks() << " to " 
                      << partition.getNum_blocks() - partition.getNum_blocks_to_merge() << std::endl;
        }
        partition = block_merge::merge_blocks(partition, graph.out_neighbors);
        std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            partition = finetune::asynchronous_gibbs(partition, graph, partition_triplet);
        else  // args.algorithm == "metropolis_hastings"
            partition = finetune::metropolis_hastings(partition, graph, partition_triplet);
        partition = partition_triplet.get_next_partition(partition);
    }
    return partition;
}

bool sbp::done_partitioning(Partition &partition, PartitionTriplet &partition_triplet, int min_num_blocks) {
    if (min_num_blocks > 0) {
        if ((partition.getNum_blocks() <= min_num_blocks) || (partition_triplet.get(2).empty == false)) {
            return true;
        }
    }
    if (partition_triplet.optimal_num_blocks_found) {
        partition_triplet.status();
        std::cout << "Optimal number of blocks was found" << std::endl;
        return true;
    }
    return false;
}
