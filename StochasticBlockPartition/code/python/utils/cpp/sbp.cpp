#include "sbp.hpp"

Partition sbp::stochastic_block_partition(int num_vertices, int num_edges, std::vector<Matrix2Column> &out_neighbors,
                                          std::vector<Matrix2Column> &in_neighbors) {
    int num_logical_cores = omp_get_num_procs();
    omp_set_num_threads(num_logical_cores);
    std::cout << "Setting #threads = #cores = " << num_logical_cores << std::endl;
    Partition partition(num_vertices, out_neighbors, BLOCK_REDUCTION_RATE);
    std::cout << "Performing stochastic block partitioning on graph with " << num_vertices << " vertices "
              << " and " << partition.getNum_blocks() << " blocks." << std::endl;
    PartitionTriplet partition_triplet = PartitionTriplet();
    while (!done_partitioning(partition, partition_triplet, 0)) {
        if (partition.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << partition.getNum_blocks() << " to " 
                      << partition.getNum_blocks() - partition.getNum_blocks_to_merge() << std::endl;
        }
        partition = block_merge::merge_blocks(partition, out_neighbors);
        partition = finetune::reassign_vertices(partition, num_vertices, num_edges, out_neighbors, in_neighbors,
                                                partition_triplet);
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
