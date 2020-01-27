/**
 * The stochastic block partitioning module.
 */
#ifndef CPPSBP_SBP_HPP
#define CPPSBP_SBP_HPP

#include <omp.h>

#include "block_merge.hpp"
#include "finetune.hpp"
#include "partition/partition_triplet.hpp"

namespace sbp {

// typedef struct new_iteration_state_t {
//     Partition &partition;
//     PartitionTriplet &partition_triplet;
// } NewIterationState;

Partition stochastic_block_partition(int num_vertices, int num_edges, std::vector<Matrix2Column> &out_neighbors,
                                     std::vector<Matrix2Column> &in_neighbors);
bool done_partitioning(Partition &partition, PartitionTriplet &partition_triplet, int min_num_blocks = 0);

} // namespace sbp

#endif // CPPSBP_SBP_HPP
