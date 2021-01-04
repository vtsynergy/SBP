/**
 * The stochastic block partitioning module.
 */
#ifndef SBP_SBP_HPP
#define SBP_SBP_HPP

#include <omp.h>

#include "args.hpp"
#include "block_merge.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "partition/partition_triplet.hpp"

namespace sbp {

/// Performs community detection on the provided graph, using the stochastic block partitioning algorithm
Partition stochastic_block_partition(Graph &graph, Args &args);

/// Returns true if the exit condition is reached based on the provided blockmodels
bool done_partitioning(Partition &partition, PartitionTriplet &partition_triplet, int min_num_blocks = 0);

} // namespace sbp

#endif // SBP_SBP_HPP
