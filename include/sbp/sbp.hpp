/**
 * The stochastic block blockmodeling module.
 */
#ifndef SBP_SBP_HPP
#define SBP_SBP_HPP

#include <omp.h>
#include <mpi.h>

#include "args.hpp"
#include "block_merge.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "partition.hpp"
#include "mpi_utils.hpp"

namespace sbp {

namespace parallel {

/// Performs community detection on the provided graph, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args);

/// Returns true if the exit condition is reached based on the provided blockmodels
bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks = 0);

} // namespace parallel

namespace naive_distributed {

/// Performs the initial communication necessary for distributed community detection
partition::GraphPartition initialize(utils::mpi::Info &mpi, Args &args);

/// Performs community detection on the provided graph, using a naive distributed stochastic block partitioning
/// algorithm
Blockmodel stochastic_block_partition(partition::GraphPartition &partition, utils::mpi::Info &mpi, Args &args);

} // namespace naive_distributed

} // namespace sbp

#endif // SBP_SBP_HPP
