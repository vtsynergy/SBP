/**
 * The stochastic block blockmodeling module.
 */
#ifndef SBP_SBP_HPP
#define SBP_SBP_HPP

#include <omp.h>

#include "args.hpp"
#include "block_merge.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"

namespace sbp {

/// Performs community detection on the provided graph, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args);

/// Returns true if the exit condition is reached based on the provided blockmodels
bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks = 0);

namespace dist {

// Blockmodel stochastic_block_partition(Graph &graph, MPI_Data &mpi, Args &args);
Blockmodel stochastic_block_partition(Graph &graph, Args &args);

} // namespace dist

} // namespace sbp

#endif // SBP_SBP_HPP
