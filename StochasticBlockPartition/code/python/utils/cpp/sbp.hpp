/**
 * The stochastic block partitioning module.
 */
#ifndef CPPSBP_SBP_HPP
#define CPPSBP_SBP_HPP

#include <numeric>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "partition/partition.hpp"
#include "util/util.hpp"


namespace py = pybind11;

namespace sbp {

struct proposal_evaluation_t {
    int proposed_block;
    float delta_entropy;
} ProposalEvaluation;

struct proposal_and_edge_counts_t {
    int proposal;
    int num_out_neighbor_edges;
    int num_in_neighbor_edges;
    int num_neighbor_edges;
} ProposalAndEdgeCounts;

/*******************
 * BLOCK MERGE
 ******************/
Partition merge_blocks(Partition partition, int num_agg_proposals_per_block, std::vector<Matrix2Column> neighbors out_neighbors);  // , py::object Evaluation);
ProposalEvaluation propose_merge(int current_block, Partition partition, Vector block_partition);  // , py::object block_merge_timings);
ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights out_blocks, EdgeWeights in_blocks, Vector block_partition, Partition partition);
int propose_random_block(int current_block, int num_blocks);
// propose_new_partition(int, pybind11::array_t<int, 16>, pybind11::array_t<int, 16>, pybind11::array_t<int, 16>, pybind11::object, bool, bool)
// py::tuple propose_new_partition(int r, py::array_t<int> neighbors_out, py::array_t<int> neighbors_in, py::array_t<int> b, py::object partition, bool agg_move, bool use_sparse);

/*******************
 * FINE-TUNE
 ******************/
}

#endif // CPPSBP_SBP_HPP
