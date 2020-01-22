/***
 * Python wrapper for the stochastic block partitioning methods.
 */
#ifndef CPPSBP_SBP_WRAPPER_HPP
#define CPPSBP_SBP_WRAPPER_HPP

#include "sbp.hpp"

void add_sbp_wrapper(py::module module) {
    py::module m = module.def_submodule("sbp")
        .def("merge_blocks", &block_merge::merge_blocks)
        .def("reassign_nodes", &finetune::reassign_vertices)
        .def("accept", &finetune::accept)
        .def("block_edge_weights", &finetune::block_edge_weights)
        .def("propose_move", &finetune::propose_move)
        .def("early_stop", &finetune::early_stop)
        .def("overall_entropy", &finetune::overall_entropy)
        .def("edge_weights", &finetune::edge_weights)
        .def("propose_new_block", &common::propose_new_block)
        .def("edge_count_updates", &finetune::edge_count_updates)
        .def("compute_new_block_degrees", &common::compute_new_block_degrees)
        .def("hastings_correction", &finetune::hastings_correction)
        .def("compute_delta_entropy", &finetune::compute_delta_entropy)
        .def("stochastic_block_partition", &sbp::stochastic_block_partition)
        ;
    py::class_<finetune::ProposalEvaluation>(m, "ProposalEvaluation")
        .def_readwrite("delta_entropy", &finetune::ProposalEvaluation::delta_entropy)
        .def_readwrite("did_move", &finetune::ProposalEvaluation::did_move)
        ;
    py::class_<EdgeWeights>(m, "EdgeWeights")
        .def_readwrite("indices", &EdgeWeights::indices)
        .def_readwrite("values", &EdgeWeights::values)
        ;
    py::class_<EdgeCountUpdates>(m, "EdgeCountUpdates")
        .def_readwrite("block_row", &EdgeCountUpdates::block_row)
        .def_readwrite("proposal_row", &EdgeCountUpdates::proposal_row)
        .def_readwrite("block_col", &EdgeCountUpdates::block_col)
        .def_readwrite("proposal_col", &EdgeCountUpdates::proposal_col)
        ;
    py::class_<common::NewBlockDegrees>(m, "NewBlockDegrees")
        .def_readwrite("block_degrees_out", &common::NewBlockDegrees::block_degrees_out)
        .def_readwrite("block_degrees_in", &common::NewBlockDegrees::block_degrees_in)
        .def_readwrite("block_degrees", &common::NewBlockDegrees::block_degrees)
        ;
    py::class_<common::ProposalAndEdgeCounts>(m, "ProposalAndEdgeCounts")
        .def_readwrite("proposal", &common::ProposalAndEdgeCounts::proposal)
        .def_readwrite("num_out_neighbor_edges", &common::ProposalAndEdgeCounts::num_out_neighbor_edges)
        .def_readwrite("num_in_neighbor_edges", &common::ProposalAndEdgeCounts::num_in_neighbor_edges)
        .def_readwrite("num_neighbor_edges", &common::ProposalAndEdgeCounts::num_neighbor_edges)
        ;
}

#endif // CPPSBP_SBP_WRAPPER_HPP
