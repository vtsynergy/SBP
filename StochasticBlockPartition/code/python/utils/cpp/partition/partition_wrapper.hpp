/***
 * Python wrapper for the partition class.
 */
#ifndef CPPSBP_PARTITION_WRAPPER_HPP
#define CPPSBP_PARTITION_WRAPPER_HPP

#include "partition.hpp"

void add_partition_wrapper(py::module module) {
    py::module partition_module = module.def_submodule("partition");
    py::class_<Partition>(partition_module, "Partition")
        .def(py::init<int, std::vector<Matrix2Column>, float>())
        .def(py::init<int, std::vector<Matrix2Column>, float, Vector>())
        .def("initialize_edge_counts", &Partition::initialize_edge_counts)
        .def("clone_with_true_block_membership", &Partition::clone_with_true_block_membership)
        .def("copy", &Partition::copy)
        .def("from_sample", &Partition::from_sample)
        .def("merge_blocks", &Partition::merge_blocks)
        .def_property("num_blocks", &Partition::getNum_blocks, &Partition::setNum_blocks)
        .def_property("blockmodel", &Partition::getBlockmodel, &Partition::setBlockmodel)
        .def_property("block_assignment", &Partition::getBlock_assignment, &Partition::setBlock_assignment)
        .def_property("block_degrees", &Partition::getBlock_degrees, &Partition::setBlock_degrees)
        .def_property("block_degrees_in", &Partition::getBlock_degrees_in, &Partition::setBlock_degrees_in)
        .def_property("block_degrees_out", &Partition::getBlock_degrees_out, &Partition::setBlock_degrees_out)
        .def_property("block_reduction_rate", &Partition::getBlock_reduction_rate, &Partition::setBlock_reduction_rate)
        .def_property("overall_entropy", &Partition::getOverall_entropy, &Partition::setOverall_entropy)
        .def_property("num_blocks_to_merge", &Partition::getNum_blocks_to_merge, &Partition::setNum_blocks_to_merge)
        ;
}

#endif // CPPSBP_PARTITION_WRAPPER_HPP
