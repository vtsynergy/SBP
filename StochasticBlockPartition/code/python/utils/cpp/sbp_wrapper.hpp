/***
 * Python wrapper for the stochastic block partitioning methods.
 */
#ifndef CPPSBP_SBP_WRAPPER_HPP
#define CPPSBP_SBP_WRAPPER_HPP

#include "sbp.hpp"

void add_sbp_wrapper(py::module module) {
    module.def_submodule("sbp")
        .def("merge_blocks", sbp::merge_blocks);
}

#endif // CPPSBP_SBP_WRAPPER_HPP
