#ifndef CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_WRAPPER_HPP
#define CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_WRAPPER_HPP

#include <boost_mapped_matrix.hpp>

void add_boost_mapped_matrix_wrapper(py::module module) {
    py::module sparse = module.def_submodule("sparse");
    py::register_exception<IndexOutOfBoundsException>(sparse, "IndexOutOfBoundsError");
    py::class_<BoostMappedMatrix>(sparse, "BoostMappedMatrix")
        .def(py::init<int, int>())
        .def("getrow", &BoostMappedMatrix::_getrow)
        .def("getcol", &BoostMappedMatrix::_getcol)
        .def("update_edge_counts", &BoostMappedMatrix::_update_edge_counts)
        .def("nonzero", &BoostMappedMatrix::_nonzero)
        .def("values", &BoostMappedMatrix::values)
        .def("sum", (int (BoostMappedMatrix::*)()) &BoostMappedMatrix::sum)
        .def("sum", (Eigen::VectorXi (BoostMappedMatrix::*)(int)) &BoostMappedMatrix::sum)
        .def("trace", &BoostMappedMatrix::trace)
        // .def("sum", (py::array_t<int> (BoostMappedMatrix::*)(int)) &BoostMappedMatrix::sum)
        .def("sub", &BoostMappedMatrix::sub)
        .def("add", (void (BoostMappedMatrix::*)(int, int, int)) &BoostMappedMatrix::add)
        .def("add", (void (BoostMappedMatrix::*)(int, py::array_t<int>, py::array_t<int>)) &BoostMappedMatrix::add)
        .def("copy", &BoostMappedMatrix::copy)
        .def("__getitem__", [](BoostMappedMatrix &matrix, py::tuple index) { return matrix[index]; })
        .def("outgoing_edges", &BoostMappedMatrix::_outgoing_edges)
        .def("incoming_edges", &BoostMappedMatrix::_incoming_edges)
        .def_readonly("shape", &BoostMappedMatrix::shape)
        ;
}

#endif // CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_WRAPPER_HPP
