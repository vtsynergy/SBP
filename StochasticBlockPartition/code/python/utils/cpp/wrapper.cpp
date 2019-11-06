#include "partition/partition_wrapper.hpp"
#include "partition/sparse/boost_mapped_matrix_wrapper.hpp"

PYBIND11_MODULE(cppsbp, m) {
    m.doc() = R"pbdoc(
        C Plus Plus Stochastic Block Partitioning Plugin
        -----------------------
        .. currentmodule:: cppsbp
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    add_partition_wrapper(m);
    add_boost_mapped_matrix_wrapper(m);
    // py::register_exception<IndexOutOfBoundsException>(m, "IndexOutOfBoundsError");
    // py::class_<BoostMappedMatrix>(m, "BoostMappedMatrix")
    //     .def(py::init<int, int>())
    //     .def("getrow", &BoostMappedMatrix::getrow)
    //     .def("getcol", &BoostMappedMatrix::getcol)
    //     .def("update_edge_counts", &BoostMappedMatrix::update_edge_counts)
    //     .def("nonzero", &BoostMappedMatrix::nonzero)
    //     .def("values", &BoostMappedMatrix::values)
    //     .def("sum", (int (BoostMappedMatrix::*)()) &BoostMappedMatrix::sum)
    //     .def("sum", (Eigen::VectorXi (BoostMappedMatrix::*)(int)) &BoostMappedMatrix::sum)
    //     // .def("sum", (py::array_t<int> (BoostMappedMatrix::*)(int)) &BoostMappedMatrix::sum)
    //     .def("sub", &BoostMappedMatrix::sub)
    //     .def("add", (void (BoostMappedMatrix::*)(int, int, int)) &BoostMappedMatrix::add)
    //     .def("add", (void (BoostMappedMatrix::*)(int, py::array_t<int>, py::array_t<int>)) &BoostMappedMatrix::add)
    //     .def("copy", &BoostMappedMatrix::copy)
    //     .def("__getitem__", [](BoostMappedMatrix &matrix, py::tuple index) { return matrix[index]; })
    //     .def("outgoing_edges", &BoostMappedMatrix::outgoing_edges)
    //     .def("incoming_edges", &BoostMappedMatrix::incoming_edges)
    //     .def_readonly("shape", &BoostMappedMatrix::shape)
    //     ;
    //======================
    // SBP submodule
    //======================
    // auto sbp = m.def_submodule("sbp")
    // py::class_<py::object>(m, "sbp")
    //     .def_static("propose_new_partition", &sbp::propose_new_partition)
    //     // .def("propose_new_partition", &sbp::propose_new_partition)
    //     ;
    // m.def_submodule("sbp")
    //     .def("propose_new_partition", &sbp::propose_new_partition)
    //     ;
    // auto mlog = m.def_submodule("log");
    // mlog.def("func1", &namespace::func1);
}