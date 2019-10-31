#include <boost_mapped_matrix.hpp>
// #include <pybind11/operators.h>

PYBIND11_MODULE(cppsbp, m) {
    m.doc() = R"pbdoc(
        C Plus Plus Stochastic Block Partitioning Plugin
        -----------------------
        .. currentmodule:: cppsbp
        .. autosummary::
           :toctree: _generate
    )pbdoc";
    py::register_exception<IndexOutOfBoundsException>(m, "IndexOutOfBoundsError");
    py::class_<BoostMappedMatrix>(m, "BoostMappedMatrix")
        .def(py::init<int, int>())
        .def("getrow", &BoostMappedMatrix::getrow)
        .def("getcol", &BoostMappedMatrix::getcol)
        .def("update_edge_counts", &BoostMappedMatrix::update_edge_counts)
        .def("nonzero", &BoostMappedMatrix::nonzero)
        .def("values", &BoostMappedMatrix::values)
        .def("sum", (int (BoostMappedMatrix::*)()) &BoostMappedMatrix::sum)
        .def("sum", (py::array_t<int> (BoostMappedMatrix::*)(int)) &BoostMappedMatrix::sum)
        .def("sub", &BoostMappedMatrix::sub)
        .def("add", (void (BoostMappedMatrix::*)(int, int, int)) &BoostMappedMatrix::add)
        .def("add", (void (BoostMappedMatrix::*)(int, py::array_t<int>, py::array_t<int>)) &BoostMappedMatrix::add)
        .def("copy", &BoostMappedMatrix::copy)
        .def("__getitem__", [](BoostMappedMatrix &matrix, py::tuple index) { return matrix[index]; })
        .def_readonly("shape", &BoostMappedMatrix::shape)
        ;
}