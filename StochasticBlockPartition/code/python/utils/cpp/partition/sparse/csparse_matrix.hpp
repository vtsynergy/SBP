/***
 * Common interface for sparse matrix types.
 */
#ifndef CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP

#include <exception>
#include <sstream>

#include <Eigen/Core>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class IndexOutOfBoundsException: public std::exception {
public:
    IndexOutOfBoundsException(int index, int max) : index(index), max(max) {
        std::ostringstream message_stream;
        message_stream << "Index " << index << " is out of bounds [0, " << max - 1 << "]";
        this->message = message_stream.str();
    }
    virtual const char* what() const noexcept override {
        return this->message.c_str();
    }
private:
    int index;
    int max;
    std::string message;
};

/****
 * C++ implementation of the sparse matrix interface
 */
class CSparseMatrix {
public:
    virtual py::array_t<int> getrow(int row) = 0;
    virtual py::array_t<int> getcol(int col) = 0;
    virtual void update_edge_counts(int current_block, int proposed_block, py::array_t<int> current_row,
        py::array_t<int> proposed_row, py::array_t<int> current_col, py::array_t<int> proposed_col) = 0;
    virtual py::tuple nonzero() = 0;
    virtual py::array_t<int> values() = 0;
    virtual int sum() = 0;
    // virtual py::array_t<int> sum(int axis = 0) = 0;
    virtual Eigen::VectorXi sum(int axis = 0) = 0;
    virtual void sub(int row, int col, int val) = 0;
    virtual void add(int row, int col, int val) = 0;
    virtual void add(int row, py::array_t<int> cols, py::array_t<int> values) = 0;
    virtual int operator[] (py::tuple index) = 0; 
    py::array_t<int> shape;
};

#endif // CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
