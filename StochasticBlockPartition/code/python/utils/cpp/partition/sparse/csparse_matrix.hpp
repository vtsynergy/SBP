/***
 * Common interface for sparse matrix types.
 */
#ifndef CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP

#include <exception>
#include <sstream>
#include <iostream>

#include <Eigen/Core>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
typedef Eigen::VectorXi Vector;

typedef struct edge_weights_t {
    std::vector<int> indices;
    std::vector<int> values;
    void print() {
        std::cout << "indices: " << Eigen::Map<Vector>(this->indices.data(), this->indices.size()).transpose();
        std::cout << " with size: " << this->indices.size() << std::endl;
        std::cout << "values: " << Eigen::Map<Vector>(this->values.data(), this->values.size()).transpose();
        std::cout << " with size: " << this->values.size() << std::endl;
    }
} EdgeWeights;

typedef struct indices_t {
    std::vector<int> rows;
    std::vector<int> cols;
} Indices;

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
class ISparseMatrix {
public:
    ISparseMatrix() {};
    virtual ~ISparseMatrix() {};
    virtual void add(int row, int col, int val) = 0;
    virtual void add(int row, py::array_t<int> cols, py::array_t<int> values) = 0;
    virtual int get(int row, int col) = 0;
    virtual Vector getcol(int col) = 0;
    virtual Vector getrow(int row) = 0;
    virtual py::array_t<int> _getcol(int col) = 0;
    virtual py::array_t<int> _getrow(int row) = 0;
    virtual Indices nonzero() = 0;
    virtual py::tuple _nonzero() = 0;
    virtual int operator[] (py::tuple index) = 0; 
    virtual void sub(int row, int col, int val) = 0;
    virtual int sum() = 0;
    virtual Eigen::VectorXi sum(int axis = 0) = 0;
    virtual void update_edge_counts(int current_block, int proposed_block, py::array_t<int> current_row,
        py::array_t<int> proposed_row, py::array_t<int> current_col, py::array_t<int> proposed_col) = 0;
    virtual Eigen::ArrayXi values() = 0;
    virtual py::array_t<int> _values() = 0;
    py::array_t<int> shape;
};

#endif // CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
