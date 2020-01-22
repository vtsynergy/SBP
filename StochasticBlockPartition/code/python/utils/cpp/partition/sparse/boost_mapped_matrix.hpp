/***
 * Sparse Matrix based on the Boost mapped_matrix
 */
#ifndef CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP

#include "csparse_matrix.hpp"

#include <Eigen/Core>

/**
 * C++ interface of the dictionary (map of maps) sparse matrix
 */
// class BoostMappedMatrix : public CSparseMatrix {
class BoostMappedMatrix {
  public:
    BoostMappedMatrix() {}
    BoostMappedMatrix(int nrows, int ncols) : ncols(ncols), nrows(nrows) {
        this->matrix = boost::numeric::ublas::mapped_matrix<int>(this->nrows, this->ncols);
        int shape_array[2] = {this->nrows, this->ncols};
        this->shape = py::array_t<int>(2, shape_array);
    }
    void add(int row, int col, int val);
    void add(int row, py::array_t<int> cols, py::array_t<int> values);
    BoostMappedMatrix copy();
    int get(int row, int col);
    Vector getcol(int col);
    Vector getrow(int row);
    py::array_t<int> _getcol(int col);
    py::array_t<int> _getrow(int row);
    EdgeWeights incoming_edges(int block);
    py::tuple _incoming_edges(int block);
    Indices nonzero();
    py::tuple _nonzero();
    int operator[](py::tuple index);
    EdgeWeights outgoing_edges(int block);
    py::tuple _outgoing_edges(int block);
    void sub(int row, int col, int val);
    int sum();
    Eigen::VectorXi sum(int axis = 0);
    void update_edge_counts(int current_block, int proposed_block, Vector current_row, Vector proposed_row,
                            Vector current_col, Vector proposed_col);
    void _update_edge_counts(int current_block, int proposed_block, py::array_t<int> current_row,
                             py::array_t<int> proposed_row, py::array_t<int> current_col,
                             py::array_t<int> proposed_col);
    Eigen::ArrayXi values();
    py::array_t<int> _values();
    py::array_t<int> shape;

  private:
    void check_row_bounds(int row);
    void check_col_bounds(int col);
    int ncols;
    int nrows;
    boost::numeric::ublas::mapped_matrix<int> matrix;
};

#endif // CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP
