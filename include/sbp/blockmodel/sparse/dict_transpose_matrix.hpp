/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP

#include <map>
#include <unordered_map>

#include "csparse_matrix.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "typedefs.hpp"
#include "../../utils.hpp"

// #include <Eigen/Core>

/**
 * C++ interface of the dictionary (map of maps) sparse matrix, with a transpose for faster column indexing
 * TODO: figure out where 0s are being added to the matrix, and whether or not we need to get rid of that
 */
class DictTransposeMatrix {
  public:
    DictTransposeMatrix() {}
    DictTransposeMatrix(int nrows, int ncols) : ncols(ncols), nrows(nrows) {
        this->matrix = std::vector<std::unordered_map<int, int>>(this->nrows, std::unordered_map<int, int>());
        this->matrix_transpose = std::vector<std::unordered_map<int, int>>(this->ncols, std::unordered_map<int, int>());
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    void add(int row, int col, int val);
    void add(int row, std::vector<int> cols, std::vector<int> values);
    /// Clears the value in a given row. Complexity ~O(number of blocks).
    void clearrow(int row);
    /// Clears the values in a given column. Complexity ~O(number of blocks).
    void clearcol(int col);
    /// Returns a copy of the current matrix.
    DictTransposeMatrix copy();
    int get(int row, int col);
    /// Returns all values in the requested column as a dense vector.
    std::vector<int> getcol(int col);
    /// Returns all values in the requested row as a dense vector.
    std::vector<int> getrow(int row);
    /// Returns all values in the requested column as a sparse vector (ordered map).
    MapVector<int> getcol_sparse(int col);
    /// Returns all values in the requested column as a sparse vector (ordered map) reference.
    const MapVector<int>& getcol_sparse(int col) const;
    /// Returns all values in the requested column as a sparse vector (ordered map).
    MapVector<int> getrow_sparse(int row);
    /// Returns all values in the requested column as a sparse vector (ordered map) reference.
    const MapVector<int>& getrow_sparse(int row) const;
    EdgeWeights incoming_edges(int block);
    Indices nonzero();
    EdgeWeights outgoing_edges(int block);
    /// Sets the values in a row equal to the input vector.
    void setrow(int row, const MapVector<int> &vector);
    /// Sets the values in a column equal to the input vector.
    void setcol(int col, const MapVector<int> &vector);
    void sub(int row, int col, int val);
    int sum();
    std::vector<int> sum(int axis = 0);
    int trace();
    void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                            std::vector<int> proposed_row, std::vector<int> current_col, std::vector<int> proposed_col);
    std::vector<int> values();
    std::pair<int, int> shape;

  private:
    void check_row_bounds(int row);
    void check_col_bounds(int col);
    void check_row_bounds(int row) const;
    void check_col_bounds(int col) const;
    int ncols;
    int nrows;
    std::vector<std::unordered_map<int, int>> matrix;
    std::vector<std::unordered_map<int, int>> matrix_transpose;
};

#endif // CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP
