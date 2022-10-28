/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP

#include <map>
#include <unordered_map>

#include "csparse_matrix.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "delta.hpp"
#include "typedefs.hpp"
#include "../../utils.hpp"

// #include <Eigen/Core>

/**
 * The list-of-maps sparse matrix, with a transpose for faster column indexing.
 * TODO: figure out where 0s are being added to the matrix, and whether or not we need to get rid of that
 */
class DictTransposeMatrix : public ISparseMatrix {
  public:
    DictTransposeMatrix() = default;
    DictTransposeMatrix(int nrows, int ncols, int buckets = 10) {
        this->ncols = ncols;
        this->nrows = nrows;
        this->matrix = std::vector<MapVector<int>>(this->nrows, MapVector<int>(buckets));
        this->matrix_transpose = std::vector<MapVector<int>>(this->ncols, MapVector<int>(buckets));
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    void add(int row, int col, int val) override;
    void add_transpose(int row, int col, int val);
    // void add(int row, std::vector<int> cols, std::vector<int> values) override;
    /// Clears the value in a given row. Complexity ~O(number of blocks).
    void clearrow(int row) override;
    /// Clears the values in a given column. Complexity ~O(number of blocks).
    void clearcol(int col) override;
    /// Returns a copy of the current matrix.
    ISparseMatrix* copy() const override;
    int distinct_edges(int block) const override;
    std::vector<std::tuple<int, int, int>> entries() const override;
    int get(int row, int col) const override;
    /// Returns all values in the requested column as a dense vector.
    std::vector<int> getcol(int col) const override;
    /// Returns all values in the requested column as a sparse vector (ordered map).
    MapVector<int> getcol_sparse(int col) const override;
    const MapVector<int>& getcol_sparseref(int col) const override;
    void getcol_sparse(int col, MapVector<int> &col_vector) const override;
    /// Returns all values in the requested row as a dense vector.
    std::vector<int> getrow(int row) const override;
    /// Returns all values in the requested column as a sparse vector (ordered map).
    MapVector<int> getrow_sparse(int row) const override;
    const MapVector<int>& getrow_sparseref(int row) const override;
    // const MapVector<int>& getrow_sparse(int row) const;
    void getrow_sparse(int row, MapVector<int> &row_vector) const override;
    EdgeWeights incoming_edges(int block) const override;
    std::set<int> neighbors(int block) const override;
    MapVector<int> neighbors_weights(int block) const override;
    Indices nonzero() const override;
    EdgeWeights outgoing_edges(int block) const override;
    /// Sets the values in a row equal to the input vector.
    void setrow(int row, const MapVector<int> &vector) override;
    /// Sets the values in a column equal to the input vector.
    void setcol(int col, const MapVector<int> &vector) override;
    void sub(int row, int col, int val) override;
    int edges() const override;
    void print() const override;
    std::vector<int> sum(int axis = 0) const override;
    int trace() const override;
    void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                                    std::vector<int> proposed_row, std::vector<int> current_col,
                                    std::vector<int> proposed_col) override;
    void update_edge_counts(int current_block, int proposed_block, MapVector<int> current_row,
                            MapVector<int> proposed_row, MapVector<int> current_col,
                            MapVector<int> proposed_col) override;
    void update_edge_counts(const Delta &delta) override;
    bool validate(int row, int col, int val) const override;
    std::vector<int> values() const override;

  private:
    // void check_row_bounds(int row);
    // void check_col_bounds(int col);
    // void check_row_bounds(int row) const;
    // void check_col_bounds(int col) const;
    // int ncols;
    // int nrows;
//    std::vector<std::unordered_map<int, int>> matrix;
//    std::vector<std::unordered_map<int, int>> matrix_transpose;
    std::vector<MapVector<int>> matrix;
    std::vector<MapVector<int>> matrix_transpose;
};

#endif // CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP
