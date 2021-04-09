/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP

#include <unordered_map>

#include "csparse_matrix.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "../../utils.hpp"

// #include <Eigen/Core>

/**
 * C++ interface of the dictionary (map of maps) sparse matrix
 */
// class DictMatrix : public CSparseMatrix {
class DictMatrix : public ISparseMatrix {
  public:
    DictMatrix() {}
    DictMatrix(int nrows, int ncols) {  // : ncols(ncols), nrows(nrows) {
        this->ncols = ncols;
        this->nrows = nrows;
        // this->matrix = boost::numeric::ublas::coordinate_matrix<int>(this->nrows, this->ncols);
        this->matrix = std::vector<std::unordered_map<int, int>>(this->nrows, std::unordered_map<int, int>());
        // this->matrix = boost::numeric::ublas::mapped_matrix<int>(this->nrows, this->ncols);
        // int shape_array[2] = {this->nrows, this->ncols};
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    virtual void add(int row, int col, int val) override;
    // virtual void add(int row, std::vector<int> cols, std::vector<int> values) override;
    virtual void clearrow(int row) override;
    virtual void clearcol(int col) override;
    virtual ISparseMatrix* copy() const override;
    virtual int get(int row, int col) const override;
    virtual std::vector<int> getcol(int col) const override;
    virtual MapVector<int> getcol_sparse(int col) const override;
    virtual void getcol_sparse(int col, MapVector<int> &col_vector) const override;
    // virtual MapVector<int> getcol_sparse(int col) override;
    // virtual const MapVector<int>& getcol_sparse(int col) const override;
    virtual std::vector<int> getrow(int row) const override;
    virtual MapVector<int> getrow_sparse(int row) const override;
    virtual void getrow_sparse(int row, MapVector<int> &row_vector) const override;
    // virtual MapVector<int> getrow_sparse(int row) override;
    // virtual const MapVector<int>& getrow_sparse(int row) const override;
    virtual EdgeWeights incoming_edges(int block) const override;
    virtual Indices nonzero() const override;
    virtual EdgeWeights outgoing_edges(int block) const override;
    /// Sets the values in a row equal to the input vector
    virtual void setrow(int row, const MapVector<int> &vector) override;
    /// Sets the values in a column equal to the input vector
    virtual void setcol(int col, const MapVector<int> &vector) override;
    virtual void sub(int row, int col, int val) override;
    virtual int sum() const override;
    virtual std::vector<int> sum(int axis = 0) const override;
    virtual int trace() const override;
    virtual void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                                    std::vector<int> proposed_row, std::vector<int> current_col,
                                    std::vector<int> proposed_col) override;
    virtual std::vector<int> values() const override;

  private:
    // void check_row_bounds(int row);
    // void check_col_bounds(int col);
    // int ncols;
    // int nrows;
    std::vector<std::unordered_map<int, int>> matrix;
    // boost::numeric::ublas::mapped_matrix<int> matrix;
    // boost::numeric::ublas::coordinate_matrix<int> matrix;
};

#endif // CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP
