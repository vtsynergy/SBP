/***
 * Common interface for sparse matrix types.
 */
#ifndef CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP

#include <exception>
#include <iostream>
#include <memory>
#include <sstream>

// #include <Eigen/Core>

#include "typedefs.hpp"

// typedef Eigen::VectorXi Vector;

typedef struct edge_weights_t {
    std::vector<int> indices;
    std::vector<int> values;
    // void print() {
    //     std::cout << "indices: " << Eigen::Map<Vector>(this->indices.data(), this->indices.size()).transpose();
    //     std::cout << " with size: " << this->indices.size() << std::endl;
    //     std::cout << "values: " << Eigen::Map<Vector>(this->values.data(), this->values.size()).transpose();
    //     std::cout << " with size: " << this->values.size() << std::endl;
    // }
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

///
/// C++ implementation of the sparse matrix interface
///
class ISparseMatrix {
public:
    // ISparseMatrix() {}
    virtual ~ISparseMatrix() {}
    virtual void add(int row, int col, int val) = 0;
    // virtual void add(int row, std::vector<int> cols, std::vector<int> values) = 0;
    virtual void clearrow(int row) = 0;
    virtual void clearcol(int col) = 0;
    virtual ISparseMatrix* copy() const = 0;
    virtual int get(int row, int col) const = 0;
    virtual std::vector<int> getcol(int col) const = 0;
    virtual MapVector<int> getcol_sparse(int col) const = 0;
    virtual void getcol_sparse(int col, MapVector<int> &col_vector) const = 0;
    // virtual const MapVector<int>& getcol_sparse(int col) const = 0;
    virtual std::vector<int> getrow(int row) const = 0;
    virtual MapVector<int> getrow_sparse(int row) const = 0;
    virtual void getrow_sparse(int row, MapVector<int> &row_vector) const = 0;
    // virtual const MapVector<int>& getrow_sparse(int row) const = 0;
    virtual EdgeWeights incoming_edges(int block) const = 0;
    virtual Indices nonzero() const = 0;
    virtual EdgeWeights outgoing_edges(int block) const = 0;
    /// Sets the values in a row equal to the input vector
    virtual void setrow(int row, const MapVector<int> &vector) = 0;
    /// Sets the values in a column equal to the input vector
    virtual void setcol(int col, const MapVector<int> &vector) = 0;
    virtual void sub(int row, int col, int val) = 0;
    virtual int sum() const = 0;
    virtual std::vector<int> sum(int axis = 0) const = 0;
    virtual int trace() const = 0;
    virtual void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                                    std::vector<int> proposed_row, std::vector<int> current_col,
                                    std::vector<int> proposed_col) = 0;
    virtual std::vector<int> values() const = 0;
    std::pair<int, int> shape;

protected:
    void check_row_bounds(int row) const {
        if (row < 0 || row >= this->nrows) {
            throw IndexOutOfBoundsException(row, this->nrows);
        }
    }
    void check_col_bounds(int col) const {
        if (col < 0 || col >= this->ncols) {
            throw IndexOutOfBoundsException(col, this->ncols);
        }
    }
    int ncols;
    int nrows;
};

///
/// C++ implementation of the distributed sparse matrix interface
///
class IDistSparseMatrix : public ISparseMatrix {
public:
    // IDistSparseMatrix() {}
    virtual ~IDistSparseMatrix() {}
    /// Returns true if this process owns this block.
    virtual bool owns(int block) const = 0;
    /// Returns a copy of this distributed matrix.
    virtual IDistSparseMatrix* copyDistSparseMatrix() const = 0;

protected:
    std::vector<int> _ownership;
    virtual void sync_ownership(const std::vector<int> &myblocks) = 0;

};

// typedef std::unique_ptr<ISparseMatrix> SparseMatrix;

#endif // CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
