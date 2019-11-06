#include "boost_mapped_matrix.hpp"

void BoostMappedMatrix::check_row_bounds(int row) {
    if (row < 0 || row >= this->nrows) {
        throw IndexOutOfBoundsException(row, this->nrows);
    }
}

void BoostMappedMatrix::check_col_bounds(int col) {
    if (col < 0 || col >= this->ncols) {
        throw IndexOutOfBoundsException(col, this->ncols);
    }
}

void BoostMappedMatrix::add(int row, int col, int val) {
    check_row_bounds(row);
    check_col_bounds(col);
    matrix(row, col) += val;
}

void BoostMappedMatrix::add(int row, py::array_t<int> cols, py::array_t<int> values) {
    check_row_bounds(row);
    auto col_indices = cols.mutable_unchecked<1>();
    auto vals = values.mutable_unchecked<1>();
    for (int i = 0; i < col_indices.size(); ++i) {
        int col = col_indices[i];
        check_col_bounds(col);
        this->matrix(row, col) += vals[i];
    }
}

void BoostMappedMatrix::sub(int row, int col, int val) {
    check_row_bounds(row);
    check_col_bounds(col);
    matrix(row, col) -= val;
}

py::array_t<int> BoostMappedMatrix::getrow(int row) {
    check_row_bounds(row);
    int row_values [this->ncols];
    for (int col = 0; col < ncols; ++col) {
        row_values[col] = matrix(row, col);
    }
    return py::array_t<int>(this->ncols, row_values);
}

py::array_t<int> BoostMappedMatrix::getcol(int col) {
    check_col_bounds(col);
    int col_values [this->nrows];
    for (int row = 0; row < nrows; ++row) {
        col_values[row] = matrix(row, col);
    }
    return py::array_t<int>(this->nrows, col_values);
}

void BoostMappedMatrix::update_edge_counts(int current_block, int proposed_block, py::array_t<int> current_row,
    py::array_t<int> proposed_row, py::array_t<int> current_col, py::array_t<int> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    auto current_row_array = current_row.mutable_unchecked<1>();
    auto proposed_row_array = proposed_row.mutable_unchecked<1>();
    for (int col = 0; col < ncols; ++col) {
        int current_val = current_row_array(col);
        if (current_val == 0)
            this->matrix.erase_element(current_block, col);
        else
            this->matrix(current_block, col) = current_val;
        int proposed_val = proposed_row_array(col);
        if (proposed_val == 0)
            this->matrix.erase_element(proposed_block, col);
        else
            this->matrix(proposed_block, col) = proposed_val;
    }
    auto current_col_array = current_col.mutable_unchecked<1>();
    auto proposed_col_array = proposed_col.mutable_unchecked<1>();
    for (int row = 0; row < nrows; ++row) {
        int current_val = current_col_array(row);
        if (current_val == 0)
            this->matrix.erase_element(row, current_block);
        else
            this->matrix(row, current_block) = current_val;
        int proposed_val = proposed_col_array(row);
        if (proposed_val == 0)
            this->matrix.erase_element(row, proposed_block);
        else
            this->matrix(row, proposed_block) = proposed_val;
    }
}

py::tuple BoostMappedMatrix::nonzero() {
    std::vector<int> row_vector;
    std::vector<int> col_vector;
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            if (matrix(row, col) != 0) {
                row_vector.push_back(row);
                col_vector.push_back(col);
            }
        }
    }
    // Note: this will copy the values from row_vector into rows
    py::array_t<int> rows(row_vector.size(), row_vector.data());
    py::array_t<int> cols(col_vector.size(), col_vector.data());
    return py::make_tuple(rows, cols);
}

py::array_t<int> BoostMappedMatrix::values() {
    std::vector<int> value_vector;
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            int value = matrix(row, col);
            if (value != 0) {
                value_vector.push_back(value);
            }
        }
    }
    return py::array_t<int>(value_vector.size(), value_vector.data());
}

int BoostMappedMatrix::sum() {
    int total = 0;
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            total += matrix(row, col);
        }
    }
    return total;
}

// py::array_t<int> BoostMappedMatrix::sum(int axis) {
Eigen::VectorXi BoostMappedMatrix::sum(int axis) {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        Eigen::VectorXi totals = Eigen::VectorXi::Zero(this->ncols);
        for (int row = 0; row < this->nrows; ++row) {
            for (int col = 0; col < this->ncols; ++col) {
                totals[col] += this->matrix(row, col);
            }
        }
        return totals;  // py::array_t<int>(this->ncols, totals);
    } else {  // (axis == 1) sum across rows
        Eigen::VectorXi totals = Eigen::VectorXi::Zero(this->nrows);
        for (int row = 0; row < this->nrows; ++row) {
            for (int col = 0; col < this->ncols; ++col) {
                totals[row] += this->matrix(row, col);
            }
        }
        return totals;
    }
}

BoostMappedMatrix BoostMappedMatrix::copy() {
    BoostMappedMatrix boost_mapped_matrix(this->nrows, this->ncols);
    boost_mapped_matrix.matrix = boost::numeric::ublas::mapped_matrix<int>(this->matrix);
    return boost_mapped_matrix;
}

int BoostMappedMatrix::operator[] (py::tuple index) {
    py::array_t<int> tuple_array(index);
    auto tuple_vals = tuple_array.mutable_unchecked<1>();
    int row = tuple_vals(0);
    int col = tuple_vals(1);
    return this->matrix(row, col);
}

EdgeWeights BoostMappedMatrix::outgoing_edges(int block) {
    check_row_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    for (int col = 0; col < this->ncols; ++col) {
        int value = this->matrix(block, col);
        if (value != 0) {
            indices.push_back(col);
            values.push_back(value);
        } else {
            this->matrix.erase_element(block, col);
        }
    }
    return std::make_pair(indices, values);
}

// py::tuple BoostMappedMatrix::outgoing_edges(int block) {
//     check_row_bounds(block);
//     std::vector<int> indices;
//     std::vector<int> values;
//     for (int col = 0; col < this->ncols; ++col) {
//         int value = this->matrix(block, col);
//         if (value != 0) {
//             indices.push_back(col);
//             values.push_back(value);
//         } else {
//             this->matrix.erase_element(block, col);
//         }
//     }
//     py::array_t<int> indices_array(indices.size(), indices.data());
//     py::array_t<int> values_array(values.size(), values.data());
//     return py::make_tuple(indices_array, values_array);
//     // return py::array_t<int>(this->ncols, row_values);
//     // std::vector<int> indices;
//     // std::vector<int> values;
//     // for (int col = 0; col < )
//     // for (int i = 0; i < )
//     // out_blocks = block_matrix.getrow(block)
//     // out_blocks_nonzero = out_blocks.nonzero()[0]  # indices
//     // out_blocks = np.vstack((out_blocks_nonzero, out_blocks[out_blocks_nonzero])).T
// }

EdgeWeights BoostMappedMatrix::incoming_edges(int block) {
    check_col_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    for (int row = 0; row < this->nrows; ++row) {
        int value = this->matrix(row, block);
        if (value != 0) {
            indices.push_back(row);
            values.push_back(value);
        } else {
            this->matrix.erase_element(row, block);
        }
    }
    return std::make_pair(indices, values);
}

// py::tuple BoostMappedMatrix::incoming_edges(int block) {
//     check_col_bounds(block);
//     std::vector<int> indices;
//     std::vector<int> values;
//     for (int row = 0; row < this->nrows; ++row) {
//         int value = this->matrix(row, block);
//         if (value != 0) {
//             indices.push_back(row);
//             values.push_back(value);
//         } else {
//             this->matrix.erase_element(row, block);
//         }
//     }
//     py::array_t<int> indices_array(indices.size(), indices.data());
//     py::array_t<int> values_array(values.size(), values.data());
//     return py::make_tuple(indices_array, values_array);
// }
