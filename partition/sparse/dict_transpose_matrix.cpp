#include "dict_transpose_matrix.hpp"

void DictTransposeMatrix::add(int row, int col, int val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    this->matrix[row][col] += val;
    this->matrix_transpose[col][row] += val;
}

void DictTransposeMatrix::check_row_bounds(int row) {
    if (row < 0 || row >= this->nrows) {
        throw IndexOutOfBoundsException(row, this->nrows);
    }
}

void DictTransposeMatrix::check_col_bounds(int col) {
    if (col < 0 || col >= this->ncols) {
        throw IndexOutOfBoundsException(col, this->ncols);
    }
}

DictTransposeMatrix DictTransposeMatrix::copy() {
    // std::vector<std::unordered_map<int, int>> dict_matrix(this->nrows, std::unordered_map<int, int>());
    DictTransposeMatrix dict_matrix(this->nrows, this->ncols);
    for (int i = 0; i < this->nrows; ++i) {
        const std::unordered_map<int, int> row = this->matrix[i];
        dict_matrix.matrix[i] = row;  // TODO: double-check that this is a copy constructor
    }
    for (int i = 0; i < this->ncols; ++i) {
        const std::unordered_map<int, int> col = this->matrix_transpose[i];
        dict_matrix.matrix_transpose[i] = col;
    }
    return dict_matrix;
}

int DictTransposeMatrix::get(int row, int col) {
    check_row_bounds(row);
    check_col_bounds(col);
    return matrix[row][col];
}

std::vector<int> DictTransposeMatrix::getcol(int col) {
    check_col_bounds(col);
    std::vector<int> col_values(this->nrows, 0);
    const std::unordered_map<int, int> &matrix_col = this->matrix_transpose[col];
    for (const std::pair<int, int> element : matrix_col) {
        col_values[element.first] = element.second;
    }
    return col_values;
}

std::map<int, int> DictTransposeMatrix::getcol_sparse(int col) {
    check_col_bounds(col);
    std::map<int, int> col_values;
    const std::unordered_map<int, int> &matrix_col = this->matrix_transpose[col];
    for (const std::pair<int, int> element : matrix_col) {
        // col_values[element.first] = element.second;
        col_values.insert(element);
    }
    return col_values;
}

std::vector<int> DictTransposeMatrix::getrow(int row) {
    check_row_bounds(row);
    std::vector<int> row_values = utils::constant<int>(this->ncols, 0);
    // int row_values [this->ncols];
    // NOTE: could save some time by pulling this->matrix[row] out, and then iterating over it using references
    // but, this could be not thread safe
    // for (int row_index = 0; row_index < this->nrows; ++row_index) {
    const std::unordered_map<int, int> &matrix_row = this->matrix[row];
    for (const std::pair<int, int> element : matrix_row) {
        row_values[element.first] = element.second;
    }
    return row_values;
}

std::map<int, int> DictTransposeMatrix::getrow_sparse(int row) {
    check_row_bounds(row);
    std::map<int, int> row_values;
    const std::unordered_map<int, int> &matrix_row = this->matrix[row];
    for (const std::pair<int, int> element : matrix_row) {
        row_values[element.first] = element.second;
    }
    return row_values;
}

EdgeWeights DictTransposeMatrix::incoming_edges(int block) {
    check_col_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    const std::unordered_map<int, int> &block_col = this->matrix_transpose[block];
    for (const std::pair<int, int> &element : block_col) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};

    // for (int row = 0; row < this->nrows; ++row) {
    //     const std::unordered_map<int, int> &matrix_row = this->matrix[row];
    //     for (const std::pair<int, int> &element : matrix_row) {
    //         if (element.first == block) {
    //             indices.push_back(row);
    //             values.push_back(element.second);
    //             break;
    //         }
    //     }
    // }
    // return EdgeWeights {indices, values};
}

Indices DictTransposeMatrix::nonzero() {
    std::vector<int> row_vector;
    std::vector<int> col_vector;
    for (int row = 0; row < nrows; ++row) {
        std::unordered_map<int, int> matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            row_vector.push_back(row);
            col_vector.push_back(element.first);
        }
        // for (int col = 0; col < ncols; ++col) {
        //     if (matrix(row, col) != 0) {
        //         row_vector.push_back(row);
        //         col_vector.push_back(col);
        //     }
        // }
    }
    return Indices{row_vector, col_vector};
}

void DictTransposeMatrix::sub(int row, int col, int val) {
    check_row_bounds(row);
    check_col_bounds(col);
    // TODO: debug mode - if matrix[row][col] doesn't exist, throw exception
    this->matrix[row][col] -= val;
    this->matrix_transpose[col][row] -= val;
}

int DictTransposeMatrix::sum() {
    int total = 0;
    for (int row = 0; row < nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            total += element.second;
        }
        // for (int col = 0; col < ncols; ++col) {
        //     total += matrix(row, col);
        // }
    }
    return total;
}

std::vector<int> DictTransposeMatrix::sum(int axis) {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<int> totals(this->ncols, 0);
        for (int row_index = 0; row_index < this->nrows; ++row_index) {
        // for (const std::unordered_map<int, int> &row : this->matrix) {
            const std::unordered_map<int, int> &row = this->matrix[row_index];
            for (const std::pair<int, int> &element : row) {
                totals[element.first] += totals[element.second];
            }
        }
        // for (int row = 0; row < this->nrows; ++row) {
        //     for (int col = 0; col < this->ncols; ++col) {
        //         totals[col] += this->matrix(row, col);
        //     }
        // }
        return totals;  // py::array_t<int>(this->ncols, totals);
    } else {  // (axis == 1) sum across rows
        std::vector<int> totals(this->nrows, 0);
        for (int row = 0; row < this->nrows; ++row) {
            const std::unordered_map<int, int> &matrix_row = this->matrix[row];
            for (const std::pair<int, int> &element : matrix_row) {
                totals[row] += element.second;
            }
            // for (int col = 0; col < this->ncols; ++col) {
            //     totals[row] += this->matrix(row, col);
            // }
        }
        return totals;
    }
}

int DictTransposeMatrix::trace() {
    int total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (int index = 0; index < this->nrows; ++index) {
        // TODO: this creates 0 elements where they don't exist. To optimize memory, could add a find call first
        total += this->matrix[index][index];
    }
    return total;
}

EdgeWeights DictTransposeMatrix::outgoing_edges(int block) {
    check_row_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    const std::unordered_map<int, int> &block_row = this->matrix[block];
    for (const std::pair<int, int> &element : block_row) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

void DictTransposeMatrix::update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
    std::vector<int> proposed_row, std::vector<int> current_col, std::vector<int> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (int col = 0; col < ncols; ++col) {
        int current_val = current_row[col];
        if (current_val == 0) {
            this->matrix[current_block].erase(col);
            this->matrix_transpose[col].erase(current_block);
        } else {
            this->matrix[current_block][col] = current_val;
            this->matrix_transpose[col][current_block] = current_val;
        }
        int proposed_val = proposed_row[col];
        if (proposed_val == 0) {
            this->matrix[proposed_block].erase(col);
            this->matrix_transpose[col].erase(proposed_block);
        } else {
            this->matrix[proposed_block][col] = proposed_val;
            this->matrix_transpose[col][proposed_block] = proposed_val;  
        }
    }
    for (int row = 0; row < nrows; ++row) {
        int current_val = current_col[row];
        if (current_val == 0) {
            this->matrix[row].erase(current_block);
            this->matrix_transpose[current_block].erase(row);
        } else {
            this->matrix[row][current_block] = current_val;
            this->matrix_transpose[current_block][row] = current_val;
        }
        int proposed_val = proposed_col[row];
        if (proposed_val == 0) {
            this->matrix[row].erase(proposed_block);
            this->matrix_transpose[proposed_block].erase(row);
        } else {
            this->matrix[row][proposed_block] = proposed_val;
            this->matrix_transpose[proposed_block][row] = proposed_val;
        }
    }
}

std::vector<int> DictTransposeMatrix::values() {
    // TODO: maybe return a sparse vector every time?
    std::vector<int> values;
    for (int row = 0; row < nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            values.push_back(element.second);
        }
    }
    return values;
}
