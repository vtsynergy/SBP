#include "dict_matrix.hpp"

void DictMatrix::add(int row, int col, int val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    matrix[row][col] += val;
}

void DictMatrix::check_row_bounds(int row) {
    if (row < 0 || row >= this->nrows) {
        throw IndexOutOfBoundsException(row, this->nrows);
    }
}

void DictMatrix::check_col_bounds(int col) {
    if (col < 0 || col >= this->ncols) {
        throw IndexOutOfBoundsException(col, this->ncols);
    }
}

DictMatrix DictMatrix::copy() {
    // std::vector<std::unordered_map<int, int>> dict_matrix(this->nrows, std::unordered_map<int, int>());
    DictMatrix dict_matrix(this->nrows, this->ncols);
    for (int i = 0; i < this->nrows; ++i) {
        const std::unordered_map<int, int> row = this->matrix[i];
        dict_matrix.matrix[i] = row;  // TODO: double-check that this is a copy constructor
    }
    return dict_matrix;
}

int DictMatrix::get(int row, int col) {
    check_row_bounds(row);
    check_col_bounds(col);
    return matrix[row][col];
}

std::vector<int> DictMatrix::getrow(int row) {
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
    // for (int col = 0; col < ncols; ++col) {
    //     row_values[col] = matrix[row][col];
    // }
    return row_values;  // py::array_t<int>(this->ncols, row_values);
}

std::vector<int> DictMatrix::getcol(int col) {
    std::vector<int> col_values(this->nrows, 0);
    for (int row = 0; row < nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            if (element.first == col) {
                col_values[row] = element.second;
                break;
            }
        }
    }
    return col_values;
}

EdgeWeights DictMatrix::incoming_edges(int block) {
    check_col_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    for (int row = 0; row < this->nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            if (element.first == block) {
                indices.push_back(row);
                values.push_back(element.second);
                break;
            }
        }
    }
    return EdgeWeights {indices, values};
}

Indices DictMatrix::nonzero() {
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

void DictMatrix::sub(int row, int col, int val) {
    check_row_bounds(row);
    check_col_bounds(col);
    // TODO: debug mode - if matrix[row][col] doesn't exist, throw exception
    matrix[row][col] -= val;
}

int DictMatrix::sum() {
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

std::vector<int> DictMatrix::sum(int axis) {
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

int DictMatrix::trace() {
    int total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (int index = 0; index < this->nrows; ++index) {
        // TODO: this creates 0 elements where they don't exist. To optimize memory, could add a find call first
        total += this->matrix[index][index];
    }
    return total;
}

EdgeWeights DictMatrix::outgoing_edges(int block) {
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

void DictMatrix::update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
    std::vector<int> proposed_row, std::vector<int> current_col, std::vector<int> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (int col = 0; col < ncols; ++col) {
        int current_val = current_row[col];
        if (current_val == 0)
            this->matrix[current_block].erase(col);
        else
            this->matrix[current_block][col] = current_val;
        int proposed_val = proposed_row[col];
        if (proposed_val == 0)
            this->matrix[proposed_block].erase(col);
        else
            this->matrix[proposed_block][col] = proposed_val;
    }
    for (int row = 0; row < nrows; ++row) {
        int current_val = current_col[row];
        if (current_val == 0)
            this->matrix[row].erase(current_block);
        else
            this->matrix[row][current_block] = current_val;
        int proposed_val = proposed_col[row];
        if (proposed_val == 0)
            this->matrix[row].erase(proposed_block);
        else
            this->matrix[row][proposed_block] = proposed_val;
    }
}

std::vector<int> DictMatrix::values() {
    // TODO: maybe return a sparse vector every time?
    std::vector<int> values;
    for (int row = 0; row < nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            values.push_back(element.second);
        }
        // for (int col = 0; col < ncols; ++col) {
        //     int value = matrix(row, col);
        //     if (value != 0) {
        //         values.push_back(value);
        //     }
        // }
    }
    return values;
}
