#include "dict_transpose_matrix.hpp"

void DictTransposeMatrix::add(int row, int col, int val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    this->matrix[row][col] += val;
    this->matrix_transpose[col][row] += val;
}

// void DictTransposeMatrix::check_row_bounds(int row) {
//     if (row < 0 || row >= this->nrows) {
//         throw IndexOutOfBoundsException(row, this->nrows);
//     }
// }

// void DictTransposeMatrix::check_col_bounds(int col) {
//     if (col < 0 || col >= this->ncols) {
//         throw IndexOutOfBoundsException(col, this->ncols);
//     }
// }

// void DictTransposeMatrix::check_row_bounds(int row) const {
//     if (row < 0 || row >= this->nrows) {
//         throw IndexOutOfBoundsException(row, this->nrows);
//     }
// }

// void DictTransposeMatrix::check_col_bounds(int col) const {
//     if (col < 0 || col >= this->ncols) {
//         throw IndexOutOfBoundsException(col, this->ncols);
//     }
// }

void DictTransposeMatrix::clearcol(int col) {
    this->matrix_transpose[col].clear();
    for (MapVector<int> &row : this->matrix) {
        row.erase(col);
    }
}

void DictTransposeMatrix::clearrow(int row) {
    this->matrix[row].clear();
    for (MapVector<int> &col : this->matrix_transpose) {
        col.erase(row);
    }
}

ISparseMatrix* DictTransposeMatrix::copy() const {
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
    return new DictTransposeMatrix(dict_matrix);
}

int DictTransposeMatrix::get(int row, int col) const {
    check_row_bounds(row);
    check_col_bounds(col);
    const MapVector<int> &row_vector = this->matrix[row];
    auto it = row_vector.find(col);
    if (it == row_vector.end())
        return 0;
    return it->second;
    // return this->matrix[row][col];
}

std::vector<int> DictTransposeMatrix::getcol(int col) const {
    check_col_bounds(col);
    std::vector<int> col_values(this->nrows, 0);
    const std::unordered_map<int, int> &matrix_col = this->matrix_transpose[col];
    for (const std::pair<int, int> element : matrix_col) {
        col_values[element.first] = element.second;
    }
    return col_values;
}

MapVector<int> DictTransposeMatrix::getcol_sparse(int col) const {
    check_col_bounds(col);
    return this->matrix_transpose[col];
}

// const MapVector<int>& DictTransposeMatrix::getcol_sparse(int col) const {
//     check_col_bounds(col);
//     return this->matrix_transpose[col];
// }

void DictTransposeMatrix::getcol_sparse(int col, MapVector<int> &col_vector) const {
    check_col_bounds(col);
    col_vector = this->matrix_transpose[col];
}

std::vector<int> DictTransposeMatrix::getrow(int row) const {
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

MapVector<int> DictTransposeMatrix::getrow_sparse(int row) const {
    check_row_bounds(row);
    return this->matrix[row];
}

// const MapVector<int>& DictTransposeMatrix::getrow_sparse(int row) const {
//     check_row_bounds(row);
//     return this->matrix[row];
// }

void DictTransposeMatrix::getrow_sparse(int row, MapVector<int> &row_vector) const {
    check_row_bounds(row);
    row_vector = this->matrix_transpose[row];
}

EdgeWeights DictTransposeMatrix::incoming_edges(int block) const {
    check_col_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    const std::unordered_map<int, int> &block_col = this->matrix_transpose[block];
    for (const std::pair<int, int> &element : block_col) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

Indices DictTransposeMatrix::nonzero() const {
    std::vector<int> row_vector;
    std::vector<int> col_vector;
    for (int row = 0; row < nrows; ++row) {
        std::unordered_map<int, int> matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            row_vector.push_back(row);
            col_vector.push_back(element.first);
        }
    }
    return Indices{row_vector, col_vector};
}

void DictTransposeMatrix::setcol(int col, const MapVector<int> &vector) {
    this->matrix_transpose[col] = MapVector<int>(vector);
    for (int row = 0; row < this->matrix.size(); ++row) {
        MapVector<int>::const_iterator value = vector.find(row);
        if (value == vector.end())  // value is not in vector
            this->matrix[row].erase(col);
        else
            this->matrix[row][col] = value->second;
    }
}

void DictTransposeMatrix::setrow(int row, const MapVector<int> &vector) {
    this->matrix[row] = MapVector<int>(vector);
    for (int col = 0; col < this->matrix.size(); ++col) {
        MapVector<int>::const_iterator value = vector.find(col);
        if (value == vector.end())  // value is not in vector
            this->matrix_transpose[col].erase(row);
        else
            this->matrix_transpose[col][row] = value->second;
    }
}

void DictTransposeMatrix::sub(int row, int col, int val) {
    check_row_bounds(row);
    check_col_bounds(col);
    // TODO: debug mode - if matrix[row][col] doesn't exist, throw exception
    this->matrix[row][col] -= val;
    this->matrix_transpose[col][row] -= val;
}

int DictTransposeMatrix::edges() const {
    int total = 0;
    for (int row = 0; row < nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            total += element.second;
        }
    }
    return total;
}

std::vector<int> DictTransposeMatrix::sum(int axis) const {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<int> totals(this->ncols, 0);
        for (int row_index = 0; row_index < this->nrows; ++row_index) {
            const std::unordered_map<int, int> &row = this->matrix[row_index];
            for (const std::pair<int, int> &element : row) {
                totals[element.first] += totals[element.second];
            }
        }
        return totals;  // py::array_t<int>(this->ncols, totals);
    } else {  // (axis == 1) sum across rows
        std::vector<int> totals(this->nrows, 0);
        for (int row = 0; row < this->nrows; ++row) {
            const std::unordered_map<int, int> &matrix_row = this->matrix[row];
            for (const std::pair<int, int> &element : matrix_row) {
                totals[row] += element.second;
            }
        }
        return totals;
    }
}

EdgeWeights DictTransposeMatrix::outgoing_edges(int block) const {
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

int DictTransposeMatrix::trace() const {
    int total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (int index = 0; index < this->nrows; ++index) {
        total += this->get(index, index);
        // total += this->matrix[index][index];
    }
    return total;
}

void DictTransposeMatrix::update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                                             std::vector<int> proposed_row, std::vector<int> current_col,
                                             std::vector<int> proposed_col) {
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

std::vector<int> DictTransposeMatrix::values() const {
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
