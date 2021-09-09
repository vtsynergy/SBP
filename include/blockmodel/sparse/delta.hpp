/***
 * Stores the current graph blockmodeling results.
 */
#ifndef SBP_BLOCKMODEL_DELTA_HPP
#define SBP_BLOCKMODEL_DELTA_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <memory>
#include <queue>

// #include <Eigen/Core>
// #include "sparse/boost_mapped_matrix.hpp"
//#include "../args.hpp"
//#include "sparse/dict_matrix.hpp"
//#include "sparse/dict_transpose_matrix.hpp"
#include "typedefs.hpp"
//#include "../utils.hpp"

class Delta {
private:
    MapVector<int> _current_block_row;
    MapVector<int> _proposed_block_row;
    MapVector<int> _current_block_col;
    MapVector<int> _proposed_block_col;
    int _current_block;
    int _proposed_block;
public:
    Delta() {
        this->_current_block = -1;
        this->_proposed_block = -1;
    }
    Delta(int current_block, int proposed_block) {
        this->_current_block = current_block;
        this->_proposed_block = proposed_block;
    }
    void add(int row, int col, int value) {
        if (row == this->_current_block)
            this->_current_block_row[col] += value;
        else if (row == this->_proposed_block)
            this->_proposed_block_row[col] += value;
        else if (col == this->_current_block)
            this->_current_block_col[row] += value;
        else if (col == this->_proposed_block)
            this->_proposed_block_col[row] += value;
        else
            throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }
    std::vector<std::tuple<int, int, int>> entries() const {
        std::vector<std::tuple<int, int, int>> result;
        for (const std::pair<const int, int> &entry : this->_current_block_row) {
            result.emplace_back(this->_current_block, entry.first, entry.second);
        }
        for (const std::pair<const int, int> &entry : this->_proposed_block_row) {
            result.emplace_back(this->_proposed_block, entry.first, entry.second);
        }
        for (const std::pair<const int, int> &entry : this->_current_block_col) {
            result.emplace_back(entry.first, this->_current_block, entry.second);
        }
        for (const std::pair<const int, int> &entry : this->_proposed_block_col) {
            result.emplace_back(entry.first, this->_proposed_block, entry.second);
        }
        return result;
    }
    int get(int row, int col) const {
        if (row == this->_current_block)
            return map_vector::get(this->_current_block_row, col);
        else if (row == this->_proposed_block)
            return map_vector::get(this->_proposed_block_row, col);
        else if (col == this->_current_block)
            return map_vector::get(this->_current_block_col, row);
        else if (col == this->_proposed_block)
            return map_vector::get(this->_proposed_block_col, row);
        throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }
    void sub(int row, int col, int value) {
        if (row == this->_current_block)
            this->_current_block_row[col] -= value;
        else if (row == this->_proposed_block)
            this->_proposed_block_row[col] -= value;
        else if (col == this->_current_block)
            this->_current_block_col[row] -= value;
        else if (col == this->_proposed_block)
            this->_proposed_block_col[row] -= value;
        else
            throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }
};

#endif // SBP_BLOCKMODEL_DELTA_HPP
