//
// Created by Frank on 9/9/2022.
//

#ifndef SBP_POINTER_DELTA_HPP
#define SBP_POINTER_DELTA_HPP

#include "typedefs.hpp"

class VertexLevelDelta {
private:
//    MapVector<int> _current_block_row;
//    MapVector<int> _proposed_block_row;
//    MapVector<int> _current_block_col;
//    MapVector<int> _proposed_block_col;
    MapVector<int> _outgoing_edges;
    MapVector<int> _incoming_edges;
    int _current_vertex;
    int _current_block;
    int _proposed_block;
    int _self_edge_weight;
public:
    VertexLevelDelta() {
        this->_current_vertex = -1;
        this->_current_block = -1;
        this->_proposed_block = -1;
        this->_self_edge_weight = 0;
    }
    VertexLevelDelta(int current_vertex, int current_block, int proposed_block, int buckets = 10) {
        this->_current_vertex = current_vertex;
        this->_current_block = current_block;
        this->_proposed_block = proposed_block;
        this->_self_edge_weight = 0;
        this->_outgoing_edges = MapVector<int>(buckets);
        this->_incoming_edges = MapVector<int>(buckets);
    }
    /// Adds `value` as the delta to edge `from` -> `to`.
    void add(int from, int to, int value) {
        if (from == this->_current_vertex)
            this->_outgoing_edges[to] += value;
        else if (to == this->_current_vertex)
            this->_incoming_edges[from] += value;
        else
            throw std::logic_error("Neither the from nor the to vertex is current_vertex.");
    }

    /// Returns all stores deltas as a list of tuples storing `from`, `to`, `delta`.
    [[nodiscard]] std::vector<std::tuple<int, int, int>> entries() const {
        std::vector<std::tuple<int, int, int>> result;
        for (const std::pair<int, int> &entry : this->_outgoing_edges) {
            result.emplace_back(this->_current_vertex, entry.first, entry.second);
        }
        for (const std::pair<int, int> &entry : this->_incoming_edges) {
            result.emplace_back(entry.first, this->_current_vertex, entry.second);
        }
        return result;
    }

    /// Returns the delta for the edge `from`, `to` without modifying the underlying data structure.
    [[nodiscard]] int get(int from, int to) const {
        if (from == this->_current_vertex)
            return map_vector::get(this->_outgoing_edges, to);
        else if (to == this->_current_vertex)
            return map_vector::get(this->_incoming_edges, from);
        throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }

    /// Returns the weight of the self edge for this move, if any.
    [[nodiscard]] int self_edge_weight() const {
        return this->_self_edge_weight;
    }

    /// Sets the weight of the self edge for this move, if any.
    void self_edge_weight(int weight) {
        this->_self_edge_weight = weight;
    }

//    /// Adds -`value` (negative `value`) as the delta to edge `from` -> `to`.
//    void sub(int row, int col, int value) {
//        if (row == this->_current_block)
//            this->_current_block_row[col] -= value;
//        else if (row == this->_proposed_block)
//            this->_proposed_block_row[col] -= value;
//        else if (col == this->_current_block)
//            this->_current_block_col[row] -= value;
//        else if (col == this->_proposed_block)
//            this->_proposed_block_col[row] -= value;
//        else
//            throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
//    }
};

#endif //SBP_POINTER_DELTA_HPP
