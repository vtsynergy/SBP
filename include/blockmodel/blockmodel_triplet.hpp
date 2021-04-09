/**
 * Stores the triplet of blockmodels needed for the fibonacci search.
 */
#ifndef CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
#define CPPSBP_PARTITION_PARTITION_TRIPLET_HPP

#include <iostream>
#include <limits>

#include "blockmodel.hpp"

class BlockmodelTriplet {

public:
    /// TODO
    BlockmodelTriplet() : optimal_num_blocks_found(false) {}
    /// TODO
    bool optimal_num_blocks_found;
    /// TODO
    void update(Blockmodel &blockmodel);
    /// TODO
    void status();
    /// TODO
    Blockmodel &get(int i) { return blockmodels[i]; }
    /// TODO
    bool golden_ratio_not_reached();
    /// TODO
    bool is_done();
    /// TODO
    Blockmodel get_next_blockmodel(Blockmodel &old_blockmodel);

private:
    /// Blockmodels arranged in order of decreasing number of blocks.
    /// If the first blockmodel is empty, then the golden ratio bracket has not yet been established.
    /// TODO
    Blockmodel blockmodels[3];
    /// TODO
    int lower_difference();
    /// TODO
    int upper_difference();
};

#endif // CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
