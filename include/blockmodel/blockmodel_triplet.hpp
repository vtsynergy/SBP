/**
 * Stores the triplet of blockmodels needed for the fibonacci search.
 */
#ifndef CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
#define CPPSBP_PARTITION_PARTITION_TRIPLET_HPP

#include <iostream>
#include <limits>

#include "blockmodel.hpp"

static const float BLOCK_REDUCTION_RATE = 0.5;

class BlockmodelTriplet {

public:
    /// Constructor for the BlockmodelTriplet. Sets the optimal_num_blocks_found to false and initializes 3 empty 
    /// blockmodels.
    BlockmodelTriplet() : optimal_num_blocks_found(false) {}
    /// True if the optimal number of blocks has been found.
    bool optimal_num_blocks_found;
    /// Updates the triplet with a new blockmodel.
    void update(Blockmodel &blockmodel);
    /// Prints out a status message showing basic info about the blockmodels in the triplet.
    void status();
    /// Returns a reference to one of the three blockmodels in the triplet.
    Blockmodel &get(int i) { return blockmodels[i]; }
    /// Returns true if the algorithm has not yet reached the fine-grained search for the optimal number of communities.
    bool golden_ratio_not_reached();
    /// Returns true if blockmodeling is done. Either the number of blocks is down to 1, or the optimal number of
    /// communities has been found.
    bool is_done();
    /// Returns a copy of the next blockmodel to run through an iteration of stochastic block partitioning.
    Blockmodel get_next_blockmodel(Blockmodel &old_blockmodel);

private:
    /// Blockmodels arranged in order of decreasing number of communities.
    /// If the blockmodel with the lowest number of communities is empty, then the golden ratio bracket has not yet
    /// been established.
    Blockmodel blockmodels[3];
    /// The difference in entropy between the two blockmodels with the lowest number of communities.
    int lower_difference();
    /// The difference in entropy between the two blockmodels with the higher number of communities.
    int upper_difference();
};

#endif // CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
