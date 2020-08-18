/**
 * Stores the triplet of partitions needed for the fibonacci search.
 */
#ifndef CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
#define CPPSBP_PARTITION_PARTITION_TRIPLET_HPP

#include <iostream>
#include <limits>

#include "partition.hpp"

static const float BLOCK_REDUCTION_RATE = 0.5;

class PartitionTriplet {

public:
    /// TODO
    PartitionTriplet() : optimal_num_blocks_found(false) {}
    /// TODO
    bool optimal_num_blocks_found;
    /// TODO
    void update(Partition &partition);
    /// TODO
    void status();
    /// TODO
    Partition &get(int i) { return partitions[i]; }
    /// TODO
    bool golden_ratio_not_reached();
    /// TODO
    bool is_done();
    /// TODO
    Partition get_next_partition(Partition &old_partition);

private:
    /// Partitions arranged in order of decreasing number of blocks.
    /// If the first partition is empty, then the golden ratio bracket has not yet been established.
    /// TODO
    Partition partitions[3];
    /// TODO
    int lower_difference();
    /// TODO
    int upper_difference();
};

#endif // CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
