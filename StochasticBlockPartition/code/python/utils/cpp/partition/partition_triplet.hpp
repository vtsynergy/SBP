/**
 * Stores the triplet of partitions needed for the fibonacci search.
 */
#ifndef CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
#define CPPSBP_PARTITION_PARTITION_TRIPLET_HPP

#include <pybind11/pybind11.h>

#include <iostream>
#include <limits>

#include "partition.hpp"

namespace py = pybind11;

static const float BLOCK_REDUCTION_RATE = 0.5;

class PartitionTriplet {

public:
    PartitionTriplet() : optimal_num_blocks_found(false) {}
    bool optimal_num_blocks_found;
    void update(Partition &partition);
    void status();
    Partition &get(int i) { return partitions[i]; }
    bool golden_ratio_not_reached();
    bool is_done();
    Partition get_next_partition(Partition &old_partition);

private:
    /// Partitions arranged in order of decreasing number of blocks.
    /// If the first partition is empty, then the golden ratio bracket has not yet been established.
    Partition partitions[3];
    int lower_difference();
    int upper_difference();
};

#endif // CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
