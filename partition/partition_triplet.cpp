#include "partition_triplet.hpp"

void PartitionTriplet::update(Partition &partition) {
    int index;
    if (this->partitions[1].empty) {
        index = 1;
    } else {
        if (partition.getOverall_entropy() <= this->partitions[1].getOverall_entropy()) {
            int old_index;
            if (this->get(1).getNum_blocks() > partition.getNum_blocks()) {
                old_index = 0;
            } else {
                old_index = 2;
            }
            this->partitions[old_index] = this->partitions[1];
            index = 1;
        } else {
            if (this->partitions[1].getNum_blocks() > partition.getNum_blocks()) {
                index = 2;
            } else {
                index = 0;
            }
        }
    }
    this->partitions[index] = partition;
}

void PartitionTriplet::status() {
    double entropies[3];
    int num_blocks[3];
    for (int i = 0; i < 3; ++i) {
        if (this->partitions[i].empty) {
            entropies[i] = std::numeric_limits<double>::min();
            num_blocks[i] = 0;
        } else {
            entropies[i] = this->partitions[i].getOverall_entropy();
            num_blocks[i] = this->partitions[i].getNum_blocks();
        }
    }
    std::cout << "Overall entropy: " << entropies[0] << " " << entropies[1] << " " << entropies[2] << std::endl;
    std::cout << "Number of blocks: " << num_blocks[0] << " " << num_blocks[1] << " " << num_blocks[2] << std::endl;
    if (this->optimal_num_blocks_found) {
        std::cout << "Optimal partition found with " << num_blocks[1] << " blocks" << std::endl;
    } else if (!(this->golden_ratio_not_reached())) {
        std::cout << "Golden ratio has been reached" << std::endl;
    }
}

bool PartitionTriplet::golden_ratio_not_reached() { return this->get(2).empty; }

bool PartitionTriplet::is_done() {
    if ((!this->get(0).empty && this->get(0).getNum_blocks() - this->get(2).getNum_blocks() == 2) ||
        (this->get(0).empty && this->get(1).getNum_blocks() - this->get(2).getNum_blocks() == 1)) {
        this->optimal_num_blocks_found = true;
    }
    return this->optimal_num_blocks_found;
}

int PartitionTriplet::lower_difference() {
    return this->get(1).getNum_blocks() - this->get(2).getNum_blocks();
}

int PartitionTriplet::upper_difference() {
    return this->get(0).getNum_blocks() - this->get(1).getNum_blocks();
}

Partition PartitionTriplet::get_next_partition(Partition &old_partition) {
    old_partition.setNum_blocks_to_merge(0);
    this->update(old_partition);
    this->status();

    // If have not yet reached golden ratio, continue from middle partition
    if (this->golden_ratio_not_reached()) {
        Partition partition = this->get(1).copy();
        partition.setNum_blocks_to_merge(int(partition.getNum_blocks() * BLOCK_REDUCTION_RATE));
        if (partition.getNum_blocks_to_merge() == 0) {
            this->optimal_num_blocks_found = true;
        }
        return partition;
    }
    // If community detection is done, return the middle (optimal) partition
    if (this->is_done()) {
        return this->get(1).copy();
    }
    // Find which Partition would serve as the starting point for the next iteration
    int index = 1;
    // TODO: if things get funky, look into this if/else statement
    if (this->get(0).empty && this->get(1).getNum_blocks() > this->get(2).getNum_blocks()) {
        index = 1;
    } else if (this->upper_difference() >= this->lower_difference()) {
        index = 0;
    } else {
        index = 1;
    }
    int next_num_blocks_to_try = this->get(index + 1).getNum_blocks();
    next_num_blocks_to_try += int((this->get(index).getNum_blocks() - this->get(index + 1).getNum_blocks()) * 0.618);
    Partition partition = this->get(index).copy();
    partition.setNum_blocks_to_merge(partition.getNum_blocks() - next_num_blocks_to_try);
    return partition;
}
