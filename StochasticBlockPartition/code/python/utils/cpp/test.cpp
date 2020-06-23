#include <iostream>
#include <Eigen/Core>
#include "finetune.hpp"

int main(int argc, char** argv) {
    int nvert = 20;
    int nblock = 10;
    Vector block_assignment;
    block_assignment << 0,0,1,1,2,3,3,3,3,3,4,5,6,6,7,1,8,9,9,0;
    std::vector<Matrix2Column> out_neighbors;
    std::vector<Matrix2Column> in_neighbors;
    Partition start = Partition(nblock, out_neighbors, 0.5, block_assignment);
    PartitionTriplet partitions;
    Partition &result = finetune::reassign_vertices(start, nvert, 30, out_neighbors, in_neighbors, partitions);
}
