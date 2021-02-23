#include "blockmodel_triplet.hpp"
#include <math.h>

void BlockmodelTriplet::update(Blockmodel &blockmodel) {
    int index;
    if (this->blockmodels[1].empty) {
        index = 1;
    } else {
        if (blockmodel.getOverall_entropy() <= this->blockmodels[1].getOverall_entropy()) {
            int old_index;
            if (this->get(1).getNum_blocks() > blockmodel.getNum_blocks()) {
                old_index = 0;
            } else {
                old_index = 2;
            }
            this->blockmodels[old_index] = this->blockmodels[1];
            index = 1;
        } else {
            if (this->blockmodels[1].getNum_blocks() > blockmodel.getNum_blocks()) {
                index = 2;
            } else {
                index = 0;
            }
        }
    }
    this->blockmodels[index] = blockmodel;
}

void BlockmodelTriplet::status() {
    double entropies[3];
    int num_blocks[3];
    for (int i = 0; i < 3; ++i) {
        if (this->blockmodels[i].empty) {
            entropies[i] = std::numeric_limits<double>::min();
            num_blocks[i] = 0;
        } else {
            entropies[i] = this->blockmodels[i].getOverall_entropy();
            num_blocks[i] = this->blockmodels[i].getNum_blocks();
        }
    }
    std::cout << "Overall entropy: " << entropies[0] << " " << entropies[1] << " " << entropies[2] << std::endl;
    std::cout << "Number of blocks: " << num_blocks[0] << " " << num_blocks[1] << " " << num_blocks[2] << std::endl;
    if (this->optimal_num_blocks_found) {
        std::cout << "Optimal blockmodel found with " << num_blocks[1] << " blocks" << std::endl;
    } else if (!(this->golden_ratio_not_reached())) {
        std::cout << "Golden ratio has been reached" << std::endl;
    }
}

bool BlockmodelTriplet::golden_ratio_not_reached() { return this->get(2).empty; }

bool BlockmodelTriplet::is_done() {
    if ((!this->get(0).empty && this->get(0).getNum_blocks() - this->get(2).getNum_blocks() == 2) ||
        (this->get(0).empty && this->get(1).getNum_blocks() - this->get(2).getNum_blocks() == 1)) {
        this->optimal_num_blocks_found = true;
    }
    return this->optimal_num_blocks_found;
}

int BlockmodelTriplet::lower_difference() {
    return this->get(1).getNum_blocks() - this->get(2).getNum_blocks();
}

int BlockmodelTriplet::upper_difference() {
    return this->get(0).getNum_blocks() - this->get(1).getNum_blocks();
}

int BlockmodelTriplet::get_mid(int min, int max) {
    float phi = (1.0 + sqrt(5.0)) / 2.0;
    float x = max - min;
    int fibo_n_floor = int(floor(log((x * sqrt(5.0)) + 0.5) / log(phi)));
    int n = fibo_n_floor - 1;
    int fibo = int(round((pow(phi, n) / sqrt(5.0))));
    int next_num_blocks = max - fibo;
    std::cout << "next num blocks: " << next_num_blocks << std::endl;
    return max - next_num_blocks;
        // Calculate next number of blocks/communities
        // def fibo(n):
        //     phi = (1 + sqrt(5)) / 2
        //     return int(round(phi ** n / sqrt(5)))

        // def fibo_n_floor(x):
        //     phi = (1 + sqrt(5)) / 2
        //     n = floor(log(x * sqrt(5) + 0.5) / log(phi))
        //     return int(n)

        // def get_mid(a, b, random=False):
        //     if random:
        //         return a + numpy.random.randint(b - a + 1)
        //     else:
        //         n = fibo_n_floor(b - a)
        //         return b - fibo(n - 1)
}

Blockmodel BlockmodelTriplet::get_next_blockmodel(Blockmodel &old_blockmodel) {
    old_blockmodel.setNum_blocks_to_merge(0);
    this->update(old_blockmodel);
    this->status();

    // If have not yet reached golden ratio, continue from middle blockmodel
    if (this->golden_ratio_not_reached()) {
        Blockmodel blockmodel = this->get(1).copy();
        blockmodel.setNum_blocks_to_merge(get_mid(1, blockmodel.getNum_blocks()));
        // blockmodel.setNum_blocks_to_merge(int(blockmodel.getNum_blocks() * BLOCK_REDUCTION_RATE));
        if (blockmodel.getNum_blocks_to_merge() == 0) {
            this->optimal_num_blocks_found = true;
        }
        return blockmodel;
    }
    // If community detection is done, return the middle (optimal) blockmodel
    if (this->is_done()) {
        return this->get(1).copy();
    }
    // Find which Blockmodel would serve as the starting point for the next iteration
    int index = 1;
    // TODO: if things get funky, look into this if/else statement
    if (this->get(0).empty && this->get(1).getNum_blocks() > this->get(2).getNum_blocks()) {
        index = 1;
    } else if (this->upper_difference() >= this->lower_difference()) {
        index = 0;
    } else {
        index = 1;
    }
    // int next_num_blocks_to_try = this->get(index + 1).getNum_blocks();
    // next_num_blocks_to_try += int((this->get(index).getNum_blocks() - this->get(index + 1).getNum_blocks()) * 0.618);
    Blockmodel blockmodel = this->get(index).copy();
    blockmodel.setNum_blocks_to_merge(get_mid(this->get(index + 1).getNum_blocks(), this->get(index).getNum_blocks()));
    // blockmodel.setNum_blocks_to_merge(blockmodel.getNum_blocks() - next_num_blocks_to_try);
    return blockmodel;
}
