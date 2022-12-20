//
// Created by Frank on 12/20/2022.
//

#ifndef SBP_RNG_HPP
#define SBP_RNG_HPP

#include "omp.h"
#include <random>

namespace rng {

extern std::vector<std::mt19937> generators;
extern std::vector<std::uniform_real_distribution<float>> distributions;

float generate();

std::mt19937 &generator();

void init_generators();

}

#endif //SBP_RNG_HPP
