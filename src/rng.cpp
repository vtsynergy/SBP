//
// Created by Frank on 12/20/2022.
//

#include "rng.hpp"

#include <iostream>

namespace rng {

std::vector<std::mt19937> generators;
std::vector<std::uniform_real_distribution<float>> distributions;

float generate() {
//    if (generators.size() < omp_get_max_threads()) {
//        init_generators();
//    }
    int thread_id = omp_get_thread_num();
    return distributions[thread_id](generators[thread_id]);
}

std::mt19937 &generator() {
//    if (generators.size() < omp_get_max_threads()) {
//        init_generators();
//    }
    return generators[omp_get_thread_num()];
}

void init_generators() {
    std::cout << "initializing generators!" << std::endl;
    std::cout << "size = " << generators.size() << " and nthreads = " << omp_get_max_threads() << std::endl;
    std::random_device seeder;
    int num_threads = omp_get_max_threads();
    for (int i = 0; i < num_threads; ++i) {
        std::mt19937 generator(seeder());
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        generators.push_back(generator);
        distributions.push_back(distribution);
    }
}

} // namespace rng
