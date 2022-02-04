#include "entropy.hpp"

#include "cmath"

namespace entropy {

double mdl(const Blockmodel &blockmodel, int num_vertices, int num_edges) {
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

}