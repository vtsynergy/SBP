/**
 * Useful type definitions and such.
 */
#ifndef SBP_TYPEDEFS_HPP
#define SBP_TYPEDEFS_HPP

#include <unordered_map>
#include <vector>

/// Used to hash a pair of integers. Source: https://codeforces.com/blog/entry/21853
struct IntPairHash {
    size_t operator() (const std::pair<int, int> &pair) const {
        return std::hash<long long>() (((long long) pair.first) ^ (((long long) pair.second) << 32));
    }
};

typedef std::vector<std::vector<int>> NeighborList;

template <typename T>
struct SparseVector {
    std::vector<int>    idx;   // The index of the corresponding element in data
    std::vector<T>      data;  // The non-zero values of the vector
    // /// Returns the sum of all elements in data.
    // inline T sum() {
    //     T result;
    //     for (const T &value: this->data) {
    //         result += value;
    //     }
    //     return result;
    // }
    inline SparseVector<T> operator/(const double &rhs) {
        SparseVector<T> result;
        for (int i = 0; i < this->idx.size(); ++i) {
            result.idx.push_back(this->idx[i]);
            result.data.push_back(this->data[i] / rhs); 
        }
        return result;
    }
};

template <typename T> 
using MapVector = std::unordered_map<int, T>;

struct Merge {
    int block = -1;
    int proposal = -1;
    double delta_entropy = std::numeric_limits<double>::max();
};

struct Membership {
    int vertex = -1;
    int block = -1;
};

typedef std::unordered_map<std::pair<int, int>, int, IntPairHash> PairIndexVector;

// template<class T>
// using SparseVector = std::vector
// typedef struct proposal_evaluation_t {
//     int proposed_block;
//     double delta_entropy;
// } ProposalEvaluation;

#endif // SBP_TYPEDEFS_HPP