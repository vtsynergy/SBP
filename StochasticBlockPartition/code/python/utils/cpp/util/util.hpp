/**
 * Utility functions
 */
#ifndef CPPSBP_UTIL_HPP
#define CPPSBP_UTIL_HPP

#include <vector>

namespace util {

/// NOTE: template functions need to be defined in the header, or at least imported in the header
template <typename T> std::vector<T> concatenate(std::vector<T> &a, std::vector<T> &b) {
    std::vector<T> result;
    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

}

#endif // CPPSBP_UTIL_HPP