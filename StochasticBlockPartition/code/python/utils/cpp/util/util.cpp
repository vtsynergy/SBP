#import "util.hpp"

template <T> std::vector<T> util::concatenate(std::vector<T> a, std::vector<T> b) {
    std::vector<T> result;
    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    return result;
}
