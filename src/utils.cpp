#include "utils.hpp"

#include "mpi_data.hpp"

namespace utils {

std::vector<int> argsort(const std::vector<int>& v) {
    if (v.empty()) {
        return {};
    }

    constexpr int num_bits = 8; // number of bits in a byte
    constexpr int num_buckets = 1 << num_bits; // number of possible byte values
    constexpr int mask = num_buckets - 1; // mask to extract the least significant byte

    int max_element = *std::max_element(v.begin(), v.end());
    int num_passes = (sizeof(int) + num_bits - 1) / num_bits; // number of passes needed for all bytes
    std::vector<int> counts(num_buckets);
    std::vector<int> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int pass = 0; pass < num_passes; pass++) {
        std::fill(counts.begin(), counts.end(), 0); // reset counts

        for (size_t i = 0; i < v.size(); i++) {
            int byte = (v[i] >> (num_bits * pass)) & mask;
            counts[byte]++;
        }

        for (int i = 1; i < num_buckets; i++) {
            counts[i] += counts[i - 1];
        }

        std::vector<int> new_indices(v.size());

        for (int i = 0; i < v.size(); i++) {
            int byte = (v[indices[i]] >> (num_bits * pass)) & mask;
            new_indices[--counts[byte] + v.size() - counts[num_buckets - 1]] = indices[i];
        }

        std::swap(indices, new_indices);
    }

    return indices;
}

std::string build_filepath() {
    std::ostringstream filepath_stream;
    filepath_stream << args.directory << "/" << args.type << "/" << args.overlap << "Overlap_" << args.blocksizevar;
    filepath_stream << "BlockSizeVar/" << args.type << "_" << args.overlap << "Overlap_" << args.blocksizevar;
    filepath_stream << "BlockSizeVar_" << args.numvertices << "_nodes";
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string filepath = filepath_stream.str();
    if (!fs::exists(filepath + ".tsv") && !fs::exists(filepath + ".mtx")) {
        std::cerr << "File doesn't exist: " << filepath + ".tsv/.mtx" << std::endl;
        exit(-1);
    }
    return filepath;
}

std::vector<std::vector<std::string>> read_csv(fs::path &filepath) {
    std::vector<std::vector<std::string>> contents;
    if (!fs::exists(filepath)) {
        if (mpi.rank == 0)
            std::cerr << "File doesn't exist: " << filepath << std::endl;
        return contents;
    }
    std::string line;
    std::ifstream file(filepath);
    int items = 0;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream line_stream(line);
        std::string value;
        while (line_stream >> value) {
            row.push_back(value);
            items++;
        }
        contents.push_back(row);
    }
    if (mpi.rank == 0)
        std::cout << "Read in " << contents.size() << " lines and " << items << " values." << std::endl;
    return contents;
}

void insert(NeighborList &neighbors, int from, int to) {
    if (from >= (int) neighbors.size()) {
        std::vector<std::vector<int>> padding(from - neighbors.size() + 1, std::vector<int>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    neighbors[from].push_back(to);
}

void insert_nodup(NeighborList &neighbors, int from, int to) {
    if (from >= (int) neighbors.size()) {
        std::vector<std::vector<int>> padding(from - neighbors.size() + 1, std::vector<int>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    for (const int &neighbor: neighbors[from])
        if (neighbor == to) return;
    neighbors[from].push_back(to);
}

bool insert(std::unordered_map<int, int> &map, int key, int value) {
    std::pair<std::unordered_map<int, int>::iterator, bool> result = map.insert(std::make_pair(key, value));
    return result.second;
}

void radix_sort(std::vector<int> &v) {
    if (v.empty()) {
        return;
    }

    constexpr int num_bits = 8; // number of bits in a byte
    constexpr int num_buckets = 1 << num_bits; // number of possible byte values
    constexpr int mask = num_buckets - 1; // mask to extract the least significant byte

    int max_element = *std::max_element(v.begin(), v.end());
    int num_passes = (sizeof(int) + num_bits - 1) / num_bits; // number of passes needed for all bytes
    std::vector<int> counts(num_buckets);

    std::vector<int> sorted_v(v.size());

    for (int pass = 0; pass < num_passes; pass++) {
        std::fill(counts.begin(), counts.end(), 0); // reset counts

        for (int elem: v) {
            int byte = (max_element - (elem >> (num_bits * pass))) &
                       mask; // changed to max_element - ... to sort in descending order
            counts[byte]++;
        }

        for (int i = num_buckets - 2; i >= 0; i--) { // changed to process buckets in reverse order
            counts[i] += counts[i + 1];
        }

        for (int i = v.size() - 1; i >= 0; i--) {
            int byte = (max_element - (v[i] >> (num_bits * pass))) &
                       mask; // changed to max_element - ... to sort in descending order
            sorted_v[--counts[byte]] = v[i];
        }

        std::swap(v, sorted_v);
    }
}

void radix_sort(std::vector<std::pair<int, int>> &v) {
    if (v.empty()) {
        return;
    }

    constexpr int num_bits = 8; // number of bits in a byte
    constexpr int num_buckets = 1 << num_bits; // number of possible byte values
    constexpr int mask = num_buckets - 1; // mask to extract the least significant byte

    int max_element = (*std::max_element(v.begin(), v.end(),
                                         [](const auto &p1, const auto &p2) { return p1.second > p2.second; })).second;
    int num_passes = (sizeof(int) + num_bits - 1) / num_bits; // number of passes needed for all bytes
    std::vector<int> counts(num_buckets);

    std::vector<std::pair<int, int>> sorted_v(v.size());

    for (int pass = 0; pass < num_passes; pass++) {
        std::fill(counts.begin(), counts.end(), 0); // reset counts

        for (const auto &elem: v) {
            int byte = (max_element - (elem.second >> (num_bits * pass))) &
                       mask; // changed to max_element - ... to sort in descending order
            counts[byte]++;
        }

        for (int i = num_buckets - 2; i >= 0; i--) { // changed to process buckets in reverse order
            counts[i] += counts[i + 1];
        }

        for (int i = v.size() - 1; i >= 0; i--) {
            int byte = (max_element - (v[i].second >> (num_bits * pass))) &
                       mask; // changed to max_element - ... to sort in descending order
            sorted_v[--counts[byte]] = v[i];
        }

        std::swap(v, sorted_v);
    }
}

}  // namespace utils
