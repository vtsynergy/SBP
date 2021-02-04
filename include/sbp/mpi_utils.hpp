/**
 * Utility functions that help with MPI communication
 */
#ifndef SBP_MPI_UTILS_HPP
#define SBP_MPI_UTILS_HPP

#include "mpi.h"

namespace utils {

namespace mpi {

/**
 * Data types for sending / receiving communication
 */
// struct VectorElement_t {
//     int index;
//     int value;
// };
// MPI_Datatype VectorElement;

/**
 * Message tags
 */
const int TAG_ROWCOL_SIZE = 0;
const int TAG_ROW = 1;
const int TAG_COL = 2;

/// Data class that holds MPI-specific information
class Info {
public:
    /// Constructor that performs MPI initialization, and initializes public variables
    Info(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &(this->rank));
        MPI_Comm_size(MPI_COMM_WORLD, &(this->num_processes));
    }
    int rank;
    int num_processes;
    MPI_Datatype VectorElement;

    /// Returns the target rank for the given `community`. Assumes round-robin distribution.
    /// TODO: extend this method to other distributions
    int get_target_rank(int community);
    /// Frees all data types
    void free_datatypes();
    /// Initializes all data types
    void initialize_datatypes();
};

}

}

#endif // SBP_MPI_UTILS_HPP