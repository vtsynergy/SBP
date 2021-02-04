#include "sbp/mpi_utils.hpp"

namespace utils {

namespace mpi {

void Info::free_datatypes() {
    MPI_Type_free(&(this->VectorElement));
}

int Info::get_target_rank(int community) {
    return community % this->num_processes;
}

void Info::initialize_datatypes() {
    MPI_Type_contiguous(2, MPI_INT, &(this->VectorElement));
    MPI_Type_commit(&(this->VectorElement));
}

} // namespace mpi

} // namespace utils