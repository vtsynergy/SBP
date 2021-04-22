/***
 * A global structure holding MPI information.
 */
#ifndef SBP_MPI_DATA_HPP
#define SBP_MPI_DATA_HPP

struct MPI_Data {
    int rank;           // The rank of the current process
    int num_processes;  // The total number of processes
};

extern MPI_Data mpi;

#endif  // SBP_MPI_DATA_HPP