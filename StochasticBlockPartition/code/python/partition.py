"""Parameters for partitioning.
"""

from collections import namedtuple
from typing import List, Dict
from argparse import Namespace

import numpy as np
from scipy import sparse as sparse

from graph import Graph
from utils.dict_transpose_matrix import DictTransposeMatrix

class Partition():
    """Stores the current partitioning results.
    """

    def __init__(self, num_blocks: int, out_neighbors: List[np.ndarray], args: Namespace,
        block_assignment: np.ndarray = None) -> None:
        """Creates a new Partition object.

            Parameters
            ---------
            num_blocks : int
                    the number of blocks in the current partition
            out_neighbors : List[np.ndarray]
                    list of outgoing edges for each node
            args : Namespace
                    the command-line arguments
            block_assignment : np.ndarray [int]
                    the provided block assignment. Default = None
        """
        self.num_blocks = num_blocks
        if block_assignment is None:
            self.block_assignment = np.array(range(num_blocks))
        else:
            self.block_assignment = block_assignment
        self.overall_entropy = np.inf
        self.interblock_edge_count = [[]]  # type: np.array
        self.block_degrees = np.zeros(num_blocks)
        self.block_degrees_out = np.zeros(num_blocks)
        self.block_degrees_in = np.zeros(num_blocks)
        self._args = args
        self.num_blocks_to_merge = int(self.num_blocks * args.blockReductionRate)
        if out_neighbors:
            self.initialize_edge_counts(out_neighbors, args.sparse)
    # End of __init__()

    def initialize_edge_counts(self, out_neighbors: List[np.ndarray], use_sparse: bool):
        """Initialize the edge count matrix and block degrees according to the current partition

            Parameters
            ----------
            out_neighbors : list of ndarray; list length is N, the number of nodes
                        each element of the list is a ndarray of out neighbors, where the first column is the node
                        indices and the second column the corresponding edge weights
            B : int
                        total number of blocks in the current partition
            b : ndarray (int)
                        array of block assignment for each node
            use_sparse : bool
                        whether the edge count matrix is stored as a sparse matrix

            Returns
            -------
            M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                        edge count matrix between all the blocks.
            d_out : ndarray (int)
                        the current out degree of each block
            d_in : ndarray (int)
                        the current in degree of each block
            d : ndarray (int)
                        the current total degree of each block

            Notes
            -----
            Compute the edge count matrix and the block degrees from scratch
        """
        if use_sparse: # store interblock edge counts as a sparse matrix
            # self.interblock_edge_count = sparse.lil_matrix((self.num_blocks, self.num_blocks), dtype=int)
            self.interblock_edge_count = DictTransposeMatrix(shape=(self.num_blocks, self.num_blocks))
        else:
            self.interblock_edge_count = np.zeros((self.num_blocks, self.num_blocks), dtype=int)
        # compute the initial interblock edge count
        for v in range(len(out_neighbors)):
            if len(out_neighbors[v]) > 0 and out_neighbors[v].size > 0:
                k1 = self.block_assignment[v]
                k2, inverse_idx = np.unique(self.block_assignment[out_neighbors[v][:, 0]], return_inverse=True)
                count = np.bincount(inverse_idx, weights=out_neighbors[v][:, 1]).astype(int)
                if use_sparse:
                    self.interblock_edge_count.add((k1, k2), count)
                else:
                    self.interblock_edge_count[k1, k2] += count
        # compute initial block degrees
        self.block_degrees_out = np.asarray(self.interblock_edge_count.sum(axis=1)).ravel()
        self.block_degrees_in = np.asarray(self.interblock_edge_count.sum(axis=0)).ravel()
        self.block_degrees = self.block_degrees_out + self.block_degrees_in
    # End of initialize_edge_counts()

    def clone_with_true_block_membership(self, out_neighbors: List[np.ndarray], 
        true_block_membership: np.ndarray) -> 'Partition':
        """Creates a new Partition object for the correctly partitioned graph.

            Parameters
            ----------
            out_neighbors : List[np.ndarray]
                    list of outgoing edges for each node
            true_block_membership : np.ndarray [int]
                    the correct block membership for every vertex
            
            Returns
            ------
            partition : Partition
                    the Partition when the partitioning is 100% accurate
        """
        num_blocks = len(np.unique(true_block_membership))
        partition = Partition(num_blocks, out_neighbors, self._args, true_block_membership)
        return partition
    # End of clone_with_true_block_membership()

    @staticmethod
    def extend_sample(num_blocks: int, out_neighbors: List[np.ndarray],
        sample_block_assignment: np.ndarray, args: 'argparse.Namespace') -> 'Partition':
        """Creates a new Partition object from the block assignment array.

            Parameters
            ----------
            num_blocks : int
                    the number of blocks in the current partition
            out_neighbors : List[np.ndarray]
                    list of outgoing edges for each node
            sample_block_assignment : np.ndarray [int]
                    the partitioning results on the sample
            args : argparse.Namespace
                    the command-line args passed to the program

            Returns
            -------
            partition : Partition
                    the partition created from the sample
        """
        block_assignment = np.full(len(out_neighbors), -1)
        block_assignment[:len(sample_block_assignment)] = sample_block_assignment
        next_block = num_blocks
        for vertex in range(len(out_neighbors)):
            if block_assignment[vertex] == -1:
                block_assignment[vertex] = next_block
                next_block += 1
        return Partition(next_block, out_neighbors, args, block_assignment)
    # End of extend_sample()

    @staticmethod
    def from_sample(num_blocks: int, out_neighbors: List[np.ndarray],
        sample_block_assignment: np.ndarray, mapping: Dict[int,int], args: 'argparse.Namespace') -> 'Partition':
        """Creates a new Partition object from the block assignment array.

            Parameters
            ----------
            num_blocks : int
                    the number of blocks in the current partition
            out_neighbors : List[np.ndarray]
                    list of outgoing edges for each node
            sample_block_assignment : np.ndarray [int]
                    the partitioning results on the sample
            mapping : Dict[int,int]
                    the mapping of sample vertex indices to full graph vertex indices
            args : argparse.Namespace
                    the command-line args passed to the program

            Returns
            -------
            partition : Partition
                    the partition created from the sample
        """
        block_assignment = np.full(len(out_neighbors), -1)
        for key, value in mapping.items():
            block_assignment[key] = sample_block_assignment[value]
        next_block = num_blocks
        for vertex in range(len(out_neighbors)):
            if block_assignment[vertex] == -1:
                block_assignment[vertex] = next_block
                next_block += 1
        for vertex in range(len(out_neighbors)):
            if block_assignment[vertex] >= num_blocks:
                # count links to each block
                block_counts = np.zeros(num_blocks)
                for neighbor in out_neighbors[vertex]:
                    neighbor_block = block_assignment[neighbor[0]]
                    if neighbor_block < num_blocks:
                        block_counts[neighbor_block] += 1
                # pick block with max link
                block_assignment[vertex] = np.argmax(block_counts)
        return Partition(num_blocks, out_neighbors, args, block_assignment)
    # End of from_sample()

    def copy(self) -> 'Partition':
        """Returns a copy of this partition.

            Returns
            -------
            partition_copy : Partition
                a copy of this partition
        """
        partition_copy = Partition(self.num_blocks, [], self._args)
        partition_copy.block_assignment = self.block_assignment.copy()
        partition_copy.overall_entropy = self.overall_entropy
        partition_copy.interblock_edge_count = self.interblock_edge_count.copy()
        partition_copy.block_degrees = self.block_degrees.copy()
        partition_copy.block_degrees_out = self.block_degrees_out.copy()
        partition_copy.block_degrees_in = self.block_degrees_in.copy()
        partition_copy.num_blocks_to_merge = 0
        return partition_copy
    # End of copy()
# End of Partition


class PartitionTriplet():
    """Used for the fibonacci search to find the optimal number of blocks. Stores 3 partitioning results, with the
    partition with the lowest entropy (overall description length) in the middle.
    """

    def __init__(self) -> None:
        self.partitions = [None, None, None]  # type: List[Partition]
        self.optimal_num_blocks_found = False
    # End of __init__()
    
    def update(self, partition: Partition):
        """If the entropy of the current partition is the best so far, moves the middle triplet
        values to the left or right depending on the current partition's block number.

        Then, updates the appropriate triplet with the results of the newest partition.

            Parameters
            ---------
            partition : Partition
                    the most recent partitioning results
        """
        if self.partitions[1] is None:
            index = 1
        elif partition.overall_entropy <= self.partitions[1].overall_entropy:
            old_index = 0 if self.partitions[1].num_blocks > partition.num_blocks else 2
            self.partitions[old_index] = self.partitions[1]
            index = 1
        else:
            index = 2 if self.partitions[1].num_blocks > partition.num_blocks else 0
        self.partitions[index] = partition
    # End of update()

    def status(self):
        """Prints the status of the partition triplet.
        """
        entropies = list()
        num_blocks = list()
        for i in [0, 1, 2]:
            if self.partitions[i] is None:
                entropies.append(-np.inf)
                num_blocks.append(0)
            else:
                entropies.append(self.partitions[i].overall_entropy)
                num_blocks.append(self.partitions[i].num_blocks)
        print("Overall entropy: {}".format(entropies))
        print("Number of blocks: {}".format(num_blocks))
        if self.optimal_num_blocks_found:
            print("Optimal partition found with {} blocks".format(self.partitions[1].num_blocks))
        if self.partitions[2] is not None:
            print("Golden ratio has been established.")
    # End of status()
# End of PartitionTriplet
