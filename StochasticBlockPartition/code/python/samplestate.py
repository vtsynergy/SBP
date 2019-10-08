"""Stores the state of a sample.
"""

from typing import List, Dict, Tuple
from copy import copy

import numpy as np

class SampleState():
    """Stores the state of a sample.
    """
    def __init__(self):
        """Creates a default, empty sample state.
        """
        self.empty = True
        self.sample_idx = np.asarray([], dtype=np.int32)
    # End of __init__()

    @staticmethod
    def create_sample_state(num_vertices: int, prev_state: 'SampleState', args: 'argparse.Namespace') -> 'SampleState':
        """Performs sampling according to the sample type in args.
        """
        if args.sample_type == "uniform_random":
            return UniformRandomSampleState(prev_state)
        elif args.sample_type == "random_walk":
            return RandomWalkSampleState(num_vertices, prev_state)
        elif args.sample_type == "random_jump":
            return RandomJumpSampleState(num_vertices, prev_state)
        elif args.sample_type == "degree_weighted":
            return Sample.degree_weighted_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                 old_true_block_assignment, args)
        elif args.sample_type == "random_node_neighbor":
            return Sample.random_node_neighbor_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                      old_true_block_assignment, args)
        elif args.sample_type == "forest_fire":
            return Sample.forest_fire_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                             old_true_block_assignment, args)
        elif args.sample_type == "expansion_snowball":
            return Sample.expansion_snowball_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                    old_true_block_assignment, args)
        else:
            raise NotImplementedError("Sample type: {} is not implemented!".format(args.sample_type))
    # End of create_sample()
# End of SampleState

class UniformRandomSampleState(SampleState):
    def __init__(self, prev_state: 'UniformRandomSampleState') -> None:
        SampleState.__init__(self)
        self.empty = False
        if not prev_state.empty:
            self.sample_idx = copy(prev_state.sample_idx)
    # End of __init__()
# End of UniformRandomSampleState

class RandomWalkSampleState(SampleState):
    def __init__(self, num_vertices: int, prev_state: 'RandomWalkSampleState') -> None:
        SampleState.__init__(self)
        self.empty = False
        self.sampled_marker = [False] * num_vertices
        self.index_set = list()  # type: List[int]
        if not prev_state.empty:
            self.sampled_marker = copy(prev_state.sampled_marker)
            self.index_set = copy(prev_state.index_set)
            self.sample_idx = copy(prev_state.sample_idx)
    # End of __init__()
# End of RandomWalkSampleState

class RandomJumpSampleState(SampleState):
    def __init__(self, num_vertices: int, prev_state: 'RandomJumpSampleState') -> None:
        SampleState.__init__(self)
        self.empty = False
        self.sampled_marker = [False] * num_vertices
        self.index_set = list()  # type: List[int]
        if not prev_state.empty:
            self.sampled_marker = copy(prev_state.sampled_marker)
            self.index_set = copy(prev_state.index_set)
            self.sample_idx = copy(prev_state.sample_idx)
    # End of __init__()
# End of RandomJumpSampleState
