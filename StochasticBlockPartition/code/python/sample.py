"""Helper functions for performing different kinds of sampling.
"""

from typing import List, Dict, Tuple
from copy import copy

import numpy as np

from samplestate import SampleState
from samplestate import UniformRandomSampleState
from samplestate import RandomWalkSampleState
from samplestate import RandomJumpSampleState
from samplestate import DegreeWeightedSampleState
from samplestate import RandomNodeNeighborSampleState
from samplestate import ForestFireSampleState
from samplestate import ExpansionSnowballSampleState


class Sample():
    """Stores the variables needed to create a subgraph.
    """

    def __init__(self, state: SampleState, old_out_neighbors: List[np.ndarray], old_in_neighbors: List[np.ndarray],
        old_true_block_assignment: np.ndarray) -> None:
        """Creates a new Sample object.
        """
        self.state = state
        self.vertex_mapping = dict([(v, k) for k,v in enumerate(state.sample_idx)])
        self.out_neighbors = list()  # type: List[np.ndarray]
        self.in_neighbors = list()  # type: List[np.ndarray]
        self.num_edges = 0
        for index in state.sample_idx:
            out_neighbors = old_out_neighbors[index]
            out_mask = np.isin(out_neighbors[:,0], state.sample_idx, assume_unique=False)
            sampled_out_neighbors = out_neighbors[out_mask]
            for out_neighbor in sampled_out_neighbors:
                out_neighbor[0] = self.vertex_mapping[out_neighbor[0]]
            self.out_neighbors.append(sampled_out_neighbors)
            in_neighbors = old_in_neighbors[index]
            in_mask = np.isin(in_neighbors[:,0], state.sample_idx, assume_unique=False)
            sampled_in_neighbors = in_neighbors[in_mask]
            for in_neighbor in sampled_in_neighbors:
                in_neighbor[0] = self.vertex_mapping[in_neighbor[0]]
            self.in_neighbors.append(sampled_in_neighbors)
            self.num_edges += np.sum(out_mask) + np.sum(in_mask)
        true_block_assignment = old_true_block_assignment[state.sample_idx]
        true_blocks = list(set(true_block_assignment))
        self.true_blocks_mapping = dict([(v, k) for k,v in enumerate(true_blocks)])
        self.true_block_assignment = np.asarray([self.true_blocks_mapping[b] for b in true_block_assignment])
        self.sample_num = len(state.sample_idx)
    # End of __init__()

    @staticmethod
    def create_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        args: 'argparse.Namespace', prev_state: SampleState = SampleState()) -> 'Sample':
        """Performs sampling according to the sample type in args.
        """
        if args.sample_type == "uniform_random":
            return Sample.uniform_random_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                old_true_block_assignment, prev_state, args)
        elif args.sample_type == "random_walk":
            return Sample.random_walk_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                             old_true_block_assignment, prev_state, args)
        elif args.sample_type == "random_jump":
            return Sample.random_jump_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                old_true_block_assignment, prev_state, args)
        elif args.sample_type == "degree_weighted":
            return Sample.degree_weighted_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                 old_true_block_assignment, prev_state, args)
        elif args.sample_type == "random_node_neighbor":
            return Sample.random_node_neighbor_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                      old_true_block_assignment, prev_state, args)
        elif args.sample_type == "forest_fire":
            return Sample.forest_fire_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                             old_true_block_assignment, prev_state, args)
        elif args.sample_type == "expansion_snowball":
            return Sample.expansion_snowball_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                    old_true_block_assignment, prev_state, args)
        else:
            raise NotImplementedError("Sample type: {} is not implemented!".format(args.sample_type))
    # End of create_sample()

    @staticmethod
    def uniform_random_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        prev_state: UniformRandomSampleState, args: 'argparse.Namespace') -> 'Sample':
        """Uniform random sampling.
        """
        state = SampleState.create_sample_state(num_vertices, prev_state, args)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        print("Sampling {} vertices from graph".format(sample_num))
        choices = np.setdiff1d(np.asarray(range(num_vertices)), state.sample_idx)
        state.sample_idx = np.concatenate(
            (state.sample_idx, np.random.choice(choices, sample_num, replace=False)),
            axis=None
        )
        return Sample(state, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of uniform_random_sampling()

    @staticmethod
    def random_walk_sample(num_vertices: int, old_out_neighbors: List[np.ndarray], old_in_neighbors: List[np.ndarray],
        old_true_block_assignment: np.ndarray, prev_state: RandomWalkSampleState, args: 'argparse.Namespace') -> 'Sample':
        """Random walk sampling.
        """
        state = SampleState.create_sample_state(num_vertices, prev_state, args)  # type: RandomWalkSampleState
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        print("Sampling {} vertices from graph".format(sample_num))
        num_tries = 0
        start = np.random.randint(sample_num)  # start with a random vertex
        vertex = start

        while len(state.index_set) == 0 or len(state.index_set) % sample_num != 0:
            num_tries += 1
            if not state.sampled_marker[vertex]:
                state.index_set.append(vertex)
                state.sampled_marker[vertex] = True
            if num_tries % sample_num == 0:  # If the number of tries is large, restart from new random vertex
                start = np.random.randint(sample_num)
                vertex = start
                num_tries = 0
            elif np.random.random() < 0.15:  # With a probability of 0.15, restart at original node
                vertex = start
            elif len(old_out_neighbors[vertex]) > 0:  # If the vertex has out neighbors, go to one of them
                vertex = np.random.choice(old_out_neighbors[vertex][:,0])
            else:  # Otherwise, restart from the original vertex
                if len(old_out_neighbors[start]) == 0:  # if original vertex has no out neighbors, change it
                    start = np.random.randint(sample_num)
                vertex = start
            
        state.sample_idx = np.asarray(state.index_set)
        return Sample(state, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of Random_walk_sampling()

    @staticmethod
    def random_jump_sample(num_vertices: int, old_out_neighbors: List[np.ndarray], old_in_neighbors: List[np.ndarray],
        old_true_block_assignment: np.ndarray, prev_state: RandomJumpSampleState, args: 'argparse.Namespace') -> 'Sample':
        """Random jump sampling.
        """
        state = SampleState.create_sample_state(num_vertices, prev_state, args)  # type: RandomJumpSampleState
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        print("Sampling {} vertices from graph".format(sample_num))
        num_tries = 0
        start = np.random.randint(sample_num)  # start with a random vertex
        vertex = start

        while len(state.index_set) == 0 or len(state.index_set) % sample_num != 0:
            num_tries += 1
            if not state.sampled_marker[vertex]:
                state.index_set.append(vertex)
                state.sampled_marker[vertex] = True
            # If the number of tries is large, or with a probability of 0.15, start from new random vertex
            if num_tries % sample_num == 0 or np.random.random() < 0.15:
                start = np.random.randint(sample_num)
                vertex = start
                num_tries = 0
            elif len(old_out_neighbors[vertex]) > 0:  # If the vertex has out neighbors, go to one of them
                vertex = np.random.choice(old_out_neighbors[vertex][:,0])
            else:  # Otherwise, restart from the original vertex
                if len(old_out_neighbors[start]) == 0:  # if original vertex has no out neighbors, change it
                    start = np.random.randint(sample_num)
                vertex = start
            
        state.sample_idx = np.asarray(state.index_set)
        return Sample(state, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of random_jump_sample()

    @staticmethod
    def degree_weighted_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        prev_state: DegreeWeightedSampleState, args: 'argparse.Namespace') -> 'Sample':
        """Degree-weighted sampling, where the probability of picking a vertex is proportional to its degree.
        """
        state = SampleState.create_sample_state(num_vertices, prev_state, args)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        print("Sampling {} vertices from graph".format(sample_num))
        vertex_degrees = np.add([len(neighbors) for neighbors in old_out_neighbors], 
                                [len(neighbors) for neighbors in old_in_neighbors])
        vertex_degrees[state.sample_idx] = 0
        state.sample_idx = np.concatenate(
            (state.sample_idx, np.random.choice(num_vertices, sample_num, replace=False, p=vertex_degrees/np.sum(vertex_degrees)))
        )
        return Sample(state, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of Random_walk_sampling()

    @staticmethod
    def random_node_neighbor_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        prev_state: RandomNodeNeighborSampleState, args: 'argparse.Namespace') -> 'Sample':
        """Random node neighbor sampling, where whenever a single node is sampled, all its out neighbors are sampled
        as well.
        """
        state = SampleState.create_sample_state(num_vertices, prev_state, args)  # type: RandomNodeNeighborSampleState
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        original_size = len(state.index_set)
        print("Sampling {} vertices from graph".format(sample_num))
        choices = np.setdiff1d(np.asarray(range(num_vertices)), state.sample_idx)
        random_samples = np.random.choice(choices, sample_num, replace=False)
        for vertex in random_samples:
            if not state.sampled_marker[vertex]:
                state.index_set.append(vertex)
                state.sampled_marker[vertex] = True
            for neighbor in old_out_neighbors[vertex]:
                if not state.sampled_marker[neighbor[0]]:
                    state.index_set.append(neighbor[0])
                    state.sampled_marker[neighbor[0]] = True
            if len(state.index_set) >= sample_num:
                break
        state.sample_idx = np.asarray(state.index_set[:original_size+sample_num])
        return Sample(state, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of random_node_neighbor_sample()

    @staticmethod
    def forest_fire_sample(num_vertices: int, old_out_neighbors: List[np.ndarray], old_in_neighbors: List[np.ndarray],
        old_true_block_assignment: np.ndarray, prev_state: ForestFireSampleState, args: 'argparse.Namespace') -> 'Sample':
        """Forest-fire sampling with forward probability = 0.7.
        """
        state = SampleState.create_sample_state(num_vertices, prev_state, args)  # type: ForestFireSampleState
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        original_size = len(state.index_set)
        print("Sampling {} vertices from graph".format(sample_num))
        while len(state.index_set) == 0 or len(state.index_set) % sample_num != 0:
            for vertex in state.current_fire_front:
                # add vertex to index set
                if not state.sampled_marker[vertex]:
                    state.sampled_marker[vertex] = True
                    state.burnt_marker[vertex] = True
                    state.index_set.append(vertex)
                # select edges to burn
                num_to_choose = np.random.geometric(0.7)
                out_neighbors = old_out_neighbors[vertex]
                if len(out_neighbors) < 1:  # If there are no outgoing neighbors
                    continue
                if len(out_neighbors) <= num_to_choose:
                    num_to_choose = len(out_neighbors)
                mask = np.zeros(len(out_neighbors))
                indexes = np.random.choice(np.arange(len(out_neighbors)), num_to_choose, replace=False)
                mask[indexes] = 1
                for index, value in enumerate(mask):
                    neighbor = out_neighbors[index][0]
                    if value == 1:  # if chosen, add to next frontier
                        if not state.burnt_marker[neighbor]:
                            state.next_fire_front.append(neighbor)
                    state.burnt_marker[neighbor] = True  # mark all neighbors as visited
            if np.sum(state.burnt_marker) == num_vertices:  # all samples are burnt, restart
                state.burnt_marker = [False] * num_vertices
                state.current_fire_front = [np.random.randint(num_vertices)]
                state.next_fire_front = list()
                continue
            if len(state.next_fire_front) == 0:  # if fire is burnt-out
                state.current_fire_front = [np.random.randint(num_vertices)]
            else:
                state.current_fire_front = copy(state.next_fire_front)
                state.next_fire_front = list()
        state.sample_idx = np.asarray(state.index_set[:original_size+sample_num])
        return Sample(state, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of forest_fire_sample()

    @staticmethod
    def expansion_snowball_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        prev_state: UniformRandomSampleState, args: 'argparse.Namespace') -> 'Sample':
        """Expansion snowball sampling. At every iterations, picks a node adjacent to the current sample that
        contributes the most new neighbors.
        """
        state = SampleState.create_sample_state(num_vertices, prev_state, args)  # type: ExpansionSnowballSampleState
        sample_num = int(num_vertices * (args.sample_size / 100))
        print("Sampling {} vertices from graph".format(sample_num))
        if not state.neighbors:
            state.neighbors = list(old_out_neighbors[state.start][:,0])
            # Set up the initial contributions counts and flag currently neighboring vertices
            for neighbor in old_out_neighbors[state.start][:,0]:
                state.neighbors_flag[neighbor] = True
                new_neighbors = 0
                for _neighbor in old_out_neighbors[neighbor][:,0]:
                    if not (state.index_flag[_neighbor] or state.neighbors_flag[_neighbor]): new_neighbors += 1
                state.contribution[neighbor] += new_neighbors
        while len(state.index_set) == 0 or len(state.index_set) % sample_num != 0:
            if len(state.neighbors) == 0 or max(state.contribution) == 0:
                vertex = np.random.choice(list(set(range(num_vertices)) - set(state.index_set)))
                state.index_set.append(vertex)
                for neighbor in old_out_neighbors[vertex][:,0]:
                    if not state.neighbors_flag[neighbor]:
                        Sample._add_neighbor(neighbor, state.contribution, state.index_flag, state.neighbors_flag,
                                             old_out_neighbors[neighbor][:,0], old_in_neighbors[neighbor][:,0],
                                             state.neighbors)
                continue
            vertex = np.argmax(state.contribution)
            state.index_set.append(vertex)
            state.index_flag[vertex] = True
            state.neighbors.remove(vertex)
            state.contribution[vertex] = 0
            for neighbor in old_in_neighbors[vertex][:,0]:
                if not state.neighbors_flag[neighbor]:
                    Sample._add_neighbor(neighbor, state.contribution, state.index_flag, state.neighbors_flag,
                                         old_out_neighbors[neighbor][:,0], old_in_neighbors[neighbor][:,0], state.neighbors)
        state.sample_idx = np.asarray(state.index_set)
        return Sample(state, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of expansion_snowball_sample()

    @staticmethod
    def _add_neighbor(vertex: int, contribution: List[int], index_flag: List[bool], neighbor_flag: List[bool],
        out_neighbors: np.ndarray, in_neighbors: np.ndarray, neighbors: List[int]) -> Tuple[List[int], List[bool]]:
        """Updates the expansion contribution for neighbors of a single vertex.
        """
        neighbors.append(vertex)
        neighbor_flag[vertex] = True
        if contribution[vertex] == 0:
            Sample._calculate_contribution(vertex, contribution, index_flag, neighbor_flag, out_neighbors, in_neighbors)
        return contribution, neighbor_flag
    # End of _add_neighbor()

    @staticmethod
    def _calculate_contribution(vertex: int, contribution: List[int], index_flag: List[bool], neighbor_flag: List[bool],
        out_neighbors: np.ndarray, in_neighbors: np.ndarray):
        # Compute contribution of this vertex
        for out_neighbor in out_neighbors:
            if not (index_flag[out_neighbor] or neighbor_flag[out_neighbor]):
                contribution[vertex] += 1
        # Decrease contribution of all neighbors with out links to this vertex
        for in_neighbor in in_neighbors:
            if contribution[in_neighbor] > 0:
                contribution[in_neighbor] -= 1
# End of Sample()
