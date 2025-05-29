from collections import defaultdict

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import BranchLengthEstimator


class NumberOfMutationsBLE(BranchLengthEstimator):
    """
    Estimate branch lengths as the number of mutations on each edge.

    A naive branch length estimator that estimates branch lengths
    as the number of mutations on that edge. This is thus a
    very naive baseline model.

    This estimator requires that the ancestral states are provided.

    Mutations that are mapped accross multiple edges are distributed in equal
    parts accross those edges ('fractional mutations'). This happens for example
    when conservative maximum parsimony is used to impute ancestral states.

    Args:
        length_of_mutationless_edges: To avoid edges with length 0, mutationless
            edges will have this length.
        make_ultrametric: If to extend branch lengths at the leaves to make the
            tree ultrametric.
    """

    def __init__(
        self,
        length_of_mutationless_edges: float = 0,
        make_ultrametric: bool = True,
    ):
        self.length_of_mutationless_edges = length_of_mutationless_edges
        self.make_ultrametric = make_ultrametric

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        """
        See base class.
        """
        length_of_mutationless_edges = self.length_of_mutationless_edges
        make_ultrametric = self.make_ultrametric

        # Tally fractional mutations
        fractional_mutations = defaultdict(float)
        # (We pre-compute all states since we will need repeated access)
        character_states_dict = {
            node: tree.get_character_states(node) for node in tree.nodes
        }
        k = tree.character_matrix.shape[1]
        for node in tree.nodes:
            if tree.is_root(node):
                continue
            parent = tree.parent(node)
            character_states = character_states_dict[node]
            parent_states = character_states_dict[parent]
            for i in range(k):
                if character_states[i] > 0 and parent_states[i] == -1:
                    # Need to go up and distribute fractional mutations to all edges
                    edges_to_distribute_frac_mutations = [(parent, node)]
                    u, v = parent, node
                    while character_states_dict[u][i] == -1:
                        u, v = tree.parent(u), u
                        edges_to_distribute_frac_mutations.append((u, v))
                    if character_states_dict[u][i] == 0:
                        # Yep, we do have a mutation to distribute!
                        for u, v in edges_to_distribute_frac_mutations:
                            fractional_mutations[(u, v)] += 1 / len(
                                edges_to_distribute_frac_mutations
                            )
                    else:
                        assert (
                            character_states_dict[u][i] == character_states[i]
                        )

        estimated_edge_lengths = {}

        for (parent, child) in tree.edges:
            parent_states = tree.get_character_states(parent)
            child_states = tree.get_character_states(child)
            num_uncuts = 0
            num_cuts = 0
            num_missing = 0
            for parent_state, child_state in zip(parent_states, child_states):
                # We only care about uncut states.
                if parent_state == 0:
                    if child_state == 0:
                        num_uncuts += 1
                    elif child_state == tree.missing_state_indicator:
                        num_missing += 1
                    else:
                        num_cuts += 1

            if num_cuts == 0 and (parent, child) not in fractional_mutations:
                estimated_edge_length = length_of_mutationless_edges
            else:
                estimated_edge_length = num_cuts + fractional_mutations.get(
                    (parent, child), 0
                )
            estimated_edge_lengths[(parent, child)] = estimated_edge_length

        times = {node: 0 for node in tree.nodes}
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = (
                times[parent] + estimated_edge_lengths[(parent, child)]
            )

        if make_ultrametric:
            max_time = max(times.values())
            for leaf in tree.leaves:
                times[leaf] = max_time

        # We smooth out epsilons that might make a parent's time greater
        # than its child
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = max(times[parent], times[child])
        tree.set_times(times)
