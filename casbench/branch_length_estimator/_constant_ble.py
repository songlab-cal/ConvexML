from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import BranchLengthEstimator


class ConstantBLE(BranchLengthEstimator):
    """
    Estimate all branch lengths as 1.

    A naive branch length estimator that estimates branch lengths
    as 1. This is thus a very naive baseline model.

    Args:
        make_ultrametric: If to extend branch lengths at the leaves to make the
            tree ultrametric.
    """

    def __init__(
        self,
        make_ultrametric: bool = True,
    ):
        self.make_ultrametric = make_ultrametric

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        """
        See base class.
        """
        make_ultrametric = self.make_ultrametric

        estimated_edge_lengths = {}

        for (parent, child) in tree.edges:
            estimated_edge_lengths[(parent, child)] = 1

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
