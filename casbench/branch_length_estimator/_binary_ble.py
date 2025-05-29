from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import BranchLengthEstimator


class BinaryBLE(BranchLengthEstimator):
    def __init__(
        self, include_missing: bool = False, make_ultrametric: bool = False
    ):
        self._include_missing = include_missing
        self._make_ultrametric = make_ultrametric

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        make_ultrametric = self._make_ultrametric

        times = {}

        def dfs(v: str, t: int):
            times[v] = t
            for u in tree.children(v):
                dfs(
                    u,
                    t
                    + 1
                    * (
                        len(
                            tree.get_mutations_along_edge(
                                v,
                                u,
                                treat_missing_as_mutations=self._include_missing,
                            )
                        )
                        > 0
                    ),
                )

        dfs(tree.root, 0)

        if make_ultrametric:
            max_time = max(times.values())
            for leaf in tree.leaves:
                times[leaf] = max_time

        # We smooth out epsilons that might make a parent's time greater
        # than its child
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = max(times[parent], times[child])

        tree.set_times(times)
