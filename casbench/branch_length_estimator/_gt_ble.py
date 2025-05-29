from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import BranchLengthEstimator


class GroundTruthBLE(BranchLengthEstimator):
    def __init__(self, tree_gt: CassiopeiaTree):
        self._tree_gt = tree_gt

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        tree.populate_tree(tree=self._tree_gt.get_tree_topology())
        times = self._tree_gt.get_times()
        tree.set_times(times)
        del self._tree_gt
