from cassiopeia.data import CassiopeiaTree
from cassiopeia.solver.CassiopeiaSolver import CassiopeiaSolver


class GroundTruthSolver(CassiopeiaSolver):
    def __init__(self, tree_gt: CassiopeiaTree):
        self._tree_gt = tree_gt

    def solve(self, tree: CassiopeiaTree) -> None:
        tree.populate_tree(tree=self._tree_gt.get_tree_topology())
        del self._tree_gt
