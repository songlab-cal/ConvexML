import cassiopeia as cas
from cassiopeia.data import CassiopeiaTree
from typing import Union


def plot_tree(
    tree: Union[CassiopeiaTree, str],
) -> None:
    """
    Plot the newick tree.
    """
    if type(tree) is str:
        cassiopeia_tree = CassiopeiaTree(tree=tree)
    else:
        cassiopeia_tree = tree
    cas.pl.plot_matplotlib(
        cassiopeia_tree,
        add_root=False,
        depth_key="time",
        extend_branches=False,
    )
