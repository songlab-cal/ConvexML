import copy
from copy import deepcopy
from queue import PriorityQueue
import networkx as nx
from cassiopeia.data import CassiopeiaTree


def _resolve_multifurcations_networkx(
    tree: nx.DiGraph,
) -> nx.DiGraph:
    """
    Resolve the multifurcations in a tree.
    Given a tree represented by a networkx DiGraph, it resolves
    multifurcations. The tree is NOT modified in-place.
    The root is made to have only one children, as in a real-life tumor
    (the founding cell never divides immediately!)
    """
    tree = copy.deepcopy(tree)
    node_names = set([n for n in tree])
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    subtree_sizes = {}

    def _dfs_subtree_sizes(tree, subtree_sizes_dict, v) -> int:
        """
        Populates subtree_sizes_dict
        """
        res = 1
        for child in tree.successors(v):
            res += _dfs_subtree_sizes(tree, subtree_sizes_dict, child)
        subtree_sizes[v] = res
        return res

    _dfs_subtree_sizes(tree, subtree_sizes, root)
    assert len(subtree_sizes) == len([n for n in tree])

    # First make the root have degree 1.
    if tree.out_degree(root) >= 2:
        children = list(tree.successors(root))
        assert len(children) == tree.out_degree(root)
        # First remove the edges from the root
        tree.remove_edges_from([(root, child) for child in children])
        # Now create the intermediate node and add edges back
        root_child = f"{root}-child"
        if root_child in node_names:
            raise Exception("Node name already exists!")
        tree.add_edge(root, root_child)
        tree.add_edges_from([(root_child, child) for child in children])

    def _dfs_resolve_multifurcations(tree, v):
        children = list(tree.successors(v))
        if len(children) >= 3:
            # Must resolve the multifurcation
            _resolve_multifurcation(tree, v, subtree_sizes, node_names)
        for child in children:
            _dfs_resolve_multifurcations(tree, child)

    _dfs_resolve_multifurcations(tree, root)
    # Check that the tree is binary
    if not (len(tree.nodes) == len(tree.edges) + 1):
        raise RuntimeError("Failed to binarize tree")
    return tree


def _resolve_multifurcation(tree, v, subtree_sizes, node_names):
    """
    node_names is used to make sure we don't create a node name that already
    exists.
    """
    children = list(tree.successors(v))
    n_children = len(children)
    assert n_children >= 3

    # Remove all edges from v to its children
    tree.remove_edges_from([(v, child) for child in children])

    # Create the new binary structure
    queue = PriorityQueue()
    for child in children:
        queue.put((subtree_sizes[child], child))

    for i in range(n_children - 2):
        # Coalesce two smallest subtrees
        subtree_1_size, subtree_1_root = queue.get()
        subtree_2_size, subtree_2_root = queue.get()
        assert subtree_1_size <= subtree_2_size
        coalesced_tree_size = subtree_1_size + subtree_2_size + 1
        coalesced_tree_root = f"{v}-coalesce-{i}"
        if coalesced_tree_root in node_names:
            raise RuntimeError("Node name already exists!")
        # For debugging:
        # print(f"Coalescing {subtree_1_root} (sz {subtree_1_size}) and"
        #       f" {subtree_2_root} (sz {subtree_2_size})")
        tree.add_edges_from(
            [
                (coalesced_tree_root, subtree_1_root),
                (coalesced_tree_root, subtree_2_root),
            ]
        )
        queue.put((coalesced_tree_size, coalesced_tree_root))
    # Hang the two subtrees obtained to v
    subtree_1_size, subtree_1_root = queue.get()
    subtree_2_size, subtree_2_root = queue.get()
    assert subtree_1_size <= subtree_2_size
    tree.add_edges_from([(v, subtree_1_root), (v, subtree_2_root)])


def resolve_multifurcations(tree: CassiopeiaTree) -> CassiopeiaTree:
    """
    Resolve the multifurcations of a CassiopeiaTree.

    The tree is NOT modified in place: a new tree is returned.
    """
    tree = deepcopy(tree)
    binary_topology = _resolve_multifurcations_networkx(
        tree.get_tree_topology()
    )
    tree.populate_tree(binary_topology)
    return tree


def pass_times_onto_original_tree(tree: CassiopeiaTree, tree_original: CassiopeiaTree) -> CassiopeiaTree:
    """
    Given a tree `tree` with resolved multifurcations and the original tree `tree_original`
    with multifurcations, pass the times from `tree` onto `tree_original`.
    """
    tree = deepcopy(tree)
    tree_original = deepcopy(tree_original)
    # If there were no multifurcations to begin with, just return the original tree
    if len(tree.internal_nodes) == len(tree_original.internal_nodes):
        return tree
    else:
        for node in tree_original.nodes:
            if node != tree.root:
                tree_original.set_time(
                    node, tree.get_time(node)
                )
        return tree_original
