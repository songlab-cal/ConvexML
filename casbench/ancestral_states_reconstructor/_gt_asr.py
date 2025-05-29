from copy import deepcopy

from cassiopeia.data import CassiopeiaTree


def ground_truth_asr(
    tree: CassiopeiaTree, tree_gt: CassiopeiaTree
) -> CassiopeiaTree:
    tree = deepcopy(tree)
    states = {
        node: tree_gt.get_character_states(node) for node in tree_gt.nodes
    }
    tree.set_all_character_states(states)
    return tree
