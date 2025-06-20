from copy import deepcopy
from cassiopeia.data import CassiopeiaTree


def maximum_parsimony(tree: CassiopeiaTree):
    tree = deepcopy(tree)
    tree.reconstruct_ancestral_characters()
    tree.set_character_states(tree.root, [0] * tree.n_character)
    return tree
