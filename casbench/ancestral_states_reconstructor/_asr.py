from copy import deepcopy
from functools import partial

from cassiopeia.data import CassiopeiaTree

from casbench.config import Config, sanity_check_config

from ._conservative_maximum_parsimony import conservative_maximum_parsimony


def maximum_parsimony(tree: CassiopeiaTree):
    tree = deepcopy(tree)
    tree.reconstruct_ancestral_characters()
    tree.set_character_states(tree.root, [0] * tree.n_character)
    return tree


def get_asr_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "conservative_maximum_parsimony":
        return partial(conservative_maximum_parsimony, **dict(args))
    elif identifier == "maximum_parsimony":
        return partial(maximum_parsimony, **dict(args))
    else:
        raise ValueError(
            f"Unknown ancestral states reconstructor: {identifier}"
        )
