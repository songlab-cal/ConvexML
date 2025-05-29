from copy import deepcopy
from functools import partial

from cassiopeia.data import CassiopeiaTree

from casbench.config import Config, sanity_check_config


def collapse(tree: CassiopeiaTree):
    tree = deepcopy(tree)
    tree.collapse_mutationless_edges(infer_ancestral_characters=True)
    return tree


def none(tree: CassiopeiaTree):
    tree = deepcopy(tree)
    return tree


def get_mutationless_edges_strategy_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "collapse":
        return partial(collapse, **dict(args))
    elif identifier == "none":
        return partial(none, **dict(args))
    else:
        raise ValueError(f"Unknown mutationless edges strategy: {identifier}")
