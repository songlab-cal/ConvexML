from copy import deepcopy
from functools import partial

from cassiopeia.data import CassiopeiaTree

from casbench.config import Config, sanity_check_config


def none(tree: CassiopeiaTree):
    tree = deepcopy(tree)
    return tree


def resolve(tree: CassiopeiaTree):
    tree = deepcopy(tree)
    tree.resolve_multifurcations()
    return tree


def get_multifurcation_resolver_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "none":
        return partial(none, **dict(args))
    elif identifier == "resolve":
        return partial(resolve, **dict(args))
    else:
        raise ValueError(f"Unknown multifurcation resolver: {identifier}")
