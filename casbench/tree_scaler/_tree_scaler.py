from copy import deepcopy
from functools import partial

from cassiopeia.data import CassiopeiaTree

from casbench.config import Config, sanity_check_config


def unit_tree_scaler(tree: CassiopeiaTree) -> CassiopeiaTree:
    tree = deepcopy(tree)
    tree.scale_to_unit_length()
    return tree


def none(tree: CassiopeiaTree) -> CassiopeiaTree:
    tree = deepcopy(tree)
    return tree


def get_tree_scaler_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "unit_tree_scaler":
        return partial(unit_tree_scaler, **dict(args))
    elif identifier == "none":
        return partial(none, **dict(args))
    else:
        raise Exception(f"Unknown tree scaler: {identifier}")
