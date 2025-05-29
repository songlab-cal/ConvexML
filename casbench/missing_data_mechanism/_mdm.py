from copy import deepcopy
from functools import partial
from typing import Callable

from cassiopeia.data import CassiopeiaTree

from casbench.config import Config, sanity_check_config
from ._handle_double_resections import handle_double_resections


MissingDataMechanismType = Callable[[CassiopeiaTree], CassiopeiaTree]


def none(tree: CassiopeiaTree) -> CassiopeiaTree:
    tree = deepcopy(tree)
    return tree


def get_missing_data_mechanism_from_config(
    config: Config
) -> MissingDataMechanismType:
    sanity_check_config(config)
    identifier, args = config
    if identifier == "none":
        return partial(none, **dict(args))
    elif identifier == "handle_double_resections":
        return partial(handle_double_resections, **dict(args))
    else:
        raise ValueError(f"Unknown missing data mechanism: {identifier}")
