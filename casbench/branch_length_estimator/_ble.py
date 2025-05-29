from typing import List, Tuple

from cassiopeia.tools.branch_length_estimator import IIDExponentialMLE

from casbench.config import Config, sanity_check_config

from ._binary_ble import BinaryBLE
from ._constant_ble import ConstantBLE
from ._number_of_mutations_ble import NumberOfMutationsBLE
from ._laml import LAML
from ._tidetree._tidetree import TiDeTree


def get_ble_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "BinaryBLE":
        return BinaryBLE(**dict(args))
    elif identifier == "ConstantBLE":
        return ConstantBLE(**dict(args))
    elif identifier == "IIDExponentialMLE":
        return IIDExponentialMLE(**dict(args))
    elif identifier == "NumberOfMutationsBLE":
        return NumberOfMutationsBLE(**dict(args))
    elif identifier == "LAML_2024_09_10_v2":
        return LAML(**dict(args))
    elif identifier == "TiDeTree_2024_09_19_v1":
        return TiDeTree(**dict(args))
    else:
        raise ValueError(f"Unknown branch length estimator: {identifier}")
