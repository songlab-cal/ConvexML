from casbench.config import Config, sanity_check_config
from .dream_challenge._dream_challenge_sub1_wrapper import dream_sub1_sims

from ._ble_paper_tree_simulator import ble_paper_tree_simulator


def get_tree_simulator_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "ble_paper_tree_simulator":
        return ble_paper_tree_simulator(**dict(args))
    elif identifier == "dream_sub1_sims__2024_09_15":
        return dream_sub1_sims(**dict(args))
    else:
        raise ValueError(f"Unknown tree simulator: {identifier}")
