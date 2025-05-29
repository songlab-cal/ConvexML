from casbench.config import Config, sanity_check_config

from ._lbi_jungle import LBIJungle


def get_fitness_estimator_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "LBIJungle":
        return LBIJungle(**dict(args))
    else:
        raise ValueError(f"Unknown fitness estimator: {identifier}")
