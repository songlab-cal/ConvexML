from cassiopeia.simulator import UniformLeafSubsampler

from casbench.config import Config, sanity_check_config


def get_leaf_subsampler_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "UniformLeafSubsampler":
        return UniformLeafSubsampler(**dict(args))
    else:
        raise ValueError(f"Unknown leaf subsampler: {identifier}")
