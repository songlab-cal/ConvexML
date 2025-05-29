from casbench.config import Config, sanity_check_config
from .dream_challenge._dream_challenge_sub1_wrapper import dream_sub1_lt

from ._reverse_engineered_cas9 import \
    ReverseEngineeredCas9LineageTracingDataSimulator


def get_lt_simulator_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "ReverseEngineeredCas9LineageTracingDataSimulator":
        return ReverseEngineeredCas9LineageTracingDataSimulator(**dict(args))
    elif identifier == "dream_sub1_lt__2024_10_13":
        return dream_sub1_lt(**dict(args))
    else:
        raise ValueError(f"Unknown lt simulator: {identifier}")
