from cassiopeia import solver

from casbench.config import Config, sanity_check_config

from . import _hardcoded


def get_solver_from_config(config: Config):
    sanity_check_config(config)
    identifier, args = config
    if identifier == "ILPSolver":
        return solver.ILPSolver(**dict(args))
    elif identifier == "MaxCutGreedySolver":
        return solver.MaxCutGreedySolver(**dict(args))
    elif identifier == "MaxCutSolver":
        return solver.MaxCutSolver(**dict(args))
    elif identifier == "NeighborJoiningSolver":
        return solver.NeighborJoiningSolver(**dict(args))
    elif identifier == "UPGMASolver":
        return solver.UPGMASolver(**dict(args))
    elif identifier == "VanillaGreedySolver":
        return solver.VanillaGreedySolver(**dict(args))
    else:
        raise ValueError(f"Unknown solver: {identifier}")
