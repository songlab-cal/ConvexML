import os
import sys
import tempfile
from typing import Optional

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "jungle"))
import jungle as jg
import numpy as np
from cassiopeia.data import CassiopeiaTree

from ._FitnessEstimator import FitnessEstimator


class LBIJungle(FitnessEstimator):
    """
    LBI as implemented by the jungle package.

    See:
    https://github.com/felixhorns/jungle/blob/master/examples/FitnessScore.ipynb
    for toturial on how to use it.

    The BLI implemented by the jungle package is invariant to tree scaling, and
    was used in the KP manuscript.
    """

    def __init__(self, random_seed: Optional[int] = None):
        self._random_seed = random_seed

    def estimate_fitness(self, tree: CassiopeiaTree) -> None:
        with tempfile.NamedTemporaryFile("w") as outfile:
            outfilename = outfile.name
            outfile.write(tree.get_newick(record_branch_lengths=True))
            outfile.flush()
            if self._random_seed is not None:
                np.random.seed(self._random_seed)
            try:
                T_empirical = jg.Tree.from_newick(outfilename)
            except Exception:
                raise Exception(
                    "Could not read newick str:\n"
                    f"{tree.get_newick(record_branch_lengths=True)}"
                )
            T_empirical.annotate_standard_node_features()
            T_empirical.infer_fitness(params={})
            res_df = T_empirical.node_features()
            node_names = res_df.name
            node_fitnesses = res_df.mean_fitness
            for v, f in zip(node_names, node_fitnesses):
                if v != "" and v[0] != "_":
                    tree.set_attribute(v, "fitness", f)
