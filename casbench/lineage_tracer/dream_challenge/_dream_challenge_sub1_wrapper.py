import unittest
import pandas as pd
import os
import re
from typing import List, Optional, Tuple

import numpy as np
from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import LineageTracingDataSimulator
from casbench.tree_simulator.dream_challenge._dream_challenge_sub1_wrapper import dream_sub1_sims
from casbench.lineage_tracer.utils import bootstrap_sites

class dream_sub1_lt(
    LineageTracingDataSimulator
):
    def __init__(
        self,
        numtree: int,
        bootstrap_sites_seed: Optional[int] = None,
    ):
        """
        Args:
            numtree: 1-76 are the training trees, and 77-106 are the testing trees.
            bootstrap_sites_seed: If not None, then the sites in the character matrix will be
                sampled with replacement to generate a bootstrapped version of the character matrix.
        """
        if not ((1 <= numtree) and (numtree <= 106)):
            raise ValueError(
                f"numtree should be between 1 and 106 inclusive, but is: {numtree}"
            )
        self.numtree = numtree
        self.bootstrap_sites_seed = bootstrap_sites_seed

    def overlay_data(self, tree: CassiopeiaTree) -> None:
        numtree = self.numtree
        dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../../data/dream/subchallenge_1"
        )

        # Get the character matrix
        if 1 <= numtree and numtree <= 76:
            character_matrix_path = os.path.join(
                dir_path,
                f"DREAM_subchallenge_1/sub1_train_{numtree}.txt",
            )
        elif 77 <= numtree and numtree <= 106:
            character_matrix_path = os.path.join(
                dir_path,
                f"DREAM_subchallenge_1/sub1_test_{numtree - 76}.txt",
            )
        else:
            raise ValueError(
                f"numtree should be between 1 and 106 inclusive, but is {numtree}"
            )
        character_matrix_df = pd.read_csv(
            character_matrix_path,
            sep="\t",
            dtype="str",
        )
        character_matrix_dict = {}
        # print(f"character_matrix_df before: {character_matrix_df}")

        def remap(v: List[int]) -> List[int]:
            """
            Remap 1 to 0 and 0 to 1
            """
            res = []
            for x in v:
                if x == 0:
                    res.append(1)
                elif x == 1:
                    res.append(0)
                elif x == 2:
                    res.append(2)
                else:
                    raise Exception("This should never get here.")
            assert(len(res) == len(v))
            return res

        for cell_id in range(character_matrix_df.shape[0]):
            cell_number = character_matrix_df.iloc[cell_id, 0]
            cell_states = [int(x) for x in character_matrix_df.iloc[cell_id, 1]]
            assert(len(cell_states) == 10)
            cell_name = f"{cell_number}_{character_matrix_df.iloc[cell_id, 1]}"
            character_matrix_dict[cell_name] = remap(cell_states)
        if sorted(list(character_matrix_dict.keys())) != sorted(list(tree.leaves)):
            raise ValueError(
                f"The character matrix and tree have different leaf sets for some reason. "
                f"character_matrix_dict: {character_matrix_dict}\n"
                f"tree.leaves = {tree.leaves}"
            )
        # print(f"character_matrix_dict after: {character_matrix_dict}")
        for node in tree.nodes:
            if node not in tree.leaves:
                character_matrix_dict[node] = [-1] * 10  # Because all casettes have length 10.
        if self.bootstrap_sites_seed is not None:
            # We bootstrap the sites.
            character_matrix_dict = bootstrap_sites(character_matrix_dict=character_matrix_dict, bootstrap_sites_seed=self.bootstrap_sites_seed)
        tree.set_all_character_states(character_matrix_dict)


def test_dream_sub1_lt():
    """
    Just tests that the DREAM subchallenge 1 trees all get created correctly.
    """
    for numtree in range(1, 107):
        tree_sim = dream_sub1_sims(numtree=numtree)
        tree = tree_sim.simulate_tree()
        lt = dream_sub1_lt(numtree=numtree)
        lt.overlay_data(tree)
    with unittest.TestCase().assertRaises(ValueError):
        tree_sim = dream_sub1_sims(numtree=0)
        tree = tree_sim.simulate_tree()
    with unittest.TestCase().assertRaises(ValueError):
        tree_sim = dream_sub1_sims(numtree=107)
        tree = tree_sim.simulate_tree()
