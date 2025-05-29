import unittest
import pandas as pd
import os
import re
from typing import Tuple

import numpy as np
from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import BirthDeathFitnessSimulator, TreeSimulator

class dream_sub1_sims(TreeSimulator):
    """
    DREAM challenge webpage:
    https://www.synapse.org/Synapse:syn20692755/wiki/597058
    """
    def __init__(
        self,
        numtree: int,
    ):
        """
        Args:
            numtree: 1-76 are the training trees, and 77-106 are the testing trees.
        """
        if not ((1 <= numtree) and (numtree <= 106)):
            raise ValueError(
                f"numtree should be between 1 and 106 inclusive, but is: {numtree}"
            )
        self.numtree = numtree

    def simulate_tree(
        self,
    ) -> CassiopeiaTree:
        numtree = self.numtree
        dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../../data/dream/subchallenge_1"
        )

        # Get the tree
        if numtree <= 76:
            tree_df_path = os.path.join(
                dir_path,
                f"DREAM_subchallenge_1/train_setDREAM2019.txt",
            )
            tree_df = pd.read_csv(tree_df_path, sep="\t")
            tree_str = tree_df.iloc[numtree - 1, 3]
        else:
            tree_df_path = os.path.join(
                dir_path,
                f"DREAM_subchallenge_1_gold_standard/Goldstandard_SC1.txt",
            )
            tree_df = pd.read_csv(tree_df_path, sep="\t")
            tree_str = tree_df.iloc[numtree - 77, 1]
        gt_tree = CassiopeiaTree(tree=tree_str)
        return gt_tree


def test_dream_sub1_sims():
    """
    Just tests that the DREAM subchallenge 1 trees all get read correctly.
    """
    for numtree in range(1, 107):
        tree_sim = dream_sub1_sims(numtree=numtree)
        tree = tree_sim.simulate_tree()
    with unittest.TestCase().assertRaises(ValueError):
        tree_sim = dream_sub1_sims(numtree=0)
        tree = tree_sim.simulate_tree()
    with unittest.TestCase().assertRaises(ValueError):
        tree_sim = dream_sub1_sims(numtree=107)
        tree = tree_sim.simulate_tree()
