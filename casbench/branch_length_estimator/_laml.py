from copy import deepcopy
import numpy as np
import networkx as nx
import subprocess
import tempfile
from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import BranchLengthEstimator
from typing import Dict, Optional
import pandas as pd


class LAML(BranchLengthEstimator):
    def __init__(
        self,
        priors: Optional[Dict] = None,
        nInitials: int = 1,
    ):
        """
        NOTE: LAML enforces the ultrametric constraint. Their flag --ultrametric was removed from the API and set to always true in their code.
        """
        self.priors = priors
        self.nInitials = nInitials

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        priors = self.priors
        nInitials = self.nInitials
        num_sites = tree.character_matrix.shape[1]
        original_nodes = sorted(list(tree.nodes))
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write files
            tree.character_matrix.replace(-1, "?").to_csv(f"{tmp_dir}/characters.csv",index = True)
            with open(f"{tmp_dir}/tree.nwk","w") as f:
                f.write(tree.get_newick())
            # Run optimization
            command = f"run_laml -c {tmp_dir}/characters.csv -t {tmp_dir}/tree.nwk -o {tmp_dir}/laml --nInitials {nInitials} --maxIters 0"

            # Check if we need to provide priors.
            if priors is not None:
                states = sorted(list(priors.keys()))
                num_states = len(states)
                priors_df = pd.DataFrame(
                    {
                        "site": [i for i in range(num_sites) for _ in range(num_states)],
                        "state": ([states[i] for i in range(num_states)]) * num_sites,
                        "prob": ([priors[states[i]] for i in range(num_states)]) * num_sites,
                    }
                )
                # print(priors_df)
                # assert(False)
                # Write out priors, and add priors to the command.
                priors_df.to_csv(f"{tmp_dir}/priors.csv", index=False)
                command += f" -p {tmp_dir}/priors.csv"

            print(f"Going to run LAML command: {command}")
            subprocess.run(command, shell=True)
            # Read output
            with open(f"{tmp_dir}/laml_trees.nwk") as f:
                nwk = f.read()
        tree_with_bls_but_wrong_internal_node_names = deepcopy(tree)
        tree_with_bls_but_wrong_internal_node_names.populate_tree(tree=nwk)

        # Now we need to create the mapping between correct and wrong node names
        # The leaf names will agree
        correct_to_wrong_node_name = {leaf: leaf for leaf in tree.leaves}
        for node in tree.depth_first_traverse_nodes(postorder=True):
            if node != tree.root:
                # Set the parent
                correct_to_wrong_node_name[tree.parent(node)] = tree_with_bls_but_wrong_internal_node_names.parent(correct_to_wrong_node_name[node])

        # Now we set the times.
        times = tree_with_bls_but_wrong_internal_node_names.get_times()
        times = {
            node: times[correct_to_wrong_node_name[node]]
            for node in tree.nodes
        }
        tree.set_times(times)
        assert(
            sorted(list(tree.nodes)) == original_nodes
        )


def test_LAML():
    """
    Perfect binary tree with "normal" amount of mutations on each edge.

    Regression test. Cannot be solved by hand. We just check that this
    test runs without error.
    """
    tree = nx.DiGraph()
    tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
    tree.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "5"),
            ("2", "6"),
        ]
    )
    tree = CassiopeiaTree(tree=tree)
    tree.set_all_character_states(
        {
            "0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "1": [1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            "2": [0, 0, 0, 0, 0, 6, 0, 0, 0, -1],
            "3": [1, 2, 0, 0, 0, 0, 0, 0, 0, -1],
            "4": [1, 0, 3, 0, 0, 0, 0, 0, 0, -1],
            "5": [0, 0, 0, 0, 5, 6, 7, 0, 0, -1],
            "6": [0, 0, 0, 4, 0, 6, 0, 8, 9, -1],
        }
    )
    model = LAML()
    model.estimate_branch_lengths(tree)


def test_LAML_with_priors():
    """
    Same as `test_LAML` but with priors.
    """
    tree = nx.DiGraph()
    tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
    tree.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "5"),
            ("2", "6"),
        ]
    )
    tree = CassiopeiaTree(tree=tree)
    tree.set_all_character_states(
        {
            "0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "1": [1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            "2": [0, 0, 0, 0, 0, 6, 0, 0, 0, -1],
            "3": [1, 2, 0, 0, 0, 0, 0, 0, 0, -1],
            "4": [1, 0, 3, 0, 0, 0, 0, 0, 0, -1],
            "5": [0, 0, 0, 0, 5, 6, 7, 0, 0, -1],
            "6": [0, 0, 0, 4, 0, 6, 0, 8, 9, -1],
        }
    )
    model = LAML(
        priors={
            i: i / (5.0 * 9.0) for i in range(1, 10)
        }
    )
    model.estimate_branch_lengths(tree)


def test_LAML_small():
    """
    Small test.
    """
    tree = nx.DiGraph()
    tree.add_nodes_from(["0", "1", "2", "3"]),
    tree.add_edges_from(
        [
            ("0", "1"),
            ("1", "2"),
            ("1", "3"),
        ]
    )
    tree = CassiopeiaTree(tree=tree)
    tree.set_all_character_states(
        {
            "0": [0, 0, 0, 0],
            "1": [0, 0, 0, 1],
            "2": [1, 0, 0, 1],
            "3": [0, 1, 0, 1],
        }
    )
    model = LAML(nInitials=1, priors={1: 0.5, 2: 0.5})
    model.estimate_branch_lengths(tree)

    # Let's see what unregularized ConvexML would do.
    from cassiopeia.tools.branch_length_estimator import IIDExponentialMLE
    convexml = IIDExponentialMLE(
        minimum_branch_length=0,
        pseudo_mutations_per_edge=0,
        pseudo_non_mutations_per_edge=0
    )
    convexml.estimate_branch_lengths(tree)
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 0.41503502789736124, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "2"), 0.5849649721026386, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "3"), 0.5849649721026386, decimal=3
    )


def test_LAML_smallest():
    """
    Small test.
    """
    tree = nx.DiGraph()
    tree.add_nodes_from(["0", "1"]),
    tree.add_edges_from(
        [
            ("0", "1"),
        ]
    )
    tree = CassiopeiaTree(tree=tree)
    tree.set_all_character_states(
        {
            "0": [0, 0],
            "1": [0, 1],
        }
    )
    model = LAML(nInitials=1)
    model.estimate_branch_lengths(tree)
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 1.0, decimal=3  #  Root edge of length 0 makes no sense.
    )
    # Let's see what unregularized ConvexML would do.
    from cassiopeia.tools.branch_length_estimator import IIDExponentialMLE
    convexml = IIDExponentialMLE(
        minimum_branch_length=0,
        pseudo_mutations_per_edge=0,
        pseudo_non_mutations_per_edge=0
    )
    convexml.estimate_branch_lengths(tree)
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 1.0, decimal=3
    )
