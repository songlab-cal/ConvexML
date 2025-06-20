"""
Test convexml.

This wraps the tests of IIDExponentialMLE in cassiopeia.tools, making sure our
public API wrapper does not change the behavior of the original code.
"""
import math
import unittest

import networkx as nx
import numpy as np
from parameterized import parameterized

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import Cas9LineageTracingDataSimulator
from convexml import convexml
from casbench.simulated_data_benchmarking import run_ble_unrolled, run_internal_node_time_metric_unrolled
from casbench.config import smart_call
from casbench.io import read_tree, read_float
import os
import tempfile
from casbench import caching


def to_newick(
    tree: nx.DiGraph,
    record_branch_lengths: bool = False,
    record_node_names: bool = False,
) -> str:
    """Converts a networkx graph to a newick string.

    Args:
        tree: A networkx tree
        record_branch_lengths: Whether to record branch lengths on the tree in
            the newick string
        record_node_names: Whether to record internal node names on the tree in
            the newick string

    Returns:
        A newick string representing the topology of the tree
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        weight_string = ""

        if record_branch_lengths and g.in_degree(node) > 0:
            parent = list(g.predecessors(node))[0]
            weight_string = ":" + str(g[parent][node]["length"])

        _name = str(node)

        name_string = ""
        if record_node_names:
            name_string = f"{_name}"

        return (
            "%s" % (_name,) + weight_string
            if is_leaf
            else (
                "("
                + ",".join(
                    _to_newick_str(g, child) for child in g.successors(node)
                )
                + ")"
                + name_string
                + weight_string
            )
        )

    root = [node for node in tree if tree.in_degree(node) == 0][0]
    return _to_newick_str(tree, root) + ";"


class TestConvexML(unittest.TestCase):
    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_no_mutations(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There is one unmutated character i.e.:
            root [state = '0']
            |
            v
            child [state = '0']
        Since the character matrix is degenerate (it has no mutations),
        an error should be raised.
        """
        tree = nx.DiGraph()
        tree.add_node("0"), tree.add_node("1")
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0], "1": [0]})
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        with self.assertRaises(ValueError):
            # model.estimate_branch_lengths(tree)
            convexml(
                tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
                ancestral_sequences={
                    internal_node_name: tree.get_character_states(internal_node_name)
                    for internal_node_name in tree.internal_nodes
                },
                leaf_sequences={
                    leaf_name: tree.get_character_states(leaf_name)
                    for leaf_name in tree.leaves
                },
                ancestral_state_reconstructor=None,
                resolve_multifurcations_before_branch_length_estimation=False,
                recover_multifurcations_after_branch_length_estimation=False,
                pseudo_mutations_per_edge=0,
                pseudo_non_mutations_per_edge=0,
            )

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_saturation(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There is one mutated character i.e.:
            root [state = '0']
            |
            v
            child [state = '1']
        Since the character matrix is degenerate (it is saturated),
        an error should be raised.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0], "1": [1]})
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        with self.assertRaises(ValueError):
            # model.estimate_branch_lengths(tree)
            convexml(
                tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
                ancestral_sequences={
                    internal_node_name: tree.get_character_states(internal_node_name)
                    for internal_node_name in tree.internal_nodes
                },
                leaf_sequences={
                    leaf_name: tree.get_character_states(leaf_name)
                    for leaf_name in tree.leaves
                },
                ancestral_state_reconstructor=None,
                resolve_multifurcations_before_branch_length_estimation=False,
                recover_multifurcations_after_branch_length_estimation=False,
                pseudo_mutations_per_edge=0,
                pseudo_non_mutations_per_edge=0,
                solver=solver,
            )

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_hand_solvable_problem_1(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There is one mutated character and one unmutated character, i.e.:
            root [state = '00']
            |
            v
            child [state = '01']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(2) ~ 0.693
        (Note that because the depth of the tree is fixed to 1, r * t0 = r * 1
        is the mutation rate.)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0, 0], "1": [0, 1]})
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
            solver=solver,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        log_likelihood = model.log_likelihood
        self.assertAlmostEqual(model.mutation_rate, np.log(2), places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(log_likelihood, -1.386, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_hand_solvable_problem_1_with_pseudomutations(self, name, solver):
        """
        Same as test_hand_solvable_problem_1 but we use pseudomutations.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0, 0], "1": [0, 1]})
        # model = IIDExponentialMLE(
        #     minimum_branch_length=1e-4,
        #     pseudo_mutations_per_edge=1,
        #     pseudo_non_mutations_per_edge=1,
        #     relative_leaf_depth=[("1", 2.0)],
        #     solver=solver,
        # )
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            pseudo_mutations_per_edge=1,
            pseudo_non_mutations_per_edge=1,
            relative_leaf_depth=[("1", 2.0)],
            solver=solver,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertAlmostEqual(model.mutation_rate, np.log(2), places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(
            model.penalized_log_likelihood, -1.386 * 2, places=2
        )
        self.assertAlmostEqual(model.log_likelihood, -1.386, places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_hand_solvable_problem_2(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There are two mutated characters and one unmutated character, i.e.:
            root [state = '000']
            |
            v
            child [state = '011']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} log(exp(-r * t0)) + 2 * log(1 - exp(-r * t0))
        The solution is r * t0 = ln(3) ~ 1.098
        (Note that because the depth of the tree is fixed to 1, r * t0 = r * 1
        is the mutation rate.)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0, 0, 0], "1": [0, 1, 1]})
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
            solver=solver,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(3), places=3)
        self.assertAlmostEqual(model.log_likelihood, -1.910, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_hand_solvable_problem_2_pseudomutations(self, name, solver):
        """
        Same as test_hand_solvable_problem_2 but with pseudomutations.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0, 0], "1": [0, 1]})
        # model = IIDExponentialMLE(
        #     minimum_branch_length=1e-4,
        #     solver=solver,
        #     pseudo_mutations_per_edge=1,
        #     pseudo_non_mutations_per_edge=0,
        #     relative_leaf_depth=[("1", 0.5)],
        # )
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            solver=solver,
            pseudo_mutations_per_edge=1,
            pseudo_non_mutations_per_edge=0,
            relative_leaf_depth=[("1", 0.5)],
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(3), places=3)
        self.assertAlmostEqual(model.penalized_log_likelihood, -1.910, places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_hand_solvable_problem_3(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There are two unmutated characters and one mutated character, i.e.:
            root [state = '000']
            |
            v
            child [state = '001']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(1.5) ~ 0.405
        (Note that because the depth of the tree is fixed to 1, r * t0 = r * 1
        is the mutation rate.)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0, 0, 0], "1": [0, 0, 1]})
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            solver=solver,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(1.5), places=3)
        self.assertAlmostEqual(model.log_likelihood, -1.910, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_small_tree_with_one_mutation(self, name, solver):
        """
        Perfect binary tree with one mutation at a node 6: Should give very
        short edges 1->3,1->4,0->2.
        The problem can be solved by hand: it trivially reduces to a
        1-dimensional problem:
            min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(1.5) ~ 0.405
        (Note that because the depth of the tree is fixed to 1, r * t0 = r * 1
        is the mutation rate.)
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
                "0": [0],
                "1": [0],
                "2": [0],
                "3": [0],
                "4": [0],
                "5": [0],
                "6": [1],
            }
        )
        # Need to make minimum_branch_length be epsilon or else SCS fails...
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            solver=solver,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "2"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("1", "3"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("1", "4"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("2", "5"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("2", "6"), 1.0, places=3)
        self.assertAlmostEqual(model.log_likelihood, -1.910, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(1.5), places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_small_tree_regression(self, name, solver):
        """
        Perfect binary tree with "normal" amount of mutations on each edge.

        Regression test. Cannot be solved by hand. We just check that this
        solution never changes.
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
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            solver=solver,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertAlmostEqual(model.mutation_rate, 0.378, places=3)
        self.assertAlmostEqual(
            tree.get_branch_length("0", "1"), 0.537, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("0", "2"), 0.219, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("1", "3"), 0.463, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("1", "4"), 0.463, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("2", "5"), 0.781, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("2", "6"), 0.781, places=3
        )
        self.assertAlmostEqual(model.log_likelihood, -22.689, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_on_simulated_data(self, name, solver):
        """
        We run the estimator on data simulated under the correct model.
        The estimator should be close to the ground truth.
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
        tree.set_times(
            {"0": 0, "1": 0.1, "2": 0.9, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0}
        )
        np.random.seed(1)
        Cas9LineageTracingDataSimulator(
            number_of_cassettes=200,
            size_of_cassette=1,
            mutation_rate=1.5,
        ).overlay_data(tree)
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            solver=solver,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertTrue(0.05 < tree.get_time("1") < 0.15)
        self.assertTrue(0.8 < tree.get_time("2") < 1.0)
        self.assertTrue(0.9 < tree.get_time("3") < 1.1)
        self.assertTrue(0.9 < tree.get_time("4") < 1.1)
        self.assertTrue(0.9 < tree.get_time("5") < 1.1)
        self.assertTrue(0.9 < tree.get_time("6") < 1.1)
        self.assertTrue(1.4 < model.mutation_rate < 1.6)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=2
        )
        # Regression test (cannot really be verified by hand that this is the
        # optima):
        self.assertAlmostEqual(model.log_likelihood, -459.797, places=2)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_on_simulated_data_with_relative_leaf_depths(self, name, solver):
        """
        We run the estimator on data simulated under the correct model,
        this time using a non-ultrametric tree. The estimator should be close
        to the ground truth.
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
        tree.set_times(
            {"0": 0, "1": 0.1, "2": 0.5, "3": 0.2, "4": 0.4, "5": 0.7, "6": 1.0}
        )
        np.random.seed(1)
        Cas9LineageTracingDataSimulator(
            number_of_cassettes=300,
            size_of_cassette=1,
            mutation_rate=1.5,
        ).overlay_data(tree)
        # model = IIDExponentialMLE(
        #     minimum_branch_length=1e-4,
        #     relative_leaf_depth=[
        #         ("3", 2),
        #         ("4", 4),
        #         ("5", 7),
        #         ("6", 10),
        #     ],
        #     solver=solver,
        # )
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            relative_leaf_depth=[
                ("3", 2),
                ("4", 4),
                ("5", 7),
                ("6", 10),
            ],
            solver=solver,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertTrue(0.05 < tree.get_time("1") < 0.15)
        self.assertTrue(0.4 < tree.get_time("2") < 0.6)
        self.assertTrue(0.1 < tree.get_time("3") < 0.3)
        self.assertTrue(0.3 < tree.get_time("4") < 0.5)
        self.assertTrue(0.6 < tree.get_time("5") < 0.8)
        self.assertTrue(0.9 < tree.get_time("6") < 1.1)
        self.assertTrue(1.4 < model.mutation_rate < 1.6)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=1
        )
        # Regression test (cannot really be verified by hand that this is the
        # optima):
        self.assertAlmostEqual(model.log_likelihood, -758.032, places=1)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_subtree_collapses_when_no_mutations(self, name, solver):
        """
        A subtree with no mutations should collapse to 0. It reduces the
        problem to the same as in 'test_hand_solvable_problem_1'
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4"]),
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3"), ("0", "4")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0], "1": [1], "2": [1], "3": [1], "4": [0]}
        )
        # model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            solver=solver,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertAlmostEqual(model.log_likelihood, -1.386, places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("1", "2"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("1", "3"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "4"), 1.0, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(2), places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_minimum_branch_length(self, name, solver):
        """
        Test that the minimum branch length feature works.

        Same as test_small_tree_with_one_mutation but now we constrain the
        minimum branch length.Should give very short edges 1->3,1->4,0->2
        and edges 0->1,2->5,2->6 close to 1.
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
                "0": [0],
                "1": [0],
                "2": [0],
                "3": [0],
                "4": [0],
                "5": [0],
                "6": [1],
            }
        )
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=0.01,
            solver=solver,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
            pendant_branch_minimum_branch_length_multiplier=1.0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)
        self.assertAlmostEqual(
            tree.get_branch_length("0", "1"), 0.990, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("0", "2"), 0.010, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("1", "3"), 0.010, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("1", "4"), 0.010, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("2", "5"), 0.990, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("2", "6"), 0.990, places=3
        )
        self.assertAlmostEqual(model.log_likelihood, -1.922, places=3)
        self.assertAlmostEqual(model.mutation_rate, 0.405, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand(
        [
            ("should_pass", [1.5, 2, 2.5], [1.5, 2, 2.5]),
            ("should_pass", [0.05, 0.08, 0.09], [0.05, 0.08, 0.09]),
            ("should_pass", [150, 200, 250], [150, 200, 250]),
            ("should_not_pass", [1.5, 2, 2.5], [1.47, 2, 2.5]),
            ("should_not_pass", [1.5, 2, 2.5], [1.5, 1.97, 2.5]),
            ("should_not_pass", [1.5, 2, 2.5], [1.5, 2, 2.52]),
        ]
    )
    def test_hand_solvable_problem_with_site_rates(
        self, name, solver_rates, math_rates
    ):
        """
        Tree topology is 0->1->2.
        The structure:
            root [state = '000']
            |
            x
            child [state = '100']
            |
            y
            child [state = '110']
        Given the site rates as rate_1, rate_2, and rate_3 respectively
        we find the two branch lengths by solving the MLE expression by hand.
        Prior to rescaling, the first branch is of length
        Ln[(rate_1+rate_2+rate_3)/(rate_2+rate_3)]/rate_1 and the other is
        of length equal to Ln[(rate_2+rate_3)/rate_3]/rate_2.
        """
        rate_1, rate_2, rate_3 = solver_rates
        math_rate_1, math_rate_2, math_rate_3 = math_rates

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2"])
        tree.add_edge("0", "1")
        tree.add_edge("1", "2")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0], "1": [1, 0, 0], "2": [1, 1, 0]}
        )
        relative_rates = [rate_1, rate_2, rate_3]
        # model = IIDExponentialMLE(
        #     minimum_branch_length=1e-4, relative_mutation_rates=relative_rates
        # )
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            relative_mutation_rates=relative_rates,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
            pendant_branch_minimum_branch_length_multiplier=1.0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)

        branch1 = (
            math.log(
                (math_rate_1 + math_rate_2 + math_rate_3)
                / (math_rate_2 + math_rate_3)
            )
            / math_rate_1
        )
        branch2 = (
            math.log((math_rate_2 + math_rate_3) / math_rate_3) / math_rate_2
        )
        total = branch1 + branch2
        branch1, branch2 = branch1 / total, branch2 / total
        mutation_rates = [x * total for x in relative_rates]

        should_be_equal = True
        for r1, r2 in zip(solver_rates, math_rates):
            if r1 != r2:
                should_be_equal = False
                break

        if should_be_equal:
            self.assertAlmostEqual(
                tree.get_branch_length("0", "1"), branch1, places=3
            )
            self.assertAlmostEqual(
                tree.get_branch_length("1", "2"), branch2, places=3
            )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)
            for x, y in zip(model.mutation_rate, mutation_rates):
                self.assertAlmostEqual(x, y, places=3)
        else:
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("0", "1"), branch1, places=3
                )
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("1", "2"), branch2, places=3
                )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)
            for x, y in zip(model.mutation_rate, mutation_rates):
                with self.assertRaises(AssertionError):
                    self.assertAlmostEqual(x, y, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand(
        [
            ("negative_rate", [1.5, -1, 2.5]),
            ("zero_rate", [1, 3, 0]),
            ("too_many_rates", [1, 1, 1, 1]),
            ("too_few_rates", [2, 2]),
            ("empty_list", []),
        ]
    )
    def test_invalid_site_rates(self, name, rates):
        """
        Tree topology is the same as test_hand_solvable_problem_with_site_rate
        but rates are misspecified so we should error out.
        """

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2"])
        tree.add_edge("0", "1")
        tree.add_edge("1", "2")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0], "1": [1, 0, 0], "2": [1, 1, 0]}
        )
        relative_rates = rates
        # model = IIDExponentialMLE(
        #     minimum_branch_length=1e-4,
        #     relative_mutation_rates=relative_rates,
        # )
        with self.assertRaises(ValueError):
            # model.estimate_branch_lengths(tree)
            res_dict = convexml(
                tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
                ancestral_sequences={
                    internal_node_name: tree.get_character_states(internal_node_name)
                    for internal_node_name in tree.internal_nodes
                },
                leaf_sequences={
                    leaf_name: tree.get_character_states(leaf_name)
                    for leaf_name in tree.leaves
                },
                ancestral_state_reconstructor=None,
                resolve_multifurcations_before_branch_length_estimation=False,
                recover_multifurcations_after_branch_length_estimation=False,
                minimum_branch_length=1e-4,
                relative_mutation_rates=relative_rates,
                pseudo_mutations_per_edge=0,
                pseudo_non_mutations_per_edge=0,
                pendant_branch_minimum_branch_length_multiplier=1.0,
            )

    @parameterized.expand(
        [
            (
                "should_pass",
                [1.5, 2, 2.5, 1.5, 2, 2.5],
                [1.5, 2, 2.5, 1.5, 2, 2.5],
            ),
            (
                "should_not_pass",
                [1.5, 2, 2.5, 1.5, 2, 2.5],
                [1.52, 1.98, 2.48, 1.52, 2.01, 2.49],
            ),
        ]
    )
    def test_larger_hand_solvable_problem_with_site_rates(
        self, name, solver_rates, math_rates
    ):
        """
        Tree topology is a duplicated version of
        test_hand_solvable_problem_with_site_rates. That is, we double the
        number of characters (while using the same site rates for each pair)
        and decouple each using missing characters as shown below. The expected
        result is the same as the aforementioned test.

        The structure: ('X' indicates missing data)
                   root [state = '0000000']
                    |
                    x
                  child [state = '100100']
                    |
           |------------------|
           y                  z
        [state=            [state=
         XXX110]            110XXX]
        """
        rate_1, rate_2, rate_3, rate_4, rate_5, rate_6 = solver_rates
        (
            math_rate_1,
            math_rate_2,
            math_rate_3,
            _,
            _,
            _,
        ) = math_rates

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edge("0", "1")
        tree.add_edge("1", "2")
        tree.add_edge("1", "3")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {
                "0": [0, 0, 0, 0, 0, 0],
                "1": [1, 0, 0, 1, 0, 0],
                "2": [-1, -1, -1, 1, 1, 0],
                "3": [1, 1, 0, -1, -1, -1],
            }
        )
        relative_rates = [rate_1, rate_2, rate_3, rate_4, rate_5, rate_6]
        # model = IIDExponentialMLE(
        #     minimum_branch_length=1e-4, relative_mutation_rates=relative_rates
        # )
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            relative_mutation_rates=relative_rates,
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
            pendant_branch_minimum_branch_length_multiplier=1.0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)

        branch1 = (
            math.log(
                (math_rate_1 + math_rate_2 + math_rate_3)
                / (math_rate_2 + math_rate_3)
            )
            / math_rate_1
        )
        branch2 = (
            math.log((math_rate_2 + math_rate_3) / math_rate_3) / math_rate_2
        )
        total = branch1 + branch2
        branch1, branch2 = branch1 / total, branch2 / total
        mutation_rates = [x * total for x in relative_rates]

        should_be_equal = True
        for r1, r2 in zip(solver_rates, math_rates):
            if r1 != r2:
                should_be_equal = False
                break

        if should_be_equal:
            self.assertAlmostEqual(
                tree.get_branch_length("0", "1"), branch1, places=3
            )
            self.assertAlmostEqual(
                tree.get_branch_length("1", "2"), branch2, places=3
            )
            self.assertAlmostEqual(
                tree.get_branch_length("1", "3"), branch2, places=3
            )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)

            for x, y in zip(model.mutation_rate, mutation_rates):
                self.assertAlmostEqual(x, y, places=3)
        else:
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("0", "1"), branch1, places=3
                )
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("1", "2"), branch2, places=3
                )
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("1", "3"), branch2, places=3
                )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)

            for x, y in zip(model.mutation_rate, mutation_rates):
                with self.assertRaises(AssertionError):
                    self.assertAlmostEqual(x, y, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    @parameterized.expand(
        [
            ("should_pass", [1.5, 2, 2.5], [1.5, 2, 2.5]),
            ("should_pass", [0.05, 0.08, 0.09], [0.05, 0.08, 0.09]),
            ("should_pass", [150, 200, 250], [150, 200, 250]),
            ("should_not_pass", [1.5, 2, 2.5], [1.47, 2, 2.5]),
            ("should_not_pass", [1.5, 2, 2.5], [1.5, 1.97, 2.5]),
            ("should_not_pass", [1.5, 2, 2.5], [1.5, 2, 2.52]),
        ]
    )
    def test_hand_solvable_problem_with_site_rates_and_long_edge_mutations(
        self, name, solver_rates, math_rates
    ):
        """
        Same as test_hand_solvable_problem_with_site_rates but we create "fake"
        internal nodes that provide no information.
        The structure:
            root [state = '000']
            |
            fake_internal_01 [state = 'X00'] ------ fake_leaf_01 [state = 'XXX']
            |
            child [state = '100']
            |
            fake_internal_12 [state = '1X0'] ------ fake_leaf_12 [state = '1XX']
            |
            child [state = '110']
        Given the site rates as rate_1, rate_2, and rate_3 respectively
        we find the two branch lengths by solving the MLE expression by hand.
        Prior to rescaling, the first branch is of length
        Ln[(rate_1+rate_2+rate_3)/(rate_2+rate_3)]/rate_1 and the other is
        of length equal to Ln[(rate_2+rate_3)/rate_3]/rate_2.
        """
        rate_1, rate_2, rate_3 = solver_rates
        math_rate_1, math_rate_2, math_rate_3 = math_rates

        tree = nx.DiGraph()
        tree.add_nodes_from(
            [
                "0",
                "1",
                "2",
                "fake_internal_01",
                "fake_internal_12",
                "fake_leaf_01",
                "fake_leaf_12",
            ]
        )
        tree.add_edge("0", "fake_internal_01")
        tree.add_edge("fake_internal_01", "1")
        tree.add_edge("fake_internal_01", "fake_leaf_01")
        tree.add_edge("1", "fake_internal_12")
        tree.add_edge("fake_internal_12", "2")
        tree.add_edge("fake_internal_12", "fake_leaf_12")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {
                "0": [0, 0, 0],
                "fake_internal_01": [-1, 0, 0],
                "fake_leaf_01": [-1, -1, -1],
                "1": [1, 0, 0],
                "fake_internal_12": [1, -1, 0],
                "fake_leaf_12": [1, -1, -1],
                "2": [1, 1, 0],
            }
        )
        relative_rates = [rate_1, rate_2, rate_3]
        # model = IIDExponentialMLE(
        #     minimum_branch_length=1e-4,
        #     relative_mutation_rates=relative_rates,
        #     relative_leaf_depth=[
        #         ("2", 10.0),
        #         ("fake_leaf_12", 9.0),
        #         ("fake_leaf_01", 8.0),
        #     ],
        # )
        # model.estimate_branch_lengths(tree)
        res_dict = convexml(
            tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
            ancestral_sequences={
                internal_node_name: tree.get_character_states(internal_node_name)
                for internal_node_name in tree.internal_nodes
            },
            leaf_sequences={
                leaf_name: tree.get_character_states(leaf_name)
                for leaf_name in tree.leaves
            },
            ancestral_state_reconstructor=None,
            resolve_multifurcations_before_branch_length_estimation=False,
            recover_multifurcations_after_branch_length_estimation=False,
            minimum_branch_length=1e-4,
            relative_mutation_rates=relative_rates,
            relative_leaf_depth=[
                ("2", 10.0),
                ("fake_leaf_12", 9.0),
                ("fake_leaf_01", 8.0),
            ],
            pseudo_mutations_per_edge=0,
            pseudo_non_mutations_per_edge=0,
            pendant_branch_minimum_branch_length_multiplier=1.0,
        )
        tree = res_dict["tree_cassiopeia"]
        tree_newick = res_dict["tree_newick"]
        model = res_dict["model"]
        assert(tree.get_newick(record_branch_lengths=True) == tree_newick)

        branch1 = (
            math.log(
                (math_rate_1 + math_rate_2 + math_rate_3)
                / (math_rate_2 + math_rate_3)
            )
            / math_rate_1
        )
        branch2 = (
            math.log((math_rate_2 + math_rate_3) / math_rate_3) / math_rate_2
        )
        total = branch1 + branch2
        branch1, branch2 = branch1 / total, branch2 / total
        mutation_rates = [x * total for x in relative_rates]

        should_be_equal = True
        for r1, r2 in zip(solver_rates, math_rates):
            if r1 != r2:
                should_be_equal = False
                break

        if should_be_equal:
            self.assertAlmostEqual(
                -tree.get_time("0") + tree.get_time("1"), branch1, places=3
            )
            self.assertAlmostEqual(
                -tree.get_time("1") + tree.get_time("2"), branch2, places=3
            )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)
            self.assertAlmostEqual(tree.get_time("fake_leaf_12"), 0.9, places=3)
            self.assertAlmostEqual(tree.get_time("fake_leaf_01"), 0.8, places=3)
            for x, y in zip(model.mutation_rate, mutation_rates):
                self.assertAlmostEqual(x, y, places=3)
        else:
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    -tree.get_time("0") + tree.get_time("1"), branch1, places=3
                )
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    -tree.get_time("1") + tree.get_time("2"), branch2, places=3
                )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)
            for x, y in zip(model.mutation_rate, mutation_rates):
                with self.assertRaises(AssertionError):
                    self.assertAlmostEqual(x, y, places=3)
        self.assertAlmostEqual(
            model.log_likelihood, model.penalized_log_likelihood, places=3
        )

    def test_intMEMOIR(self):
        tmpdir = tempfile.mkdtemp()
        caching.set_cache_dir(tmpdir)
        trees = {}
        metrics = {}

        for numtree in [1]:
            # print(f"***** numtree = {numtree} *****")
            for ble_name in ["GT", "ConvexML"]:
                tree_simulator_config = {
                    "identifier": "dream_sub1_sims__2024_09_15",
                    "args": {"numtree": numtree},
                }
                leaf_subsampler_config = {
                    "identifier": "UniformLeafSubsampler",
                    "args": {
                        "ratio": 1.0,
                        "random_seed": 42,
                        "collapse_unifurcations": False,
                    },
                }
                tree_scaler_config = {
                    "identifier": "unit_tree_scaler",
                    "args": {},
                }
                lt_simulator_config = {
                    "identifier": "dream_sub1_lt__2024_10_13",
                    "args": {"numtree": numtree},
                }
                missing_data_mechanism_config = {
                    "identifier": "none",
                    "args": {},
                }
                missing_data_imputer_config = {"identifier": "none", "args": {}}
                solver_config = {"identifier": "GroundTruthSolver", "args": {}}
                mutationless_edges_strategy_config = {"identifier": "none", "args": {}}
                multifurcation_resolver_config = {"identifier": "none", "args": {}}
                ancestral_states_reconstructor_config = {"identifier": "maximum_parsimony", "args": {}}  # Since DREAM challenge has no missing data, MP and CMP agree.

                ble_config = None
                if ble_name == "GT":
                    ble_config = {
                        "identifier": "GroundTruthBLE",
                        "args": {},
                    }
                elif ble_name == "ConvexML":
                    ble_config = {
                        "identifier": "IIDExponentialMLE",
                        "args": {
                            "minimum_branch_length": 0.15,
                            "pseudo_mutations_per_edge": 0.1,
                            "pseudo_non_mutations_per_edge": 0.1,
                            "solver": "CLARABEL",
                            "pendant_branch_minimum_branch_length_multiplier": 0.5,
                        },
                    }
                elif ble_name == "LAML":
                    ble_config = {
                        "identifier": "LAML_2024_09_10_v2",
                        "args": {"priors": {1:0.5, 2:0.5}},
                    }
                elif ble_name == "TiDeTree":
                    ble_config = {
                        "identifier": "TiDeTree_2024_09_19_v1",
                        "args": {
                            "priors": {1:0.5, 2:0.5},
                            "experiment_duration": 54.0,
                            "edit_duration": 54.0,
                            "chain_length": 1000000,
                        },
                    }
                else:
                    raise ValueError(f"Unknown ble_name: {ble_name}")

                ble_tree_scaler_config = {"identifier": "unit_tree_scaler", "args": {}}
                internal_node_time_predictor_config = {"identifier": "mrca_impute", "args": {"aggregation": "mean"}}
                metric_config = {"identifier": "mae", "args": {}}

                configs = {
                    "tree_simulator_config": tree_simulator_config,
                    "leaf_subsampler_config": leaf_subsampler_config,
                    "tree_scaler_config": tree_scaler_config,
                    "lt_simulator_config": lt_simulator_config,
                    "missing_data_mechanism_config": missing_data_mechanism_config,
                    "missing_data_imputer_config": missing_data_imputer_config,
                    "solver_config": solver_config,
                    "mutationless_edges_strategy_config": mutationless_edges_strategy_config,
                    "multifurcation_resolver_config": multifurcation_resolver_config,
                    "ancestral_states_reconstructor_config": ancestral_states_reconstructor_config,
                    "ble_config": ble_config,
                    "ble_tree_scaler_config": ble_tree_scaler_config,
                    "internal_node_time_predictor_config": internal_node_time_predictor_config,
                    "metric_config": metric_config,
                }

                tree_dir = smart_call(
                    run_ble_unrolled,
                    configs
                )
                trees[f"{numtree}__{ble_name}"] = read_tree(tree_dir["output_tree_dir"] + "/result.txt")

                metric_dir = smart_call(
                    run_internal_node_time_metric_unrolled,
                    configs
                )
                metrics[f"{numtree}__{ble_name}"] = read_float(metric_dir["output_metric_dir"] + "/result.txt")

            tree = trees[f"{numtree}__GT"]
            res_dict = convexml(
                tree_newick=to_newick(tree.get_tree_topology(), record_node_names=True),
                ancestral_sequences=None,
                leaf_sequences={
                    leaf_name: tree.get_character_states(leaf_name)
                    for leaf_name in tree.leaves
                },
                resolve_multifurcations_before_branch_length_estimation=False,
                recover_multifurcations_after_branch_length_estimation=False,
                minimum_branch_length=0.15,
            )
            tree = res_dict["tree_cassiopeia"]
            tree_newick = res_dict["tree_newick"]
            model = res_dict["model"]
            assert(tree.get_newick(record_branch_lengths=True) == tree_newick)

            # Compare against tree from paper's experiment.
            tree.scale_to_unit_length()
            assert(tree.get_newick(record_branch_lengths=True) == trees[f"{numtree}__ConvexML"].get_newick(record_branch_lengths=True))
