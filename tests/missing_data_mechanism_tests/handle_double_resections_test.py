import pandas as pd
import unittest

import networkx as nx
from cassiopeia.data import CassiopeiaTree

from casbench.missing_data_mechanism import get_missing_data_mechanism_from_config


class Test_conservative_maximum_parsimony(unittest.TestCase):
    def test_rooted_binary_tree_only_leaves(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {
                "0": [100000005, 100000005, 100000005, 100000005, 100000005, 100000005, 0, 0, 0],
                "1": [0, 1, 0, 0, 100000006, 100000006, 100000005, 100000005, 100000005],
                "2": [100000005, 100000005, 100000005, 0, 100000006, 100000006, 100000005, 100000005, 100000005],
                "3": [100000003, 100000003, 0, 0, 1, 0, 100000005, 100000005, 100000005],
            }
        )
        mdm = get_missing_data_mechanism_from_config(
            [
                "handle_double_resections",
                [("missing_data_indicator", -3), ("size_of_cassette", 3)],
            ]
        )
        tree = mdm(tree)

        assert tree.get_character_states("0") == [100000005, 100000005, 100000005, 100000005, 100000005, 100000005, 0, 0, 0]
        assert tree.get_character_states("1") == [0, 1, 0, 0, 100000006, 100000006, 100000005, 100000005, 100000005]
        assert tree.get_character_states("2") == [
            100000005,
            -3,
            100000005,
            0,
            100000006,
            100000006,
            100000005,
            -3,
            100000005,
        ]
        assert tree.get_character_states("3") == [100000003, 100000003, 0, 0, 1, 0, 100000005, -3, 100000005]
        assert all(
            tree.character_matrix.loc["2"] == [
                100000005,
                -3,
                100000005,
                0,
                100000006,
                100000006,
                100000005,
                -3,
                100000005,
            ]
        )
        assert all(
            tree.character_matrix.loc["3"] == [100000003, 100000003, 0, 0, 1, 0, 100000005, -3, 100000005]
        )

    def test_rooted_binary_tree_all_nodes(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {
                "0": [100000005, 100000005, 100000005, 100000005, 100000005, 100000005, 0, 0, 0],
                "1": [0, 1, 0, 0, 100000006, 100000006, 100000005, 100000005, 100000005],
                "2": [100000005, 100000005, 100000005, 0, 100000006, 100000006, 100000005, 100000005, 100000005],
                "3": [100000003, 100000003, 0, 0, 1, 0, 100000005, 100000005, 100000005],
            }
        )
        mdm = get_missing_data_mechanism_from_config(
            [
                "handle_double_resections",
                [("also_internal_nodes", True), ("missing_data_indicator", -3), ("size_of_cassette", 3)],
            ]
        )
        tree = mdm(tree)

        assert tree.get_character_states("0") == [100000005, -3, 100000005, 100000005, -3, 100000005, 0, 0, 0]
        assert tree.get_character_states("1") == [0, 1, 0, 0, 100000006, 100000006, 100000005, -3, 100000005]
        assert tree.get_character_states("2") == [
            100000005,
            -3,
            100000005,
            0,
            100000006,
            100000006,
            100000005,
            -3,
            100000005,
        ]
        assert tree.get_character_states("3") == [100000003, 100000003, 0, 0, 1, 0, 100000005, -3, 100000005]
        assert all(
            tree.character_matrix.loc["2"] == [
                100000005,
                -3,
                100000005,
                0,
                100000006,
                100000006,
                100000005,
                -3,
                100000005,
            ]
        )
        assert all(
            tree.character_matrix.loc["3"] == [100000003, 100000003, 0, 0, 1, 0, 100000005, -3, 100000005]
        )
