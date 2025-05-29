import os
import time
import unittest
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from cassiopeia.data import CassiopeiaTree

from casbench import caching
from casbench.ancestral_states_reconstructor import \
    conservative_maximum_parsimony
from casbench.io import read_tree
from casbench.papers.paper_ble.global_variables import CACHE_PAPER_BLE
from casbench.papers.paper_ble.slice_benchmark_paper_ble import (
    get_ble_configs, get_default_regime_args)
from casbench.simulated_data_benchmarking import \
    run_missing_data_imputer_unrolled


def cmp_with_sankoff(
    tree: CassiopeiaTree,
    smart: bool = False,
    smartest: bool = False,
) -> CassiopeiaTree:
    """
    CMP implemented with the Sankoff black-box algorithm in time O(n k^2)

    If 'smart', then it will use irreversibility to check less child states,
    giving time O(n k)

    If 'smartest', then it will further analyze the states below to check less
    node states, giving O(n) time. (TODO: Not implemented)
    """
    if smartest:
        raise NotImplementedError(f"'smartest' option still not implemented")
    tree = deepcopy(tree)
    num_characters = tree.character_matrix.shape[1]

    def cost(x, y):
        """
        Transition cost matrix
        """
        if x == y:
            return 0
        elif x == 0 and y != 0:
            return 1
        elif x >= 1 and y == -1:
            return 1
        else:
            return np.inf

    all_character_states_leaves = {
        leaf: tree.get_character_states(leaf) for leaf in tree.leaves
    }

    cmpr_states_dict = {node: [] for node in tree.nodes}
    for k in range(num_characters):
        character_states_leaves = {
            leaf: all_character_states_leaves[leaf][k] for leaf in tree.leaves
        }
        alphabet_set = set(tree.character_matrix.iloc[:, k].tolist())
        alphabet_set.update({-1, 0})
        alphabet = list(alphabet_set)
        dp = {node: {x: np.inf for x in alphabet} for node in tree.nodes}
        # print(alphabet)
        # assert(False)
        for node in tree.depth_first_traverse_nodes():
            if tree.is_leaf(node):
                gt_state = character_states_leaves[node]
                dp[node][gt_state] = 0
            else:
                valid_states = alphabet if node != tree.root else [0]
                for state in valid_states:
                    child_alphabet = alphabet
                    if smart:
                        if state != 0:
                            child_alphabet = list(set([-1, state]))
                    dp[node][state] = sum(
                        [
                            min(
                                [
                                    dp[child][state_2] + cost(state, state_2)
                                    for state_2 in child_alphabet
                                ]
                            )
                            for child in tree.children(node)
                        ]
                    )

        visited = {
            node: {state: False for state in alphabet} for node in tree.nodes
        }

        def dfs(node, state, optimal_states_for_node, visited):
            """
            Given a node with optimal state, tell its children which of their states are optimal
            """
            if visited[node][state]:
                return
            optimal_states_for_node[node].append(state)
            visited[node][state] = True
            for child in tree.children(node):
                child_alphabet = alphabet
                if smart:
                    if state != 0:
                        child_alphabet = list(set([-1, state]))
                optimal_cost = min(
                    [
                        dp[child][state_2] + cost(state, state_2)
                        for state_2 in child_alphabet
                    ]
                )
                for state_2 in child_alphabet:
                    if optimal_cost == dp[child][state_2] + cost(
                        state, state_2
                    ):
                        dfs(child, state_2, optimal_states_for_node, visited)

        optimal_states_for_node = {node: [] for node in tree.nodes}
        dfs(tree.root, 0, optimal_states_for_node, visited)

        for node in tree.nodes:
            assert len(optimal_states_for_node[node]) >= 1
            if len(optimal_states_for_node[node]) == 1:
                cmpr_states_dict[node].append(optimal_states_for_node[node][0])
            else:
                cmpr_states_dict[node].append(-1)

    tree.set_all_character_states(cmpr_states_dict)
    return tree


class Test_conservative_maximum_parsimony(unittest.TestCase):
    def test_rooted_binary_tree(self):
        tree = nx.DiGraph()
        tree.add_nodes_from([str(i) for i in range(16)])
        tree.add_edges_from(
            [("0", "1")]
            + [(str(i), str(2 * i)) for i in range(1, 8)]
            + [(str(i), str(2 * i + 1)) for i in range(1, 8)]
        )
        tree = CassiopeiaTree(tree=tree)
        cm = pd.DataFrame.from_dict(
            {
                "8": [99, 99, 00, 00, 99, 99, 99],
                "9": [-1, -1, 11, -1, -1, -1, -1],
                "10": [99, -1, 22, -1, 99, 99, 99],
                "11": [-1, -1, 33, 99, -1, -1, 11],
                "12": [-1, 99, 44, -1, 99, 22, 99],
                "13": [-1, -1, -1, 99, -1, -1, -1],
                "14": [-1, -1, 55, 99, -1, -1, 11],
                "15": [-1, -1, -1, -1, -1, -1, 22],
            },
            orient="index",
        )
        tree.character_matrix = cm
        tree.set_character_states_at_leaves(cm)
        tree = conservative_maximum_parsimony(tree)
        tree_sankoff = cmp_with_sankoff(tree)

        def dfs(a_tree, v):
            if v == 0:
                return [(0, a_tree.get_character_states("0"))] + dfs(a_tree, 1)
            elif 2 * v >= len(tree.nodes):
                return [(v, a_tree.get_character_states(str(v)))]
            else:
                return (
                    [(v, a_tree.get_character_states(str(v)))]
                    + dfs(a_tree, 2 * v)
                    + dfs(a_tree, 2 * v + 1)
                )

        for a_tree in [tree, tree_sankoff]:
            cs = dfs(a_tree, 0)
            self.assertEqual(
                cs,
                [
                    (0, [00, 00, 00, 00, 00, 00, 00]),
                    (1, [-1, 99, 00, 00, 99, 00, 00]),
                    (2, [99, 99, 00, 00, 99, 99, 00]),
                    (4, [99, 99, 00, 00, 99, 99, -1]),
                    (8, [99, 99, 00, 00, 99, 99, 99]),
                    (9, [-1, -1, 11, -1, -1, -1, -1]),
                    (5, [99, -1, 00, -1, 99, 99, 00]),
                    (10, [99, -1, 22, -1, 99, 99, 99]),
                    (11, [-1, -1, 33, 99, -1, -1, 11]),
                    (3, [-1, 99, 00, 99, 99, -1, 00]),
                    (6, [-1, 99, -1, 99, 99, -1, -1]),
                    (12, [-1, 99, 44, -1, 99, 22, 99]),
                    (13, [-1, -1, -1, 99, -1, -1, -1]),
                    (7, [-1, -1, -1, 99, -1, -1, 00]),
                    (14, [-1, -1, 55, 99, -1, -1, 11]),
                    (15, [-1, -1, -1, -1, -1, -1, 22]),
                ],
            )

    def test_perfect_binary_tree(self):
        tree = nx.DiGraph()
        tree.add_nodes_from([str(i) for i in range(15)])
        tree.add_edges_from(
            [(str(i), str(2 * i + 1)) for i in range(7)]
            + [(str(i), str(2 * i + 2)) for i in range(7)]
        )
        tree = CassiopeiaTree(tree=tree)
        cm = pd.DataFrame.from_dict(
            {
                "7": [00, 99],
                "8": [00, -1],
                "9": [00, 99],
                "10": [00, -1],
                "11": [00, 99],
                "12": [00, -1],
                "13": [00, -1],
                "14": [00, -1],
            },
            orient="index",
        )
        tree.character_matrix = cm
        tree.set_character_states_at_leaves(cm)
        tree = conservative_maximum_parsimony(tree)
        tree_sankoff = cmp_with_sankoff(tree)

        def dfs(a_tree, v):
            if 2 * v >= len(tree.nodes) - 1:
                return [(v, tree.get_character_states(str(v)))]
            else:
                return (
                    [(v, tree.get_character_states(str(v)))]
                    + dfs(a_tree, 2 * v + 1)
                    + dfs(a_tree, 2 * v + 2)
                )

        for a_tree in [tree, tree_sankoff]:
            cs = dfs(a_tree, 0)
            self.assertEqual(
                cs,
                [
                    (0, [00, 00]),
                    (1, [00, 99]),
                    (3, [00, 99]),
                    (7, [00, 99]),
                    (8, [00, -1]),
                    (4, [00, 99]),
                    (9, [00, 99]),
                    (10, [00, -1]),
                    (2, [00, -1]),
                    (5, [00, -1]),
                    (11, [00, 99]),
                    (12, [00, -1]),
                    (6, [00, -1]),
                    (13, [00, -1]),
                    (14, [00, -1]),
                ],
            )

    @pytest.mark.slow
    def test_large_simulated_tree(self):
        """
        On trees from the BLE paper, check that CMP algorithm gives the same
        result as Sankoff.
        """
        caching.set_cache_dir(CACHE_PAPER_BLE)
        caching.set_dir_levels(3)
        caching.set_log_level(9)

        for repetition in range(50):
            regime_args = get_default_regime_args()
            ble_configs = get_ble_configs(
                regime_args=regime_args,
                model_name="gt__c__r__gt__gt",
                repetition=repetition,
            )
            ble_configs.pop("solver_config")
            ble_configs.pop("mutationless_edges_strategy_config")
            ble_configs.pop("multifurcation_resolver_config")
            ble_configs.pop("ancestral_states_reconstructor_config")
            ble_configs.pop("ble_config")
            ble_configs.pop("ble_tree_scaler_config")
            output_tree_dir = run_missing_data_imputer_unrolled(**ble_configs)
            tree = read_tree(
                os.path.join(output_tree_dir["output_tree_dir"], "result.txt")
            )

            st = time.time()
            tree_cmp_algo = conservative_maximum_parsimony(tree)
            # print(f"time cmp_algo = {time.time() - st}")

            st = time.time()
            tree_sankoff = cmp_with_sankoff(tree, smart=True)
            # print(f"time cmp_sankoff = {time.time() - st}")

            states_cmp_algo = [
                tree_cmp_algo.get_character_states(node)
                for node in tree_cmp_algo.nodes
            ]
            states_sankoff = [
                tree_sankoff.get_character_states(node)
                for node in tree_sankoff.nodes
            ]
            assert states_cmp_algo == states_sankoff
