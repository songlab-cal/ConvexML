from typing import Dict, List

import numpy as np


def _bootstrap_sites_given_chosen_indices(
    character_matrix_dict: Dict[str, List[int]],
    chosen_indices: List[int],
) -> Dict[str, List[int]]:
    res = {
        node_name: [node_states[i] for i in sorted(chosen_indices)]
        for node_name, node_states in character_matrix_dict.items()
    }
    return res


def test__bootstrap_sites_given_chosen_indices():
    character_matrix_dict = {
        "leaf_1": [0, 1, 2, 2, 2],
        "leaf_2": [0, 1, 2, -1, 2],
        "leaf_3": [1, 0, 2, 2, -1],
        "leaf_4": [1, 0, 2, -1, -1],
        "internal_node_1": [0, 1, 2, 2, 2],
        "internal_node_2": [1, 0, 2, 2, -1],
        "internal_node_3": [0, 0, 2, 2, 2],
        "root": [0, 0, 0, 0, 0],
    }
    res = _bootstrap_sites_given_chosen_indices(
        character_matrix_dict=character_matrix_dict,
        chosen_indices=[0, 4, 1, 3, 2]
    )
    assert(
        res == character_matrix_dict
    )
    res = _bootstrap_sites_given_chosen_indices(
        character_matrix_dict=character_matrix_dict,
        chosen_indices=[0, 0, 1, 4, 4]
    )
    assert(
        res == {
            "leaf_1": [0, 0, 1, 2, 2],
            "leaf_2": [0, 0, 1, 2, 2],
            "leaf_3": [1, 1, 0, -1, -1],
            "leaf_4": [1, 1, 0, -1, -1],
            "internal_node_1": [0, 0, 1, 2, 2],
            "internal_node_2": [1, 1, 0, -1, -1],
            "internal_node_3": [0, 0, 0, 2, 2],
            "root": [0, 0, 0, 0, 0],
        }
    )


test__bootstrap_sites_given_chosen_indices()


def bootstrap_sites(
    character_matrix_dict: Dict[str, List[int]],
    bootstrap_sites_seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(bootstrap_sites_seed)
    indices = list(
        range(
            len(
                list(
                    character_matrix_dict.values()
                )[0]
            )
        )
    )
    chosen_indices = rng.choice(indices, size=len(indices), replace=True)
    res = _bootstrap_sites_given_chosen_indices(
        character_matrix_dict=character_matrix_dict,
        chosen_indices=chosen_indices,
    )
    return res


def test_bootstrap_sites():
    character_matrix_dict = {
        "leaf_1": [0, 1, 2, 2, 2],
        "leaf_2": [0, 1, 2, -1, 2],
        "leaf_3": [1, 0, 2, 2, -1],
        "leaf_4": [1, 0, 2, -1, -1],
        "internal_node_1": [0, 1, 2, 2, 2],
        "internal_node_2": [1, 0, 2, 2, -1],
        "internal_node_3": [0, 0, 2, 2, 2],
        "root": [0, 0, 0, 0, 0],
    }
    res = bootstrap_sites(
        character_matrix_dict=character_matrix_dict,
        bootstrap_sites_seed=42,
    )
    assert(
        res == {
            "leaf_1": [0, 2, 2, 2, 2],
            "leaf_2": [0, 2, 2, -1, -1],
            "leaf_3": [1, 2, 2, 2, 2],
            "leaf_4": [1, 2, 2, -1, -1],
            "internal_node_1": [0, 2, 2, 2, 2],
            "internal_node_2": [1, 2, 2, 2, 2],
            "internal_node_3": [0, 2, 2, 2, 2],
            "root": [0, 0, 0, 0, 0],
        }
    )


test_bootstrap_sites()
