from typing import List

from copy import deepcopy
import numpy as np

from cassiopeia.data import CassiopeiaTree


def _handle_double_resections_for_one_cell(
    states: np.array,
    size_of_cassette: int,
    missing_data_indicator: int,
) -> np.array:
    """
    Given the states of one cell, handle double resections.
    """
    res = deepcopy(states)
    num_sites = len(states)
    assert(num_sites % size_of_cassette == 0)
    num_cassettes = int(num_sites / size_of_cassette)
    for cassette in range(num_cassettes):
        cassette_start = cassette * size_of_cassette
        cassette_end = (cassette + 1) * size_of_cassette
        cassette_states = states[cassette_start:cassette_end]
        start_idx = 0
        while start_idx < size_of_cassette:
            end_idx = start_idx
            if cassette_states[start_idx] > 10**8:
                # Need to find the end of the double resection
                while end_idx + 1 < size_of_cassette and cassette_states[end_idx + 1] == cassette_states[start_idx]:
                    end_idx += 1
                # Now, we have the start and end of the double resection
                # We need to make the interior target sites missing
                res[(cassette_start + start_idx + 1):(cassette_start + end_idx)] = missing_data_indicator
            start_idx = end_idx + 1
    return res


def handle_double_resections(
    tree: CassiopeiaTree,
    size_of_cassette: int,
    missing_data_indicator: int,
    also_internal_nodes: bool = False,
) -> CassiopeiaTree:
    """
    Handles double resections by making the interior target sites go missing.

    So, for example, in a cassette of size 3 with states:
    [100000005, 100000005, 100000005]
    missing data is introduced to obtain:
    [100000005, -1, 100000005]
    In general, the -1 can be changed to `missing_data_indicator`.
    It is assumed that double resections are encoded as contiguous
    integers > 10^8 that are all the same; maximal stretches of
    integers > 10^8 that are all the same are assumed to be double
    resections.

    To also do this for the internal nodes in the tree, use
    `also_internal_nodes` = True
    """
    tree = deepcopy(tree)
    character_matrix_dict = {}
    cm = tree.character_matrix
    new_cm = deepcopy(cm)
    for cell in new_cm.index:
        states = np.array(new_cm.loc[cell].values)
        new_states = _handle_double_resections_for_one_cell(
            states=states,
            size_of_cassette=size_of_cassette,
            missing_data_indicator=missing_data_indicator,
        )
        character_matrix_dict[cell] = list(new_states)

    # Now we need to handle the internal nodes (including the root)
    for node in tree.internal_nodes:
        states = np.array(tree.get_character_states(node))
        if also_internal_nodes:
            new_states = _handle_double_resections_for_one_cell(
                states=states,
                size_of_cassette=size_of_cassette,
                missing_data_indicator=missing_data_indicator,
            )
        else:
            new_states = states
        character_matrix_dict[node] = list(new_states)
    tree.set_all_character_states(character_matrix_dict)
    return tree
