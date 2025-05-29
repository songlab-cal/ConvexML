"""
Node-level prediction for the ground truth tree.

The prediction module enables obtaining predictions for a specific subset of
nodes in the ground truth tree, leveraging MRCAs.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from cassiopeia.data import CassiopeiaTree


def _check_is_single_cell_phylogeny(tree: CassiopeiaTree) -> None:
    """
    Check whether the given tree represents a single-cell phylogeny.

    A single cell phylogeny is a tree where the root has degree exactly 1,
    and each other internal node has degree exactly 2.

    Args:
        tree: The given tree

    Raises:
        ValueError if the tree does not represent a single-cell phylogeny.
    """
    if len(tree.children(tree.root)) != 1:
        raise ValueError(
            "The ground truth tree should be a single cell phylogeny, but the "
            f"root has more than one child ({len(tree.children(tree.root))}) "
            f"children. The root is {tree.root}"
        )
    for node in tree.internal_nodes:
        if node != tree.root:
            if len(tree.children(node)) != 2:
                raise ValueError(
                    "The ground truth tree should be a single cell phylogeny, "
                    f"but an internal node has {len(tree.children(node))} "
                    f"children instead of 2. The node is {node}"
                )
    # This last check should be unnecessary, but just in case.
    for leaf in tree.leaves:
        if len(tree.children(leaf)) != 0:
            raise ValueError(
                "The ground truth tree should be a single cell phylogeny, "
                "but strangely, there is a leaf with "
                f"{len(tree.children(leaf))} children. The leaf is {leaf}"
            )


def _check_is_weak_single_cell_phylogeny(tree: CassiopeiaTree) -> None:
    """
    Check whether the given tree represents a single-cell phylogeny,
    except possibly for the root node having degree 1.

    Args:
        tree: The given tree

    Raises:
        ValueError if the tree does not represent a single-cell phylogeny,
        except possibly for the root node having degree 1.
    """
    if len(tree.children(tree.root)) not in [1, 2]:
        raise ValueError(
            "The ground truth tree should be a single cell phylogeny, but the "
            f"root has ({len(tree.children(tree.root))}) "
            f"children. The root is {tree.root}"
        )
    for node in tree.internal_nodes:
        if node != tree.root:
            if len(tree.children(node)) != 2:
                raise ValueError(
                    "The ground truth tree should be a single cell phylogeny, "
                    f"but an internal node has {len(tree.children(node))} "
                    f"children instead of 2. The node is {node}"
                )
    # This last check should be unnecessary, but just in case.
    for leaf in tree.leaves:
        if len(tree.children(leaf)) != 0:
            raise ValueError(
                "The ground truth tree should be a single cell phylogeny, "
                "but strangely, there is a leaf with "
                f"{len(tree.children(leaf))} children. The leaf is {leaf}"
            )


def node_attribute_predictions_via_mrca_aggregation(
    tree_true: CassiopeiaTree,
    tree_inferred: CassiopeiaTree,
    attribute_name: str,
    aggregation: str,
) -> Dict[str, float]:
    """
    Predicted attribute for all nodes in the ground truth tree.

    The trees should represent a single-cell phylogeny.

    Because the inferred tree might have a different topology from the ground
    truth tree, each pair of leaves contributes to the MRCA in the ground truth
    tree by means of the value observed in the inferred tree.

    Args:
        tree_true: The ground truth tree
        tree_inferred: The inferred tree
        attribute_name: The name of the attrubute, e.g. "time" or "fitness"
        aggregation: 'mean' or 'median'. Specifies how the attribute for the
            internal nodes of the ground truth tree will be computed from the
            MRCA information.

    Returns:
        A dictionary with the predicted attribute value for each internal node
            of the ground truth tree.
    """
    if aggregation not in ["mean", "median"]:
        raise ValueError("Aggregation should be either 'mean' or 'median'")
    # _check_is_single_cell_phylogeny(tree_true)  # This is too pedantic for the DREAM subchallenge 1 because the root doesn't have degree 1...
    _check_is_weak_single_cell_phylogeny(tree_true)

    aggregation_function = None
    if aggregation == "mean":
        aggregation_function = np.mean
    elif aggregation == "median":
        aggregation_function = np.median

    if sorted(tree_true.leaves) != sorted(tree_inferred.leaves):
        raise ValueError(
            "The ground truth tree and the inferred tree should"
            " have the same set of leaves."
        )

    # Create all MRCAs
    leaves = tree_true.leaves
    pairs_of_leaves = [
        (leaves[i], leaves[j])
        for i in range(len(leaves))
        for j in range(i + 1, len(leaves))
    ]
    mrcas_inferred = dict(
        list(tree_inferred.find_lcas_of_pairs(pairs_of_leaves))
    )

    inferred_attribute_aux = defaultdict(list)
    for ((leaf_1, leaf_2), mrca) in tree_true.find_lcas_of_pairs(
        pairs_of_leaves
    ):
        mrca_inferred = mrcas_inferred[(leaf_1, leaf_2)]
        inferred_attribute_aux[mrca].append(
            tree_inferred.get_attribute(mrca_inferred, attribute_name)
        )

    predictions = {}  # Type: Dict[str, float]
    # For the root and the leaves, we just copy over the values
    # Root:
    predictions[tree_true.root] = tree_inferred.get_attribute(
        tree_inferred.root, attribute_name
    )
    # Leaves:
    for leaf in tree_true.leaves:
        predictions[leaf] = tree_inferred.get_attribute(leaf, attribute_name)
    # For the internal nodes, we use the MRCA aggregation:
    for node in tree_true.internal_nodes:
        if node == tree_true.root:
            continue
        if len(inferred_attribute_aux[node]) == 0:
            raise ValueError(
                "There should have been at least one pair of"
                f" leaves whose MRCA is {node}."
            )
        predictions[node] = aggregation_function(inferred_attribute_aux[node])
    return predictions


def _non_root_internal_nodes(tree: CassiopeiaTree) -> List[str]:
    return [
        internal_node
        for internal_node in tree.internal_nodes
        if internal_node != tree.root
    ]


def get_true_vs_predicted_attribute(
    tree_true: CassiopeiaTree,
    tree_inferred: CassiopeiaTree,
    attribute_name: str,
    aggregation: str,
    include_leaves: bool,
    include_non_root_internal_nodes: bool,
    include_root: bool,
) -> Dict[str, Tuple[float, float]]:
    """
    True and predicted attribute for the chosen set of nodes.

    Args:
        tree_true: The ground truth tree
        tree_inferred: The inferred tree
        attribute_name: The name of the attrubute, e.g. "time" or "fitness"
        aggregation: 'mean' or 'median'. Specifies how the attribute for the
            internal nodes of the ground truth tree will be computed from the
            MRCA information.
        include_leaves: Whether to include the leaves
        include_non_root_internal_nodes: Whether to include non-root internal
            nodes.
        include_root: Whether to include the root.

    Returns:
        A dictionary mapping each node of interest to its true and predicted
            value.
    """
    # First, we compute the predictions for the nodes we potentially care about.
    predictions = {}
    if (not include_root) and (not include_non_root_internal_nodes):
        # We only care about the predictions at the leaves.
        predictions = {
            leaf: tree_inferred.get_attribute(leaf, attribute_name)
            for leaf in tree_inferred.leaves
        }
    else:
        # More complicated: we need to perform MRCA aggregation.
        predictions = node_attribute_predictions_via_mrca_aggregation(
            tree_true=tree_true,
            tree_inferred=tree_inferred,
            aggregation=aggregation,
            attribute_name=attribute_name,
        )

    # Now we subset the predictions we actually care about, and pair them with
    # their ground truth.
    nodes_to_consider = []
    if include_leaves:
        nodes_to_consider += tree_true.leaves
    if include_non_root_internal_nodes:
        nodes_to_consider += [
            node
            for node in _non_root_internal_nodes(tree_true)
            if node != tree_true.root
        ]
    if include_root:
        nodes_to_consider += [tree_true.root]
    res = {
        node: (tree_true.get_attribute(node, attribute_name), predictions[node])
        for node in nodes_to_consider
    }
    return res
