"""
This module contains the public API for the ConvexML method.
"""
from typing import Dict, List, Optional, Tuple
from cassiopeia.data import CassiopeiaTree
import networkx as nx
import numpy as np
from copy import deepcopy
from ._parsimony import maximum_parsimony, conservative_maximum_parsimony
from ._iid_exponential_mle import IIDExponentialMLE
from ._multifurcations import resolve_multifurcations, pass_times_onto_original_tree


class ConvexMLValueError(Exception):
    pass


def convexml(
    tree_newick: str,
    leaf_sequences: Dict[str, List[int]],
    ancestral_sequences: Optional[Dict[str, List[int]]] = None,
    ancestral_state_reconstructor: Optional[str] = "conservative_maximum_parsimony",
    resolve_multifurcations_before_branch_length_estimation: bool = True,
    recover_multifurcations_after_branch_length_estimation: bool = True,
    minimum_branch_length: float = 0.01,
    pseudo_mutations_per_edge: float = 0.1,
    pseudo_non_mutations_per_edge: float = 0.1,
    relative_leaf_depth: Optional[List[Tuple[str, float]]] = None,
    relative_mutation_rates: Optional[List[float]] = None,
    verbose: bool = False,
    solver: str = "CLARABEL",
    backup_solver: Optional[str] = "SCS",
    pendant_branch_minimum_branch_length_multiplier: float = 0.5,
    _use_vectorized_implementation: bool = True,
    compute_data_log_likelihood_after_solving: bool = True,
) -> Dict[str, object]:
    """
    ConvexML method for branch length estimation under an irreversible mutation model.

    Arguments:
        tree_newick: The Newick string representation of the tree topology.
            For example: "((D,F),(B,H));".
        leaf_sequences: A dictionary mapping leaf names to their sequences,
            where sequences are represented as lists of integers. Missing data
            should be represented as -1.
        ancestral_sequences: Optionally, the ancestral states can be provided
            too. If not provided (i.e. None), then the algorithm to
            reconstruct the ancestral sequences can be provided with
            `ancestral_state_reconstructor`. If you provide ancestral sequences,
            then your newick tree should name the internal nodes too, e.g.
            "((D,F)E,(B,H)B);".
        ancestral_state_reconstructor: Either "maximum_parsimony" or
            "conservative_maximum_parsimony".
            Use None when `ancestral_sequences` are provided.
            If "maximum_parsimony", the maximum parsimony ancestral states are
                computed. `ancestral_sequences` must be None.
            If "conservative_maximum_parsimony", the conservative maximum
                parsimony ancestral states are computed. `ancestral_sequences`
                must be None.
        resolve_multifurcations_before_branch_length_estimation: Whether to
            resolve multifurcations before branch length estimation. The
            multifurcations may be recovered later by using
            `recover_multifurcations_after_branch_length_estimation=True`.
        recover_multifurcations_after_branch_length_estimation: Whether to
            recover multifurcations after branch length estimation. This can
            only be used with
            `resolve_multifurcations_before_branch_length_estimation=True`.
        minimum_branch_length: Estimated branch lengths will be constrained to
            have length at least this value. By default it is set to 0.01,
            since the MLE tends to collapse mutationless edges to length 0.
        pseudo_mutations_per_edge: Regularization whereby we add this number of
            fictitious mutations to each edge in the tree.
        pseudo_non_mutations_per_edge: Regularization whereby we add this number
            of fictitious non-mutations to each edge in the tree.
        relative_leaf_depth: If provided, the relative depth of each leaf in the
            tree. This allows relaxing the ultrametric assumption to deal with
            the case where the tree is not ultrametric but the relative leaf
            depths are known.
        relative_mutation_rates: List of positive floats of length equal to the
            number of character sites. Number at each character site indicates
            the relative mutation rate at that site. Must be fully specified or
            None in which case all sites are assumed to evolve at the same rate.
            None is the default value for this argument.
        verbose: Verbosity.
        solver: Convex optimization solver to use. Can be "SCS", "ECOS", or
            "MOSEK". Note that "MOSEK" solver should be installed separately.
            We recommend "ECOS" (which is the default).
        backup_solver: In case the main solver fails, this backup solver will
            be tried instead. Useful for applying a faster but less
            sophisticated solver first, and only falling back to a very
            reliable but slower solver if needed. We recommend "SCS" (which is
            the default). (If `backup_solver=None` is provided, no retry will be
            attempted and an error will be raised immediately if the main solver
            fails.)
        pendant_branch_minimum_branch_length_multiplier: For pendant edges in
            the tree (i.e. those corresponding to leaf nodes), the minimum
            branch length constraint does not really apply since leaves do not
            correspond to cell divisions. Thus we set
            pendant_branch_minimum_branch_length_multiplier=0.5 by
            default.
        _use_vectorized_implementation: Deprecated -- we only support the
            vectorized implementation as of ConvexML version 1.0.0
        compute_data_log_likelihood_after_solving: If True, after solving the
            optimization problem, the data log-likelihood (*without pseudocounts*)
            will be computed and stored in the `log_likelihood` attribute.
    Returns:
        A dictionary containing:
            - "tree_newick": The Newick string representation of the tree with
                estimated branch lengths.
            - "tree_cassiopeia": The CassiopeiaTree object with the estimated
                branch lengths and ancestral character states.
            - "model": The branch length estimation model, which is of the
                class IIDExponentialMLE. This is for advanced usage and testing
                only, e.g. if you want to extract the estimated mutation rate
                of the model.
    Raises:
        ConvexMLValueError: If the arguments are not compatible with each other.
    """
    branch_length_estimation_model = IIDExponentialMLE(
        minimum_branch_length=minimum_branch_length,
        pseudo_mutations_per_edge=pseudo_mutations_per_edge,
        pseudo_non_mutations_per_edge=pseudo_non_mutations_per_edge,
        relative_leaf_depth=relative_leaf_depth,
        relative_mutation_rates=relative_mutation_rates,
        verbose=verbose,
        solver=solver,
        backup_solver=backup_solver,
        pendant_branch_minimum_branch_length_multiplier=pendant_branch_minimum_branch_length_multiplier,
        _use_vectorized_implementation=_use_vectorized_implementation,
        compute_data_log_likelihood_after_solving=compute_data_log_likelihood_after_solving,
    )

    tree = CassiopeiaTree(tree=tree_newick)

    # Case 1 (less commons scenario): Ancestral states provided
    if ancestral_sequences is not None:
        # In this case, there are several constraints on the arguments:
        if ancestral_state_reconstructor is not None:
            raise ConvexMLValueError(
                "If `ancestral_sequences` are provided, "
                "`ancestral_state_reconstructor` must be None."
            )
        if resolve_multifurcations_before_branch_length_estimation:
            raise ConvexMLValueError(
                "If `ancestral_sequences` are provided, "
                "`resolve_multifurcations_before_branch_length_estimation` must be False."
            )
        if recover_multifurcations_after_branch_length_estimation:
            raise ConvexMLValueError(
                "If `ancestral_sequences` are provided, "
                "`recover_multifurcations_after_branch_length_estimation` must be False."
            )
        all_states = {**leaf_sequences, **ancestral_sequences}
        tree.set_all_character_states(all_states)
        branch_length_estimation_model.estimate_branch_lengths(tree)
    elif ancestral_sequences is None:
        ancestral_sequences = {internal_node: [-1] * len(next(iter(leaf_sequences.values())))
                                 for internal_node in tree.internal_nodes}
        all_states = {**leaf_sequences, **ancestral_sequences}
        tree.set_all_character_states(all_states)
        # Case 2 (most common scenario): Ancestral states not provided
        # First figure out if we need to resolve multifurcations
        if resolve_multifurcations_before_branch_length_estimation:
            if recover_multifurcations_after_branch_length_estimation:
                # We need to retain the original tree structure
                tree_newick_original = deepcopy(tree_newick)
                tree_original = deepcopy(tree)
            tree = resolve_multifurcations(tree)
        # Now we need to reconstruct the ancestral states
        if ancestral_state_reconstructor is None:
            raise ConvexMLValueError(
                "If `ancestral_sequences` are None, "
                "`ancestral_state_reconstructor` must be provided."
            )
        if ancestral_state_reconstructor == "maximum_parsimony":
            tree = maximum_parsimony(tree)
        elif ancestral_state_reconstructor == "conservative_maximum_parsimony":
            tree = conservative_maximum_parsimony(tree)
        else:
            raise ConvexMLValueError(
                "Invalid value for `ancestral_state_reconstructor`. "
                "Must be one of 'maximum_parsimony', 'conservative_maximum_parsimony', or None."
            )
        # Now we perform branch length estimation
        branch_length_estimation_model.estimate_branch_lengths(tree)
        # Now we need to recover multifurcations if requested.
        if recover_multifurcations_after_branch_length_estimation:
            tree = pass_times_onto_original_tree(tree, tree_original)
    else:
        raise Exception("This should never happen. Please report this bug to the developers.")
    tree_newick = tree.get_newick(record_branch_lengths=True)
    res_dict = {
        "tree_newick": tree_newick,
        "tree_cassiopeia": tree,
        "model": branch_length_estimation_model,
    }
    return res_dict
