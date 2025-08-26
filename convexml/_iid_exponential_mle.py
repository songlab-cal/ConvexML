"""
This file stores a subclass of BranchLengthEstimator, the IIDExponentialMLE.
Briefly, this model assumes that CRISPR/Cas9 mutates each site independently
and identically, with an exponential waiting time.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cvxpy as cp
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import IIDExponentialMLEError

from ._branch_length_estimator import BranchLengthEstimator


def _get_edge_depth(tree: CassiopeiaTree) -> int:
    """
    Maximum number of edges from root to leaf in the tree.
    """
    def dfs(v):
        res = 0
        for u in tree.children(v):
            res = max(res, 1 + dfs(u))
        return res

    return dfs(tree.root)


class IIDExponentialMLE(BranchLengthEstimator):
    """
    MLE under a model of IID memoryless CRISPR/Cas9 mutations.

    In more detail, this model assumes that CRISPR/Cas9 mutates each site
    independently and identically, with an exponential waiting time. The
    tree is assumed to have depth exactly 1, and the user can provide a
    minimum branch length. Pseudocounts in the form of fictitious mutations and
    non-mutations can be added to regularize the MLE.

    The relative depth of each leaf can be specified to relax the ultrametric
    constraint. Also, the identicality assumption can be relaxed by providing
    the relative mutation rate of each site.

    The MLE under this set of assumptions is a special kind of convex
    optimization problem known as an exponential cone program, which can be
    readily solved with off-the-shelf (open source) solvers.

    Ancestral states may or may not all be provided. We recommend imputing them
    using the cassiopeia.tools.conservative_maximum_parsimony function.

    Missing states are treated as missing always completely at random (MACAR) by
    the model.

    The estimated mutation rate(s) will be stored as an attribute called
    `mutation_rate`. The log-likelihood will be stored as an attribute
    called `log_likelihood`. The penalized log-likelihood will be stored as an
    attribute called `penalized_log_likelihood` (the penalized log-likelihood
    includes the pseudocounts, whereas the log-likelihood does not).

    Args:
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
            correspond to cell divisions. By default the constraint will be
            applied. Thus, we can turn off the constraint
            by setting pendant_branch_minimum_branch_length_multiplier=0.
            An intermediate option would be
            pendant_branch_minimum_branch_length_multiplier=0.5.
        _use_vectorized_implementation: Deprecated -- we only support the
            vectorized implementation as of ConvexML version 1.0.0
        compute_data_log_likelihood_after_solving: If True, after solving the
            optimization problem, the data log-likelihood (*without pseudocounts*)
            will be computed and stored in the `log_likelihood` attribute.
        verbose: Verbosity level.

    Attributes:
        mutation_rate: The estimated CRISPR/Cas9 mutation rate(s), assuming that
            the tree has depth exactly 1.
        log_likelihood: The log-likelihood of the training data under the
            estimated model.
        penalized_log_likelihood: The penalized log-likelihood (i.e., with
            pseudocounts) of the training data under the estimated model.
        minimum_branch_length: The minimum branch length (which was provided
            during initialization).
        pseudo_mutations_per_edge: The number of fictitious mutations added to
            each edge to regularize the MLE (which was provided during
            initialization).
        pseudo_non_mutations_per_edge: The number of fictitious non-mutations
            added to each edge to regularize the MLE (which was provided during
            initialization).
    """

    def __init__(
        self,
        minimum_branch_length: float = 0.01,
        pseudo_mutations_per_edge: float = 0.0,
        pseudo_non_mutations_per_edge: float = 0.0,
        relative_leaf_depth: Optional[List[Tuple[str, float]]] = None,
        relative_mutation_rates: Optional[List[float]] = None,
        verbose: bool = False,
        solver: str = "ECOS",
        backup_solver: Optional[str] = "SCS",
        pendant_branch_minimum_branch_length_multiplier: float = 1.0,
        compute_data_log_likelihood_after_solving: bool = True,
        _use_vectorized_implementation: bool = True,
    ):
        allowed_solvers = ["ECOS", "SCS", "MOSEK", "CLARABEL"]
        if solver not in allowed_solvers:
            raise ValueError(
                f"Solver {solver} not allowed. "
                f"Allowed solvers: {allowed_solvers}"
            )  # pragma: no cover
        self._minimum_branch_length = minimum_branch_length
        self._pseudo_mutations_per_edge = pseudo_mutations_per_edge
        self._pseudo_non_mutations_per_edge = pseudo_non_mutations_per_edge
        self._relative_leaf_depth = relative_leaf_depth
        self._relative_mutation_rates = relative_mutation_rates
        self._verbose = verbose
        self._solver = solver
        self._backup_solver = backup_solver
        self._pendant_branch_minimum_branch_length_multiplier = pendant_branch_minimum_branch_length_multiplier
        self._use_vectorized_implementation = _use_vectorized_implementation
        self._mutation_rate = None
        self._penalized_log_likelihood = None
        self._log_likelihood = None
        self._backup_solver_was_needed = None
        self._compute_data_log_likelihood_after_solving = compute_data_log_likelihood_after_solving

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        MLE under a model of IID memoryless CRISPR/Cas9 mutations.

        The only caveat is that this method raises an IIDExponentialMLEError
        if the underlying convex optimization solver fails, or a
        ValueError if the character matrix is degenerate (fully mutated,
        or fully unmutated).

        Raises:
            IIDExponentialMLEError
            ValueError
        """
        # Extract parameters
        minimum_branch_length = self._minimum_branch_length
        pseudo_mutations_per_edge = self._pseudo_mutations_per_edge
        pseudo_non_mutations_per_edge = self._pseudo_non_mutations_per_edge
        relative_leaf_depth = self._relative_leaf_depth
        relative_mutation_rates = self._relative_mutation_rates
        solver = self._solver
        backup_solver = self._backup_solver
        pendant_branch_minimum_branch_length_multiplier = self._pendant_branch_minimum_branch_length_multiplier
        _use_vectorized_implementation = self._use_vectorized_implementation
        verbose = self._verbose

        # # # # # Check that the character has at least one mutation # # # # #
        if (tree.character_matrix == 0).all().all() and pseudo_mutations_per_edge == 0.0:
            raise ValueError(
                "The character matrix has no mutations. Please check your data, or use "
                "pseudo_mutations_per_edge > 0."
            )

        # # # # # Check that the character is not saturated # # # # #
        if (tree.character_matrix != 0).all().all() and pseudo_non_mutations_per_edge == 0.0:
            raise ValueError(
                "The character matrix is fully mutated. The MLE does not "
                "exist unless pseudo_non_mutations_per_edge > 0."
            )

        # # # # # Check that the minimum_branch_length makes sense # # # # #
        if _get_edge_depth(tree) * minimum_branch_length >= 1.0:
            raise ValueError(
                "The minimum_branch_length is too large. Please reduce it."
            )

        # # # # # Check that the relative_mutation_rates list is valid # # # # #
        is_rates_specified = False
        if relative_mutation_rates is not None:
            is_rates_specified = True
            if tree.character_matrix.shape[1] != len(relative_mutation_rates):
                raise ValueError(
                    "The number of character sites does not match the length \
                    of the provided relative_mutation_rates list. Please check \
                    your data."
                )
            for x in relative_mutation_rates:
                if x <= 0:
                    raise ValueError(
                        f"Relative mutation rates must be strictly positive, \
                        but you provided: {relative_mutation_rates}"
                    )
        else:
            relative_mutation_rates = [1.0] * tree.character_matrix.shape[1]

        # Group together sites having the same rate
        sites_by_rate = defaultdict(list)
        for i in range(len(relative_mutation_rates)):
            rate = relative_mutation_rates[i]
            sites_by_rate[rate].append(i)

        # # # # # Get and check relative_leaf_depth # # # # #
        if relative_leaf_depth is None:
            relative_leaf_depth = [(leaf, 1.0) for leaf in tree.leaves]
        if sorted([leaf for (leaf, _) in relative_leaf_depth]) != sorted(
            tree.leaves
        ):
            raise ValueError(
                "All leaves - and only leaves - must be specified in "
                f"relative_leaf_depth. You provided: relative_leaf_depth = "
                f"{relative_leaf_depth} but the leaves in the tree are: "
                f"{tree.leaves}"
            )

        deepest_leaf = max(
            (
                (relative_depth, leaf) for (leaf, relative_depth) in relative_leaf_depth
            )
        )[1]
        relative_leaf_depth_dict = dict(relative_leaf_depth)
        del relative_leaf_depth

        # # # # # Create variables of the optimization problem, in a vectorized way # # # # #
        nodes = tree.nodes
        num_nodes = len(nodes)
        node_to_idx = {node: i for (i, node) in enumerate(nodes)}
        t_variables = cp.Variable(num_nodes, name="t")

        # # # # # Create constraints of the optimization problem, in a vectorized way # # # # #
        all_constraints = []

        t_deepest_leaf = t_variables[node_to_idx[deepest_leaf]]

        root_has_time_0_constraint = t_variables[node_to_idx[tree.root]] == 0
        all_constraints.append(root_has_time_0_constraint)

        leaf_set = set(tree.leaves)
        # Map edges to two vectors: one of parents, and one of children.
        # This will allow us to vectorize all the constraints.
        # The order of the edges is given by that in the list of `all_edges`
        all_edges = tree.edges
        parent_idxs = np.array([node_to_idx[parent] for (parent, _) in all_edges])
        child_idxs = np.array([node_to_idx[child] for (_, child) in all_edges])
        edge_expressions = t_variables[child_idxs] - t_variables[parent_idxs]
        # We need to determine which child nodes are leaves, as the minimum branch length constraint
        # gets scaled for these
        leaf_set = set(tree.leaves)
        child_is_leaf = np.array([child in leaf_set for (_, child) in all_edges], dtype=bool)
        del leaf_set
        # Compute the minimum branch length factor in each case
        minimum_branch_length_factor = np.where(
            child_is_leaf,
            minimum_branch_length * pendant_branch_minimum_branch_length_multiplier,
            minimum_branch_length,
        )
        # Now we are ready to apply the minimum branch length constraint in a vectorized way
        minimum_branch_length_constraints = edge_expressions >= minimum_branch_length_factor * t_deepest_leaf
        all_constraints.append(minimum_branch_length_constraints)

        # Now we compute the ultrametric constraints. This applies to all leaves
        # (except the deepest one, which is used as the reference)
        leaf_idxs_ultrametric_constraint = np.array(
            [
                node_to_idx[leaf]
                for leaf in tree.leaves
                if leaf != deepest_leaf
            ],
            dtype=int
        )
        leaf_depth_factor = [
            relative_leaf_depth_dict[nodes[i]] / relative_leaf_depth_dict[deepest_leaf]
            for i in leaf_idxs_ultrametric_constraint
        ]
        if leaf_idxs_ultrametric_constraint.size > 0:
            ultrametric_constraints = t_variables[leaf_idxs_ultrametric_constraint] == t_deepest_leaf * leaf_depth_factor
            all_constraints.append(ultrametric_constraints)

        # # # # # Compute the penalized log-likelihood for edges # # # # #
        penalized_log_likelihood = 0
        num_sites = tree.character_matrix.shape[1]
        assert (
            sum([len(sites_by_rate[rate]) for rate in sites_by_rate.keys()])
            == num_sites
        )

        # To vectorize computation of the number of mutations, we first create
        # a matrix of size #nodes x #states
        num_states = tree.character_matrix.shape[1]
        node_states_matrix = np.zeros(shape=(num_nodes, num_states))
        for node in nodes:
            node_states_matrix[node_to_idx[node]] = tree.get_character_states(node)
        # Now we get it for parents and children of shape #edges x #states
        parent_states_matrix = node_states_matrix[parent_idxs, :]
        child_states_matrix = node_states_matrix[child_idxs, :]
        # Now we are ready to compute the number of mutations and non-mutations on each edge
        non_missing_transition = (parent_states_matrix != tree.missing_state_indicator) & (child_states_matrix != tree.missing_state_indicator)
        is_non_mutation = (parent_states_matrix == 0) & (child_states_matrix == 0)
        is_mutation = (child_states_matrix != parent_states_matrix) & non_missing_transition

        for (rate, site_rate_idxs) in sites_by_rate.items():
            # NOTE: The pseudo-mutations are equally divided amongst all sites.
            # This is why we use the factor of len(site_rate_idxs) / num_sites
            site_rate_idxs_np = np.array(site_rate_idxs, dtype=int)
            num_non_mutations_for_rate = is_non_mutation[:, site_rate_idxs_np].sum(axis=1) + pseudo_non_mutations_per_edge * len(site_rate_idxs) / num_sites
            num_mutations_for_rate = is_mutation[:, site_rate_idxs_np].sum(axis=1) + pseudo_mutations_per_edge * len(site_rate_idxs) / num_sites
            penalized_log_likelihood += cp.sum(
                cp.multiply(
                    -rate * num_non_mutations_for_rate,
                    edge_expressions
                )
            ) + cp.sum(
                cp.multiply(
                    num_mutations_for_rate,
                    cp.log(
                        1.0 - cp.exp(-rate * edge_expressions - 1e-5)
                    )
                )
            )

        # # # # # Add in log-likelihood of long-edge mutations # # # #
        long_edge_mutations = self._get_long_edge_mutations(tree, sites_by_rate)
        for rate in long_edge_mutations:
            parents__childs__num_mutated = long_edge_mutations[rate].items()
            if len(parents__childs__num_mutated) == 0:
                continue
            parents__childs = [x[0] for x in parents__childs__num_mutated]
            num_mutated = [x[1] for x in parents__childs__num_mutated]
            parents = [x[0] for x in parents__childs]
            childs = [x[1] for x in parents__childs]
            parent_idxs_long_edges = np.array([node_to_idx[parent] for parent in parents])
            child_idxs_long_edges = np.array([node_to_idx[child] for child in childs])
            edge_expressions_long_edges = t_variables[child_idxs_long_edges] - t_variables[parent_idxs_long_edges]
            penalized_log_likelihood += cp.sum(
                cp.multiply(
                    num_mutated,
                    cp.log(
                        1.0 - cp.exp(
                            -edge_expressions_long_edges * rate - 1e-5
                        )
                    )
                )
            )

        # # # Normalize penalized_log_likelihood by the number of sites # # #
        # This is just to keep the log-likelihood on a similar scale
        # regardless of the number of characters.
        penalized_log_likelihood /= num_sites

        # # # # # Solve the problem # # # # #
        obj = cp.Maximize(penalized_log_likelihood)
        prob = cp.Problem(obj, all_constraints)
        try:
            prob.solve(solver=solver, verbose=verbose)
            self._backup_solver_was_needed = False
        except cp.SolverError as err:  # pragma: no cover
            # We try the backup_solver
            try:
                if backup_solver is None:
                    # We don't retry; just raise the original error
                    raise err
                prob.solve(solver=backup_solver, verbose=verbose)
                self._backup_solver_was_needed = True
            except cp.SolverError:  # pragma: no cover
                raise IIDExponentialMLEError("Third-party solver(s) failed")

        # # # # # Extract the mutation rate # # # # #
        scaling_factor = float(t_variables[node_to_idx[deepest_leaf]].value)
        if scaling_factor < 1e-8 or scaling_factor > 15.0:
            # Note: when passing in very small relative mutation rates, this
            # check will fail even though everything is OK. Still worth checking
            # and raising an error.
            raise IIDExponentialMLEError(
                "The solver failed when it shouldn't have."
            )
        if is_rates_specified:
            self._mutation_rate = tuple(
                [rate * scaling_factor for rate in relative_mutation_rates]
            )
        else:
            self._mutation_rate = scaling_factor

        # # # # # Extract the log-likelihood # # # # #
        # Need to re-scale by the number of characters
        penalized_log_likelihood = (
            float(penalized_log_likelihood.value)
            * tree.character_matrix.shape[1]
        )
        if np.isnan(penalized_log_likelihood):
            penalized_log_likelihood = -np.inf
        self._penalized_log_likelihood = penalized_log_likelihood

        # # # # # Populate the tree with the estimated branch lengths # # # # #
        times = {
            node: float(t_variables[node_to_idx[node]].value) / scaling_factor
            for node in tree.nodes
        }
        # Make sure that the root has time 0 (avoid epsilons)
        times[tree.root] = 0.0
        # We smooth out epsilons that might make a parent's time greater
        # than its child (which can happen if minimum_branch_length=0)
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = max(times[parent], times[child])
        tree.set_times(times)
        if self._compute_data_log_likelihood_after_solving:
            self._log_likelihood = self.model_log_likelihood(
                tree=tree, mutation_rate=self._mutation_rate
            )

    @property
    def penalized_log_likelihood(self):
        """
        The penalized log-likelihood of the training data.
        """
        return self._penalized_log_likelihood

    @property
    def log_likelihood(self):
        """
        The log-likelihood of the training data.
        """
        return self._log_likelihood

    @property
    def mutation_rate(self):
        """
        The estimated CRISPR/Cas9 mutation rate(s) under the given model. If
        relative_mutation_rates is specified, we return a list of rates (one per
        site). Otherwise all sites have the same rate and that rate is returned.
        """
        return self._mutation_rate

    @property
    def minimum_branch_length(self):
        """
        The minimum_branch_length.
        """
        return self._minimum_branch_length

    @property
    def pseudo_mutations_per_edge(self):
        """
        The pseudo_mutations_per_edge.
        """
        return self._pseudo_mutations_per_edge

    @property
    def pseudo_non_mutations_per_edge(self):
        """
        The pseudo_non_mutations_per_edge.
        """
        return self._pseudo_non_mutations_per_edge

    @staticmethod
    def _get_long_edge_mutations(
        tree,
        sites_by_rate: Dict[float, List[int]],
    ) -> Dict[float, Dict[Tuple[str, str], int]]:
        """
        Mutations mapped across multiple edges, by rate.
        """
        long_edge_mutations = {
            rate: defaultdict(float) for rate in sites_by_rate.keys()
        }
        # Note that if there is no missing data, there is nothing to do.
        # So we just check for that first.
        if (tree.character_matrix == tree.missing_state_indicator).sum().sum() == 0:
            return long_edge_mutations
        # We pre-compute all states since we will need repeated access
        character_states_dict = {
            node: tree.get_character_states(node) for node in tree.nodes
        }
        for node in tree.nodes:
            if tree.is_root(node):
                continue
            parent = tree.parent(node)
            character_states = character_states_dict[node]
            parent_states = character_states_dict[parent]
            for rate in sites_by_rate.keys():
                for i in sites_by_rate[rate]:
                    if (
                        character_states[i] > 0
                        and parent_states[i] == tree.missing_state_indicator
                    ):
                        # Need to go up the tree and determine if we have a long
                        # edge mutation.
                        u = parent
                        while (
                            character_states_dict[u][i]
                            == tree.missing_state_indicator
                        ):
                            u = tree.parent(u)
                        if character_states_dict[u][i] != character_states[i]:
                            # We have identified a 'long-edge' mutation
                            long_edge_mutations[rate][(u, node)] += 1
        return long_edge_mutations

    @staticmethod
    def model_log_likelihood(
        tree: CassiopeiaTree, mutation_rate: Union[float, List[float]]
    ) -> float:
        """
        Model log-likelihood.

        The log-likelihood of the given character states under the model,
        up to constants (the q distribution is ignored).

        Used for cross-validation.

        Args:
            tree: The given tree with branch lengths
            mutation_rate: Either the mutation rate of all sites (a float) or a
                list of mutation rates, one per site.
        """
        num_sites = tree.character_matrix.shape[1]
        if type(mutation_rate) is float:
            mutation_rate = [mutation_rate] * num_sites
        if len(mutation_rate) != tree.character_matrix.shape[1]:
            raise ValueError(
                "mutation_rate must have the same length as the number of "
                f"sites in the tree, but mutation_rate = {mutation_rate} "
                f"whereas the tree has {num_sites} sites."
            )

        # Group together sites having the same rate
        sites_by_rate = defaultdict(list)
        for i in range(len(mutation_rate)):
            rate = mutation_rate[i]
            sites_by_rate[rate].append(i)

        # # # # # Compute the log-likelihood for edges # # # # #
        log_likelihood = 0
        for (parent, child) in tree.edges:
            edge_length = tree.get_time(child) - tree.get_time(parent)
            parent_states = tree.get_character_states(parent)
            child_states = tree.get_character_states(child)
            for rate in sites_by_rate.keys():
                num_mutated = 0
                num_unmutated = 0
                for site in sites_by_rate[rate]:
                    if parent_states[site] == 0 and child_states[site] == 0:
                        num_unmutated += 1
                    elif parent_states[site] != child_states[site]:
                        if (
                            parent_states[site] != tree.missing_state_indicator
                            and child_states[site]
                            != tree.missing_state_indicator
                        ):
                            num_mutated += 1
                if num_unmutated > 0:
                    log_likelihood += num_unmutated * (-edge_length * rate)
                if num_mutated > 0:
                    log_likelihood += num_mutated * np.log(
                        1 - np.exp(-edge_length * rate - 1e-5)
                    )

        # # # # # Add in log-likelihood of long-edge mutations # # # # #
        long_edge_mutations = IIDExponentialMLE._get_long_edge_mutations(
            tree, sites_by_rate
        )
        for rate in long_edge_mutations:
            for ((parent, child), num_mutated) in long_edge_mutations[
                rate
            ].items():
                edge_length = tree.get_time(child) - tree.get_time(parent)
                log_likelihood += num_mutated * np.log(
                    1
                    - np.exp(
                        -edge_length * rate - 1e-5
                    )  # We add eps for stability.
                )

        if np.isnan(log_likelihood):
            raise ValueError("tree has nan log-likelihood.")

        return log_likelihood
