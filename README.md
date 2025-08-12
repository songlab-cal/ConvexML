# ConvexML: Branch length estimation under irreversible mutation models.

This package implements the `ConvexML` branch length estimation method. Aditionally, if allows seamlessly reproducing all results from our `ConvexML` paper.

## Installation

ConvexML requires Python version 3.10 or greater. If needed, you can create and activate a virtual enviroment for `convexml` as follows:
```
conda create --name convexml-env python=3.10
conda activate convexml-env
```

First install the Cassiopeia package, as follows:
```
pip install git+https://github.com/YosefLab/Cassiopeia@master#egg=cassiopeia-lineage
```

Then, to install the `convexml` package, just do:

```
pip install convexml
```

See `Example.ipynb` for an example of how to run ConvexML on your data. The full API of the `convexml` function is as follows:

```
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
        _use_vectorized_implementation: Toggles between vectorized and
            non-vectorized implementations. Only used for profiling purposes.
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
```

## Reproducing all results from our paper

To reproduce all results, first create a python environment and install all requirements. For instance:

```
conda create --name convexml-repro python=3.10
conda activate convexml-repro
pip install -r requirements.txt
```

If you have any issues setting up the environment, you can use the `pip_freeze.txt` instead.

Make sure the tests are passing:

```
pip install pytest
python -m pytest tests/
```

NOTE: If the TiDeTree tests fail, you may need to update your version of Java to a more recent version (please take a look at the TiDeTree documentation).

Then, you can just run:

```
time python -m casbench.papers.paper_ble.figures
```

NOTE: You can specify the number of processes used to parallelize computation by changing the variable `NUM_PROCESSES = 32` in `figures.py`.

Each function call in the `figures.py` file reproduces one set of figures. All figures will be written to the directory `paper_ble_figures/`:
- `fig_intMEMOIR(), fig_intMEMOIR_errors_by_depth()` reproduces the results on the intMEMOIR data. The figures will be written to the subdirectory `intMEMOIR`.
- `fig_branch_length_distribution()` reproduces the figure showing the distribution of branch lengths of the simulated ground truth trees. The two figures will be written to the subdirectory `branch_length_distribution`.
- The calls to `fig_model_misspecification_errors_only_states()` reproduce the six figures showing the maximum parsimony bias as a function of the number of states. Each figure will be written to a directory `fig_model_misspecification_errors_only_states_[...]`.
- `fig_mutation_rates_splined_from_real_data()` reproduces the figure showing the mutation rates used in our simulation, which have high variability (and thus violate the IID assumption). The figure will be written to `fig_mutation_rates_splined_from_real_data.png`.
- `fig_q_distribution()` reproduces the figure showing the indel state distribution used. The figure will be written to `fig_q_distribution.png`.
- `fig_full_results(with_gt_topology=True, use_ranks=False)` reproduces the full simulation results, using the ground truth tree topology. The figures will be written to the folder `full_results_with_gt_topology`. There are three different `contexts` being evaluated: 400 leaves, 40 leaves, and 2000 leaves. These are contexts 0, 1, and 2 respectively. The call with `with_gt_topology=False` reproduces the results when the tree topology is estimated as well (using the MaxCut tree topology estimator for all methods). The figures will be written to the folder `full_results_without_gt_topology`.
- `fig_tree_simulator_parameters()` reproduces the parameters of the birth-death-process simulation, given in the section `Tree Simulation Details` of our paper.
- `fig_brief_results()` reproduces the brief benchmarking results shown in our main text, which are only for the `default` lineage tracing regime. The figures will be written to the `fig_brief_results` subdirectory.
- The calls to `fig_gt_trees(), fig_gt_trees_40(), fig_gt_trees_2000()` reproduces the figures showing examples of ground truth simulated trees. The trees will be saved to the `gt_trees` subdirectory; the sub-subdirectories `40/` and `2000/` contain the trees with 40 and 2000 leaves respectively.

The codebase uses caching to make benchmarking faster and seamless. The data caches are set to `_cache`, `_cache_paper_ble`, `_cache_paper_ble_model_mis`. Feel free to delete these cache directories to free up space after you are done reproducing our results.

Note that you do not need to download any data files to reproduce our results this way -- all data will be generated de-novo, thus reproducing our results from scratch. In particular, you can change any parameters of the simulation to construct your own new benchmarks.

## Data Availability

Simulated data files used for our benchmark can be found in the Dryad data repository: https://doi.org/10.5061/dryad.qrfj6q5nz
