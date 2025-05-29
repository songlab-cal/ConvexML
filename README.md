# ConvexML

This repository allows reproducing all results from our `ConvexML` paper.

To reproduce all results, first create a python environment and install all requirements. For instance:

```
$ conda create --name convexml-repro python=3.10
$ conda activate convexml-repro
$ pip install -r requirements.txt
```

If you have any issues setting up the environment, you can use the `pip_freeze.txt` instead.

Make sure the tests are passing:

```
$ pip install pytest
$ python -m pytest tests/
```

NOTE: If the TiDeTree tests fail, you may need to update your version of Java to a more recent version (please take a look at the TiDeTree documentation).

Then, you can just run:

```
$ time python -m casbench.papers.paper_ble.figures
```

NOTE: You can specify the number of processes used to parallelize computation by changing the variable `NUM_PROCESSES = 8` in `figures.py`.

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
