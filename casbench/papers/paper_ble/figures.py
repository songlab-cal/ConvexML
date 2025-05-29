"""
To reproduce all figures in the paper, run this script:
$ time python -m casbench.papers.paper_ble.figures
"""
import os
from collections import defaultdict
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import logging

import cassiopeia as cas

from casbench import caching
from casbench.config import create_config_from_dict
from casbench.io import read_float, read_tree
from casbench.lineage_tracer._reverse_engineered_cas9 import \
    reverse_engineer_non_iid_mutation_rates
from casbench.simulated_data_benchmarking import (
    run_ble_unrolled, run_internal_node_time_metric_unrolled,
    run_tree_simulator_unrolled)
from casbench.tree_simulator._ble_paper_tree_simulator import \
    reverse_engineer_birth_and_death_rate

from .global_variables import (BLE_DISPLAY_NAMES, CACHE_PAPER_BLE,
                               CACHE_PAPER_BLE_MODEL_MISSPECIFICATION_ERRORS,
                               METRIC_DISPLAY_NAMES,
                               PAPER_BLE_FIGURES_DIR, PSEUDOMUTATIONS_LIST,
                               SOLVER_DISPLAY_NAMES)
from .slice_benchmark_paper_ble import (eval_function, get_ble_configs,
                                        get_default_regime_args,
                                        get_tree_simulator_configs,
                                        slice_benchmark_paper_ble)
from casbench.config import smart_call
from casbench.branch_length_estimator._tidetree._tidetree import TiDeTreeError

NUM_PROCESSES = 32


def fig_model_misspecification_errors(
    plot_dir: str,
    plot_trees: bool = False,
    number_of_cassettes: int = 33334,
    size_of_cassette: int = 3,
    random_seed: int = 0,
    simulate_double_resections: bool = False,
    handle_double_resections: bool = False,
    mp_strategies=[
        "ground_truth_asr",
        "conservative_maximum_parsimony",
        "maximum_parsimony",
    ],
    title="",
    regime_dict_list: List = [],
):
    """
    Effect of model misspecification on estimated branch lengths.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE_MODEL_MISSPECIFICATION_ERRORS)
    caching.set_dir_levels(3)
    caching.set_log_level(9)

    mp_strategies_display_names = {
        "ground_truth_asr": "With known ancestral states",
        "conservative_maximum_parsimony": "With conservative maximum parsimony",
        "maximum_parsimony": "With naive maximum parsimony",
    }
    mp_strategies_colors = {
        "ground_truth_asr": "black",
        "conservative_maximum_parsimony": "blue",
        "maximum_parsimony": "red",
    }

    regime_list = [x[0] for x in regime_dict_list]

    mae_list = []
    for regime_id, (regime_name, regime_args) in enumerate(regime_dict_list):
        for mp_strategy in mp_strategies:
            print("*" * 30)
            print("Regime:", regime_name, mp_strategy)
            print("*" * 30)

            tree_simulator_config = create_config_from_dict(
                {
                    "identifier": "ble_paper_tree_simulator",
                    "args": {
                        "n_cells": 400,
                        "fitness": "high",
                        "sampling_probability": 0.01,
                        "offset": 0.01,
                        "bd_ratio": 10.0,
                        "random_seed": random_seed,
                    },
                }
            )
            leaf_subsampler_config = create_config_from_dict(
                {
                    "identifier": "UniformLeafSubsampler",
                    "args": {
                        "number_of_leaves": 400,
                        "random_seed": random_seed,
                    },
                }
            )
            tree_scaler_config = create_config_from_dict(
                {"identifier": "unit_tree_scaler", "args": {}}
            )
            lt_simulator_config = create_config_from_dict(
                {
                    "identifier": "ReverseEngineeredCas9LineageTracingDataSimulator",  # noqa
                    "args": {
                        "number_of_cassettes": number_of_cassettes,
                        "size_of_cassette": size_of_cassette,
                        "number_of_states": regime_args.get(
                            "number_of_states", 100
                        ),
                        "expected_proportion_heritable": regime_args.get(
                            "expected_proportion_heritable", 0
                        ),  # noqa
                        "expected_proportion_stochastic": regime_args.get(
                            "expected_proportion_stochastic", 0
                        ),  # noqa
                        "expected_proportion_mutated": 50,
                        "random_seed": random_seed,
                        "iid_mutation_rates": regime_args.get(
                            "iid_mutation_rates", True
                        ),  # noqa
                        "collapse_sites_on_cassette": simulate_double_resections,
                        "create_allele_when_collapsing_sites_on_cassette": simulate_double_resections,
                    },
                }
            )
            if handle_double_resections:
                missing_data_mechanism_config = create_config_from_dict(
                    {
                        "identifier": "handle_double_resections",
                        "args": {
                            "size_of_cassette": size_of_cassette,
                            "missing_data_indicator": -3,
                            "also_internal_nodes": True,
                        }
                    }
                )
            else:
                missing_data_mechanism_config = create_config_from_dict(
                    {
                        "identifier": "none",
                        "args": {
                        }
                    }
                )
            missing_data_imputer_config = create_config_from_dict(
                {"identifier": "none", "args": {}}
            )
            solver_config = create_config_from_dict(
                {"identifier": "GroundTruthSolver", "args": {}}
            )
            mutationless_edges_strategy_config = create_config_from_dict(
                {"identifier": "none", "args": {}}
            )
            multifurcation_resolver_config = create_config_from_dict(
                {"identifier": "none", "args": {}}
            )
            ancestral_states_reconstructor_config = create_config_from_dict(
                {"identifier": mp_strategy, "args": {}}
            )
            ble_config = create_config_from_dict(
                {
                    "identifier": "IIDExponentialMLE",
                    "args": {
                        "minimum_branch_length": 0.001,
                        "pseudo_mutations_per_edge": 0.1,
                        "pseudo_non_mutations_per_edge": 0.1,
                    },
                }
            )
            ble_tree_scaler_config = create_config_from_dict(
                {"identifier": "unit_tree_scaler", "args": {}}
            )
            internal_node_time_predictor_config = create_config_from_dict(
                {"identifier": "mrca_impute", "args": {"aggregation": "mean"}}
            )
            metric_config = create_config_from_dict(
                {"identifier": "mae", "args": {}}
            )

            if plot_trees and regime_id == 0:
                # Only used for plotting
                gt_tree_path = (
                    run_tree_simulator_unrolled(
                        tree_simulator_config=tree_simulator_config,
                        leaf_subsampler_config=leaf_subsampler_config,
                        tree_scaler_config=tree_scaler_config,
                    )["output_tree_dir"]
                    + "/result.txt"
                )
            if plot_trees:
                # Only used for plotting
                ble_tree_path = (
                    run_ble_unrolled(
                        tree_simulator_config=tree_simulator_config,
                        leaf_subsampler_config=leaf_subsampler_config,
                        tree_scaler_config=tree_scaler_config,
                        lt_simulator_config=lt_simulator_config,
                        missing_data_mechanism_config=missing_data_mechanism_config,  # noqa
                        missing_data_imputer_config=missing_data_imputer_config,
                        solver_config=solver_config,
                        mutationless_edges_strategy_config=mutationless_edges_strategy_config,  # noqa
                        multifurcation_resolver_config=multifurcation_resolver_config,  # noqa
                        ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
                        ble_config=ble_config,
                        ble_tree_scaler_config=ble_tree_scaler_config,
                    )["output_tree_dir"]
                    + "/result.txt"
                )
                tree = read_tree(ble_tree_path)
                cm = tree.character_matrix
                print(f"Percent missing: {(cm < 0).mean().mean()}")

            mae_metric_path = (
                run_internal_node_time_metric_unrolled(
                    tree_simulator_config=tree_simulator_config,
                    leaf_subsampler_config=leaf_subsampler_config,
                    tree_scaler_config=tree_scaler_config,
                    lt_simulator_config=lt_simulator_config,
                    missing_data_mechanism_config=missing_data_mechanism_config,
                    missing_data_imputer_config=missing_data_imputer_config,
                    solver_config=solver_config,
                    mutationless_edges_strategy_config=mutationless_edges_strategy_config,  # noqa
                    multifurcation_resolver_config=multifurcation_resolver_config,  # noqa
                    ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
                    ble_config=ble_config,
                    ble_tree_scaler_config=ble_tree_scaler_config,
                    internal_node_time_predictor_config=internal_node_time_predictor_config,  # noqa
                    metric_config=metric_config,
                )["output_metric_dir"]
                + "/result.txt"
            )
            mae_metric = read_float(mae_metric_path)

            print(f"mae_metric = {mae_metric}")

            mae_list.append(mae_metric)

            if plot_trees:
                outdir = os.path.join(
                    PAPER_BLE_FIGURES_DIR,
                    plot_dir,
                )
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                if regime_id == 0 and mp_strategy == mp_strategies[0]:
                    print("***** GT Tree *****")
                    fig, ax = cas.pl.plot_matplotlib(
                        read_tree(gt_tree_path),
                        add_root=False,
                        depth_key="time",
                        extend_branches=False,
                    )
                    plt.savefig(f"{outdir}/gt.png")
                    plt.close()
                print("***** BLE Tree *****")
                fig, ax = cas.pl.plot_matplotlib(
                    read_tree(ble_tree_path),
                    add_root=False,
                    depth_key="time",
                    extend_branches=False,
                )
                plt.savefig(
                    f"{outdir}/regime_id-{regime_id}__mp_strat-{mp_strategy}.png"
                )
                plt.close()

    print(f"mae_list = {mae_list}")

    display_names = regime_list[:]
    fig = plt.figure()
    data = [
        [mae_list[len(mp_strategies) * i + j] for i in range(len(regime_list))]
        for j in range(len(mp_strategies))
    ]
    X = np.arange(len(regime_list))
    ax = fig.add_axes([0, 0, 1, 1])
    for i, mp_strategy in enumerate(mp_strategies):
        n_plus_1 = len(mp_strategies) + 1
        ax.barh(
            X + (n_plus_1 - i) / n_plus_1 + 1 / 6,
            data[i][::-1],
            color=mp_strategies_colors[mp_strategy],
            height=1 / n_plus_1,
        )
    plt.yticks(X + 1, display_names[::-1])
    handles = [
        mpatches.Patch(
            color=mp_strategies_colors[mp_strategy],
            label=mp_strategies_display_names[mp_strategy],
        )
        for mp_strategy in mp_strategies
    ]
    fontsize = 16
    plt.legend(
        handles=handles,
        fontsize=fontsize,
        loc='upper left',
        bbox_to_anchor=(0, -0.15),
    )
    plt.title(title, fontsize=fontsize)
    metric_name = "mae"
    metric_display_names = {"mae": "Internal Node Time, Mean Absolute Error"}
    plt.xlabel(metric_display_names[metric_name], fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    outdir = os.path.join(
        PAPER_BLE_FIGURES_DIR,
        plot_dir,
    )
    os.makedirs(outdir, exist_ok=True)
    plt.grid()
    plt.savefig(f"{outdir}/{metric_name}.png", bbox_inches="tight", dpi=300)
    plt.close()


def fig_model_misspecification_errors_only_states(
    plot_trees: bool = False,
    number_of_cassettes: int = 33334,
    size_of_cassette: int = 3,
    expected_proportion_stochastic: int = 30,
    expected_proportion_heritable: int = 30,
    simulate_double_resections: bool = False,
    handle_double_resections: bool = False,
    plot_dir: str = "fig_model_misspecification_errors_only_states",
    random_seed: int = 0,
):
    """
    Supplementary figure showing that increasing the number of
    states makes CMP work really well.
    """
    regime_dict_list = [
        [
            "1 state",
            {
                "number_of_states": 1,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "2 states",
            {
                "number_of_states": 2,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "3 states",
            {
                "number_of_states": 3,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "4 states",
            {
                "number_of_states": 4,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "5 states",
            {
                "number_of_states": 5,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "6 states",
            {
                "number_of_states": 6,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "7 states",
            {
                "number_of_states": 7,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "8 states",
            {
                "number_of_states": 8,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "9 states",
            {
                "number_of_states": 9,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "10 states",
            {
                "number_of_states": 10,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "20 states",
            {
                "number_of_states": 20,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "30 states",
            {
                "number_of_states": 30,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "40 states",
            {
                "number_of_states": 40,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "50 states",
            {
                "number_of_states": 50,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "60 states",
            {
                "number_of_states": 60,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "70 states",
            {
                "number_of_states": 70,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "80 states",
            {
                "number_of_states": 80,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "90 states",
            {
                "number_of_states": 90,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
        [
            "100 states",
            {
                "number_of_states": 100,
                "expected_proportion_stochastic": expected_proportion_stochastic,
                "expected_proportion_heritable": expected_proportion_heritable,
            },
        ],
    ]
    fig_model_misspecification_errors(
        plot_dir=plot_dir,
        plot_trees=plot_trees,
        number_of_cassettes=number_of_cassettes,
        size_of_cassette=size_of_cassette,
        random_seed=random_seed,
        regime_dict_list=regime_dict_list,
        simulate_double_resections=simulate_double_resections,
        handle_double_resections=handle_double_resections,
        title="Maximum parsimony bias by number of states",
    )


def fig_brief_results(
    fontsize: int = 24,
    legend_fontsize: int = 16,
    xticks_fontsize: int = 22,
):
    """
    Plot model performance on the default regime, with either ground
    truth topology or reconstructed topology.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE)
    caching.set_dir_levels(3)
    caching.set_log_level(9)

    repetitions = 50
    ble_names = [
        f"{mp_strategy}__MLE_mbl-1_pm-{p}_pnm-{p}"
        for p in PSEUDOMUTATIONS_LIST
        for mp_strategy in ["cmp", "mp"]
    ] + [
        "cmp__muts_extended-0.5",
        "mp__muts_extended-0.5",
    ] + [
        "mp__LAML",
    ]
    display_names = [BLE_DISPLAY_NAMES[ble_name] for ble_name in ble_names]

    solver_names = [
        "gt__c__r",
        "maxcut__C__R",
    ]

    model_perfs = defaultdict(list)

    metric_names = [
        "ble__ancestors-0.5-mre",
        "ble__node_time-mae",
        "fe__sr2",
    ]

    for ble_name in ble_names:
        for solver_name in solver_names:
            metric_values = defaultdict(list)
            for repetition in range(repetitions):
                for metric_name in metric_names:
                    metric_value = eval_function(
                        regime_args=get_default_regime_args(),
                        model_name=solver_name + "__" + ble_name,
                        metric_name=metric_name,
                        repetition=repetition,
                    )
                    metric_values[metric_name].append(metric_value)
            for metric_name in metric_names:
                aggregation_over_repetitions_func = np.mean
                model_perfs[metric_name].append(
                    aggregation_over_repetitions_func(
                        metric_values[metric_name]
                    )
                )

    for metric_name in model_perfs.keys():
        fig = plt.figure()
        data = [
            [
                model_perfs[metric_name][2 * i]
                for i in range(len(ble_names))
            ],  # GT topology perfs
            [
                model_perfs[metric_name][2 * i + 1]
                for i in range(len(ble_names))
            ],  # Reconstructed topology perfs
        ]
        X = np.arange(len(ble_names))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.barh(X + 3 / 3 + 1 / 6, data[0][::-1], color="black", height=1 / 3)
        ax.barh(X + 2 / 3 + 1 / 6, data[1][::-1], color="blue", height=1 / 3)
        plt.yticks(X + 1, display_names[::-1], fontsize=fontsize)
        plt.xticks(fontsize=xticks_fontsize)
        if metric_name == "ble__node_time-mae":
            plt.legend(
                handles=[
                    mpatches.Patch(
                        color="black", label="Ground truth\ntopology"
                    ),
                    mpatches.Patch(
                        color="blue", label="Reconstructed\ntopology"
                    ),
                ],
                fontsize=legend_fontsize,
            )
        plt.title(
            METRIC_DISPLAY_NAMES[metric_name]
            + f"\nsummarized over {repetitions} repetitions",
            fontsize=fontsize,
        )
        plt.xlabel(METRIC_DISPLAY_NAMES[metric_name], fontsize=fontsize)
        outdir = os.path.join(PAPER_BLE_FIGURES_DIR, "fig_brief_results")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/{metric_name}.png", bbox_inches="tight", dpi=300)
        plt.close()


def fig_full_results(with_gt_topology, use_ranks=False):
    """
    Supplementary figure showing full results.

    Plot performance of all models on all regimes, using either ground truth
    or reconstructed tree topologies. To get the respective results, call with
    `with_gt_topology=True` and `with_gt_topology=False` respectively.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE)
    caching.set_dir_levels(3)
    caching.set_log_level(9)

    solver_name = "gt__c__r" if with_gt_topology else "maxcut__C__R"
    ble_names = (
        [
            f"{mp_strategy}__MLE_mbl-1_pm-{p}_pnm-{p}"
            for p in PSEUDOMUTATIONS_LIST
            for mp_strategy in ["gt", "cmp", "mp"]
        ]
        + [
            "gt__muts_extended-0.5",
            "cmp__muts_extended-0.5",
            "mp__muts_extended-0.5",
        ]
        + [
            "mp__LAML",
            # "mp__LAML_GT_prior",
            # "mp__TiDeTree",
        ]
        if with_gt_topology
        else [
            f"{mp_strategy}__MLE_mbl-1_pm-{p}_pnm-{p}"
            for p in PSEUDOMUTATIONS_LIST
            for mp_strategy in ["cmp", "mp"]
        ]
        + [
            "cmp__muts_extended-0.5",
            "mp__muts_extended-0.5",
        ]
        + [
            "mp__LAML",
            # "mp__LAML_GT_prior",
            # "mp__TiDeTree",
        ]
    )
    model_names = [f"{solver_name}__{ble_name}" for ble_name in ble_names]
    plot_model_params = {
        model_name: {"display_name": BLE_DISPLAY_NAMES[ble_name]}
        for (ble_name, model_name) in zip(ble_names, model_names)
    }
    metric_display_names = {
        "ble__node_time-mae": METRIC_DISPLAY_NAMES["ble__node_time-mae"],
        "ble__ancestors-0.5-mre": METRIC_DISPLAY_NAMES[
            "ble__ancestors-0.5-mre"
        ],
        "fe__sr2": METRIC_DISPLAY_NAMES["fe__sr2"],
        "ble__runtime": METRIC_DISPLAY_NAMES["ble__runtime"],
    }
    plot_dir = "full_results"
    if with_gt_topology:
        plot_dir += "_with_gt_topology"
    else:
        plot_dir += "_without_gt_topology"
    if use_ranks:
        plot_dir += "_ranks"
    slice_benchmark_paper_ble(
        model_names=model_names,
        plot_model_params=plot_model_params,
        metric_names=list(metric_display_names.keys()),
        metric_display_names=metric_display_names,
        repetitions=range(0, 50),
        outdir=os.path.join(PAPER_BLE_FIGURES_DIR, plot_dir),
        use_ranks=use_ranks,
        num_processes=NUM_PROCESSES,
        cache_dir=CACHE_PAPER_BLE,
    )


def fig_gt_trees(num_trees=9):
    """
    Supplementary figure showing ground truth trees.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE)
    caching.set_dir_levels(3)
    caching.set_log_level(9)
    for repetition in range(num_trees):
        print(f"***** {repetition} *****")
        regime_args = get_default_regime_args()
        configs = get_ble_configs(
            regime_args=regime_args,
            model_name="gt__c__r__gt__gt",
            repetition=repetition,
        )
        gt_tree_dir = run_ble_unrolled(**configs)["output_tree_dir"]
        tree_true = read_tree(os.path.join(gt_tree_dir, "result.txt"))
        outdir = os.path.join(PAPER_BLE_FIGURES_DIR, "gt_trees")
        os.makedirs(outdir, exist_ok=True)
        cas.pl.plot_matplotlib(
            tree_true,
            add_root=False,
            depth_key="time",
            extend_branches=False
        )
        plt.savefig(f"{outdir}/{repetition}.png")
        plt.close()


def fig_gt_trees_40(num_trees=9):
    """
    Supplementary figure showing ground truth trees.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE)
    caching.set_dir_levels(3)
    caching.set_log_level(9)
    for repetition in range(num_trees):
        print(f"***** {repetition} *****")
        regime_args = get_default_regime_args(
            n_cells=40,
            fitness="neutral",
        )
        configs = get_ble_configs(
            regime_args=regime_args,
            model_name="gt__c__r__gt__gt",
            repetition=repetition,
        )
        gt_tree_dir = run_ble_unrolled(**configs)["output_tree_dir"]
        tree_true = read_tree(os.path.join(gt_tree_dir, "result.txt"))
        outdir = os.path.join(PAPER_BLE_FIGURES_DIR, "gt_trees/40")
        os.makedirs(outdir, exist_ok=True)
        cas.pl.plot_matplotlib(
            tree_true,
            add_root=False,
            depth_key="time",
            extend_branches=False
        )
        plt.savefig(f"{outdir}/{repetition}.png")
        plt.close()


def fig_gt_trees_2000(num_trees=9):
    """
    Supplementary figure showing ground truth trees.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE)
    caching.set_dir_levels(3)
    caching.set_log_level(9)
    for repetition in range(num_trees):
        print(f"***** {repetition} *****")
        regime_args = get_default_regime_args(
            n_cells=2000,
            sampling_probability=0.1,
        )
        configs = get_ble_configs(
            regime_args=regime_args,
            model_name="gt__c__r__gt__gt",
            repetition=repetition,
        )
        gt_tree_dir = run_ble_unrolled(**configs)["output_tree_dir"]
        tree_true = read_tree(os.path.join(gt_tree_dir, "result.txt"))
        outdir = os.path.join(PAPER_BLE_FIGURES_DIR, "gt_trees/2000")
        os.makedirs(outdir, exist_ok=True)
        cas.pl.plot_matplotlib(
            tree_true,
            add_root=False,
            depth_key="time",
            extend_branches=False
        )
        plt.savefig(f"{outdir}/{repetition}.png")
        plt.close()


def fig_sampling_effect_on_branch_lengths():
    """
    Sampling cells makes branches at the leaves longer.

    This was done manually in IToL.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE)
    caching.set_dir_levels(3)
    caching.set_log_level(9)

    raise NotImplementedError


def fig_mutation_rates_splined_from_real_data():
    """
    Supplementary figures showing our site rate variation in simulations.
    """
    rates = reverse_engineer_non_iid_mutation_rates(
        number_of_cassettes=13,
        size_of_cassette=3,
        expected_proportion_mutated=50 / 100,
    )

    plt.bar(range(1, len(rates) + 1), rates)
    plt.ylabel("Mutation Rate")
    plt.xlabel("Site")
    plt.title("Mutation Rates Splined from Real Data")

    os.makedirs(PAPER_BLE_FIGURES_DIR, exist_ok=True)
    plt.grid()
    plt.savefig(
        f"{PAPER_BLE_FIGURES_DIR}/fig_mutation_rates_splined_from_real_data.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def fig_branch_length_distribution(num_trees: int = 50):
    """
    Supplementary figure showing ground truth tree branch length distribution.

    This justifies the use of a 0.01 minimum branch length hyperparameter.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE)
    caching.set_dir_levels(3)
    caching.set_log_level(9)
    branch_lengths_all = []
    for repetition in range(num_trees):
        regime_args = get_default_regime_args()
        configs = get_ble_configs(
            regime_args=regime_args,
            model_name="gt__c__r__gt__gt",
            repetition=repetition,
        )
        gt_tree_dir = run_ble_unrolled(**configs)["output_tree_dir"]
        tree_true = read_tree(os.path.join(gt_tree_dir, "result.txt"))
        branch_lengths_tree = [tree_true.get_branch_length(parent, child) for (parent, child) in tree_true.edges]
        branch_lengths_all += branch_lengths_tree

    plt.hist(branch_lengths_all, bins=200)
    plt.title("Branch Length Distribution")
    plt.ylabel("Count")
    plt.xlabel("Branch Length")
    outdir = os.path.join(PAPER_BLE_FIGURES_DIR, "branch_length_distribution")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/hist.png", bbox_inches="tight", dpi=300)
    plt.close()

    branch_lengths_small = [x for x in branch_lengths_all if x <= 0.03]
    plt.hist(branch_lengths_small, bins=30)
    plt.title("Branch Length Distribution Zoom-In")
    plt.ylabel("Count")
    plt.xlabel("Branch Length")
    outdir = os.path.join(PAPER_BLE_FIGURES_DIR, "branch_length_distribution")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/hist_small.png", bbox_inches="tight", dpi=300)
    plt.close()


def fig_q_distribution():
    """
    Supplementary figure showing the skewed distribution of
    indel probabilities using in our simulations.
    """
    number_of_states = 100
    state_priors = scipy.stats.expon.ppf(
        (np.array(range(number_of_states)) + 0.5) / number_of_states,
        loc=0,
        scale=1e-5,
    )
    state_priors /= sum(state_priors)
    state_priors = {
        number_of_states - i: state_priors[i] for i in range(number_of_states)
    }

    qs = [state_priors[i + 1] for i in range(number_of_states)]
    plt.bar(range(1, len(qs) + 1), qs)
    plt.ylabel("Probability")
    plt.xlabel("Indel State")
    plt.title("Indel State Probability Distribution")

    os.makedirs(PAPER_BLE_FIGURES_DIR, exist_ok=True)
    plt.grid()
    plt.savefig(
        f"{PAPER_BLE_FIGURES_DIR}/fig_q_distribution.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def fig_tree_simulator_parameters():
    """
    Show the concrete values used for simulating the GT trees
    (which were reverse-engineered)
    """
    expected_number_of_increased_fitness_changes = 256
    n_cells = 400
    sampling_probability = 0.01
    deleterious_prob = 0.9

    mutation_prob = (
        expected_number_of_increased_fitness_changes
        / n_cells
        * sampling_probability
        / (1.0 - deleterious_prob)
    )

    print(f"mutation_prob = {mutation_prob}")

    print("Fitness decrease:", 2.0**-0.1)
    print("Fitness increase:", 2.0**1.1)

    offset = 0.01
    bd_ratio = 10.0
    birth_rate, death_rate = reverse_engineer_birth_and_death_rate(
        expected_population_size=n_cells,
        offset=offset,
        birth_to_death_rate_ratio=bd_ratio,
        sampling_probability=sampling_probability,
    )
    print(f"birth_rate, death_rate = {birth_rate, death_rate}")


def fig_choice_of_maxcut(use_ranks, c__r__scheme: str = "__C__R"):
    """
    Supplementary figure showing the MaxCut solver is among the best.
    """
    caching.set_cache_dir(CACHE_PAPER_BLE)
    caching.set_dir_levels(3)
    caching.set_log_level(9)

    solver_names = [x + "__C__R" for x in list(SOLVER_DISPLAY_NAMES.keys()) if "ilp" not in x and x != "gt"]
    ble_name = "mp__muts_extended-0.5"

    model_names = [f"{solver_name}__{ble_name}" for solver_name in solver_names]
    plot_model_params = {
        model_name: {"display_name": SOLVER_DISPLAY_NAMES[solver_name.split('__')[0]]}
        for (solver_name, model_name) in zip(solver_names, model_names)
    }
    metric_display_names = {
        "solver__triplets-500": METRIC_DISPLAY_NAMES["solver__triplets-500"],
        "solver__rf": METRIC_DISPLAY_NAMES["solver__rf"],
    }
    plot_dir = "choice_of_maxcut"
    if use_ranks:
        plot_dir += "_ranks"
    slice_benchmark_paper_ble(
        model_names=model_names,
        plot_model_params=plot_model_params,
        metric_names=list(metric_display_names.keys()),
        metric_display_names=metric_display_names,
        repetitions=range(0, 50),
        outdir=os.path.join(PAPER_BLE_FIGURES_DIR, plot_dir),
        use_ranks=use_ranks,
        num_processes=NUM_PROCESSES,
        cache_dir=CACHE_PAPER_BLE,
    )


TIDETREE_FAILED_TREES = [7, 19, 31, 44, 60, 66, 74, 87, 101, 105]


def fig_intMEMOIR():
    caching.set_cache_dir("_cache")
    outdir = os.path.join(PAPER_BLE_FIGURES_DIR, "intMEMOIR")
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    trees = {}
    metrics = {}
    num_trees = 106
    failed_trees = TIDETREE_FAILED_TREES[:]  # Manually exclude TiDeTree failed trees
    for numtree in [x for x in range(1, num_trees + 1) if x not in TIDETREE_FAILED_TREES[:]]:  # Manually exclude TiDeTree failed trees
        # print(f"***** numtree = {numtree} *****")
        for ble_name in ["GT", "ConvexML", "LAML", "TiDeTree"]:
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

            try:
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
            except TiDeTreeError:
                failed_trees.append(numtree)

    print(f"failed_trees = {failed_trees}")

    plt.scatter(
        [metrics[f"{numtree}__ConvexML"] for numtree in range(1, num_trees + 1) if numtree not in failed_trees],
        [metrics[f"{numtree}__TiDeTree"] for numtree in range(1, num_trees + 1) if numtree not in failed_trees],
        alpha=0.3,
    )
    plt.plot([0, 1], [0, 1], color="r")
    plt.xlabel("ConvexML MAE")
    plt.ylabel("TiDeTree MAE")
    beats_on = 100.0 * np.mean([metrics[f"{numtree}__ConvexML"] < metrics[f"{numtree}__TiDeTree"] for numtree in range(1, num_trees + 1) if numtree not in failed_trees])
    plt.title(f"intMEMOIR benchmark\nConvexML vs TiDeTree\nConvexML better on {beats_on:.0f}% of trees")
    plt.savefig(f"{outdir}/ConvexML_vs_TiDeTree.png", bbox_inches="tight", dpi=300)
    plt.close()

    plt.scatter(
        [metrics[f"{numtree}__ConvexML"] for numtree in range(1, num_trees + 1)  if numtree not in failed_trees],
        [metrics[f"{numtree}__LAML"] for numtree in range(1, num_trees + 1)  if numtree not in failed_trees],
        alpha=0.3,
    )
    plt.plot([0, 1], [0, 1], color="r")
    plt.xlabel("ConvexML MAE")
    plt.ylabel("LAML MAE")
    beats_on = 100.0 * np.mean([metrics[f"{numtree}__ConvexML"] < metrics[f"{numtree}__LAML"] for numtree in range(1, num_trees + 1) if numtree not in failed_trees])
    plt.title(f"intMEMOIR benchmark\nConvexML vs LAML\nConvexML better on {beats_on:.0f}% of trees")
    plt.savefig(f"{outdir}/ConvexML_vs_LAML.png", bbox_inches="tight", dpi=300)
    plt.close()

    numtree = 1

    cas.pl.plot_matplotlib(
        trees[f"{numtree}__GT"],
        add_root=False,
        depth_key="time",
        extend_branches=False
    )
    plt.savefig(f"{outdir}/tree_{numtree}_GT.png", bbox_inches="tight", dpi=300)
    plt.close()

    cas.pl.plot_matplotlib(
        trees[f"{numtree}__ConvexML"],
        add_root=False,
        depth_key="time",
        extend_branches=False
    )
    plt.savefig(f"{outdir}/tree_{numtree}_ConvexML.png", bbox_inches="tight", dpi=300)
    plt.close()

    cas.pl.plot_matplotlib(
        trees[f"{numtree}__TiDeTree"],
        add_root=False,
        depth_key="time",
        extend_branches=False
    )
    plt.savefig(f"{outdir}/tree_{numtree}_TiDeTree.png", bbox_inches="tight", dpi=300)
    plt.close()

    cas.pl.plot_matplotlib(
        trees[f"{numtree}__LAML"],
        add_root=False,
        depth_key="time",
        extend_branches=False
    )
    plt.savefig(f"{outdir}/tree_{numtree}_LAML.png", bbox_inches="tight", dpi=300)
    plt.close()


def fig_intMEMOIR_errors_by_depth():
    caching.set_cache_dir("_cache")
    outdir = os.path.join(PAPER_BLE_FIGURES_DIR, "intMEMOIR")
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    trees = {}
    metrics = {}
    num_trees = 106
    failed_trees = TIDETREE_FAILED_TREES[:]  # Manually exclude TiDeTree failed trees
    for numtree in [x for x in range(1, num_trees + 1) if x not in TIDETREE_FAILED_TREES[:]]:  # Manually exclude TiDeTree failed trees
        for ble_name in ["GT", "ConvexML", "LAML", "TiDeTree"]:
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
                    "args": {"priors": {1:0.5, 2:0.5}, "nInitials": 5},
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

            try:
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
            except TiDeTreeError:
                failed_trees.append(numtree)

    print(f"failed_trees = {failed_trees}")

    gt_times = []
    ble_times = []
    errors = []
    abs_errors = []
    for i in range(1, num_trees + 1):
        if f"{i}__GT" not in trees:
            continue
        tree_gt = trees[f"{i}__GT"]
        tree_ble = trees[f"{i}__ConvexML"]
        for node in tree_gt.nodes:
            gt_time = tree_gt.get_time(node)
            leaf_and_root_set = set(tree_gt.leaves + [tree_gt.root])
            if node not in leaf_and_root_set:
                # Is internal node
                ble_time = tree_ble.get_time(node)
                gt_times.append(gt_time)
                ble_times.append(ble_time)
                errors.append(ble_time - gt_time)
                abs_errors.append(abs(ble_time - gt_time))
    print(f"Done!")
    plt.figure(figsize=(6,4))
    plt.scatter(gt_times, abs_errors, alpha=0.3)
    plt.xlabel("True time")
    plt.ylabel("Absolute error")
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    pairs = list(zip(gt_times, abs_errors))
    pairs = sorted(pairs)
    gt_times_soted = sorted(gt_times)
    moving_average = []
    s = 0
    denom = 0
    window_size = 30
    for i in range(len(pairs)):
        gt_err = pairs[i]
        err = gt_err[1]
        s += err
        denom += 1
        if denom > window_size:
            s -= pairs[i - window_size][1]
            denom -= 1
        moving_average.append(s / denom)

    plt.plot(gt_times_soted, moving_average, color="r", lw=4, label="moving average")
    plt.legend()
    plt.title("Internal node time error as a function of node depth")
    plt.savefig(f"{outdir}/depth_vs_error.png", bbox_inches="tight", dpi=300)
    plt.close()


def repro_all_figures():
    """
    Note: running this from a jupyter notebook produces figures with
    slightly different size. I ran this on a notebook.

    Note: Excluding figure benchmarking solvers because ILP solver too slow.
    """
    caching.set_read_only(False)
    fig_intMEMOIR()
    fig_intMEMOIR_errors_by_depth()

    # For less verbose logging, uncomment the following two lines.
    # logger = logging.getLogger("casbench.slice_benchmark")
    # logger.setLevel(logging.ERROR)

    fig_full_results(with_gt_topology=True, use_ranks=False)
    fig_full_results(with_gt_topology=False, use_ranks=False)
    fig_branch_length_distribution()
    fig_gt_trees()
    fig_gt_trees_40()
    fig_gt_trees_2000()
    fig_mutation_rates_splined_from_real_data()
    fig_q_distribution()
    fig_tree_simulator_parameters()
    fig_brief_results()

    # Turn off double resections
    fig_model_misspecification_errors_only_states(
        number_of_cassettes=33334,
        size_of_cassette=3,
        expected_proportion_stochastic=30,
        expected_proportion_heritable=30,
        simulate_double_resections=False,
        handle_double_resections=False,
        plot_trees=False,
        plot_dir="fig_model_misspecification_errors_only_states_NO_DR_100002",
    )
    # Turn on double-resections
    fig_model_misspecification_errors_only_states(
        number_of_cassettes=33334,
        size_of_cassette=3,  # Leads to ~1% double-resection missing data, but note that the number of double resections, and thus of double resection alleles, is much higher due to double-resections at neighboring sites
        expected_proportion_stochastic=30,
        expected_proportion_heritable=30,
        simulate_double_resections=True,
        handle_double_resections=True,
        plot_trees=False,
        plot_dir="fig_model_misspecification_errors_only_states_W_DR_100002",
    )
    # Increase cassette size to 10.
    fig_model_misspecification_errors_only_states(
        number_of_cassettes=10000,
        size_of_cassette=10,  # Leads to ~10% double-resection missing data.
        expected_proportion_stochastic=30,
        expected_proportion_heritable=30,
        simulate_double_resections=True,
        handle_double_resections=True,
        plot_trees=False,
        plot_dir="fig_model_misspecification_errors_only_states_W_DR_100000",
    )
    # Don't handle DRs, cassette size 3
    fig_model_misspecification_errors_only_states(
        number_of_cassettes=33334,
        size_of_cassette=3,  # Leads to ~1% double-resection missing data, but note that the number of double resections, and thus of double resection alleles, is much higher due to double-resections at neighboring sites
        expected_proportion_stochastic=30,
        expected_proportion_heritable=30,
        simulate_double_resections=True,
        handle_double_resections=False,
        plot_trees=False,
        plot_dir="fig_model_misspecification_errors_only_states_W_DR_100002_DR_unhandled",
    )
    # Don't handle DRs, cassette size 10
    fig_model_misspecification_errors_only_states(
        number_of_cassettes=10000,
        size_of_cassette=10,  # Leads to ~10% double-resection missing data.
        expected_proportion_stochastic=30,
        expected_proportion_heritable=30,
        simulate_double_resections=True,
        handle_double_resections=False,
        plot_trees=False,
        plot_dir="fig_model_misspecification_errors_only_states_W_DR_100000_DR_unhandled",
    )

    # Now repeat for the reviewer who asked about what happens when there is only stochastic missing data.
    # Total time: 4084m1.279s (simulations are super slow)
    fig_model_misspecification_errors_only_states(
        number_of_cassettes=33334,
        size_of_cassette=3,
        expected_proportion_stochastic=60,
        expected_proportion_heritable=0,
        simulate_double_resections=False,
        handle_double_resections=False,
        plot_trees=False,
        plot_dir="fig_model_misspecification_errors_only_states_NO_DR_100002_60",
    )


if __name__ == "__main__":
    repro_all_figures()
