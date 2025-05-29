import argparse
import os
import socket
import time
from typing import Any, Dict, List, Optional

from casbench import caching
from casbench.config import Config, create_config_from_dict
from casbench.io import read_float
from casbench.simulated_data_benchmarking import (
    run_ancestral_lineages_metric_unrolled, run_ble_runtime_metric_unrolled,
    run_fitness_metric_unrolled, run_internal_node_time_metric_unrolled,
    run_solver_metric_unrolled)
from casbench.slice_benchmark import slice_benchmark

from .global_variables import (ASR_CONFIG_DICTS, BLE_CONFIG_DICTS,
                               CACHE_PAPER_BLE, FE_CONFIG_DICTS,
                               KNOB_DISPLAY_NAMES, METRIC_DISPLAY_NAMES,
                               SOLVER_CONFIG_DICTS, NUMBER_OF_CASSETTES)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model to run",
    )
    parser.add_argument(
        "--min_seed",
        type=int,
        default=0,
        help="Minimum seed",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        default=10,
        help="Maximum seed",
    )
    args = parser.parse_args()
    return args


def get_default_regime_args(
    # Tree simulator args (context values)
    n_cells: int = 400,
    fitness: str = "high",
    sampling_probability: float = 0.01,
    offset: float = 0.01,
    bd_ratio: float = 10.0,
    iid_mutation_rates: bool = False,
    double_resections: bool = True,
    # Lineage tracing args (knobs)
    number_of_cassettes: int = 13,
    size_of_cassette: int = 3,
    number_of_states: int = 100,
    expected_prop_missing: int = 20,
    expected_proportion_mutated: int = 50,
) -> Dict[str, Any]:
    """
    Arguments for the default lineage tracing regime
    """
    res = {
        # Tree simulator args (context values)
        "n_cells": n_cells,
        "fitness": fitness,
        "sampling_probability": sampling_probability,
        "offset": offset,
        "bd_ratio": bd_ratio,
        "iid_mutation_rates": iid_mutation_rates,
        "double_resections": double_resections,
        # Lineage tracing args (knobs)
        "number_of_cassettes": number_of_cassettes,
        "size_of_cassette": size_of_cassette,
        "number_of_states": number_of_states,
        "expected_prop_missing": expected_prop_missing,
        "expected_proportion_mutated": expected_proportion_mutated,
    }
    return res


def get_tree_simulator_configs(
    regime_args: Dict[str, Any], repetition: int
) -> Dict[str, Config]:
    res = {
        "tree_simulator_config": create_config_from_dict(
            {
                "identifier": "ble_paper_tree_simulator",
                "args": {
                    "n_cells": regime_args["n_cells"],
                    "fitness": regime_args["fitness"],
                    "sampling_probability": regime_args["sampling_probability"],
                    "offset": regime_args["offset"],
                    "bd_ratio": regime_args["bd_ratio"],
                    "random_seed": repetition,
                },
            },
        ),
        "leaf_subsampler_config": create_config_from_dict(
            {
                "identifier": "UniformLeafSubsampler",
                "args": {
                    "number_of_leaves": regime_args["n_cells"],
                    "random_seed": repetition,
                    "collapse_unifurcations": False,
                },
            }
        ),
        "tree_scaler_config": create_config_from_dict(
            {"identifier": "unit_tree_scaler", "args": {}}
        ),
    }
    return res


def get_ble_configs(
    regime_args: Dict[str, Any], model_name: str, repetition: int
) -> Dict[str, Config]:
    """
    Configs associated to a regime up to and including BLE.

    Given all the high-level arguments characterizing a lineage tracing
    regime up to and including branch length estimation, returns the
    associated configs instantiating all the concrete classes needed to realize
    said higher-level description.
    """
    expected_args = [  # Note: need to be sorted!
        "bd_ratio",
        "double_resections",
        "expected_prop_missing",
        "expected_proportion_mutated",
        "fitness",
        "iid_mutation_rates",
        "n_cells",
        "number_of_cassettes",
        "number_of_states",
        "offset",
        "sampling_probability",
        "size_of_cassette",
    ]
    if sorted(list(regime_args.keys())) != expected_args:
        raise ValueError(f"Keys of regime_args should be: {expected_args}")
    (
        solver_name,
        collapse_mutationless_edges,
        resolve_multifurcations,
        mp_strategy,
        ble_name,
    ) = model_name.split("__")
    if collapse_mutationless_edges not in ["c", "C"]:
        raise ValueError(
            "Unknown collapse_mutationless_edges = "
            f"{collapse_mutationless_edges}"
        )
    if resolve_multifurcations not in ["r", "R"]:
        raise ValueError(
            f"Unknown resolve_multifurcations = {resolve_multifurcations}"
        )
    if mp_strategy not in ["gt", "mp", "cmp"]:
        raise ValueError(f"Unknown mp_strategy = {mp_strategy}")
    configs = {
        **get_tree_simulator_configs(
            regime_args=regime_args,
            repetition=repetition,
        ),
        "lt_simulator_config": create_config_from_dict(
            {
                "identifier": "ReverseEngineeredCas9LineageTracingDataSimulator",  # noqa
                "args": {
                    "number_of_cassettes": regime_args["number_of_cassettes"],
                    "size_of_cassette": regime_args["size_of_cassette"],
                    "number_of_states": regime_args["number_of_states"],
                    "expected_proportion_heritable": max(
                        0, regime_args["expected_prop_missing"] - 10
                    ),
                    "expected_proportion_stochastic": min(
                        10, regime_args["expected_prop_missing"]
                    ),
                    "expected_proportion_mutated": regime_args[
                        "expected_proportion_mutated"
                    ],
                    "random_seed": repetition,
                    "iid_mutation_rates": regime_args["iid_mutation_rates"],
                    "collapse_sites_on_cassette": regime_args["double_resections"],
                    "create_allele_when_collapsing_sites_on_cassette": True,
                },
            }
        ),
        "missing_data_mechanism_config": create_config_from_dict(
            {
                "identifier": "handle_double_resections",
                "args": {
                    "size_of_cassette": regime_args["size_of_cassette"],
                    "missing_data_indicator": -3,
                    "also_internal_nodes": True,
                }
            }
        ),
        "missing_data_imputer_config": create_config_from_dict(
            {"identifier": "none", "args": {}}
        ),
        "solver_config": create_config_from_dict(
            SOLVER_CONFIG_DICTS[solver_name]
        ),
        "mutationless_edges_strategy_config": create_config_from_dict(
            {"identifier": "collapse", "args": {}}
        )
        if collapse_mutationless_edges == "C"
        else create_config_from_dict({"identifier": "none", "args": {}}),
        "multifurcation_resolver_config": create_config_from_dict(
            {"identifier": "resolve", "args": {}}
        )
        if resolve_multifurcations == "R"
        else create_config_from_dict({"identifier": "none", "args": {}}),
        "ancestral_states_reconstructor_config": create_config_from_dict(
            ASR_CONFIG_DICTS[mp_strategy]
        ),
        "ble_config": create_config_from_dict(BLE_CONFIG_DICTS[ble_name]),
        "ble_tree_scaler_config": create_config_from_dict(
            {"identifier": "unit_tree_scaler", "args": {}}
        ),
    }
    return configs


def eval_function_solver(
    regime_args: Dict[str, Any],
    model_name: str,
    metric_name: str,
    repetition: int,
) -> float:
    """
    Evaluate a solver on a lineage tracing regime.
    """
    configs = get_ble_configs(regime_args, model_name, repetition)
    # Get rid of BLE configs since it is not needed for solver evaluation.
    configs.pop("mutationless_edges_strategy_config")
    configs.pop("multifurcation_resolver_config")
    configs.pop("ancestral_states_reconstructor_config")
    configs.pop("ble_config")
    configs.pop("ble_tree_scaler_config")

    configs[
        "ancestral_states_reconstructor_gt_config"
    ] = create_config_from_dict({"identifier": "ground_truth_asr", "args": {}})
    configs["mutationless_edges_strategy_gt_config"] = create_config_from_dict(
        {"identifier": "none", "args": {}}
    )
    configs["multifurcation_resolver_gt_config"] = create_config_from_dict(
        {"identifier": "none", "args": {}}
    )
    configs[
        "ancestral_states_reconstructor_recon_config"
    ] = create_config_from_dict({"identifier": "maximum_parsimony", "args": {}})
    configs[
        "mutationless_edges_strategy_recon_config"
    ] = create_config_from_dict({"identifier": "none", "args": {}})
    configs["multifurcation_resolver_recon_config"] = create_config_from_dict(
        {"identifier": "none", "args": {}}
    )

    if metric_name.startswith("triplets-"):
        _, number_of_trials = metric_name.split("-")
        configs["metric_config"] = create_config_from_dict(
            {
                "identifier": "triplets",
                "args": {"number_of_trials": int(number_of_trials)},
            }
        )
        output_metric_dir = run_solver_metric_unrolled(**configs)[
            "output_metric_dir"
        ]
    elif metric_name == "rf":
        configs["metric_config"] = create_config_from_dict(
            {"identifier": "rf", "args": {}}
        )
        output_metric_dir = run_solver_metric_unrolled(**configs)[
            "output_metric_dir"
        ]
    else:
        raise ValueError(f"Unknown metric_name = {metric_name}")
    return read_float(os.path.join(output_metric_dir, "result.txt"))


def eval_function_ble(
    regime_args: Dict[str, Any],
    model_name: str,
    metric_name: str,
    repetition: int,
) -> float:
    """
    Evaluate a branch length estimator on a lineage tracing regime.
    """
    configs = get_ble_configs(regime_args, model_name, repetition)
    if metric_name.startswith("ancestors-"):
        _, t, metric_name = metric_name.split("-")
        configs["ancestral_lineages_config"] = create_config_from_dict(
            {"identifier": "ancestral_time", "args": {"t": float(t)}}
        )
        configs["metric_config"] = create_config_from_dict(
            {"identifier": metric_name, "args": {}}
        )
        output_metric_dir = run_ancestral_lineages_metric_unrolled(**configs)[
            "output_metric_dir"
        ]
    elif metric_name.startswith("node_time-"):
        _, metric_name = metric_name.split("-")
        configs[
            "internal_node_time_predictor_config"
        ] = create_config_from_dict(
            {"identifier": "mrca_impute", "args": {"aggregation": "mean"}}
        )
        configs["metric_config"] = create_config_from_dict(
            {"identifier": metric_name, "args": {}}
        )
        output_metric_dir = run_internal_node_time_metric_unrolled(**configs)[
            "output_metric_dir"
        ]
    elif metric_name == "runtime":
        output_metric_dir = run_ble_runtime_metric_unrolled(**configs)[
            "output_metric_dir"
        ]
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")
    return read_float(os.path.join(output_metric_dir, "result.txt"))


def eval_function_fe(
    regime_args: Dict[str, Any],
    model_name: str,
    metric_name: str,
    repetition: int,
) -> float:
    """
    Evaluate fitness estimation on a lineage tracing regime.
    """
    configs = get_ble_configs(regime_args, model_name, repetition)
    configs["fitness_estimator_config"] = create_config_from_dict(
        FE_CONFIG_DICTS["lbi"]
    )
    configs["fitness_subsetting_config"] = create_config_from_dict(
        {"identifier": "subset_nodes", "args": {"leaves": True}}
    )
    configs["metric_config"] = create_config_from_dict(
        {"identifier": metric_name, "args": {}}
    )
    output_metric_dir = run_fitness_metric_unrolled(**configs)[
        "output_metric_dir"
    ]
    return read_float(os.path.join(output_metric_dir, "result.txt"))


def eval_function(
    regime_args: Dict[str, Any],
    model_name: str,
    metric_name: str,
    repetition: int,
) -> float:
    """
    Evaluate a metric on a lineage tracing regime.
    """
    task_name = metric_name.split("__")[0]
    metric_name = metric_name[(len(task_name) + 2) :]  # noqa
    if task_name == "solver":
        return eval_function_solver(
            regime_args, model_name, metric_name, repetition
        )
    elif task_name == "ble":
        return eval_function_ble(
            regime_args, model_name, metric_name, repetition
        )
    elif task_name == "fe":
        return eval_function_fe(
            regime_args, model_name, metric_name, repetition
        )
    else:
        raise ValueError(
            f"Unknown task_name: {task_name} when evaluating metric_name "
            f"{metric_name}"
        )


def slice_benchmark_paper_ble(
    model_names: List[str],
    plot_model_params: Optional[Dict[str, Dict[str, str]]] = None,
    metric_names=list(METRIC_DISPLAY_NAMES.keys()),
    metric_display_names=METRIC_DISPLAY_NAMES,
    knob_display_names=KNOB_DISPLAY_NAMES,
    eval_function=eval_function,
    context_names=[
        "n_cells",
        "fitness",
        "sampling_probability",
        "offset",
        "bd_ratio",
        "iid_mutation_rates",
        "double_resections",
    ],
    context_values=[
        {  # High fitness regime, w/double-resections
            "n_cells": 400,
            "fitness": "high",
            "sampling_probability": 0.01,
            "offset": 0.01,
            "bd_ratio": 10.0,
            "iid_mutation_rates": False,
            "double_resections": True,
        },
        {  # Now 40 cells (but no fitness)
            "n_cells": 40,
            "fitness": "neutral",
            "sampling_probability": 0.01,
            "offset": 0.01,
            "bd_ratio": 10.0,
            "iid_mutation_rates": False,
            "double_resections": True,
        },
        {  # Now 2000 cells, but sampling 10% instead of 1% to make it faster.
            "n_cells": 2000,
            "fitness": "high",
            "sampling_probability": 0.1,
            "offset": 0.01,
            "bd_ratio": 10.0,
            "iid_mutation_rates": False,
            "double_resections": True,
        },
    ],
    knobs=[
        "number_of_cassettes",
        "size_of_cassette",
        "expected_proportion_mutated",
        "number_of_states",
        "expected_prop_missing",
    ],
    defaults={
        "number_of_cassettes": NUMBER_OF_CASSETTES[::-1][3],
        "size_of_cassette": 3,
        "expected_proportion_mutated": 50,
        "number_of_states": 100,
        "expected_prop_missing": 20,
    },
    ranges={
        "number_of_cassettes": NUMBER_OF_CASSETTES,
        "size_of_cassette": [],
        "expected_proportion_mutated": [10, 30, 50, 70, 90],
        "number_of_states": [5, 10, 25, 50, 100, 500, 1000],
        "expected_prop_missing": [0, 10, 20, 30, 40, 50, 60],
    },
    repetitions=range(50),
    plot: bool = True,
    log: bool = True,
    use_tqdm: bool = True,
    aggregation: str = "mean",
    use_ranks: bool = False,
    plot_type: str = "heatmap",
    fmt: Optional[str] = None,
    figsize=None,
    ylim=None,
    outdir=None,
    just_return_regimes_and_do_not_run: bool = False,
    num_processes: int = 1,
    cache_dir: Optional[str] = None,
):
    """
    Wrapper around slice_benchmark function adding all the boilerplate code for
    the BLE paper benchmarking.
    """
    return slice_benchmark(
        model_names=model_names,
        plot_model_params=plot_model_params,
        metric_names=metric_names,
        metric_display_names=metric_display_names,
        eval_function=eval_function,
        context_names=context_names,
        context_values=context_values,
        knobs=knobs,
        knob_display_names=knob_display_names,
        defaults=defaults,
        ranges=ranges,
        repetitions=repetitions,
        plot=plot,
        log=log,
        use_tqdm=use_tqdm,
        aggregation=aggregation,
        use_ranks=use_ranks,
        plot_type=plot_type,
        fmt=fmt,
        figsize=figsize,
        ylim=ylim,
        outdir=outdir,
        just_return_regimes_and_do_not_run=just_return_regimes_and_do_not_run,
        num_processes=num_processes,
        cache_dir=cache_dir,
    )
