"""
slice_benchmark module for seamless model benchmarking.

This module contains the slice_benchmark function for seamless model
benchmarking.
"""
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from casbench import caching


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


import multiprocessing


def _worker(
    regime_args_list,
    model_name,
    metric_name,
    repetition,
    eval_function,
    cache_dir,
):
    if cache_dir is not None:
        caching.set_cache_dir(cache_dir)
        caching.set_log_level(9)
    print(f"***** Running worker on repetition {repetition}... *****")
    return eval_function(
        regime_args=dict(regime_args_list),
        model_name=model_name,
        metric_name=metric_name,
        repetition=repetition,
    )


from casbench import caching
# caching.set_read_only(True)
@caching.cached(
    exclude=["eval_function", "eval_function_module", "cache_dir", "num_processes"],
)
def eval_function_on_all_repetitions(
    regime_args_list,
    model_name,
    metric_name,
    repetitions,
    eval_function,
    eval_function_name,
    eval_function_module,
    cache_dir,
    num_processes,
):
    if num_processes == 1:
        metric_values = []
        for repetition in repetitions:
            print(f"***** Running repetition {repetition}... *****")
            metric_values.append(
                eval_function(
                    regime_args=dict(regime_args_list),
                    model_name=model_name,
                    metric_name=metric_name,
                    repetition=repetition,
                )
            )
    else:
        # Use multiprocessing Pool to parallelize the execution
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Pass the function and arguments to the pool's map function
            metric_values = pool.starmap(
                _worker, [(regime_args_list, model_name, metric_name, repetition, eval_function, cache_dir) for repetition in repetitions]
            )

    return metric_values



def slice_benchmark(
    model_names: List[str],
    metric_names: List[str],
    eval_function: Callable,
    context_names: List[str],
    context_values: List[Dict[str, Any]],
    knobs: List[str],
    defaults: Dict[str, Any],
    ranges: Dict[str, List[Any]],
    repetitions: List[int],
    plot: bool = True,
    plot_model_params: Optional[Dict[str, Dict[str, str]]] = None,
    metric_display_names: Optional[Dict[str, str]] = None,
    knob_display_names: Optional[Dict[str, str]] = None,
    metric_lower_is_better: Optional[Dict[str, bool]] = None,
    log: bool = False,
    use_tqdm: bool = True,
    aggregation: str = "mean",
    use_ranks: bool = False,
    plot_type: str = "heatmap",
    cmap: str = "YlGnBu",
    fmt: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ylim=None,
    outdir=None,
    cache_dir: Optional[str] = None,
    num_processes: int = 1,
    show_plot: bool = False,
    just_return_regimes_and_do_not_run: bool = False,
):
    """
    Run a "slice benchmark" and plot results.

    A "slice benchmark" consists of a "default" regime of parameters, as
    specified by the `defaults` dictionary. For example, for a Cas9 lineage
    tracing experiment, the defaults might be:

        defaults={
            "number_of_cassettes": 13,
            "size_of_cassette": 3,
            "expected_proportion_mutated": 50,
            "number_of_states": 100,
            "expected_prop_missing": 20,
        }

    The keys of this dictionary are called the `knobs`. Each of these parameters
    will be varied in turn to explore the impact on performance. To specify the
    values of these knobs that we want to explore, we use the `ranges` dict. For
    example, if we want to explore how model performance varies when the
    number_of_cassettes varies from 3 to 50; and also how model performance
    varies when the expected_prop_missing varies from 0 to 60; etc., we may
    specify:

        ranges={
            "number_of_cassettes": [3, 6, 13, 20, 30, 50],
            "size_of_cassette": [],
            "expected_proportion_mutated": [10, 30, 50, 70, 90],
            "number_of_states": [5, 10, 25, 50, 100, 500, 1000],
            "expected_prop_missing": [0, 10, 20, 30, 40, 50, 60],
        },

    If we specify an empty list such as "size_of_cassette": [], this means that
    the knob will be skipped and the plot for this know will not be generated
    (this is an easy way of skipping parts of the experiment which are no longer
    considered interesting)

    Note that the results not only depend on the lineage tracing parameters, but
    also on the tree simulator parameters. If we simulate trees under a
    high-fitness regime, we might get very different results than if we simulate
    trees under a neutral regime. The parameters that specify the tree regime
    are called the `context_names` and their values are specified in the
    `context_values` dictionary. For example, if we want to explore a
    high-fitness regime and a low-fitness regime, we would specify:

        context_names=[
            "n_cells",
            "fitness",
            "sampling_probability",
            "offset",
            "bd_ratio",
            "iid_mutation_rates",
        ]
        context_values=[
            {  # High fitness regime
                "n_cells": 400,
                "fitness": "high",
                "sampling_probability": 0.01,
                "offset": 0.01,
                "bd_ratio": 10.0,
                "iid_mutation_rates": False,
            },
            {  # Neutral regime
                "n_cells": 400,
                "fitness": "neutral",
                "sampling_probability": 0.01,
                "offset": 0.0,
                "bd_ratio": 1000.0,
                "iid_mutation_rates": True,
            },
        ]

    We usually want to run repeated simulations with different random seeds to
    get reliable results. This is specified by the `repetitions` list giving
    the random seeds to iterate over. So, if we wanted to use 50 repetitions, we
    would do:

        repetitions=range(50)

    The values of all repetitions are summarized by specifying the
    `aggregation`, which can be "mean" or "median".

    The models benchmarked are provided in the `model_names` list. The list of
    metrics evaluated is specified in the `metric_names` list.

    Finally, the most important part of the slice benchmark is the evaluation
    function `eval_function` which, given all the knob values, parameter values,
    repetition (random seed), model name, metric name, computes the metric for
    the given model in the given setting. This is where all the hard work
    happens, and where all the strings we have specified so far take their
    meaning. This function typically will create suitable Configs and then just
    forward them to one of the evaluation function in the
    simulated_data_benchmarking module, such as: run_solver_metric_unrolled,
    run_internal_node_time_metric_unrolled, run_ble_runtime_metric_unrolled,
    run_ancestral_lineages_metric_unrolled, run_fitness_metric_unrolled.
    These functions do even harder work for you: they simulate the tree,
    run the lineage tracer, run your model, compute your metric, etc.,
    all while caching intermediate computations in a seamless way.

    A cool feature of the slice benchmark is that we can choose to show either
    the metric value of each model, or the *rank* of each model among all the
    models benchmarked. This is achieved with the `use_ranks` boolean indicator.

    By default the plot will be a heatmap, but this can be changes to lines by
    specifying `plot_type="lines"` instead.

    When it comes to plotting, `plot_model_params` is a dictionary which
    indicates what color and linestyle to use for each model (if plotting
    lines), and the model's display name. For example, we might specify:

        plot_model_params = {
            "nj": {
                "display_name": "Neighbor-Joining",
                "color": "r",
                "linestyle": "-",
            },
            "upgma": {
                "display_name": "UPGMA",
                "color": "b",
                "linestyle": "--",
            }
        }

    The reason the model's display_name might be different from its model_name
    is that internally, in the code we might want to choose nice short strings,
    but when displaying the model name in the figure we might want to show a
    longer, more involved strings which will commonly include LaTeX!

    Similarly, the `metric_display_names` dictionary specifies how to display
    metric names (which appear in the y-axis of the plot). For example:

        metric_display_names = {
            "solver__triplets-500": "Triplets Correct",
            "solver__rf": "Robinson Foulds",
            "ble__node_time-mae": "Internal Node Time, Mean Absolute Error",
            "ble__ancestors-0.5-mre": "Number of Ancestral Lineages, Relative Error",
            "ble__runtime": "Runtime (s)",
            "fe__sr2": "Leaf Fitness, Spearman Correlation",
        }

    Similarly, the `knob_display_names` dictionary specifies how to display
    knob names (which appear on the x-axis of the plot). For example:

        knob_display_names = {
            "number_of_cassettes": "Number of Barcodes",
            "size_of_cassette": "Barcode Size",
            "expected_proportion_mutated": "Expected Proportion of Character Matrix Mutated",
            "number_of_states": "Number of Indel States",
            "expected_prop_missing": "Expected Proportion of Character Matrix Missing",
        }

    Because some metrics are better when they are lower and some when they are
    higher, we can specify this with the `metric_lower_is_better` dictionary,
    such as:

        metric_lower_is_better = {
            "solver__triplets-500": False,
            "solver__rf": True,
            "ble__node_time-mae": True,
            "ble__ancestors-0.5-mre": True,
            "ble__runtime": True,
            "fe__sr2": False,
        }

    This will allow the plotting code to decide if it needs to reverse the
    colormap of the heatmap.

    The colormap of the heatmap can be customized with the `cmap` string.

    The way that metric numbers are displayed can be controlled with the `fmt`
    string, but the default shouldwork nicely.

    The figure size can be customized with the `figsize` parameter, but the
    default should work fine.

    If you want to force the y-limits of all plots to the same value, you can
    specify this with `ylim`.

    Sometimes you want to skip saving the plots completely, for example when
    warming up the cache on slurm. In this case, you can specify `plot=False`.

    Independently of `plot`, you might want to skip showing the plots, in which
    case you can specify `show_plot=False`.

    Logging can be turned off by setting `log=False`. Similarly, tqdm can be
    turned off by specifying `use_tqdm=False`.

    The output directory for all images is `outdir`.

    If `cache_dir` is passed in, then the cache dir will be set to this. This is
    needed when using multiprocessing (i.e. when `num_processes > 1`); to use
    multiprocessing, simply provide `num_processes`.
    """  # noqa
    assert sorted(list(defaults.keys())) == sorted(list(knobs))
    assert sorted(list(ranges.keys())) == sorted(list(knobs))

    def get_model_display_name(model_name: str) -> str:
        res = model_name
        if (
            plot_model_params is not None
            and model_name in plot_model_params
            and "display_name" in plot_model_params[model_name]
        ):
            res = plot_model_params[model_name]["display_name"]
        return res

    def get_model_display_color(model_name: str) -> str:
        res = None
        #print(model_name, plot_model_params, plot_model_params[model_name])
        if (
            plot_model_params is not None
            and model_name in plot_model_params
            and "color" in plot_model_params[model_name]
        ):
            res = plot_model_params[model_name]["color"]
        return res

    def get_model_linestyle(model_name: str) -> str:
        res = "-"
        if (
            plot_model_params is not None
            and model_name in plot_model_params
            and "linestyle" in plot_model_params[model_name]
        ):
            res = plot_model_params[model_name]["linestyle"]
        return res

    def get_metric_display_name(metric_name: str) -> str:
        res = metric_name
        if (
            metric_display_names is not None
            and metric_name in metric_display_names
        ):
            res = metric_display_names[metric_name]
        return res

    def get_knob_display_name(knob_name: str) -> str:
        res = knob_name
        if knob_display_names is not None and knob_name in knob_display_names:
            res = knob_display_names[knob_name]
        return res

    def get_metric_lower_is_better(metric_name: str) -> bool:
        res = True
        if (
            metric_lower_is_better is not None
            and metric_name in metric_lower_is_better
        ):
            res = metric_lower_is_better[metric_name]
        return res

    if figsize is None:
        figsize = (12, 8) if plot_type == "lines" else (6, 4)
    if fmt is None:
        fmt = ".2g" if use_ranks else ".2g"
    logger = logging.getLogger(__name__)
    if not log:
        logger.setLevel(logging.ERROR)
    logger.info("slice_benchmark starting ...")

    if just_return_regimes_and_do_not_run:
        regimes_to_return = []

    results = []
    total = (
        len(context_values)
        * sum([len(ranges[knob]) for knob in knobs])
        * len(model_names)
        * len(metric_names)
    )
    with tqdm(
        total=total,
        disable=not use_tqdm,
        miniters=int(total / 100),
        desc="tqdm Progress",
    ) as pbar:
        for context_id, context_value in enumerate(context_values):
            assert sorted(list(context_value.keys())) == sorted(
                list(context_names)
            )
            logger.info(f"***** Exploring context: {context_value} *****")
            for knob in knobs:
                logger.info(f"Exploring knob: {knob}")
                for knob_value in ranges[knob]:
                    logger.info(f"Setting knob to value: {knob} = {knob_value}")
                    knob_values = deepcopy(defaults)
                    knob_values[knob] = knob_value
                    if len(set({**context_value, **knob_values})) != len(
                        context_value
                    ) + len(knob_values):
                        raise ValueError(
                            "Context names and knob names overlap!"
                        )
                    regime_args_list = sorted(
                        ({**context_value, **knob_values}).items()
                    )
                    if just_return_regimes_and_do_not_run:
                        regimes_to_return.append(
                            [
                                context_id,
                                context_value,
                                knob,
                                knob_value,
                                regime_args_list,
                            ]
                        )
                        continue
                    for model_name in model_names:
                        logger.info(f"Running model {model_name}")
                        for metric_name in metric_names:
                            logger.info(f"Evaluating metric {metric_name}")
                            metric_values = eval_function_on_all_repetitions(
                                regime_args_list,
                                model_name,
                                metric_name,
                                repetitions,
                                eval_function,
                                eval_function_name=eval_function.__name__,
                                eval_function_module=eval_function.__module__,
                                cache_dir=cache_dir,
                                num_processes=num_processes,
                            )
                            results += [
                                [context_id]
                                + [
                                    context_value[context_name]
                                    for context_name in context_names
                                ]
                                + [
                                    knob,
                                    knob_value,
                                    model_name,
                                    metric_name,
                                    metric_value,
                                    metric_value is not None,
                                    repetition,
                                ]
                                for (repetition, metric_value) in zip(
                                    repetitions, metric_values
                                )
                            ]
                            pbar.update(1)
    if just_return_regimes_and_do_not_run:
        return regimes_to_return
    res_df = pd.DataFrame(
        results,
        columns=["context_id"]
        + context_names
        + [
            "knob",
            "knob_value",
            "model_name",
            "metric_name",
            "metric_value",
            "reco_succeeded",
            "repetition",
        ],
    )
    assert len(res_df) == total * len(repetitions)
    plots = {}

    for context_id, (context_value, sub_df) in enumerate(
        sorted(res_df.groupby(["context_id"] + context_names))
    ):
        logger.info(f"Plotting context: {context_value}")
        for metric_name in metric_names:
            logger.info(f"***** Plotting metric = {metric_name} *****")
            for knob in knobs:
                if len(ranges[knob]) == 0:
                    continue
                plt.figure(figsize=figsize)
                if ylim:
                    plt.ylim(ylim)

                model_perfs = []
                for knob_value in ranges[knob]:
                    model_metrics = []
                    for model_name in model_names:
                        aux = sub_df[
                            (sub_df["knob"] == knob)
                            & (sub_df["knob_value"] == knob_value)
                            & (sub_df["model_name"] == model_name)
                            & (sub_df["metric_name"] == metric_name)
                        ].copy()
                        aux.sort_values(["repetition"], inplace=True)
                        model_metrics.append(aux["metric_value"])
                    model_metrics = np.array(model_metrics)
                    assert model_metrics.shape == (
                        len(model_names),
                        len(repetitions),
                    )

                    if aggregation == "median":
                        aggregation_func = np.median
                    elif aggregation == "mean":
                        aggregation_func = np.mean
                    else:
                        raise ValueError(f"Unknown aggregation: {aggregation}")

                    if use_ranks:
                        # Some metrics are better when they are higher, others
                        # when they are lower. We want the colorscheme of the
                        # heatmap to be consistent between plots, so we adjust
                        # for this.
                        multiplier = (
                            1.0
                            if get_metric_lower_is_better(metric_name)
                            else -1.0
                        )
                        model_summarized_ranks = (
                            aggregation_func(
                                np.argsort(
                                    np.argsort(
                                        multiplier * model_metrics, axis=0
                                    ),
                                    axis=0,
                                ),
                                axis=1,
                            )
                            + 1
                        )
                        model_perfs.append(model_summarized_ranks)
                    else:
                        model_perfs.append(
                            aggregation_func(model_metrics, axis=1)
                        )

                if len(ranges[knob]) > 1:
                    if plot_type == "heatmap":
                        model_perfs = np.array(model_perfs).T
                        yticklabels = [
                            get_model_display_name(model_name)
                            for model_name in model_names
                        ]

                        vmin = None
                        if metric_name == "median_bias":
                            vmin = -0.1
                        elif metric_name == "bias_binary_symmetric":
                            vmin = -1.0
                        vmax = None
                        if metric_name == "median_bias":
                            vmax = 0.1
                        elif metric_name == "bias_binary_symmetric":
                            vmax = 1.0
                        center = (
                            0
                            if metric_name == "bias_binary_symmetric"
                            or metric_name == "median_bias"
                            else None
                        )

                        sns.heatmap(
                            model_perfs,
                            yticklabels=yticklabels,
                            xticklabels=ranges[knob],
                            cmap=cmap,
                            annot=True,
                            fmt=fmt,
                            vmin=vmin,
                            vmax=vmax,
                            center=center,
                        )
                        plt.yticks(rotation=0)
                        plt.ylabel(get_metric_display_name(metric_name))
                        plt.xlabel(get_knob_display_name(knob))
                    elif plot_type == "lines":
                        model_perfs = np.array(model_perfs).T
                        plt.title(f"context: {context_value}\nVarying: {knob}")
                        plt.xlabel(get_knob_display_name(knob))
                        plt.ylabel(get_metric_display_name(metric_name))
                        for i, model_name in enumerate(model_names):
                            display_name = get_model_display_name(model_name)
                            plt.plot(
                                ranges[knob],
                                model_perfs[i, :],
                                label=display_name,
                                color=get_model_display_color(model_name),
                                linestyle=get_model_linestyle(model_name),
                            )
                        plt.legend()
                    else:
                        raise ValueError(f"Unknown plot type: {plot_type}")
                else:
                    plt.bar(
                        [
                            get_model_display_name(model_name)
                            for model_name in model_names
                        ],
                        np.array(model_perfs).reshape([-1]),
                        color=[
                            get_model_display_color(model_name)
                            for model_name in model_names
                        ],
                    )
                    plt.xticks(rotation=90)
                    plt.legend()
                plots[
                    tuple(list(context_value) + [metric_name, knob])
                ] = plt.gcf()
                if plot:
                    if outdir is not None:
                        os.makedirs(outdir, exist_ok=True)
                        plt.savefig(
                            f"{outdir}/ctx_{context_id}_metric_name_"
                            f"{metric_name}_knob_{knob}.png",
                            bbox_inches="tight",
                            dpi=300,
                        )
                if show_plot:
                    plt.show()
                plt.close()
    return res_df, plots
