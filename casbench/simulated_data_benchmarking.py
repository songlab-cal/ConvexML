import os
import time
from functools import partial
from typing import Optional

import cassiopeia
import numpy as np
from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import IIDExponentialMLE

from casbench import caching
from casbench.config import Config

from .ancestral_states_reconstructor import (get_asr_from_config,
                                             ground_truth_asr)
from .branch_length_estimator import GroundTruthBLE, get_ble_from_config
from .fitness_estimator import get_fitness_estimator_from_config
from .internal_node_time_prediction import get_true_vs_predicted_attribute
from .io import (read_pickle, read_str, read_tree, write_ble, write_float,
                 write_pickle, write_str, write_tree)
from .leaf_subsampler import get_leaf_subsampler_from_config
from .lineage_tracer import get_lt_simulator_from_config
from .lineage_tracer._reverse_engineered_cas9 import \
    reverse_engineer_non_iid_mutation_rates
from .metric import get_metric_by_config
from .missing_data_imputer import get_missing_data_imputer_from_config
from .missing_data_mechanism import get_missing_data_mechanism_from_config
from .multifurcation_resolver import get_multifurcation_resolver_from_config
from .mutationless_edges_strategy import \
    get_mutationless_edges_strategy_from_config
from .solver import GroundTruthSolver, get_solver_from_config
from .tree_scaler import get_tree_scaler_from_config
from .tree_simulator import get_tree_simulator_from_config
from casbench.branch_length_estimator._laml import LAML
import scipy


@caching.cached_computation(
    output_dirs=["output_tree_dir"],
)
def run_tree_simulator_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    output_tree_dir: Optional[str] = None,
):
    profiling_str = ""

    tree_simulator = get_tree_simulator_from_config(tree_simulator_config)
    st = time.time()
    tree = tree_simulator.simulate_tree()
    profiling_str += f"Time simulate tree: {time.time() - st}\n"

    leaf_subsampler = get_leaf_subsampler_from_config(leaf_subsampler_config)
    st = time.time()
    tree = leaf_subsampler.subsample_leaves(tree)
    profiling_str += f"Time subsample leaves: {time.time() - st}\n"

    tree_scaler = get_tree_scaler_from_config(tree_scaler_config)
    st = time.time()
    tree = tree_scaler(tree)
    profiling_str += f"Time scale tree after simulation: {time.time() - st}\n"

    write_tree(tree, os.path.join(output_tree_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_tree_dir, "profiling.txt"))


@caching.cached_computation(
    output_dirs=["output_tree_dir"],
)
def run_lt_simulator_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    output_tree_dir: Optional[str] = None,
):
    prev_output_tree_dir = run_tree_simulator_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
    )["output_tree_dir"]
    tree = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
    profiling_str = read_str(
        os.path.join(prev_output_tree_dir, "profiling.txt")
    )

    lt_simulator = get_lt_simulator_from_config(lt_simulator_config)
    st = time.time()
    lt_simulator.overlay_data(tree)
    profiling_str += f"Time simulate lineage tracing data: {time.time() - st}\n"

    # Need to finally collapse unifurcations (they were made available to the
    # lineage tracer to properly handle double-resections). I will use the
    # UniformLeafSubsampler not to have to deal with the casework of the
    # singular root edge.
    st = time.time()
    leaf_subsampler = get_leaf_subsampler_from_config([
        "UniformLeafSubsampler",
        [
            ("collapse_unifurcations", True),  # This is what we want.
            ("number_of_leaves", tree.character_matrix.shape[0]),  # I.e. no subsampling.
            ("random_seed", 42),  # Does not matter.
        ]
    ])
    tree = leaf_subsampler.subsample_leaves(tree)
    profiling_str += f"Time collapse unifurcations after simulating LT data: {time.time() - st}\n"

    write_tree(tree, os.path.join(output_tree_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_tree_dir, "profiling.txt"))


@caching.cached_computation(
    output_dirs=["output_tree_dir"],
)
def run_missing_data_mechanism_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    output_tree_dir: Optional[str] = None,
):
    prev_output_tree_dir = run_lt_simulator_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
    )["output_tree_dir"]
    tree = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
    profiling_str = read_str(
        os.path.join(prev_output_tree_dir, "profiling.txt")
    )

    missing_data_mechanism = get_missing_data_mechanism_from_config(
        missing_data_mechanism_config
    )
    st = time.time()
    tree = missing_data_mechanism(tree)
    profiling_str += f"Time missing data mechanism: {time.time() - st}\n"

    # Replace all missing data (epigenetic and heritable) by -1
    write_tree(
        tree,
        os.path.join(
            output_tree_dir, "result_with_disentangled_epi_and_stoch.txt"
        ),
    )
    st = time.time()
    assert tree.missing_state_indicator == -1
    new_states = {
        node: [max(x, -1) for x in tree.get_character_states(node)]
        for node in tree.nodes
    }
    tree.set_all_character_states(new_states)
    profiling_str += f"Time make all missing data -1: {time.time() - st}\n"

    write_tree(tree, os.path.join(output_tree_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_tree_dir, "profiling.txt"))


@caching.cached_computation(
    output_dirs=["output_tree_dir"],
)
def run_missing_data_imputer_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    output_tree_dir: Optional[str] = None,
):
    prev_output_tree_dir = run_missing_data_mechanism_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
    )["output_tree_dir"]
    tree = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
    profiling_str = read_str(
        os.path.join(prev_output_tree_dir, "profiling.txt")
    )

    missing_data_imputer = get_missing_data_imputer_from_config(
        missing_data_imputer_config
    )
    st = time.time()
    tree = missing_data_imputer(tree)
    profiling_str += f"Time impute missing data: {time.time() - st}\n"

    write_tree(tree, os.path.join(output_tree_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_tree_dir, "profiling.txt"))


def get_solver_from_configs(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
):
    """
    Enables oracle models.
    """
    identifier, _ = solver_config
    if identifier == "GroundTruthSolver":
        prev_output_tree_dir = run_missing_data_imputer_unrolled(
            tree_simulator_config=tree_simulator_config,
            leaf_subsampler_config=leaf_subsampler_config,
            tree_scaler_config=tree_scaler_config,
            lt_simulator_config=lt_simulator_config,
            missing_data_mechanism_config=missing_data_mechanism_config,
            missing_data_imputer_config=missing_data_imputer_config,
        )["output_tree_dir"]
        tree_gt = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
        return GroundTruthSolver(tree_gt)
    else:
        return get_solver_from_config(solver_config)


@caching.cached_computation(
    output_dirs=["output_tree_dir"],
)
def run_solver_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    output_tree_dir: Optional[str] = None,
):
    prev_output_tree_dir = run_missing_data_imputer_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
    )["output_tree_dir"]
    tree = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
    profiling_str = read_str(
        os.path.join(prev_output_tree_dir, "profiling.txt")
    )

    solver = get_solver_from_configs(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
    )
    st = time.time()
    # In case some solvers are non-deterministic but do not allow setting the
    # random seed, so we set it ourselves.
    np.random.seed(0)
    solver.solve(tree)
    profiling_str += f"Time solve tree topology: {time.time() - st}\n"

    write_tree(tree, os.path.join(output_tree_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_tree_dir, "profiling.txt"))


def get_asr_from_configs(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    ancestral_states_reconstructor_config: Config,
):
    """
    Enables oracle models.
    """
    identifier, args = ancestral_states_reconstructor_config
    if identifier == "ground_truth_asr":
        if len(args) > 0:
            raise ValueError(
                f"ground_truth_asr takes no arguments. You provided: {args}"
            )
        prev_output_tree_dir = run_missing_data_mechanism_unrolled(
            tree_simulator_config=tree_simulator_config,
            leaf_subsampler_config=leaf_subsampler_config,
            tree_scaler_config=tree_scaler_config,
            lt_simulator_config=lt_simulator_config,
            missing_data_mechanism_config=missing_data_mechanism_config,
        )["output_tree_dir"]
        tree_gt = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
        return partial(ground_truth_asr, tree_gt=tree_gt)
    else:
        return get_asr_from_config(ancestral_states_reconstructor_config)


def get_ble_from_configs(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    ble_config: Config,
):
    """
    Enables oracle models.
    """
    identifier, args = ble_config
    if identifier == "GroundTruthBLE":
        if len(args) > 0:
            raise ValueError(
                f"GroundTruthBLE takes no arguments. You provided: {args}"
            )
        prev_output_tree_dir = run_lt_simulator_unrolled(
            tree_simulator_config=tree_simulator_config,
            leaf_subsampler_config=leaf_subsampler_config,
            tree_scaler_config=tree_scaler_config,
            lt_simulator_config=lt_simulator_config,
        )["output_tree_dir"]
        tree_gt = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
        return GroundTruthBLE(tree_gt=tree_gt)
    elif identifier == "LAML_GT_prior_2024_09_10_v2":
        lt_simulator_identifier, lt_simulator_args = lt_simulator_config
        lt_simulator_args_dict = dict(lt_simulator_args)
        if lt_simulator_identifier != "ReverseEngineeredCas9LineageTracingDataSimulator":
            raise ValueError(
                f"Only know how to get priors for ReverseEngineeredCas9LineageTracingDataSimulator ."
            )
        number_of_states = lt_simulator_args_dict["number_of_states"]
        state_priors = scipy.stats.expon.ppf(
            (np.array(range(number_of_states)) + 0.5) / number_of_states,
            loc=0,
            scale=1e-5,
        )
        state_priors /= sum(state_priors)
        state_priors = {
            number_of_states - i: state_priors[i]
            for i in range(number_of_states)
        }
        # Need to create some prior for the double resection events. We set them to match the most likely state.
        if lt_simulator_args_dict["collapse_sites_on_cassette"]:
            if lt_simulator_args_dict["size_of_cassette"] != 3:
                raise ValueError(
                    f"Only probabilities for double resection states 100000003, 100000005, 100000006 are being added, "
                    "i.e. this code is hardcoded to work with cassettes of size 3."
                )
            state_priors[100000003] = state_priors[1]
            state_priors[100000005] = state_priors[1]
            state_priors[100000006] = state_priors[1]
            # Need to renormalize the new priors which include the double resection events.
            sm = sum(state_priors.values())
            state_priors = {
                x: y / sm
                for (x, y) in state_priors.items()
            }
        assert(
            abs(sum(state_priors.values()) - 1.0) < 1e-6
        )
        # Done creating the priors!

        args_dict = dict(args)
        args_dict["priors"] = state_priors
        return LAML(**args_dict)
    else:
        return get_ble_from_config(ble_config)


@caching.cached_computation(
    output_dirs=["output_tree_dir"],
    write_extra_log_files=True,
)
def run_ble_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    mutationless_edges_strategy_config: Config,
    multifurcation_resolver_config: Config,
    ancestral_states_reconstructor_config: Config,
    ble_config: Config,
    ble_tree_scaler_config: Config,
    output_tree_dir: Optional[str] = None,
):
    prev_output_tree_dir = run_solver_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
    )["output_tree_dir"]
    tree = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
    profiling_str = read_str(
        os.path.join(prev_output_tree_dir, "profiling.txt")
    )

    st = time.time()
    mutationless_edges_strategy = get_mutationless_edges_strategy_from_config(
        mutationless_edges_strategy_config
    )
    tree = mutationless_edges_strategy(tree)
    profiling_str += f"Time mutationless edges strategy: {time.time() - st}\n"

    st = time.time()
    multifurcation_resolver = get_multifurcation_resolver_from_config(
        multifurcation_resolver_config
    )
    tree = multifurcation_resolver(tree)
    profiling_str += f"Time multifurcation resolver: {time.time() - st}\n"

    st = time.time()
    ancestral_states_reconstructor = get_asr_from_configs(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
    )
    tree = ancestral_states_reconstructor(tree)
    profiling_str += (
        f"Time ancestral states reconstructor: {time.time() - st}\n"
    )

    st = time.time()
    ble = get_ble_from_configs(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        ble_config=ble_config,
    )
    ble.estimate_branch_lengths(tree)
    profiling_str += f"Time branch length estimation: {time.time() - st}\n"

    st = time.time()
    tree_scaler = get_tree_scaler_from_config(ble_tree_scaler_config)
    tree = tree_scaler(tree)
    profiling_str += (
        f"Time scale tree after branch length estimation: {time.time() - st}\n"
    )

    write_ble(ble, os.path.join(output_tree_dir, "ble.txt"))
    write_tree(tree, os.path.join(output_tree_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_tree_dir, "profiling.txt"))


@caching.cached_computation(
    output_dirs=["output_predictions_dir"],
)
def run_predict_times_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    mutationless_edges_strategy_config: Config,
    multifurcation_resolver_config: Config,
    ancestral_states_reconstructor_config: Config,
    ble_config: Config,
    ble_tree_scaler_config: Config,
    internal_node_time_predictor_config: Config,
    output_predictions_dir: Optional[str] = None,
):
    tree_true = read_tree(
        os.path.join(
            run_lt_simulator_unrolled(
                tree_simulator_config=tree_simulator_config,
                leaf_subsampler_config=leaf_subsampler_config,
                tree_scaler_config=tree_scaler_config,
                lt_simulator_config=lt_simulator_config,
            )["output_tree_dir"],
            "result.txt",
        )
    )

    tree_inferred_dir = run_ble_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
        mutationless_edges_strategy_config=mutationless_edges_strategy_config,
        multifurcation_resolver_config=multifurcation_resolver_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
        ble_config=ble_config,
        ble_tree_scaler_config=ble_tree_scaler_config,
    )["output_tree_dir"]
    tree_inferred = read_tree(os.path.join(tree_inferred_dir, "result.txt"))
    profiling_str = read_str(os.path.join(tree_inferred_dir, "profiling.txt"))

    identifier, args = internal_node_time_predictor_config
    if identifier != "mrca_impute":
        raise ValueError(f"Unknown internal node time predictor: {identifier}")

    st = time.time()
    predicted_times_dict = get_true_vs_predicted_attribute(
        tree_true=tree_true,
        tree_inferred=tree_inferred,
        attribute_name="time",
        include_leaves=False,
        include_non_root_internal_nodes=True,
        include_root=False,
        **dict(args),
    )
    profiling_str += f"Time predict internal node times: {time.time() - st}\n"

    write_pickle(
        list(zip(*list(predicted_times_dict.values()))),
        os.path.join(output_predictions_dir, "result.txt"),
    )
    write_str(
        profiling_str, os.path.join(output_predictions_dir, "profiling.txt")
    )


@caching.cached_computation(
    output_dirs=["output_metric_dir"],
)
def run_internal_node_time_metric_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    mutationless_edges_strategy_config: Config,
    multifurcation_resolver_config: Config,
    ancestral_states_reconstructor_config: Config,
    ble_config: Config,
    ble_tree_scaler_config: Config,
    internal_node_time_predictor_config: Config,
    metric_config: Config,
    output_metric_dir: Optional[str] = None,
):
    output_predictions_dir = run_predict_times_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
        mutationless_edges_strategy_config=mutationless_edges_strategy_config,
        multifurcation_resolver_config=multifurcation_resolver_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
        ble_config=ble_config,
        ble_tree_scaler_config=ble_tree_scaler_config,
        internal_node_time_predictor_config=internal_node_time_predictor_config,
    )["output_predictions_dir"]

    true_times, inferred_times = read_pickle(
        os.path.join(output_predictions_dir, "result.txt")
    )
    profiling_str = read_str(
        os.path.join(output_predictions_dir, "profiling.txt")
    )

    metric = get_metric_by_config(metric_config)
    st = time.time()
    res = metric(true_times, inferred_times)
    profiling_str += (
        f"Time compute internal node time metric: {time.time() - st}\n"
    )

    write_float(res, os.path.join(output_metric_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_metric_dir, "profiling.txt"))


@caching.cached_computation(
    output_dirs=["output_metric_dir"],
)
def run_ble_runtime_metric_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    mutationless_edges_strategy_config: Config,
    multifurcation_resolver_config: Config,
    ancestral_states_reconstructor_config: Config,
    ble_config: Config,
    ble_tree_scaler_config: Config,
    output_metric_dir: Optional[str] = None,
):
    ble_tree_dir = run_ble_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
        mutationless_edges_strategy_config=mutationless_edges_strategy_config,
        multifurcation_resolver_config=multifurcation_resolver_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
        ble_config=ble_config,
        ble_tree_scaler_config=ble_tree_scaler_config,
    )["output_tree_dir"]
    res = None
    profiling_str = read_str(os.path.join(ble_tree_dir, "profiling.txt"))
    for line in profiling_str.split("\n"):
        if line.startswith("Time branch length estimation:"):
            res = float(line.split(" ")[4])
    assert res is not None
    write_float(res, os.path.join(output_metric_dir, "result.txt"))


def number_of_ancestral_lineages(
    tree: CassiopeiaTree,
    t: float,
) -> int:
    res = 0
    for (p, c) in tree.edges:
        if tree.get_time(p) < t and tree.get_time(c) >= t:
            res += 1
    return res


@caching.cached_computation(
    output_dirs=["output_metric_dir"],
)
def run_ancestral_lineages_metric_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    mutationless_edges_strategy_config: Config,
    multifurcation_resolver_config: Config,
    ancestral_states_reconstructor_config: Config,
    ble_config: Config,
    ble_tree_scaler_config: Config,
    ancestral_lineages_config: Config,
    metric_config: Config,
    output_metric_dir: Optional[str] = None,
):
    gt_tree_dir = run_lt_simulator_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
    )["output_tree_dir"]
    gt_tree = read_tree(os.path.join(gt_tree_dir, "result.txt"))

    ble_tree_dir = run_ble_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
        mutationless_edges_strategy_config=mutationless_edges_strategy_config,
        multifurcation_resolver_config=multifurcation_resolver_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
        ble_config=ble_config,
        ble_tree_scaler_config=ble_tree_scaler_config,
    )["output_tree_dir"]
    ble_tree = read_tree(os.path.join(ble_tree_dir, "result.txt"))
    profiling_str = read_str(os.path.join(ble_tree_dir, "profiling.txt"))

    identifier, args = ancestral_lineages_config
    args_dict = dict(args)
    if identifier != "ancestral_time":
        raise ValueError(f"Unknown ancestral lineages identifier: {identifier}")
    if list(args_dict.keys()) != ["t"]:
        raise ValueError("ancestral_time takes exactly one argument: 't'")
    t = args_dict["t"]

    st = time.time()
    true_number_of_ancestral_lineages = number_of_ancestral_lineages(
        gt_tree,
        t,
    )
    inferred_number_of_ancestral_lineages = number_of_ancestral_lineages(
        ble_tree,
        t,
    )
    metric = get_metric_by_config(metric_config)
    res = metric(
        [true_number_of_ancestral_lineages],
        [inferred_number_of_ancestral_lineages],
    )
    profiling_str += (
        f"Time compute ancestral lineages metric: {time.time() - st}\n"
    )

    write_float(res, os.path.join(output_metric_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_metric_dir, "profiling.txt"))


@caching.cached_computation(
    output_dirs=["output_tree_dir"],
)
def run_fitness_estimation_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    mutationless_edges_strategy_config: Config,
    multifurcation_resolver_config: Config,
    ancestral_states_reconstructor_config: Config,
    ble_config: Config,
    ble_tree_scaler_config: Config,
    fitness_estimator_config: Config,
    output_tree_dir: Optional[str] = None,
):
    prev_output_tree_dir = run_ble_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
        mutationless_edges_strategy_config=mutationless_edges_strategy_config,
        multifurcation_resolver_config=multifurcation_resolver_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
        ble_config=ble_config,
        ble_tree_scaler_config=ble_tree_scaler_config,
    )["output_tree_dir"]
    tree = read_tree(os.path.join(prev_output_tree_dir, "result.txt"))
    profiling_str = read_str(
        os.path.join(prev_output_tree_dir, "profiling.txt")
    )

    fe = get_fitness_estimator_from_config(fitness_estimator_config)
    st = time.time()
    fe.estimate_fitness(tree)
    profiling_str += f"Time fitness estimation: {time.time() - st}\n"

    write_tree(tree, os.path.join(output_tree_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_tree_dir, "profiling.txt"))


@caching.cached_computation(
    output_dirs=["output_metric_dir"],
)
def run_fitness_metric_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    mutationless_edges_strategy_config: Config,
    multifurcation_resolver_config: Config,
    ancestral_states_reconstructor_config: Config,
    ble_config: Config,
    ble_tree_scaler_config: Config,
    fitness_estimator_config: Config,
    fitness_subsetting_config: Config,
    metric_config: Config,
    output_metric_dir: Optional[str] = None,
):
    gt_tree_dir = run_lt_simulator_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
    )["output_tree_dir"]
    gt_tree = read_tree(os.path.join(gt_tree_dir, "result.txt"))

    fe_tree_dir = run_fitness_estimation_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
        mutationless_edges_strategy_config=mutationless_edges_strategy_config,
        multifurcation_resolver_config=multifurcation_resolver_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_config,  # noqa
        ble_config=ble_config,
        ble_tree_scaler_config=ble_tree_scaler_config,
        fitness_estimator_config=fitness_estimator_config,
    )["output_tree_dir"]
    fe_tree = read_tree(os.path.join(fe_tree_dir, "result.txt"))
    profiling_str = read_str(os.path.join(fe_tree_dir, "profiling.txt"))

    st = time.time()
    identifier, args = fitness_subsetting_config
    if identifier != "subset_nodes":
        raise ValueError(f"Unknown fitness subsetting: {identifier}")
    if args != [("leaves", True)]:
        raise ValueError("The only supported node_subset is the leaves")
    true_fitness = [
        gt_tree.get_attribute(node, "fitness") for node in gt_tree.leaves
    ]
    pred_fitness = [
        fe_tree.get_attribute(node, "fitness") for node in gt_tree.leaves
    ]
    metric = get_metric_by_config(metric_config)
    res = metric(true_fitness, pred_fitness)
    profiling_str += f"Time fitness metric: {time.time() - st}\n"

    write_float(res, os.path.join(output_metric_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_metric_dir, "profiling.txt"))


@caching.cached_computation(
    output_dirs=["output_metric_dir"],
)
def run_solver_metric_unrolled(
    tree_simulator_config: Config,
    leaf_subsampler_config: Config,
    tree_scaler_config: Config,
    lt_simulator_config: Config,
    missing_data_mechanism_config: Config,
    missing_data_imputer_config: Config,
    solver_config: Config,
    ancestral_states_reconstructor_gt_config: Config,
    mutationless_edges_strategy_gt_config: Config,
    multifurcation_resolver_gt_config: Config,
    ancestral_states_reconstructor_recon_config: Config,
    mutationless_edges_strategy_recon_config: Config,
    multifurcation_resolver_recon_config: Config,
    metric_config: Config,
    output_metric_dir: Optional[str] = None,
):
    ground_truth_tree_dir = run_missing_data_mechanism_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
    )["output_tree_dir"]
    reconstructed_tree_dir = run_solver_unrolled(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        missing_data_imputer_config=missing_data_imputer_config,
        solver_config=solver_config,
    )["output_tree_dir"]
    ground_truth_tree = read_tree(
        os.path.join(ground_truth_tree_dir, "result.txt")
    )
    reconstructed_tree = read_tree(
        os.path.join(reconstructed_tree_dir, "result.txt")
    )
    profiling_str = read_str(
        os.path.join(reconstructed_tree_dir, "profiling.txt")
    )

    # First post-process the GT tree.

    st = time.time()
    asr_gt = get_asr_from_configs(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_gt_config,  # noqa
    )
    ground_truth_tree = asr_gt(ground_truth_tree)
    del asr_gt
    profiling_str += "Time solver metric ASR GT tree: " f"{time.time() - st}\n"

    st = time.time()
    mutationless_edges_strategy_gt = (
        get_mutationless_edges_strategy_from_config(
            mutationless_edges_strategy_gt_config
        )
    )
    ground_truth_tree = mutationless_edges_strategy_gt(ground_truth_tree)
    del mutationless_edges_strategy_gt
    profiling_str += (
        "Time solver metric mutationless edges strategy GT tree: "
        f"{time.time() - st}\n"
    )

    st = time.time()
    multifurcation_resolver_gt = get_multifurcation_resolver_from_config(
        multifurcation_resolver_gt_config
    )
    ground_truth_tree = multifurcation_resolver_gt(ground_truth_tree)
    del multifurcation_resolver_gt
    profiling_str += (
        "Time solver metric multifurcation resolver "
        f"GT tree: {time.time() - st}\n"
    )

    # Now post-process the reconstructed tree

    st = time.time()
    asr_recon = get_asr_from_configs(
        tree_simulator_config=tree_simulator_config,
        leaf_subsampler_config=leaf_subsampler_config,
        tree_scaler_config=tree_scaler_config,
        lt_simulator_config=lt_simulator_config,
        missing_data_mechanism_config=missing_data_mechanism_config,
        ancestral_states_reconstructor_config=ancestral_states_reconstructor_recon_config,  # noqa
    )
    reconstructed_tree = asr_recon(reconstructed_tree)
    del asr_recon
    profiling_str += (
        "Time solver metric ASR reconstructed tree: " f"{time.time() - st}\n"
    )

    st = time.time()
    mutationless_edges_strategy_recon = (
        get_mutationless_edges_strategy_from_config(
            mutationless_edges_strategy_recon_config
        )
    )
    reconstructed_tree = mutationless_edges_strategy_recon(reconstructed_tree)
    del mutationless_edges_strategy_recon
    profiling_str += (
        "Time solver metric mutationless edges strategy "
        f"reconstructed tree: {time.time() - st}\n"
    )

    st = time.time()
    multifurcation_resolver_recon = get_multifurcation_resolver_from_config(
        multifurcation_resolver_recon_config
    )
    reconstructed_tree = multifurcation_resolver_recon(reconstructed_tree)
    del multifurcation_resolver_recon
    profiling_str += (
        "Time solver metric multifurcation resolver "
        f"reconstructed tree: {time.time() - st}\n"
    )

    identifier, args = metric_config
    args_dict = dict(args)
    st = time.time()
    if identifier == "rf":
        rf, rf_max = cassiopeia.critique.compare.robinson_foulds(
            ground_truth_tree, reconstructed_tree, **args_dict
        )
        res = rf / rf_max
    elif identifier == "triplets":
        triplets = cassiopeia.critique.compare.triplets_correct(
            ground_truth_tree, reconstructed_tree, **args_dict
        )
        res = np.mean(list(triplets[0].values()))
    profiling_str += f"Time compute solver metric: {time.time() - st}\n"

    write_float(res, os.path.join(output_metric_dir, "result.txt"))
    write_str(profiling_str, os.path.join(output_metric_dir, "profiling.txt"))