import itertools


NUMBER_OF_CASSETTES = [3, 6, 13, 20, 30, 50]

CACHE_PAPER_BLE = "_cache_paper_ble"
CACHE_PAPER_BLE_MODEL_MISSPECIFICATION_ERRORS = "_cache_paper_ble_model_mis"
PAPER_BLE_FIGURES_DIR = "./paper_ble_figures/"

PSEUDOMUTATIONS_LIST = [0.0, 0.1, 0.5]

BLE_DISPLAY_NAMES = {
    "gt__gt": "GT",  # noqa
    "gt__muts_extended-0.5": r"$\mathrm{GT\ States\ +\ Number\ of\ Mutations}$",  # noqa
    "mp__muts_extended-0.5": r"$\mathrm{MP\ +\ Number\ of\ Mutations}$",  # noqa
    "cmp__muts_extended-0.5": r"$\mathrm{Conservative\ MP\ +\ Number\ of\ Mutations}$",  # noqa
    "mp__LAML": "LAML",  # noqa
    "mp__LAML_GT_prior": "LAML_GT_prior",
    "mp__TiDeTree": "TiDeTree",
}
MP_STRATEGY_DISPLAY_NAMES = {
    "gt": r"GT\ States",
    "cmp": r"Conservative\ MP",
    "mp": r"MP",
}
PSEUDOCOUNTS_DISPLAY_NAMES = {
    0.0: r"0",
    0.1: r"0.1",
    0.5: r"0.5",
}
for mp_strategy in ["gt", "mp", "cmp"]:
    for mbl_pct in ["0.1", "1", "2"]:
        for p in PSEUDOMUTATIONS_LIST:
            BLE_DISPLAY_NAMES[
                f"{mp_strategy}__MLE_mbl-{mbl_pct}_pm-{p}_pnm-{p}"
            ] = (
                r"$\mathrm{"
                + MP_STRATEGY_DISPLAY_NAMES[mp_strategy]
                + r"\ +\ MLE"
                + r",\ \lambda="
                + PSEUDOCOUNTS_DISPLAY_NAMES[p]
                + r"}$"
            )
BLE_CONFIG_DICTS = {
    "gt": {
        "identifier": "GroundTruthBLE",
        "args": {},
    },
    **{
        f"MLE_mbl-{mbl_pct}_pm-{p}_pnm-{p}": {
            "identifier": "IIDExponentialMLE",
            "args": {
                "minimum_branch_length": mbl,
                "pseudo_mutations_per_edge": p,
                "pseudo_non_mutations_per_edge": p,
                "solver": "CLARABEL",
                "backup_solver": "SCS",
                "pendant_branch_minimum_branch_length_multiplier": 0.5,
            },
        }
        for (mbl_pct, mbl) in [("0.1", 0.001), ("1", 0.01), ("2", 0.02)]
        for p in PSEUDOMUTATIONS_LIST
    },
    "muts-0.5": {
        "identifier": "NumberOfMutationsBLE",
        "args": {
            "length_of_mutationless_edges": 0.5,
            "make_ultrametric": False,
        },
    },
    "muts_extended-0.5": {
        "identifier": "NumberOfMutationsBLE",
        "args": {
            "length_of_mutationless_edges": 0.5,
            "make_ultrametric": True,
        },
    },
    "muts_extended-0.0": {
        "identifier": "NumberOfMutationsBLE",
        "args": {
            "length_of_mutationless_edges": 0.0,
            "make_ultrametric": True,
        },
    },
    "const": {
        "identifier": "ConstantBLE",
        "args": {"make_ultrametric": False},
    },
    "const_extended": {
        "identifier": "ConstantBLE",
        "args": {"make_ultrametric": True},
    },
    "binary": {
        "identifier": "BinaryBLE",
        "args": {"include_missing": False, "make_ultrametric": False},
    },
    "binary_extended": {
        "identifier": "BinaryBLE",
        "args": {"include_missing": False, "make_ultrametric": True},
    },
    "LAML": {
        "identifier": "LAML_2024_09_10_v2",
        "args": {},
    },
    "LAML_GT_prior": {
        "identifier": "LAML_GT_prior_2024_09_10_v2",
        "args": {},
    },
    "TiDeTree": {
        "identifier": "TiDeTree_2024_09_19_v1",
        "args": {
            "priors": None,
            "subsampling_probability": 0.01,
            "silencing_rate": 0.01,
        },
    },
}

SOLVER_DISPLAY_NAMES = {
    "gt": "GT",
    "greedy": "Greedy",
    "nj": "NJ",
    "upgma": "UPGMA",
    "maxcut_greedy": "Maxcut Greedy",
    "maxcut": "Maxcut",
    # "ilp_20_unw": "ILP",  # Takes too long to run
}
SOLVER_CONFIG_DICTS = {
    "gt": {"identifier": "GroundTruthSolver", "args": {}},
    "greedy": {"identifier": "VanillaGreedySolver", "args": {}},
    "nj": {"identifier": "NeighborJoiningSolver", "args": {"add_root": True}},
    "upgma": {"identifier": "UPGMASolver", "args": {}},
    "maxcut_greedy": {"identifier": "MaxCutGreedySolver", "args": {}},
    "maxcut": {"identifier": "MaxCutSolver", "args": {}},
    # "ilp_20_unw": {
    #     "identifier": "ILPSolver",
    #     "args": {
    #         "maximum_potential_graph_lca_distance": 20,
    #         "weighted": False,
    #     },
    # },  # Takes too long to run
}

FE_CONFIG_DICTS = {
    "lbi": {"identifier": "LBIJungle", "args": {"random_seed": 0}},
}

METRIC_DISPLAY_NAMES = {
    "solver__triplets-500": "Triplets Correct",
    "solver__rf": "Robinson Foulds",
    "ble__node_time-mae": "Internal Node Time, Mean Absolute Error",
    "ble__node_time-mean_bias": "Internal Node Time, Mean Bias",
    "ble__ancestors-0.5-mre": "Number of Ancestral Lineages, Relative Error",  # noqa
    "ble__runtime": "Runtime (s)",
    "fe__sr2": "Leaf Fitness, Spearman Correlation",
}

KNOB_DISPLAY_NAMES = {
    "number_of_cassettes": "Number of Barcodes",
    "size_of_cassette": "Barcode Size",
    "expected_proportion_mutated": "Expected Proportion of Character Matrix Mutated",  # noqa
    "number_of_states": "Number of Indel States",
    "expected_prop_missing": "Expected Proportion of Character Matrix Missing",
}

ASR_CONFIG_DICTS = {
    "mp": {"identifier": "maximum_parsimony", "args": {}},
    "cmp": {"identifier": "conservative_maximum_parsimony", "args": {}},
    "gt": {"identifier": "ground_truth_asr", "args": {}},
}
