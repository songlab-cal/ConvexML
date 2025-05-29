from typing import List, Tuple

import numpy as np
import scipy.stats
from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import (Cas9LineageTracingDataSimulator,
                                  LineageTracingDataSimulator)


def reverse_engineer_heritable_silencing_rate(
    tree: CassiopeiaTree,
    expected_proportion_heritable: float,
) -> float:
    """
    Heritable silencing rate to achieve a given expected_proportion_heritable.

    Given a (NOT subsampled) single-cell tree and a target proportion of
    heritable missing data, returns the heritable silencing rate that would
    lead, in expectation, to the desired amount of missing heritable data.

    Complexity: O(tree.n_cell * log_2(1000000)) (binary search)

    Args:
        tree: The single-cell phylogeny.

    Returns:
        The heritable silencing rate that would lead, in expectation, to the
            desired amount of missing heritable data.
    """
    lo = 0.0
    hi = 100.0
    while lo < hi - 0.0001:
        mid = (lo + hi) / 2.0

        def dfs(v: str, rate: float) -> Tuple[float, int]:
            """
            The expected_number_mutated starting the tracer from here,
            as well as the size of the subtree.
            """

            def get_p(v: str, rate: float):
                """
                Probability of a casette going missing at v.
                """
                if v == tree.root:
                    life_time = 0.0
                else:
                    pa = tree.parent(v)
                    life_time = tree.get_branch_length(pa, v)
                p = 1 - np.exp(-life_time * rate)
                return p

            p = get_p(v, rate)
            ch = tree.children(v)
            if len(ch) == 0:
                # Base case
                expected_number_mutated = p
                size = 1
            else:
                # Recursion
                expected_number_mutated = 0
                size = 0
                for u in tree.children(v):
                    sub_expected_number_mutated, sub_size = dfs(u, rate)
                    expected_number_mutated += (
                        1.0 - p
                    ) * sub_expected_number_mutated
                    size += sub_size
                expected_number_mutated += p * size
            return (expected_number_mutated, size)

        expected_number_mutated, size = dfs(tree.root, mid)
        assert size == tree.n_cell
        if expected_number_mutated < expected_proportion_heritable * size:
            lo = mid
        else:
            hi = mid
    return lo


def spline_out(
    ys: List[float],
    n_points: int,
) -> List[float]:
    """
    Spline out a function from another one.

    Given a piecewise linear function given by f(k) = ys[k], returns its version
    splined to n_points.

    Args:
        ys: List of y values of the function to spline.
        n_points: How many points to spline the function to.

    Returns:
        The y values for the splined function.
    """
    res = []
    for i in range(n_points):
        # calculate res[i]
        i_adj = (i * (len(ys) - 1)) / (n_points - 1.0)
        i_adj_int = int(i_adj)
        i_adj_frac = i_adj - i_adj_int
        assert 0 <= i_adj
        assert i_adj <= len(ys) - 1
        del i_adj
        assert 0 <= i_adj_frac
        assert i_adj_frac < 1
        res_i = (
            ys[i_adj_int] * (1.0 - i_adj_frac)
            + ys[min(i_adj_int + 1, len(ys) - 1)] * i_adj_frac
        )
        res.append(res_i)
    return res


def reverse_engineer_non_iid_mutation_rates(
    number_of_cassettes: int,
    size_of_cassette: int,
    expected_proportion_mutated: float,
) -> List[float]:
    """
    Mutation rates needed to obtains the desired expected proportion mutated.

    For a tree of depth 1, empirically-derived non-iid rates that lead to the
    expected_proportion_mutated. The rates come from real data (plate data
    clone #2)

    Complexity: O(50 * log_2(10^6)) (binary search)

    Args:
        number_of_cassettes: Number of casettes
        size_of_cassette: Size of each casette
        expected_proportion_mutated: The desired expected proporiton mutated.

    Returns:
        The mutation rate per site needed to achieve the desired expected
            proportion mutated.
    """
    # The empirical rates were estimated from clone #2 in the plate data from the Cassiopeia paper.
    # They have high variability, which importantly violates the IID assumption.
    empirical_rates = [
        0.004756080162253702,
        0.11223978421162137,
        0.1302435236654361,
        0.1665905054258625,
        0.17821783904087865,
        0.197712707114252,
        0.21900004958405614,
        0.2372578424600062,
        0.239297505529848,
        0.24323030988094713,
        0.24574573944637987,
        0.26292459796547346,
        0.2688445463555396,
        0.28589688899642846,
        0.29650055912620066,
        0.30448127433404065,
        0.307329922948694,
        0.3102212875062263,
        0.3321698433456854,
        0.3344813814515644,
        0.3362422127552227,
        0.34506378301105445,
        0.3454579227153953,
        0.3468140751731784,
        0.35719974059207843,
        0.3841975473875777,
        0.39929910016450615,
        0.40228834119783513,
        0.42887468409698143,
        0.44399678688248817,
        0.47657652688037955,
        0.5075797624479093,
        0.5098376691315422,
        0.521955426973116,
        0.5307622364579627,
        0.5618035661501062,
        0.570566978935683,
        0.633090149336745,
        0.6509249524386804,
        0.7155931326333668,
        0.7338699145709124,
        0.7600194250437365,
        0.7697468763217338,
        0.8468600000576509,
        0.8544940300987346,
        0.9278997326176991,
        1.0364860040145172,
        1.1071360361494877,
        1.2205558538249277,
        1.2857257418549366,
        1.6488404603001292,
    ]
    n_characters = number_of_cassettes * size_of_cassette
    # First spline out the distribution to our number of characters.
    rates = spline_out(
        ys=empirical_rates,
        n_points=n_characters,
    )
    # Now scale the rates to achieve the desired expected_proportion_mutated.
    lo = 0
    hi = 100.0
    while lo < hi - 0.0001:
        mid = (lo + hi) / 2.0
        curr_expected_proportion_mutated = sum(
            [(1.0 - np.exp(-r * mid)) for r in rates]
        ) / len(rates)
        if curr_expected_proportion_mutated < expected_proportion_mutated:
            lo = mid
        else:
            hi = mid
    return [r * mid for r in rates]


class ReverseEngineeredCas9LineageTracingDataSimulator(
    LineageTracingDataSimulator
):
    def __init__(
        self,
        number_of_cassettes: int,
        size_of_cassette: int,
        number_of_states: int,
        expected_proportion_heritable: float,
        expected_proportion_stochastic: float,
        expected_proportion_mutated: float,
        iid_mutation_rates: bool,
        collapse_sites_on_cassette: bool,
        random_seed: int,
        heritable_missing_data_state: int = -1,
        stochastic_missing_data_state: int = -2,
        create_allele_when_collapsing_sites_on_cassette: bool = False,
    ):
        self._number_of_cassettes = number_of_cassettes
        self._size_of_cassette = size_of_cassette
        self._number_of_states = number_of_states
        self._expected_proportion_heritable = expected_proportion_heritable
        self._expected_proportion_stochastic = expected_proportion_stochastic
        self._expected_proportion_mutated = expected_proportion_mutated
        self._iid_mutation_rates = iid_mutation_rates
        self._collapse_sites_on_cassette = collapse_sites_on_cassette
        self._random_seed = random_seed
        self._heritable_missing_data_state = heritable_missing_data_state
        self._stochastic_missing_data_state = stochastic_missing_data_state
        self._create_allele_when_collapsing_sites_on_cassette = create_allele_when_collapsing_sites_on_cassette

    def overlay_data(self, tree: CassiopeiaTree) -> None:
        number_of_cassettes = self._number_of_cassettes
        size_of_cassette = self._size_of_cassette
        number_of_states = self._number_of_states
        expected_proportion_heritable = self._expected_proportion_heritable
        expected_proportion_stochastic = self._expected_proportion_stochastic
        expected_proportion_mutated = self._expected_proportion_mutated
        iid_mutation_rates = self._iid_mutation_rates
        collapse_sites_on_cassette = self._collapse_sites_on_cassette
        random_seed = self._random_seed
        heritable_missing_data_state = self._heritable_missing_data_state
        stochastic_missing_data_state = self._stochastic_missing_data_state
        create_allele_when_collapsing_sites_on_cassette = self._create_allele_when_collapsing_sites_on_cassette

        expected_proportion_stochastic = expected_proportion_stochastic / 100.0
        expected_proportion_heritable = expected_proportion_heritable / 100.0

        if iid_mutation_rates:
            mutation_rate = -np.log(1.0 - expected_proportion_mutated / 100.0)
        else:
            # Spline from realistic mutation rate distribution.
            mutation_rate = reverse_engineer_non_iid_mutation_rates(
                number_of_cassettes=number_of_cassettes,
                size_of_cassette=size_of_cassette,
                expected_proportion_mutated=expected_proportion_mutated / 100.0,
            )
        heritable_silencing_rate = reverse_engineer_heritable_silencing_rate(
            tree=tree,
            expected_proportion_heritable=expected_proportion_heritable,
        )
        stochastic_silencing_rate = expected_proportion_stochastic / (
            1.0 - expected_proportion_heritable
        )
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
        Cas9LineageTracingDataSimulator(
            number_of_cassettes=number_of_cassettes,
            size_of_cassette=size_of_cassette,
            mutation_rate=mutation_rate,
            state_generating_distribution=None,
            number_of_states=number_of_states,
            state_priors=state_priors,
            stochastic_silencing_rate=stochastic_silencing_rate,
            heritable_silencing_rate=heritable_silencing_rate,
            heritable_missing_data_state=heritable_missing_data_state,
            stochastic_missing_data_state=stochastic_missing_data_state,
            random_seed=random_seed,
            collapse_sites_on_cassette=collapse_sites_on_cassette,
            create_allele_when_collapsing_sites_on_cassette=create_allele_when_collapsing_sites_on_cassette,
        ).overlay_data(tree)
        self._mutation_rate = mutation_rate
        self._stochastic_silencing_rate = stochastic_silencing_rate
        self._heritable_silencing_rate = heritable_silencing_rate
