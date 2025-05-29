import re
from typing import Tuple

import numpy as np
from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins.errors import TreeSimulatorError
from cassiopeia.simulator import BirthDeathFitnessSimulator, TreeSimulator


def compute_extinction_probability(
    lam: float,
    offset: float,
    mu: float,
    sampling_probability: float,
) -> float:
    """
    Extinction probability for a subsampled birth-death process.

    Given a birth-death process with birth following an exponential distribution
    with rate 'lambda' offset by 'offset', and death following an exponential
    distribution with rate 'mu', which is subsequently sampled down by
    sampling_probability, goes extinct. The process is run for 1 time unit.

    Args:
        lam: birth rate
        offset: Holding time before birth events can start
        mu: Death rate.
        sampling_probability: Probability of a leaf getting sampled.
    """
    T = 10000  # Discretization level
    dt = 1.0 / T
    c = int(T * offset)
    p_ext = [1.0 - sampling_probability]
    for t in range(1, T + 1):
        next_t = max(t - 1 - c, 0)
        p_ext_t = (
            dt
            * lam
            * (
                (
                    1.0 - np.exp(-mu * (t - next_t) / T)
                )  # Extinction before birth can start
                + np.exp(-mu * (t - next_t) / T)
                * p_ext[
                    next_t
                ]  # Extinction once birth starts (or end of process)
            )
            ** 2
            + dt * mu
            + (1.0 - dt * lam - dt * mu) * p_ext[t - 1]
        )
        p_ext.append(p_ext_t)
    next_t = max(T - c, 0)
    res = (1.0 - np.exp(-mu * (T - next_t) / T)) + np.exp(
        -mu * (T - next_t) / T
    ) * p_ext[next_t]
    return res


def reverse_engineer_birth_and_death_rate(
    expected_population_size: float,
    offset: float,
    birth_to_death_rate_ratio: float,
    sampling_probability: float,
) -> Tuple[float, float]:
    """
    Birth and death rate parameters needed to achieve a given population size.

    Returns the birth and death rate that would lead to the expected population
    size after 1 time unit, given that birth happens following an exponential
    distribution with rate parameter lambda, offset by c, and that death
    happens following and exponential distribution with rate parameter mu =
    lambda / birth_to_death_rate_ratio, where at the end we subsample each leaf
    with sampling_probability. We also condition on there not being extinction
    of the lineage.

    Complexity: O(1000 * log_2(10^6)) (binary search)

    Args:
        expected_population_size: The expected population size after 1 time
            unit.
        offset: The holding time before birth events can kick in.
        birth_to_death_rate_ratio: Ratio of birth to death rate.
        sampling_probability: Probability that a leaf is sampled at the end of
            the process.

    Returns:
        The birth and death ratesneeded to achieve the given population size
            under the birth-to-death-ratio constraint.
    """
    lo = 0.0
    hi = 100.0
    while lo < hi - 0.0001:
        mid = (lo + hi) / 2.0

        def get_expected_population_size(
            lam: float,
            offset: float,
            mu: float,
            sampling_probability: float,
        ) -> float:
            """
            The expected population size after 1 time unit of
            running a birth-death process where birth happens
            following an exponential distribution with parameter
            lam, offset by c, and death happens following an
            exponential distribution with parameter mu, and
            at the end we subsample each leaf with sampling_probability.
            We also condition on there not being extinction of the lineage.
            """
            T = 10000  # Discretization level
            dt = 1.0 / T
            c = int(T * offset)
            res = [sampling_probability]
            for t in range(1, T + 1):
                next_t = max(t - 1 - c, 0)
                res_t = (
                    dt
                    * lam
                    * 2.0
                    * np.exp(-mu * (t - next_t) / T)
                    * res[next_t]
                    + dt * mu * 0.0
                    + (1.0 - dt * lam - dt * mu) * res[t - 1]
                )
                res.append(res_t)
            p_ext = compute_extinction_probability(
                lam=lam,
                offset=offset,
                mu=mu,
                sampling_probability=sampling_probability,
            )
            return res[max(T - c, 0)] / (1.0 - p_ext)

        if (
            get_expected_population_size(
                lam=mid,
                offset=offset,
                mu=mid / birth_to_death_rate_ratio,
                sampling_probability=sampling_probability,
            )
            < expected_population_size
        ):
            lo = mid
        else:
            hi = mid
    return (lo, lo / birth_to_death_rate_ratio)


class BirthDeathFitnessSimulatorWithoutExtinction(BirthDeathFitnessSimulator):
    """
    Since BirthDeathFitnessSimulator can fail to simulate a tree due to full
    extinction, we need to retry.
    """

    def set_extinction_random_seed(self, random_seed):
        self._extinction_random_seed = random_seed

    def simulate_tree(self) -> CassiopeiaTree:
        np.random.seed(self._extinction_random_seed)
        while True:
            try:
                return super().simulate_tree()
            except TreeSimulatorError:
                # We need to retry
                print("TreeSimulatorError: Retrying ...")
                pass


def get_tree_simulator_from_config(
    tree_simulator_name: str,
    offset: float,
    bd_ratio: float,
    random_seed: int,
    verbose: bool = False,
) -> TreeSimulator:
    m = re.search(
        r"^(high|medium|low|neutral)_fitness"
        r"-(10|40|400|2000)_cells"
        r"-(1.0|0.1|0.01)_sampling_probability$",
        tree_simulator_name,
    )
    if m is not None:
        fitness_regime = str(m.group(1))
        n_cells = int(m.group(2))
        sampling_probability = float(m.group(3))

        birth_rate, death_rate = reverse_engineer_birth_and_death_rate(
            expected_population_size=n_cells,
            offset=offset,
            birth_to_death_rate_ratio=bd_ratio,
            sampling_probability=sampling_probability,
        )
        if verbose:
            print(f"Using birth_rate, death_rate = {birth_rate}, {death_rate}")

        def low_fitness_distribution():
            if fitness_regime == "high":
                return 0.1
            elif fitness_regime == "medium":
                return 0.05
            elif fitness_regime == "low":
                return 0.03
            elif fitness_regime == "neutral":
                return 0.0
            else:
                raise ValueError(f"Unknown fitness_regime: {fitness_regime}")

        def high_fitness_distribution():
            if fitness_regime == "high":
                return -1.1
            elif fitness_regime == "medium":
                return -0.7
            elif fitness_regime == "low":
                return -0.5
            elif fitness_regime == "neutral":
                return 0.0
            else:
                raise ValueError(f"Unknown fitness_regime: {fitness_regime}")

        def fitness_distribution():
            if np.random.uniform() < 0.9:
                return low_fitness_distribution()
            else:
                return high_fitness_distribution()

        if n_cells == 400:
            if sampling_probability == 1.0:
                expected_number_of_increased_fitness_changes = 8.0
            elif sampling_probability == 0.1:
                expected_number_of_increased_fitness_changes = 32.0
            elif sampling_probability == 0.01:
                expected_number_of_increased_fitness_changes = 256.0
            else:
                raise ValueError(
                    f"Unknown sampling_probability: {sampling_probability}"
                )
        elif n_cells == 2000:
            if sampling_probability == 1.0:
                expected_number_of_increased_fitness_changes = 24.0  # OK: Low fitness shows it's super reliable, with ~0.20 SR2
            elif sampling_probability == 0.1:
                expected_number_of_increased_fitness_changes = 96.0  # OK: Low fitness shows it's quite reliable, with ~0.15 SR2
            elif sampling_probability == 0.01:
                expected_number_of_increased_fitness_changes = (
                    512.0  # OK: Looks good.
                )
            else:
                raise ValueError(
                    f"Unknown sampling_probability: {sampling_probability}"
                )
        elif n_cells == 40 or n_cells == 10:
            assert fitness_regime == "neutral"
            expected_number_of_increased_fitness_changes = 8.0  # Doesn't matter
        else:
            raise ValueError(f"Unknown n_cells = {n_cells}")

        deleterious_prob = 0.9
        mutation_prob = (
            expected_number_of_increased_fitness_changes
            / n_cells
            * sampling_probability
            / (1.0 - deleterious_prob)
        )

        tree_simulator = BirthDeathFitnessSimulatorWithoutExtinction(
            birth_waiting_distribution=lambda scale: offset
            + np.random.exponential(scale),
            initial_birth_scale=1.0 / birth_rate,
            death_waiting_distribution=lambda: np.random.exponential(
                scale=1.0 / death_rate
            ),
            mutation_distribution=lambda: 1
            if np.random.uniform() < mutation_prob
            else 0,
            fitness_distribution=fitness_distribution,
            fitness_base=2.0,
            num_extant=int(n_cells / sampling_probability),
            experiment_time=None,
            collapse_unifurcations=True,
            random_seed=None,
        )
        tree_simulator.set_extinction_random_seed(random_seed)
    else:
        raise ValueError(f"Unknown tree simulator: {tree_simulator_name}")
    return tree_simulator


def ble_paper_tree_simulator(
    fitness: str,
    n_cells: int,
    sampling_probability: str,
    bd_ratio: float,
    offset: float,
    random_seed: int,
):
    return get_tree_simulator_from_config(
        tree_simulator_name=f"{fitness}_fitness-{n_cells}_cells-{sampling_probability}_sampling_probability",
        offset=offset,
        bd_ratio=bd_ratio,
        random_seed=random_seed,
    )
