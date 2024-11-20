"""Generate dataset for several simulations at different levels of fidelity."""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from chem_mfbo.simulations import config
from chem_mfbo.simulations.implementations.factory import sim_factory


def compute_round_samples(levels: list, n_samples: int) -> np.array:
    """Given a list of fidelity levels and a number of samples, return an array
    with the reciprocal number of samples for each level of fidelity
    """

    reciprocals = 1.0 / np.array(levels)

    # Normalize these reciprocals so that they sum to 1
    weights = reciprocals / np.sum(reciprocals)

    # Calculate raw sample counts
    raw_counts = weights * n_samples

    # Round counts to nearest integer
    rounded_counts = np.round(raw_counts).astype(int)

    # Adjust rounding errors if necessary
    while np.sum(rounded_counts) != n_samples:
        if np.sum(rounded_counts) < n_samples:
            # Add to the smallest rounding error
            differences = raw_counts - rounded_counts
            index_to_increment = np.argmax(differences)
            rounded_counts[index_to_increment] += 1
        elif np.sum(rounded_counts) > n_samples:
            # Subtract from the largest rounding error
            differences = rounded_counts - raw_counts
            index_to_decrement = np.argmax(differences)
            rounded_counts[index_to_decrement] -= 1

    return rounded_counts


def compute_points(sim_module_name: str, n_points: int) -> pd.DataFrame:
    """Compute points randomly sampling for the different input parameters and the different levels of fidelity to
    build a dataset for multifidelity regression.
    """

    config.init_config()

    sim_config = config.get_simulation_config(sim_module_name)

    simulation = sim_factory(sim_config)

    fidelity_levels = sim_config.normalized_fidelities

    # if ising, remove first level because it is most likely uninformative
    if sim_module_name == "ising":
        fidelity_levels = fidelity_levels[1:]

    # create points to sample
    samples = [10, 10]

    points = []

    for i, sample in enumerate(samples):

        for _ in range(sample):
            data = {}
            fid = {}

            fid[sim_config.fidelity.name] = fidelity_levels[i]

            for param in sim_config.input_parameters:
                # only considering numerical values
                low = param.low_value
                high = param.high_value
                value = random.uniform(low, high)
                data[param.name] = value

            points.append((data, {"fidelity": fidelity_levels[0]}))
            points.append((data, {"fidelity": fidelity_levels[1]}))

    # # generate extra hf for testing
    # for _ in range(samples[-1]):
    #     data = {}
    #     fid = {}

    #     fid[sim_config.fidelity.name] = fidelity_levels[-1]

    #     for param in sim_config.input_parameters:
    #         # only considering numerical values
    #         low = param.low_value
    #         high = param.high_value
    #         value = random.uniform(low, high)
    #         data[param.name] = value

    #     points.append((data, fid))

    results_list = Parallel(n_jobs=-1)(
        delayed(simulation.simulate)(data, fid) for data, fid in points
    )

    final_dataset = [
        {**inp, **fid, **res} for (inp, fid), res in zip(points, results_list)
    ]

    dataset = pd.DataFrame(final_dataset)

    return dataset


if __name__ == "__main__":

    random.seed(33)
    torch.manual_seed(33)

    results_dir = Path("data/regression_datasets/")

    if not results_dir.exists():
        os.makedirs(results_dir, exist_ok=True)

    SIMULATIONS = ["hartmann", "branin", "rosenbrock", "park"]

    for simulation in SIMULATIONS:
        df = compute_points(simulation, 50)
        df.to_csv(results_dir.joinpath(f"{simulation}.csv"))
