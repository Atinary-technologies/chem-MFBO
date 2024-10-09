"""Generate dataset for several simulations at different levels of fidelity."""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from mf_kmc.simulations import config
from mf_kmc.simulations.implementations.factory import sim_factory


def compute_points(
    sim_module_name: str, n_points: int, fidelity: float = None
) -> pd.DataFrame:
    """Compute points randomly sampling for the different input parameters and the different levels of fidelity to
    build a dataset for multifidelity regression.
    """

    config.init_config()

    sim_config = config.get_simulation_config(sim_module_name)

    simulation = sim_factory(sim_config)

    fidelities = [0.2, 0.4, 0.6, 0.8, 1]

    # if ising, remove first level because it is most likely uninformative
    if sim_module_name == "ising":
        fidelity_levels = fidelity_levels[1:]

    points = []

    for _ in range(n_points):

        data = {}

        for param in sim_config.input_parameters:
            # only considering numerical values
            low = param.low_value
            high = param.high_value
            value = random.uniform(low, high)
            data[param.name] = value

        for fid in fidelities:
            points.append((data, {"fidelity": fid}))

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

    SIMULATIONS = ["hartmann", "branin", "rosenbrock"]

    for simulation in SIMULATIONS:

        df = compute_points(simulation, 50)

        results_dir = Path(f"data/regression_datasets")

        if not results_dir.exists():
            os.makedirs(results_dir, exist_ok=True)

        df.to_csv(results_dir.joinpath(f"{simulation}.csv"))
