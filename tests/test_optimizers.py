"""Test and verify optimizers recommendations."""

import random

import numpy as np
import pytest
import torch

from chem_mfbo.optimization.factory import optim_factory
from chem_mfbo.simulations import config
from chem_mfbo.simulations.config import get_optimizer_config, get_simulation_config


@pytest.mark.parametrize(
    "name, optimizer_name, cost_ratio, experiments, output, seed",
    [
        (
            "hartmann",
            "single_fidelity_ei",
            0.1,
            [
                {
                    'x0': 0.5331155450215787,
                    'x1': 0.1579747759624835,
                    'x2': 0.5881093208036141,
                    'x3': 0.051599383868120074,
                    'x4': 0.20833997653826203,
                    'x5': 0.9798908895476997,
                    'fidelity': 1.0,
                    'result': 0.20935234323230525,
                    'step': 0,
                    'cost': 1.0,
                },
                {
                    'x0': 0.144256587941121,
                    'x1': 0.7154870728683209,
                    'x2': 0.08702138891480038,
                    'x3': 0.7339197632628279,
                    'x4': 0.5933779523179038,
                    'x5': 0.0014948534637541089,
                    'fidelity': 1.0,
                    'result': 0.17535542669340848,
                    'step': 0,
                    'cost': 1.0,
                },
            ],
            [
                {
                    'x0': 0.702712441364793,
                    'x1': 0.08785828645070609,
                    'x2': 0.7959529124535719,
                    'x3': 0.00765606064650572,
                    'x4': 0.4935586165791415,
                    'x5': 0.8334799761131669,
                },
                {'fidelity': 1.0},
            ],
            0,
        )
    ],
)
def test_optimizer(name, optimizer_name, cost_ratio, experiments, output, seed):

    config.init_config()

    # Load data from .csv in assets
    sim_config = get_simulation_config(name)

    optim_config = get_optimizer_config(optimizer_name)

    # Load optimizer
    optimizer = optim_factory(optim_config, sim_config, cost_ratio)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # get recommendations
    new_points = optimizer.recommend(inputs=experiments, batch_size=1)

    truth_value = torch.tensor(list(output[0].values()))

    new_value = torch.tensor(list(new_points[0][0].values()))

    assert torch.isclose(
        truth_value, new_value
    ).all(), f"Output not matching recommendation {experiments}: {output}"
