"""Test simulators are working properly."""
import pytest
import torch

from mf_kmc.simulations import config
from mf_kmc.simulations.config import get_simulation_config
from mf_kmc.simulations.implementations.factory import sim_factory


@pytest.fixture(scope="session", autouse=True)
def get_config():
    config.init_config()
    return config.CFG


@pytest.fixture(autouse=True)
def seed_experiments():
    torch.manual_seed(33)


@pytest.mark.parametrize(
    "name, description, inputs, fidelity, output",
    [
        (
            "hartmann",
            "maximum",
            {
                "x0": 0.20169,
                "x1": 0.150011,
                "x2": 0.476874,
                "x3": 0.275332,
                "x4": 0.311652,
                "x5": 0.6573,
            },
            {"fidelity": 1},
            3.3224,
        ),
        (
            "branin",
            "maximum",
            {"x0": -3.141592, "x1": 12.275},
            {"fidelity": 1.0},
            -0.3979,
        ),
        (
            "branin",
            "maximum_lowfid",
            {"x0": 3.141592, "x1": 1.7815197},
            {"fidelity": 0.8},
            -0.4369,
        ),
        (
            "park",
            "maximum_highfid",
            {"x0": 1, "x1": 1, "x2": 1, "x3": 1},
            {"fidelity": 1},
            25.5893,
        ),
        (
            "park",
            "maximum_lowfid",
            {"x0": 1, "x1": 1, "x2": 1, "x3": 1},
            {"fidelity": 0.4},
            11.9687,
        ),
    ],
)
def test_simulations(name, description, inputs, fidelity, output):
    """Testing different cases in the simulation"""

    sim_config = get_simulation_config(name)

    sim = sim_factory(sim_config)

    sim_output = sim.simulate(inputs, fidelity)

    results = list(sim_output.values())[0]

    assert (
        round(results, 4) == output
    ), f"wrong result for {name} simulation {description} with {inputs}: {output}"
