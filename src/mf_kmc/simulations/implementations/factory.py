"""Import simulator dynamically from config.py."""

from mf_kmc.simulations.config import SimulationConfig
from mf_kmc.simulations.implementations.base import BaseSimulator
from mf_kmc.simulations.implementations.branin.branin import BraninSimulator
from mf_kmc.simulations.implementations.hartmann.hartmann import HartmannSimulator
from mf_kmc.simulations.implementations.park.park import ParkSimulator


def sim_factory(simulation_config: SimulationConfig) -> BaseSimulator:
    """Reading config file module name to load simulator"""

    if simulation_config.module_name == "hartmann":
        return HartmannSimulator(simulation_config)

    elif simulation_config.module_name == "branin":
        return BraninSimulator(simulation_config)

    elif simulation_config.module_name == "park":
        return ParkSimulator(simulation_config)
