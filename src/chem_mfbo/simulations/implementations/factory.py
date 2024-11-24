"""Import simulator dynamically from config.py."""

from chem_mfbo.simulations.config import SimulationConfig
from chem_mfbo.simulations.implementations.base import BaseSimulator
from chem_mfbo.simulations.implementations.branin.branin import BraninSimulator
from chem_mfbo.simulations.implementations.hartmann.hartmann import HartmannSimulator
from chem_mfbo.simulations.implementations.park.park import ParkSimulator


def sim_factory(simulation_config: SimulationConfig) -> BaseSimulator:
    """Reading config file module name to load simulator"""

    if simulation_config.module_name == "hartmann":
        return HartmannSimulator(simulation_config)

    elif simulation_config.module_name == "branin":
        return BraninSimulator(simulation_config)

    elif simulation_config.module_name == "park":
        return ParkSimulator(simulation_config)
