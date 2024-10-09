# Base class for the simulations
from abc import ABC, abstractmethod
from typing import Any, Dict

from mf_kmc.simulations.models.simulation_config import SimulationConfig


class BaseSimulator(ABC):
    def __init__(self, config: SimulationConfig) -> None:

        self.config = config

    @classmethod
    @abstractmethod
    def _simulate(
        self, input_parameters: Dict[str, Any], fidelity: Dict[str, Any]
    ) -> None:
        """Specific modifications required to run a given type of simulation"""
        pass

    def simulate(
        self, input_parameters: Dict[str, Any], fidelity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Method to run the simulation using _simulate(), from inputs x get output y.
        Take input_parameters from SimulationConfig"""

        outs = self._simulate(input_parameters, fidelity)

        return outs
