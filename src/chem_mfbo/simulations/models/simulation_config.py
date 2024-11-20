# Shows list in simulation_config.yml using Pydantic
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class Parameter(BaseModel):
    name: str
    high_value: Optional[Union[float, int]] = None
    low_value: Optional[Union[float, int]] = None
    options: Optional[List[str]] = None


class Measurement(BaseModel):
    name: str
    high_value: Optional[Union[float, int]]
    low_value: Optional[Union[float, int]]


class FidelityType(str, Enum):
    discrete = 'discrete'
    continuous = 'continuous'


class CostModel(str, Enum):
    affine = "affine"
    ising = "ising"


class Fidelity(BaseModel):
    name: str
    fidelity_type: FidelityType
    best_fidelity: Union[int, float]
    options: List[Union[int, float]] = None
    high_value: float = None
    low_value: float = None


class SimulationConfig(BaseModel):
    name: str
    module_name: str
    input_parameters: List[Parameter]
    measurements: List[Measurement]
    fidelity: Fidelity
    bias_lowfid: float
    objectives: List[Dict[str, str]]
    cost_model: CostModel
    noise: Union[float, None]

    @property
    def normalized_fidelities(self):
        """Normalized fidelities to the [0, 1] interval to use internally in the
        optimizers"""
        if self.fidelity.fidelity_type == "discrete":

            options = self.fidelity.options

            norm_options = [opt / max(options) for opt in options]

            return norm_options

    @property
    def fidelities_map(self):
        "Dictionary mapping fidelities in the [0, 1] space to the original space"

        fids_map = {
            norm: orig
            for norm, orig in zip(self.normalized_fidelities, self.fidelity.options)
        }

        return fids_map
