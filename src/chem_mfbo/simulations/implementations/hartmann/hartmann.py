"""Implementation of Hartmann 6d function from BOTorch documentation. In this an the other synthetic functions, the fidelity
dimension distorts the value of the real function by substracting or adding a certain quantity that is smaller the closer you are
to the best fidelity.

Source: https://botorch.org/tutorials/discrete_multi_fidelity_bo
"""


from typing import Dict, Optional

import torch
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from pydantic import BaseModel
from torch import Tensor

from chem_mfbo.simulations.implementations.base import BaseSimulator
from chem_mfbo.simulations.models.simulation_config import SimulationConfig


class HartmannInput(BaseModel):
    """Class to take the inputs for Hartmann function, take them as floats and transform them
    into a Tensor."""

    x0: float
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    fidelity: float
    bias: float

    @property
    def torch_input(self) -> Tensor:

        simulation_bias = self.bias if self.fidelity != 1.0 else 1.0

        return torch.tensor(
            [self.x0, self.x1, self.x2, self.x3, self.x4, self.x5, simulation_bias],
            dtype=torch.double,
        )


class Hartmann(AugmentedHartmann):
    """Min max scaled Hartmann multi fidelity simulation"""

    def __init__(self, noise_std: Optional[float] = None):
        super().__init__(noise_std=noise_std)
        self._max_ = 3.322367993160791
        self._min_ = 0.0
        self._optimal_value = 1.0
        self._pis = "hartmann"
        self._ais = "hartmann_low"

    def __call__(self, X: Tensor) -> Tensor:
        H = -super().evaluate_true(X)  # non-negated is not interesting objective

        if X[-1] == 1.0:
            H = H + self.noise_std * torch.randn_like(H)

        return H


class HartmannSimulator(BaseSimulator):
    """Simulate Hartmann function from BOTorch."""

    def _simulate(
        self, input_parameters: Dict[str, int], fidelity: Dict[str, float]
    ) -> Tensor:

        # get inputs
        inputs = {**input_parameters, **fidelity}

        # define bias (this is the real number affecting to the simulation)
        bias = self.config.bias_lowfid

        inputs['bias'] = bias

        # combine them into input
        hartmann_inp = HartmannInput(**inputs)

        # get output
        sim = Hartmann(noise_std=self.config.noise)

        out = sim(hartmann_inp.torch_input)

        # store result as a float on a dictionary
        result = {"result": float(out)}

        return result
