"""Implementation of park function following this reference:
https://proceedings.neurips.cc/paper_files/paper/2020/file/60e1deb043af37db5ea4ce9ae8d2c9ea-Supplemental.pdf.

We modify the function to have a parameter that controls correlation.
"""

from typing import Dict, Optional

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from pydantic import BaseModel
from torch import Tensor

from mf_kmc.simulations.implementations.base import BaseSimulator


class ParkInput(BaseModel):
    """Class to take the inputs for Park, take them as floats and transform them
    into a Tensor. We use bias as the equivalent factor for r2 in our simulation."""

    x0: float
    x1: float
    x2: float
    x3: float
    fidelity: float
    bias: float

    @property
    def torch_input(self) -> Tensor:

        simulation_bias = self.bias if self.fidelity != 1.0 else 1.0

        return torch.tensor(
            [self.x0, self.x1, self.x2, self.x3, simulation_bias], dtype=torch.double
        ).reshape(1, -1)


class Park(SyntheticTestFunction):
    """Modified Park function with one level of fidelity.
    The low-fidelity function can be modulated depending on the fidelity
    dimension in the tensor (when 1.0, the function is high fidelity)"""

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False):
        self.dim = 4
        # last dimension is fidelity
        self._bounds = [(0.0001, 1), (0, 1), (0, 1), (0, 1)]
        self._optimizers = [1, 1, 1, 1]
        self._max_ = 25.5893
        self._min_ = 0
        self._optimal_value = 25.5893
        self._pis = "park"
        self._ais = "park_low"
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass, X should be 4D and have an extra dimension for the fidelity."""

        term1 = (
            X[:, 0]
            / 2
            * (
                torch.sqrt(1 + ((X[:, 1]) + X[:, 2] ** 2) * X[:, 3] / (X[:, 0] ** 2))
                - 1
            )
        )
        term2 = (X[:, 0] + (3 - 4 * (1 - X[:, 4])) * X[:, 3]) * torch.exp(
            1 + torch.sin(X[:, 2])
        )

        return term1 + term2


class ParkSimulator(BaseSimulator):
    """Simulate Park function from BOTorch"""

    def _simulate(
        self, input_parameters: Dict[str, int], fidelity: Dict[str, float]
    ) -> Tensor:

        # get inputs
        inputs = {**input_parameters, **fidelity}

        # define bias (this is the real number affecting the simulation, probably there is a better way
        # of defining it)
        bias = self.config.bias_lowfid

        inputs['bias'] = bias

        # combine them into input
        park_inp = ParkInput(**inputs)

        # get output
        sim = Park(noise_std=self.config.noise)

        out = sim(park_inp.torch_input)

        # store result as a float on a dictionary
        result = {"result": float(out)}

        return result
