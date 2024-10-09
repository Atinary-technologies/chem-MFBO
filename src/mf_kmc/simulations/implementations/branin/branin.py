"""Implementation of branin 2d function from BOTorch in a discrete setting.

Source: https://github.com/pytorch/botorch/blob/7c28a20037b67d2e753504519839cd05205ef85a/botorch/test_functions/multi_fidelity.py#L21
"""

from typing import Dict, Optional

import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin
from pydantic import BaseModel
from torch import Tensor

from mf_kmc.simulations.implementations.base import BaseSimulator


class BraninInput(BaseModel):
    """Class to take the inputs for Branin, take them as floats and transform them
    into a Tensor."""

    x0: float
    x1: float
    fidelity: float
    bias: float

    @property
    def torch_input(self) -> Tensor:

        simulation_bias = self.bias if self.fidelity != 1.0 else 1.0

        return torch.tensor(
            [self.x0, self.x1, simulation_bias], dtype=torch.double
        ).reshape(1, -1)


class Branin(AugmentedBranin):
    """Min max scaled Branin function with one level of fidelity"""

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False):
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        self._pis = "branin"
        self._ais = "branin_lowfid"
        if not self.negate:
            self._max_ = 308.12909601160663
            self._min_ = 0.3978873577401032
        else:
            self._min_ = -308.12909601160663
            self._max_ = -0.3978873577401032
        self._optimal_value = 1.0

    def __call__(self, X: Tensor) -> Tensor:
        H = super().evaluate_true(X)

        if self.negate:
            H = -H

        if X[:, -1] == 1.0:
            H = H + self.noise_std * torch.randn_like(H)

        return H


class BraninSimulator(BaseSimulator):
    """Simulate Branin function from BOTorch"""

    def _simulate(
        self, input_parameters: Dict[str, int], fidelity: Dict[str, float]
    ) -> Tensor:

        # get inputs
        inputs = {**input_parameters, **fidelity}

        # define bias (this is the real number affecting to the simulation)
        bias = self.config.bias_lowfid

        inputs['bias'] = bias

        # combine them into input
        branin_inp = BraninInput(**inputs)

        # get output
        sim = Branin(negate=True, noise_std=self.config.noise)

        out = sim(branin_inp.torch_input)

        # store result as a float on a dictionary
        result = {"result": float(out)}

        # computing time based on fidelity can be added

        return result
