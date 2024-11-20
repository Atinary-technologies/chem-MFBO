"""Single Fidelity optimizer, BOTorch based. Uses qExpectedImprovement

Definition:

https://arxiv.org/pdf/1712.00424.pdf
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)
from botorch.models import SingleTaskGP
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor

from chem_mfbo.optimization.optimizer_config import OptimizerConfig
from chem_mfbo.simulations.models.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleFidelityOptimizer:

    """Optimizer class for Single Fidelity Optimization BOTorch routines.

    The optimizer takes the same input than for Multi Fidelity routines but it
    only runs on highest fidelity.

    Acquisition function is qExpectedImprovement.

    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        optim_config: OptimizerConfig,
        cost_model: AffineFidelityCostModel,
        n_samples: int = 10,
    ) -> None:

        """Initialize optimizer."""

        self.config = sim_config
        self.optim_config = optim_config
        self.n_samples = n_samples
        self.cost_model = cost_model
        self.acquisition = optim_config.acquisition

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cpu"),
        }

    # define these properties from the config so they can be dynamically loaded if the config changes
    @property
    def name(self):
        return self.config.name

    @property
    def input_parameters(self):
        return self.config.input_parameters

    @property
    def measurements(self):
        return self.config.measurements

    @property
    def n_dimensions(self):
        """We suppose that there is only one fidelity level here."""
        return len(self.input_parameters)

    @property
    def input_names(self):
        return [param.name for param in self.input_parameters]

    @property
    def bounds(self):
        """Get bounds tensor (from input parameters)"""

        low_bound = []
        high_bound = []

        for param in self.input_parameters:
            low_bound.append(param.low_value)
            high_bound.append(param.high_value)

        bounds = torch.tensor([low_bound, high_bound], **self.tkwargs)

        return bounds

    @property
    def candidates(self):
        """Candidate points for acquisition function"""

        n_samps = 1000
        candidate_set = torch.rand(n_samps, self.bounds.size(1), **self.tkwargs)

        return candidate_set

    def _build_init_torch_data(
        self, model_inputs: List[Dict[str, float]]
    ) -> Tuple[Tensor, Tensor]:
        """Get torch tensors for GP model from dictionary containing experiments.
        We build the tensors by building the X variables (ignoring fidelity in this case).
        """

        df = pd.DataFrame(model_inputs)

        # get only X data (inputs)
        x_names = self.input_names

        # get y names
        y_names = [m.name for m in self.measurements]

        X_input = torch.tensor(df[x_names].values, **self.tkwargs)
        y_input = torch.tensor(df[y_names].values, **self.tkwargs)

        return X_input, y_input

    def _update_from_tensor(
        self, X: torch.Tensor
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Get new points in the structure of the inputs ready for the simulation engine.

        Take X tensor from the recommendation of the acquisition function and transforms it
        to the correct input structure. We add best fidelity from simulation by default.
        """

        X_list = X.detach().tolist()

        updated_inputs = []

        for data in X_list:
            input_dict = {name: data[i] for i, name in enumerate(self.input_names)}
            fidelity = {self.config.fidelity.name: 1.0}
            updated_inputs.append((input_dict, fidelity))

        return updated_inputs

    def compute_cost(self, samples: List[Tuple]) -> List[float]:
        """Compute cost from samples of simulations."""

        # convert list of tuples into list of dictionaries
        dict_samples = [{**tup[0], **tup[1]} for tup in samples]

        df = pd.DataFrame(dict_samples)

        # get only X data (inputs + fidelity)
        x_names = self.input_names
        x_names.append(self.config.fidelity.name)

        X_input = torch.tensor(df[x_names].values, **self.tkwargs)

        cost = self.cost_model(X_input).reshape(-1).detach().tolist()

        return cost

    def recommend(
        self,
        inputs: List[Tuple[Dict[str, Any]]] = None,
        batch_size: int = 2,
    ) -> List[Dict[str, float]]:

        """Recommend points based on candidates (surrogate update + af).

        Returns candidates for next iteration according to the acquisition function.

        """

        X_train, y_train = self._build_init_torch_data(inputs)

        # scaling
        # min max normalize X_train
        X_train = (X_train - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

        # get model and mll
        model = SingleTaskGP(
            X_train,
            y_train,
            covar_module=ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.n_dimensions,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            ),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(mll)

        # get acquisition and new candidates
        if self.acquisition == "ei":
            af = ExpectedImprovement(model, best_f=y_train.max())

        elif self.acquisition == "mes":
            af = qMaxValueEntropy(model=model, candidate_set=self.candidates)

        elif self.acquisition == "gibbon":
            af = qLowerBoundMaxValueEntropy(model=model, candidate_set=self.candidates)

        elif self.acquisition == "kg":
            af = qKnowledgeGradient(model, num_fantasies=self.optim_config.n_fantasies)

        new_x, _ = optimize_acqf(
            acq_function=af,
            bounds=torch.tensor(
                [[0] * self.n_dimensions, [1] * self.n_dimensions], **self.tkwargs
            ),
            # bounds=self.bounds,
            q=batch_size,
            num_restarts=self.optim_config.n_restarts,
            raw_samples=self.optim_config.n_raw_samples,
            options={
                "batch_limit": self.optim_config.batch_limit,
                "maxiter": self.optim_config.max_iter,
            },
        )

        # rescale
        new_x = new_x * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        new_x = self._update_from_tensor(new_x)

        return new_x
