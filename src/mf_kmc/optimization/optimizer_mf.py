"""Multi Fidelity optimizer, BOTorch based.

Check: https://botorch.org/tutorials/discrete_multi_fidelity_bo
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim.optimize import optimize_acqf_mixed
from torch import Tensor
from torch.quasirandom import SobolEngine

from mf_kmc.optimization.acquisition import (
    CostMultiFidelityEI,
    get_full_fidelity_recommendation,
    get_mfgibbon,
    get_mfkg,
    get_mfmes,
)
from mf_kmc.optimization.model import initialize_model
from mf_kmc.optimization.optimizer_config import OptimizerConfig
from mf_kmc.simulations.models.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MultiFidelityOptimizer:

    """Optimizer class for Multi-Fidelity Optimization BOTorch routines.

    TO DO: Add information about the algorithm.

    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        optim_config: OptimizerConfig,
        target_fids: Dict[int, int],
        cost_model: AffineFidelityCostModel,
        cost_aware_utility: InverseCostWeightedUtility,
        n_samples: int = 10,
    ) -> None:

        """Initialize optimizer."""

        self.config = sim_config
        self.optim_config = optim_config
        self.target_fids = target_fids
        self.cost_model = cost_model
        self.cost_aware_utility = cost_aware_utility
        self.n_samples = n_samples
        self.acquisition = optim_config.acquisition
        self.gp = optim_config.gp
        self.multitask = True if self.gp == "multitask" else False

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cpu"),
        }

    # define these properties from the config so they can be dynamically loaded if the config changes
    @property
    def fidelity(self):
        return self.config.fidelity

    @property
    def name(self):
        return self.config.name

    @property
    def fidelity_dimension(self):
        return len(self.config.input_parameters)

    @property
    def input_parameters(self):
        return self.config.input_parameters

    @property
    def measurements(self):
        return self.config.measurements

    @property
    def n_dimensions(self):
        """We suppose that there is only one fidelity level here."""
        return len(self.input_parameters) + 1

    @property
    def input_names(self):
        return [param.name for param in self.input_parameters]

    @property
    def bounds(self):
        """Get bounds tensor (from input parameters and fidelity)"""

        low_bound = []
        high_bound = []

        for param in self.input_parameters:
            low_bound.append(param.low_value)
            high_bound.append(param.high_value)

        # append fidelity (probably it needs refactor)
        low_bound.append(0)
        high_bound.append(1)

        bounds = torch.tensor([low_bound, high_bound], **self.tkwargs)

        return bounds

    @property
    def features_list(self):
        """Get fixed features list for the acquisition function."""

        levels = self.config.normalized_fidelities

        features_list = [{self.fidelity_dimension: level} for level in levels]

        return features_list

    @property
    def candidates(self):
        """Candidate points for acquisition function"""

        n_samps = 1000

        sampler = SobolEngine(dimension=self.n_dimensions - 1)
        candidates = sampler.draw(n_samps)
        candidate_set = torch.cat((candidates, torch.ones(n_samps, 1)), dim=1)

        return candidate_set

    def _build_init_torch_data(
        self, model_inputs: List[Dict[str, float]]
    ) -> Tuple[Tensor, Tensor]:
        """Get torch tensors for GP model from dictionary containing experiments.
        We build the tensors by building the X variables
        """

        df = pd.DataFrame(model_inputs)

        # get only X data (inputs + fidelity)
        x_names = self.input_names
        x_names.append(self.fidelity.name)

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
        to the correct input structure.
        """

        X_list = X.detach().tolist()

        updated_inputs = []

        for data in X_list:
            input_dict = {name: data[i] for i, name in enumerate(self.input_names)}
            fidelity_dict = {self.fidelity.name: data[-1]}
            updated_inputs.append((input_dict, fidelity_dict))

        return updated_inputs

    def compute_cost(self, samples: List[Tuple]) -> List[float]:
        """Compute cost from samples of simulations."""

        # convert list of tuples into list of dictionaries
        dict_samples = [{**tup[0], **tup[1]} for tup in samples]

        df = pd.DataFrame(dict_samples)

        # get only X data (inputs + fidelity)
        x_names = self.input_names
        x_names.append(self.fidelity.name)

        X_input = torch.tensor(df[x_names].values, **self.tkwargs)

        cost = self.cost_model(X_input).reshape(-1).detach().tolist()

        return cost

    def recommend(
        self,
        inputs: List[Dict[str, Any]] = None,
        batch_size: int = 2,
        use_full_fidelity: bool = False,
    ) -> List[Dict[str, float]]:

        """Recommend points based on candidates (surrogate update + acquisition function optimization).

        It returns the candidate points.

        """

        X_train, y_train = self._build_init_torch_data(inputs)

        # min max normalize X_train
        X_train = (X_train - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

        # get model and mll
        mll, model = initialize_model(
            X_train, y_train, self.fidelity_dimension, multitask=self.multitask
        )

        fit_gpytorch_mll(mll)

        # if we are on full fidelity, we want a full fidelity evaluation
        if use_full_fidelity:

            new_x = get_full_fidelity_recommendation(
                model,
                self.bounds,
                self.n_dimensions,
                [self.fidelity_dimension],
                [1.0],
            )

            new_x = self._update_from_tensor(new_x)

        # else we apply MF proposal
        else:

            if self.acquisition == "mfkg":
                # get knowledge gradient AF
                af = get_mfkg(
                    model,
                    self.bounds,
                    self.cost_aware_utility,
                    self.target_fids,
                    self.n_dimensions,
                    [self.fidelity_dimension],
                    [1.0],
                    self.optim_config.n_fantasies,
                )

            elif self.acquisition == "mfmes":

                af = get_mfmes(
                    model,
                    self.candidates,
                    self.cost_aware_utility,
                    self.target_fids,
                    self.optim_config.n_fantasies,
                    multitask=self.multitask,
                )

            elif self.acquisition == "mfgibbon":

                af = get_mfgibbon(
                    model,
                    self.candidates,
                    self.cost_aware_utility,
                    self.target_fids,
                    self.optim_config.n_fantasies,
                )

            elif self.acquisition == "mfei":

                # get max value at highest fidelity
                y_max = y_train[X_train[..., -1] == 1.0].max().item()

                af = CostMultiFidelityEI(
                    model,
                    best_f=y_max,
                    cost_model=self.cost_model,
                    multitask=self.multitask,
                )

            # optimize AF
            candidates, _ = optimize_acqf_mixed(
                acq_function=af,
                bounds=torch.tensor(
                    [[0] * self.n_dimensions, [1] * self.n_dimensions],
                    **self.tkwargs,
                ),
                fixed_features_list=self.features_list,
                q=batch_size,
                num_restarts=self.optim_config.n_restarts,
                raw_samples=self.optim_config.n_raw_samples,
                options={
                    "batch_limit": self.optim_config.batch_limit,
                    "maxiter": self.optim_config.max_iter,
                },
            )

            # observe new values
            new_x = candidates.detach()

            # inverse transform
            new_x = new_x * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

            # we get new inputs in the form we want
            new_inputs = self._update_from_tensor(new_x)
            logger.info(f"candidates:\n{new_inputs}\n")

        return new_inputs
