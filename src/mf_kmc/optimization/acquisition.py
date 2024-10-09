# SAME THING WITH THE ACQUISITION (get_mfkg)
from typing import Dict, List, Tuple

import torch
from botorch.acquisition import ExpectedImprovement, PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.optim.optimize import optimize_acqf
from torch import Tensor


class CostMultiFidelityEI(ExpectedImprovement):
    """Adapted version of EI for MF BO inspired from MFSKO paper.

    ref: https://link.springer.com/content/pdf/10.1007/s00158-005-0587-0.pdf"""

    def __init__(self, model, best_f, cost_model, **kwargs):
        super().__init__(model=model, best_f=best_f, **kwargs)
        self.cost_model = cost_model
        self.af_old = ExpectedImprovement(self.model, best_f=self.best_f)

    def _compute_correlation(self, X: torch.Tensor) -> torch.Tensor:
        """Compute correlation between methods"""
        X = X.squeeze(1)

        X_hf = torch.clone(X)

        X_hf[..., -1] = 1.0

        X_all = torch.cat((X, X_hf), dim=0)

        # variance for each fidelity
        var_hf = torch.flatten(self.model.posterior(X_hf).variance)
        var = torch.flatten(self.model.posterior(X).variance)

        # posterior covariance
        cov = torch.diag(
            self.model(X_all).covariance_matrix[: X.size()[0], X.size()[0] :]
        )

        corr = cov / (torch.sqrt(var) * torch.sqrt(var_hf))

        return corr

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # compute expected improvement at highest fidelity only
        X_hf = torch.clone(X)

        X_hf[..., -1] = 1

        ei = super().forward(X_hf).reshape(-1, 1)

        # compute correlation between HF and LF levels
        correlation = self._compute_correlation(X).reshape(-1, 1).detach()

        # compute cost and cost ratio
        costs = self.cost_model(X)

        cost_ratio = 1 / costs

        # combine everyhting to get good function
        alpha = ei * cost_ratio * correlation

        return alpha.flatten()


class IsingCostModel(AffineFidelityCostModel):

    """Class to model specific Ising cost according to the polynomial fit of
    the fidelities."""

    def __init__(self, fidelity_weights: Dict[int, float]):
        super().__init__(fidelity_weights, fixed_cost=0)

    def forward(self, X: Tensor):

        lin_cost = torch.einsum(
            "...f,f", X[..., self.fidelity_dims], self.weights.to(X)
        )
        #fids = X[..., -1]
        final_cost = lin_cost**2.6 * 171.38 - 4.9323 * lin_cost + 0.8122

        return final_cost


class AffineModifiedCostModel(AffineFidelityCostModel):

    """Class to model cost for two levels of fidelities based on
    a cost ratio. For the moment it only works with two levels of
    fidelities.
    """

    def __init__(
        self,
        fidelity_weights: Dict[int, float],
        fixed_cost: float = 0,
        cost_ratio: float = 0.1,
    ):
        super().__init__(fidelity_weights, fixed_cost)
        self.cost_ratio = cost_ratio

    def forward(self, X: Tensor) -> Tensor:

        fids = X[..., -1]

        final_cost = torch.where(fids != 1.0, self.cost_ratio, fids)

        return final_cost


def get_cost_function(
    cost_model: str, fidelity_weights: Dict[int, float], cost_ratio: float = None
) -> Tuple[AffineFidelityCostModel, InverseCostWeightedUtility]:
    """Gets associated cost utility from a given simulation type. We have to implement
    a specific cost function for each simulation type.
    """

    if cost_model == "affine":

        cost_model = AffineModifiedCostModel(
            fidelity_weights=fidelity_weights, fixed_cost=0, cost_ratio=cost_ratio
        )
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        return cost_model, cost_aware_utility

    elif cost_model == "ising":
        cost_model = IsingCostModel(fidelity_weights=fidelity_weights)

        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        return cost_model, cost_aware_utility


def get_mfkg(
    model: SingleTaskMultiFidelityGP,
    bounds: Tensor,
    cost_function: InverseCostWeightedUtility,
    target_fidelities: Dict[int, float],
    dimensions: int,
    columns: List[int],
    values: List[int],
    n_fantasies: int,
) -> qMultiFidelityKnowledgeGradient:
    def _project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=dimensions,
        columns=columns,
        values=values,
    )

    # Hyperparameters here are fixed
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 10, "maxiter": 200},
    )

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=n_fantasies,
        current_value=current_value,
        cost_aware_utility=cost_function,
        project=_project,
    )


def get_full_fidelity_recommendation(
    model: SingleTaskMultiFidelityGP,
    bounds: Tensor,
    dimensions: int,
    columns: List[int],
    values: List[int],
) -> Tensor:
    """Get full fidelity recommendation from a fitted GP with acquired points at
    different fidelity levels.
    """

    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=dimensions,
        columns=columns,
        values=values,
    )

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "max_iter": 200},
    )

    final_rec = rec_acqf._construct_X_full(final_rec)

    return final_rec


def get_mfmes(
    model: SingleTaskMultiFidelityGP,
    candidates: Tensor,
    cost_function: InverseCostWeightedUtility,
    target_fidelities: Dict[int, float],
    n_fantasies: int,
) -> qMultiFidelityMaxValueEntropy:
    """Get Multi FIdelity Max Value Entropy acquisition function"""

    def _project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    return qMultiFidelityMaxValueEntropy(
        model=model,
        num_fantasies=n_fantasies,
        cost_aware_utility=cost_function,
        project=_project,
        candidate_set=candidates,
    )


def get_mfgibbon(
    model: SingleTaskMultiFidelityGP,
    candidates: Tensor,
    cost_function: InverseCostWeightedUtility,
    target_fidelities: Dict[int, float],
    n_fantasies: int,
) -> qMultiFidelityLowerBoundMaxValueEntropy:
    """Get Multi Fidelity GIBBON acquisition function"""

    def _project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    return qMultiFidelityLowerBoundMaxValueEntropy(
        model=model,
        num_fantasies=n_fantasies,
        cost_aware_utility=cost_function,
        project=_project,
        candidate_set=candidates,
    )
