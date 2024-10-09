"""Script to run Reizmann Suzuki MF simulation"""
import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models import SingleTaskGP
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf_mixed
from botorch.optim.optimize import optimize_acqf
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from joblib import Parallel, delayed
from scipy.stats.qmc import LatinHypercube
from summit import *
from summit.benchmarks import get_pretrained_reizman_suzuki_emulator
from torch import Tensor
from torch.quasirandom import SobolEngine

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class ReizmanSuzuki:
    """Class to get result from Reizmann Suzuki. Take tensor
    as input."""

    def __init__(self, bias: float = 0) -> None:

        self.experiment = get_pretrained_reizman_suzuki_emulator(case=1)
        self.domain = self.experiment.domain
        self.bias = bias

    def _encode_tensor(self, X: torch.Tensor) -> DataSet:
        """Encode tensor to dictionary"""

        categorical = self.domain.get_categorical_combinations()

        # this is for the categorical variables
        catalyst_dict = {
            i: value for i, value in enumerate(categorical['catalyst'].values)
        }

        # create dictionary to compute inputs
        new = {}

        new[('catalyst', 'DATA')] = [catalyst_dict[torch.argmax(X[:8]).item()]]
        new[('t_res', 'DATA')] = [X[8]]
        new[('temperature', 'DATA')] = [X[9]]
        new[('catalyst_loading', 'DATA')] = [X[10]]

        dataset = DataSet(new)

        return dataset

    def _simulate(self, X: torch.Tensor) -> torch.Tensor:
        """Get non distorted result"""

        inp = self._encode_tensor(X)

        result = self.experiment.run_experiments(inp)

        output = torch.tensor(result['yld'].values[0])

        return output

    def _compute_result(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the result for a single input tensor"""
        y = self._simulate(X)

        if X[11] != 1.0:
            y = y + torch.abs(torch.normal(0, self.bias, size=y.size()))
            y = y.clamp(0, 100)

        return y.clamp(0, 100)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Get final result depending on the fidelity."""

        if X.dim() == 1:
            # If X is a single 1D tensor, handle it directly
            return self._compute_result(X)
        else:
            # If X is a batch of 2D tensors, process each row
            results = []
            for i in range(X.shape[0]):
                y = self._compute_result(X[i])
                results.append(y)
            return torch.stack(results).type(tkwargs['dtype'])


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


class AffineModifiedCostModel(AffineFidelityCostModel):

    """Class to model cost for two levels of fidelities based on
    a cost ratio. For the moment it only works with two levels of
    fidelities.
    """

    def __init__(
        self,
        cost_ratio: float = 0.1,
    ):
        super().__init__()
        self.cost_ratio = cost_ratio

    def forward(self, X: Tensor) -> Tensor:

        fids = X[..., -1]

        final_cost = torch.where(fids != 1.0, self.cost_ratio, fids)

        return final_cost


def get_bounds(dom):
    """Get sampling of initial points from a given domain"""

    parameters = dom.to_dict()

    low_bounds = []
    high_bounds = []

    num_cats = []

    for p in parameters:
        if p["is_objective"]:
            continue

        else:
            if p["type"] == "ContinuousVariable":
                low_bounds.append(p["bounds"][0])
                high_bounds.append(p["bounds"][1])

            else:
                num_cats.append(len(p['levels']))

    bounds = torch.tensor([low_bounds, high_bounds])

    return bounds, num_cats


def sample_tensors(bound, num_cats, n_samps, seed):

    sampler = LatinHypercube(d=len(bound[0]), seed=seed)

    samples = sampler.random(n=n_samps)

    scaled = (bound[1] - bound[0]) * samples + bound[0]

    cats = torch.randint(0, num_cats[0], (n_samps,))

    one_hot = torch.nn.functional.one_hot(cats, num_classes=num_cats[0])

    samples = torch.cat((one_hot, scaled), axis=1)

    return samples


def _project(X):
    return project_to_target_fidelity(X=X, target_fidelities={-1: 1.0})


def run_experiment(
    mode: str = "mf",
    lowfid: float = 0.7,
    seed: int = 33,
    af_name: str = "MES",
    budget: int = 5,
    bias: float = 10.0,
    cost_ratio: float = 0.1,
):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # here summit simulation
    reizman = ReizmanSuzuki(bias=bias)

    cost_model = AffineModifiedCostModel(cost_ratio=cost_ratio)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    # Initial sampling
    n_low = 10
    n_high = 2

    if mode == "sf":
        n_high = 3

    domain = reizman.domain

    bound, num_cats = get_bounds(domain)

    samps_hf = sample_tensors(bound, num_cats, n_high, seed)
    samps_lf = sample_tensors(bound, num_cats, n_low, seed)

    samps_hf = torch.cat((samps_hf, torch.ones((n_high, 1))), axis=1)
    samps_lf = torch.cat((samps_lf, torch.ones((n_low, 1)) * lowfid), axis=1)

    X_init = torch.concatenate((samps_hf, samps_lf), axis=0)

    y_init = reizman(X_init).unsqueeze(1).type(torch.float64)

    init_costs = cost_model(X_init)

    init_cost = sum(init_costs).item()

    steps = [0] * len(y_init)

    # if single fidelity, take only the tensors where X[-1] == 1
    # TO DO
    if mode == "sf":
        X_init = X_init[:n_high, :-1]
        steps = steps[:n_high]
        y_init = y_init[:n_high]

    step = 0

    while init_cost < budget:

        step += 1

        # normalize inputs
        X_init[:, 8:11] = (X_init[:, 8:11] - bound[0, :]) / (bound[1, :] - bound[0, :])

        if mode == "mf":
            model = SingleTaskMultiFidelityGP(
                X_init,
                y_init,
                linear_truncated=False,
                outcome_transform=Standardize(m=1),
                data_fidelities=[11],
            )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_mll(mll)

            if af_name == "MES":
                # this candidates are used in MES
                n_samps = 1000
                sampler = SobolEngine(dimension=11)
                candidates = sampler.draw(n_samps)
                candidates = torch.cat((candidates, torch.ones(n_samps, 1)), dim=1)

                af = qMultiFidelityMaxValueEntropy(
                    model=model,
                    num_fantasies=128,
                    cost_aware_utility=cost_aware_utility,
                    project=_project,
                    candidate_set=candidates,
                )

            elif af_name == "EI":

                best_y = (y_init[X_init[..., -1] == 1.0]).max().item()

                af = CostMultiFidelityEI(model, best_f=best_y, cost_model=cost_model)

            candidates, _ = optimize_acqf_mixed(
                acq_function=af,
                bounds=torch.tensor([[0] * 12, [1] * 12], **tkwargs),
                fixed_features_list=[{11: lowfid}, {11: 1.0}],
                q=1,
                num_restarts=5,
                raw_samples=128,
                options={
                    "batch_limit": 5,
                    "maxiter": 200,
                },
            )

        # HERE SF LOOP, where we ignore the last index
        elif mode == "sf":

            # get model and mll
            model = SingleTaskGP(
                X_init,
                y_init,
                covar_module=ScaleKernel(
                    RBFKernel(
                        ard_num_dims=11,
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                ),
                outcome_transform=Standardize(m=1),
            )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_mll(mll)

            if af_name == "MES":

                sampler = torch.quasirandom.SobolEngine(dimension=11)

                candidates = sampler.draw(1000)

                af = qMaxValueEntropy(
                    model,
                    candidate_set=candidates,
                    num_fantasies=128,
                )

            elif af_name == "EI":

                best_y = y_init.max().item()

                af = ExpectedImprovement(model=model, best_f=best_y)

            candidates, _ = optimize_acqf(
                acq_function=af,
                bounds=torch.tensor([[0] * 11, [1] * 11], **tkwargs),
                q=1,
                num_restarts=5,
                raw_samples=128,
                options={
                    "batch_limit": 5,
                    "maxiter": 200,
                },
            )

        # THIS TRANSFORMATION IS SPECIFIC TO THIS BATCH 1, IT MUST BE CHANGED FOR BATCH>1
        new_x = candidates.detach()

        max_ind = torch.argmax(new_x[:, 0:8])

        new_cat = torch.zeros_like(candidates[:, 0:8])

        new_cat[0, max_ind] = 1.0

        new_x[:, 0:8] = new_cat

        # we append the new_x in the normalized range to X and then scale back both
        X_init = torch.cat((X_init, new_x))

        new_x[:, 8:11] = new_x[:, 8:11] * (bound[1, :] - bound[0, :]) + bound[0, :]

        X_init[:, 8:11] = X_init[:, 8:11] * (bound[1, :] - bound[0, :]) + bound[0, :]

        # here else-if to consider SF cost
        if mode == "sf":
            new_x = torch.cat((new_x, torch.tensor([1]).unsqueeze(1)), axis=1)

            new_y = reizman(new_x).unsqueeze(1)

        else:
            new_y = reizman(new_x).unsqueeze(1)

        y_init = torch.cat((y_init, new_y))

        new_cost = cost_model(new_x).item()

        init_cost += new_cost

        print(f"Step: {step}")
        print(f"Cost: {new_cost}")

    if mode == "sf":
        X_init = torch.cat((X_init, torch.ones(size=(len(X_init), 1))), axis=1)

    data = torch.cat((X_init, y_init), dim=1)

    d = pd.DataFrame(data)
    d.rename(columns={12: "result", 11: 'fidelity'}, inplace=True)
    d["cost"] = d["fidelity"].apply(lambda x: 1 if x == 1.0 else lowfid)
    d['step'] = steps + list(range(step))

    return d


if __name__ == "__main__":

    modes = ["mf", "sf"]

    af_names = ["MES", "EI"]

    seeds = list(range(10))

    lowfid = 0.3

    budget = 30
    cost_ratio = 0.2
    parallel = True

    for mode in modes:
        for af_name in af_names:

            path = f"benchmark/results_reizman/{mode}_{af_name}"

            if os.path.exists(path):
                shutil.rmtree(path)

            os.makedirs(path)

            if parallel:

                results_list = Parallel(n_jobs=-1)(
                    delayed(run_experiment)(
                        mode=mode,
                        af_name=af_name,
                        seed=seed,
                        budget=budget,
                        lowfid=lowfid,
                        cost_ratio=cost_ratio,
                    )
                    for seed in seeds
                )

                for result, seed in zip(results_list, seeds):
                    result.to_csv(f"{path}/{seed}.csv")

            else:

                for seed in seeds:

                    results = run_experiment(
                        mode=mode,
                        af_name=af_name,
                        seed=seed,
                        budget=budget,
                        lowfid=lowfid,
                        cost_ratio=cost_ratio,
                    )

                    results.to_csv(f"{path}/{seed}.csv")
