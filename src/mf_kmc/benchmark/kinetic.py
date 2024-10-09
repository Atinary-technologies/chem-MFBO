"""Script to run kinetic negative result benchmark"""
import os

import hydra
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
from omegaconf import DictConfig
from scipy.stats.qmc import LatinHypercube
from summit import *
from torch import Tensor
from torch.quasirandom import SobolEngine

from mf_kmc.optimization.acquisition import CostMultiFidelityEI

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class KineticFidelity:
    """Class to get result from Reizmann Suzuki. Take tensor
    as input."""

    def __init__(self, hf_time: float = 600, lf_time: float = 60) -> None:

        self.experiment = MIT_case2()
        self.domain = self.experiment.domain
        self.hf_time = hf_time
        self.lf_time = lf_time

    def _encode_tensor(self, X: torch.Tensor) -> DataSet:
        """Encode tensor to dictionary"""

        categorical = self.domain.get_categorical_combinations()

        # this is for the categorical variables
        catalyst_dict = {
            i: value for i, value in enumerate(categorical['cat_index'].values)
        }

        # create dictionary to compute inputs
        new = {}

        new[('cat_index', 'DATA')] = [catalyst_dict[torch.argmax(X[:8]).item()]]
        new[('conc_cat', 'DATA')] = [X[8]]
        new[('temperature', 'DATA')] = [X[9]]

        # time is the HF/LF variable
        if X[-1] == 1:
            new[('t', 'DATA')] = self.hf_time

        else:
            new[('t', 'DATA')] = self.lf_time

        dataset = DataSet(new)

        return dataset

    def _simulate(self, X: torch.Tensor) -> torch.Tensor:
        """Get result for one experiment"""

        inp = self._encode_tensor(X)

        result = self.experiment.run_experiments(inp)

        output = torch.tensor(result['y'].values[0])

        return output

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Get final result depending on the fidelity."""

        if X.dim() == 1:
            # If X is a single 1D tensor, handle it directly
            return self._simulate(X)
        else:
            # If X is a batch of 2D tensors, process each row
            results = []
            for i in range(X.shape[0]):
                y = self._simulate(X[i])
                results.append(y)
            return torch.stack(results).type(tkwargs['dtype'])


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
            if p["name"] == "t":
                continue

            elif p["type"] == "ContinuousVariable":
                low_bounds.append(p["bounds"][0])
                high_bounds.append(p["bounds"][1])

            else:
                num_cats.append(len(p['levels']))

    bounds = torch.tensor([low_bounds, high_bounds])

    return bounds, num_cats


def sample_random(bound, num_cats, n_samps):

    samp_tens = torch.rand(size=(n_samps, len(bound[0])))

    cats = torch.randint(0, num_cats, (n_samps,))

    one_hot = torch.nn.functional.one_hot(cats, num_classes=num_cats)

    samples = torch.cat((one_hot, samp_tens), axis=1)

    return samples


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


def normalize_ys(X, y, mode):
    """Min max normalize outputs, if multi-fidelity, in different scales."""

    if mode == "mf":

        y_hf = y[X[..., -1] == 1]

        y_lf = y[X[..., -1] != 1]

        y_hf = (y_hf - y_hf.min()) / (y_hf.max() - y_hf.min())
        y_lf = (y_lf - y_lf.min()) / (y_lf.max() - y_lf.min())

        X_hf = X[X[..., -1] == 1]
        X_lf = X[X[..., -1] != 1]

        X = torch.cat((X_hf, X_lf))
        y = torch.cat((y_hf, y_lf))

    elif mode == "sf":
        y = (y - y.min()) / (y.max() - y.min())

    return X, y


def run_experiment(
    mode: str = "mf",
    lowfid: float = 0.7,
    seed: int = 33,
    af_name: str = "MES",
    budget: int = 5,
    cost_ratio: float = 0.1,
):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # here summit simulation
    kinetic = KineticFidelity()

    cost_model = AffineModifiedCostModel(cost_ratio=cost_ratio)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    # Initial sampling
    n_low = 10
    n_high = 2

    if mode == "sf" or mode == "random":
        n_high = 3

    domain = kinetic.domain

    bound, num_cats = get_bounds(domain)

    samps_hf = sample_tensors(bound, num_cats, n_high, seed)
    samps_lf = sample_tensors(bound, num_cats, n_low, seed)

    samps_hf = torch.cat((samps_hf, torch.ones((n_high, 1))), axis=1)
    samps_lf = torch.cat((samps_lf, torch.ones((n_low, 1)) * lowfid), axis=1)

    X_init = torch.concatenate((samps_hf, samps_lf), axis=0)

    y_init = kinetic(X_init).unsqueeze(1).type(torch.float64)

    init_costs = cost_model(X_init)

    init_cost = sum(init_costs).item()

    steps = [0] * len(y_init)

    # if single fidelity, take only the tensors where X[-1] == 1
    # TO DO
    if mode == "sf" or mode == "random":
        X_init = X_init[:n_high, :-1]
        steps = steps[:n_high]
        y_init = y_init[:n_high]

    step = 0

    while init_cost < budget:

        step += 1

        # normalize inputs
        X_init[:, 8:10] = (X_init[:, 8:10] - bound[0, :]) / (bound[1, :] - bound[0, :])

        # normalize outputs
        X, y = normalize_ys(X_init, y_init, mode)

        if mode == "mf":
            model = SingleTaskMultiFidelityGP(
                X,
                y,
                linear_truncated=False,
                outcome_transform=Standardize(m=1),
                data_fidelities=[10],
            )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_mll(mll)

            if af_name == "MES":
                # this candidates are used in MES
                n_samps = 1000
                sampler = SobolEngine(dimension=10)
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
                bounds=torch.tensor([[0] * 11, [1] * 11], **tkwargs),
                fixed_features_list=[{10: lowfid}, {10: 1.0}],
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
                        ard_num_dims=10,
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                ),
                outcome_transform=Standardize(m=1),
            )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_mll(mll)

            if af_name == "MES":

                sampler = torch.quasirandom.SobolEngine(dimension=10)

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
                bounds=torch.tensor([[0] * 10, [1] * 10], **tkwargs),
                q=1,
                num_restarts=5,
                raw_samples=128,
                options={
                    "batch_limit": 5,
                    "maxiter": 200,
                },
            )

        elif mode == "random":
            candidates = sample_random(bound, 8, 1)

        # THIS TRANSFORMATION IS SPECIFIC TO THIS BATCH 1, IT MUST BE CHANGED FOR BATCH>1
        new_x = candidates.detach()

        max_ind = torch.argmax(new_x[:, 0:8])

        new_cat = torch.zeros_like(candidates[:, 0:8])

        new_cat[0, max_ind] = 1.0

        new_x[:, 0:8] = new_cat

        # we append the new_x in the normalized range to X and then scale back both
        X_init = torch.cat((X_init, new_x))

        new_x[:, 8:10] = new_x[:, 8:10] * (bound[1, :] - bound[0, :]) + bound[0, :]

        X_init[:, 8:10] = X_init[:, 8:10] * (bound[1, :] - bound[0, :]) + bound[0, :]

        # here else-if to consider SF cost
        if mode == "sf" or mode == "random":
            new_x = torch.cat((new_x, torch.tensor([1]).unsqueeze(1)), axis=1)

            new_y = kinetic(new_x).unsqueeze(1)

        else:
            new_y = kinetic(new_x).unsqueeze(1)

        y_init = torch.cat((y_init, new_y))

        new_cost = cost_model(new_x).item()

        init_cost += new_cost

        print(f"Step: {step}")
        print(f"Cost: {new_cost}")

    if mode == "sf" or mode == "random":
        X_init = torch.cat((X_init, torch.ones(size=(len(X_init), 1))), axis=1)

    data = torch.cat((X_init, y_init), dim=1)

    d = pd.DataFrame(data)
    d.rename(columns={10: "fidelity", 11: 'result'}, inplace=True)
    d["cost"] = d["fidelity"].apply(lambda x: 1 if x == 1.0 else cost_ratio)
    d['step'] = steps + list(range(step))

    return d


@hydra.main(version_base=None, config_path="config", config_name="kinetic")
def main(cfg: DictConfig) -> None:

    seeds = list(range(cfg.seeds))

    results_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for mode in cfg.modes:
        for af_name in cfg.af_names:

            if mode == "random":
                run_dir = f"{results_dir}/{mode}"

            else:
                run_dir = f"{results_dir}/{mode}_{af_name}"

            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

            os.makedirs(run_dir)

            if cfg.parallel:

                results_list = Parallel(n_jobs=-1)(
                    delayed(run_experiment)(
                        mode=mode,
                        af_name=af_name,
                        seed=seed,
                        budget=cfg.budget,
                        cost_ratio=cfg.cost_ratio,
                        lowfid=cfg.lowfid,
                    )
                    for seed in seeds
                )

                for result, seed in zip(results_list, seeds):
                    result.to_csv(f"{run_dir}/{seed}.csv")

            else:

                for seed in seeds:

                    results = run_experiment(
                        mode=mode,
                        af_name=af_name,
                        seed=seed,
                        budget=cfg.budget,
                        cost_ratio=cfg.cost_ratio,
                        lowfid=cfg.lowfid,
                    )

                    results.to_csv(f"{run_dir}/{seed}.csv")


if __name__ == "__main__":

    main()
