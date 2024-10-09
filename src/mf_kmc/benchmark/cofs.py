import os
import random
import shutil
from copy import copy
from typing import Dict

import hydra
import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models import AffineFidelityCostModel, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from joblib import Parallel, delayed
from omegaconf import DictConfig
from torch import Tensor

from mf_kmc.optimization.acquisition import CostMultiFidelityEI


class FixedCostFids(AffineFidelityCostModel):
    """Class to model specific Ising cost according to the polynomial fit of
    the fidelities."""

    def __init__(
        self, fidelity_weights: Dict[int, float], fixed_cost=0, min_cost=0.065
    ):
        super().__init__(fidelity_weights, fixed_cost)
        self.min_cost = min_cost

    def forward(self, X: Tensor):

        fids = X[..., -1]

        final_cost = torch.where(fids != 1.0, self.min_cost, fids)

        return final_cost


def diverse_set(X, seed_cof, train_size):
    """Initialization method that takes a candidate randomly and then samples the most
    diverse ones.
    """
    # initialize with one random point; pick others in a max diverse fashion
    nb_COFs = X.shape[0]
    ids_train = copy(seed_cof)
    # select remaining training points
    for j in range(train_size - 1):
        # for each point in data set, compute its min dist to training set
        dist_to_train_set = np.linalg.norm(X - X[ids_train, None, :], axis=2)
        assert np.shape(dist_to_train_set) == (len(ids_train), nb_COFs)
        min_dist_to_a_training_pt = np.min(dist_to_train_set, axis=0)
        assert np.size(min_dist_to_a_training_pt) == nb_COFs

        # acquire point with max(min distance to train set) i.e. Furthest from train set
        ids_train.append(np.argmax(min_dist_to_a_training_pt))
    assert np.size(np.unique(ids_train)) == train_size  # must be unique
    return np.array(ids_train)


def run_experiment(
    mode: str = "mf",
    seed: int = 33,
    af_name: str = "MES",
    total_budget: int = 5,
    cost_ratio: float = 0.065,
    lowfid=0.85,
):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    if mode == "mf":
        cost_model = FixedCostFids(fidelity_weights={-1: 1.0}, min_cost=cost_ratio)
        cost_aware = InverseCostWeightedUtility(cost_model=cost_model)

    df = pd.read_csv("data/converted_data_raw.csv")

    f_names = df.columns[1:15]

    df[f_names] = (df[f_names] - df[f_names].min()) / (
        df[f_names].max() - df[f_names].min()
    )

    df[f_names]

    X = torch.tensor(df[f_names].to_numpy())
    y_hf = torch.tensor(df['gcmc_y']).unsqueeze(1)
    y_lf = torch.tensor(df['henry_y']).unsqueeze(1)

    X_hf = torch.cat((X, torch.ones(X.size()[0], 1)), dim=1)
    X_lf = torch.cat((X, torch.ones(X.size()[0], 1) * lowfid), dim=1)

    budget = 0

    # same initialization strategy than them (sample one random cofs and then select diverse ones)

    init_cof = [rng.integers(0, 608)]

    if mode == "sf" or mode == "random":
        indices_hf = diverse_set(X, init_cof, 3)

    else:
        indices_hf = diverse_set(X, init_cof, 2)

    indices_lf = rng.integers(0, 608, round(1/cost_ratio))

    X_hf_init = X_hf[indices_hf]
    y_mf_init = y_hf[indices_hf]
    X_lf_init = X_lf[indices_lf]
    y_sf_init = y_lf[indices_lf]

    # Create masks for selecting the indices
    mask_hf = torch.ones(len(X_hf), dtype=torch.bool)
    mask_lf = torch.ones(len(X_hf), dtype=torch.bool)
    mask_hf[indices_hf] = False
    mask_lf[indices_lf] = False

    X_hf_rest = X_hf[mask_hf]
    y_hf_rest = y_hf[mask_hf]
    X_lf_rest = X_lf[mask_lf]
    y_lf_rest = y_lf[mask_lf]

    X_init = torch.cat((X_lf_init, X_hf_init))
    y_init = torch.cat((y_sf_init, y_mf_init))

    # the rest
    X_rest = torch.cat((X_hf_rest, X_lf_rest))
    y_rest = torch.cat((y_hf_rest, y_lf_rest))

    if mode == "sf" or mode == "random":
        X_init = X_hf_init
        X_rest = X_hf_rest
        y_init = y_mf_init
        y_rest = y_hf_rest

    steps = [0, 0, 0]

    if mode == "mf":
        steps = [0, 0] + [0] * len(indices_lf)

    step = 0

    while budget < total_budget:

        step += 1

        print(f"Step: {step}")
        steps.append(step)

        if mode == "mf":

            model = SingleTaskMultiFidelityGP(
                X_init,
                y_init,
                data_fidelities=[-1],
                linear_truncated=False,
                outcome_transform=Standardize(m=1),
            )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_mll(mll)

            if af_name == "MES":
                sampler = torch.quasirandom.SobolEngine(dimension=14)

                candidates = sampler.draw(1000)

                candidates = torch.cat((candidates, torch.ones(1000, 1)), dim=1)

                af = qMultiFidelityMaxValueEntropy(
                    model,
                    candidate_set=candidates,
                    num_fantasies=128,
                    project=project_to_target_fidelity,
                    cost_aware_utility=cost_aware,
                )

            elif af_name == "EI":

                best_y = (y_init[X_init[..., -1] == 1.0]).max().item()

                af = CostMultiFidelityEI(model, best_f=best_y, cost_model=cost_model)

        elif mode == "sf":

            model = SingleTaskGP(X_init, y_init, outcome_transform=Standardize(m=1))

            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_mll(mll)

            if af_name == "MES":

                sampler = torch.quasirandom.SobolEngine(dimension=14)

                candidates = sampler.draw(2000)

                candidates = torch.cat((candidates, torch.ones(2000, 1)), dim=1)

                af = qMaxValueEntropy(
                    model,
                    candidate_set=candidates,
                    num_fantasies=128,
                )

            elif af_name == "EI":

                best_y = y_init.max().item()

                af = ExpectedImprovement(model=model, best_f=best_y)

        if mode == "random":
            best = rng.integers(0, X_rest.size()[0])

        else:
            best = af(X_rest.unsqueeze(1)).argmax()

        new_x = X_rest[best].unsqueeze(0)
        new_y = y_rest[best].unsqueeze(0)

        # add new point to old
        X_init = torch.cat((X_init, new_x))
        y_init = torch.cat((y_init, new_y))

        # remove point from olds
        X_rest = torch.cat((X_rest[:best], X_rest[best + 1 :]))
        y_rest = torch.cat((y_rest[:best], y_rest[best + 1 :]))

        if mode == "mf":
            cost = cost_model(new_x).item()

        else:
            cost = 1.0

        budget += cost

    xs = pd.DataFrame(X_init.detach().numpy())
    ys = pd.DataFrame(y_init.detach().numpy())

    results = pd.concat((xs, ys), axis=1)

    results.columns = list(f_names) + ["fidelity", "selectivity"]
    results["step"] = steps

    results["cost"] = results["fidelity"].apply(
        lambda x: 1 if x == 1.0 else cost_ratio
    )

    #assert not results.duplicated().any(), "Df contains duplicates"

    return results


@hydra.main(version_base=None, config_path="config", config_name="cofs")
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
                        total_budget=cfg.budget,
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
                        total_budget=cfg.budget,
                        cost_ratio=cfg.cost_ratio,
                        lowfid=cfg.lowfid
                    )

                    results.to_csv(f"{run_dir}/{seed}.csv")


if __name__ == "__main__":

    main()
