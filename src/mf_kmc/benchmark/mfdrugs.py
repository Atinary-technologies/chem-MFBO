import os
import random
import shutil
from random import sample
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    qLogExpectedImprovement,
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
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Descriptors import CalcMolDescriptors
from sklearn.decomposition import PCA
from torch import Tensor


class FixedCostFids(AffineFidelityCostModel):
    """Class to model specific cost of the problem."""

    def __init__(self, fidelity_weights: Dict[int, float], fixed_cost=0, min_cost=0.5):
        super().__init__(fidelity_weights, fixed_cost)
        self.min_cost = min_cost

    def forward(self, X: Tensor):

        fids = X[..., -1]

        final_cost = torch.where(fids == 0.5, self.min_cost, fids)

        return final_cost


class CostScaledqLogEI(qLogExpectedImprovement):
    """Adapted version of qLogEI where you simply scale it by the cost of the fidelity"""

    def __init__(self, model, best_f, cost_model, **kwargs):
        super().__init__(model=model, best_f=best_f, **kwargs)
        self.cost_model = cost_model

    def forward(self, X):

        # compute expected improvement at highest fidelity only
        X_hf = torch.clone(X)

        X_hf[..., -1] = 1
        ei = super().forward(X_hf)

        costs = self.cost_model(X).squeeze(1)

        scaled = ei / costs

        return scaled


class CostMultiFidelityEI(ExpectedImprovement):
    """Adapted version of qLogEI for MF BO taken from MFSKO paper.

    ref: https://link.springer.com/content/pdf/10.1007/s00158-005-0587-0.pdf"""

    def __init__(self, model, best_f, cost_model, **kwargs):
        super().__init__(model=model, best_f=best_f, **kwargs)
        self.cost_model = cost_model
        self.af_old = ExpectedImprovement(self.model, best_f=self.best_f)

    def _compute_correlation(self, X):
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

    def forward(self, X):

        # compute expected improvement at highest fidelity only
        X_hf = torch.clone(X)

        X_hf[..., -1] = 1

        ei = super().forward(X_hf).reshape(-1, 1)

        correlation = self._compute_correlation(X).reshape(-1, 1)

        costs = self.cost_model(X)

        cost_ratio = 1 / costs

        alpha = ei * cost_ratio * correlation

        return alpha.unsqueeze(1)


# def encode_mols(smiles: list[str]) -> pd.DataFrame:
#     """Encode list of smiles using rdkit descriptors
#     and return pandas Dataframe"""

#     mols = [Chem.MolFromSmiles(sm) for sm in smiles]

#     desc = [CalcMolDescriptors(mol) for mol in mols]

#     desc_df = pd.DataFrame(desc)

#     return desc_df


def encode_fps(smiles: list[str]) -> np.array:
    """Encode SMILES as fps"""

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]

    fps = np.array([MACCSkeys.GenMACCSKeys(mol) for mol in mols])

    pca = PCA(n_components=20)

    fps = pca.fit_transform(fps)

    fps = (fps - np.min(fps, axis=0)) / (np.max(fps, axis=0) - np.min(fps, axis=0))

    return fps


# def preprocess_mols(path: str) -> pd.DataFrame:
#     """Preprocess molecules from a given dataset and get
#     DataFrame ready for tensor transformation"""


#     df = pd.read_csv(path)

#     #drop molecules that don't have DR values
#     df = df.dropna(subset="DR")

#     #take only SMILES, SD and DR columns
#     df = df[["SD", "DR", "neut-smiles"]].reset_index()

#     # #min max scale DR and SR
#     # df["SD"] = (df["SD"] - df["SD"].min()) / (df["SD"].max() - df["SD"].min())

#     # df["DR"] = (df["DR"] - df["DR"].min()) / (df["DR"].max() - df["DR"].min())

#     #encode descriptors
#     df_descs = encode_mols(df["neut-smiles"])

#     final_df = pd.concat((df, df_descs), axis=1)

#     final_df = final_df.drop(columns=["index", "neut-smiles"])

#     #min max normalize
#     cols = final_df.columns

#     final_df[cols] = (final_df[cols] - final_df[cols].min()) / (final_df[cols].max() - final_df[cols].min())

#     final_df = final_df.T.drop_duplicates().dropna(axis=1).T

#     return final_df


def encode_descriptors(smiles: list[str]) -> np.array:
    """Encode SMILES as descriptors"""

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]

    desc = [CalcMolDescriptors(mol) for mol in mols]

    desc_df = pd.DataFrame(desc)

    desc_arr = desc_df.to_numpy()

    pca = PCA(n_components=10)

    fps = pca.fit_transform(desc_arr)

    fps = (fps - np.min(fps, axis=0)) / (np.max(fps, axis=0) - np.min(fps, axis=0))

    return fps


def data_to_input(path: str) -> pd.DataFrame:
    """Preprocess molecules from a given dataset and get
    tensors for computation with fps"""

    df = pd.read_csv(path)

    # drop molecules that don't have DR values
    df = df.dropna(subset="DR")

    # take only SMILES, SD and DR columns
    df = df[["SD", "DR", "neut-smiles"]].reset_index()

    # min max scale DR and SR
    df["SD"] = (df["SD"] - df["SD"].min()) / (df["SD"].max() - df["SD"].min())

    df["DR"] = (df["DR"] - df["DR"].min()) / (df["DR"].max() - df["DR"].min())

    # encode fps
    X = torch.tensor(encode_descriptors(df["neut-smiles"]), dtype=torch.float64)

    y_lf = torch.tensor(df["SD"], dtype=torch.float64).unsqueeze(1)
    y_hf = torch.tensor(df["DR"], dtype=torch.float64).unsqueeze(1)

    return X, y_hf, y_lf


# def inputs_to_tensor(inps: pd.DataFrame) -> Union[torch.Tensor]:
#     """Build torch data from dataframe"""

#     y_low_fid = torch.tensor(inps.iloc[:, 0].to_numpy())
#     y_high_fid = torch.tensor(inps.iloc[:, 1].to_numpy())
#     X = torch.tensor(inps.iloc[:, 2:].to_numpy())

#     return y_low_fid, y_high_fid, X

# def min_max_normalize_y(y_init: torch.tensor,
#                         X_init: torch.tensor) -> torch.tensor:
#     """Min-max normalize y according to the level of fidelity
#     """
#     # Separate indices for each group
#     indices_group_0 = (X_init[:, -1] != 1)
#     indices_group_1 = (X_init[:, -1] == 1)

#     # Group 0
#     y_group_0 = y_init[indices_group_0]
#     min_0 = y_group_0.min()
#     max_0 = y_group_0.max()
#     y_group_0_normalized = (y_group_0 - min_0) / (max_0 - min_0)

#     # Group 1
#     y_group_1 = y_init[indices_group_1]
#     min_1 = y_group_1.min()
#     max_1 = y_group_1.max()
#     y_group_1_normalized = (y_group_1 - min_1) / (max_1 - min_1)

#     # Create the output tensor and place the normalized values in the correct positions
#     y_normalized = torch.empty_like(y_init)
#     y_normalized[indices_group_0] = y_group_0_normalized
#     y_normalized[indices_group_1] = y_group_1_normalized

#     return y_normalized


def run_experiment(
    path: str,
    mode: str = "mf",
    seed: int = 33,
    af_name: str = "MES",
    total_budget: int = 5,
):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    if mode == "mf":
        cost_model = FixedCostFids(fidelity_weights={-1: 1.0}, min_cost=0.2)
        cost_aware = InverseCostWeightedUtility(cost_model=cost_model)

    X, y_hf, y_lf = data_to_input(path)

    X_hf = torch.cat((X, torch.ones(X.size()[0], 1)), dim=1)
    X_lf = torch.cat((X, torch.ones(X.size()[0], 1) * 0.5), dim=1)

    budget = 0

    # same initialization strategy than them (sample one random cofs and then select diverse ones)

    indices = rng.integers(X.shape[0], endpoint=True, size=3)

    X_hf_init = X_hf[indices]
    y_mf_init = y_hf[indices]
    X_lf_init = X_lf[indices]
    y_sf_init = y_lf[indices]

    # Create masks for selecting the indices
    mask = torch.ones(len(X_hf), dtype=torch.bool)
    mask[indices] = False

    X_hf_rest = X_hf[mask]
    y_hf_rest = y_hf[mask]
    X_lf_rest = X_lf[mask]
    y_lf_rest = y_lf[mask]

    X_init = torch.cat((X_lf_init, X_hf_init))
    y_init = torch.cat((y_sf_init, y_mf_init))

    X_rest = torch.cat((X_hf_rest, X_lf_rest))
    y_rest = torch.cat((y_hf_rest, y_lf_rest))

    if mode == "sf" or mode == "random":
        X_init = X_hf_init
        X_rest = X_hf_rest
        y_init = y_mf_init
        y_rest = y_hf_rest

    steps = [0] * X_hf_init.size()[0]

    if mode == "mf":
        steps = [0] * X_lf_init.size()[0] + steps

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
                sampler = torch.quasirandom.SobolEngine(dimension=X.shape[1])

                candidates = sampler.draw(2000)

                candidates = torch.cat((candidates, torch.ones(2000, 1)), dim=1)

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

                sampler = torch.quasirandom.SobolEngine(dimension=20)

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
            # select best candidate randomly
            best = rng.integers(low=0, high=X_rest.size()[0], size=1)[0]

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
    ys = pd.DataFrame(y_init.detach().numpy()).rename(columns={0: "DR"})

    results = pd.concat((xs, ys), axis=1)

    results.rename(columns={X.shape[1]: "fidelity"}, inplace=True)

    results["step"] = steps

    results["cost"] = results["fidelity"].apply(
        lambda x: 1 if x == 1.0 else cost_model.min_cost
    )

    assert not results.duplicated().any(), "Df contains duplicates"

    return results


if __name__ == "__main__":

    name = "data/504329.csv"

    modes = ["mf", "sf", "random"]
    # modes = ["mf"]
    af_names = ["EI"]  # , "MES"]

    seeds = list(range(8))
    budget = 35

    parallel = True

    for mode in modes:
        for af_name in af_names:

            if mode == "random":
                path = f"src/mf_kmc/benchmark/results_drugs/{mode}"

            else:
                path = f"src/mf_kmc/benchmark/results_drugs/{mode}_{af_name}"

            if os.path.exists(path):
                shutil.rmtree(path)

            os.makedirs(path)

            if parallel:

                # joblib.parallel_config("multiprocessing")
                results_list = Parallel(n_jobs=-1)(
                    delayed(run_experiment)(
                        name,
                        mode=mode,
                        af_name=af_name,
                        seed=seed,
                        total_budget=budget,
                    )
                    for seed in seeds
                )

                for result, seed in zip(results_list, seeds):
                    result.to_csv(f"{path}/{seed}.csv")

            else:

                for seed in seeds:

                    results = run_experiment(
                        name, mode=mode, af_name=af_name, seed=seed, total_budget=budget
                    )

                    results.to_csv(f"{path}/{seed}.csv")
