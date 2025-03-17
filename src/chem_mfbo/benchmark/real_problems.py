import os
import random
import shutil
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

from chem_mfbo.optimization.acquisition import CostMultiFidelityEI
from chem_mfbo.benchmark.utils import FixedCostFids, encode_descriptors, diverse_set



def load_and_preprocess_data(file_path: str, data_type: str):
    df = pd.read_csv(file_path)
    
    if data_type == "molecular":
        smiles = df["smiles"].tolist()
        X = torch.tensor(encode_descriptors(smiles))
    else:
        f_names = df.columns[:-2]
        df[f_names] = (df[f_names] - df[f_names].min()) / (df[f_names].max() - df[f_names].min())
        X = torch.tensor(df[f_names].to_numpy())
    
    y_hf = torch.tensor(df['HF'], dtype=torch.float64).unsqueeze(1)
    y_lf = torch.tensor(df['LF'], dtype=torch.float64).unsqueeze(1)
    
    return X, y_hf, y_lf


def run_experiment(
    mode: str = "mf",
    seed: int = 33,
    af_name: str = "MES",
    total_budget: int = 5,
    sampling_budget: float = 0.1,
    cost_ratio: float = 0.065,
    noise: float = None,
    highfid=2/3,
    lowfid=1/3,
    data_type: str = "molecular",
    file_path: str = "data/clean/cofs.csv"
):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    if mode == "mf":
        cost_model = FixedCostFids(fidelity_weights={-1: 1.0}, min_cost=cost_ratio)
        cost_aware = InverseCostWeightedUtility(cost_model=cost_model)

    X, y_hf, y_lf = load_and_preprocess_data(file_path, data_type)

    # if noise, we add Gaussian noise to the LF samples
    if noise:
        y_lf = y_lf + torch.randn(y_lf.size()) * 0.10

    # add fidelities
    X_hf = torch.cat((X, torch.ones(X.size()[0], 1) * highfid), dim=1)
    X_lf = torch.cat((X, torch.ones(X.size()[0], 1) * lowfid), dim=1)

    budget = 0
    
    # total number of samples to take
    n_samps = int(np.ceil(total_budget * sampling_budget))

    # initialization strategy: sobol sampling at HF and LF based on max budget and % of this budget
    init_sample = [rng.integers(0, X.size()[0])]

    # this is used to select HF and LF samples (in the MF case we take 50% HF and 50% LF)
    n_hf = int(np.ceil(n_samps/2))
    n_lf = int(np.floor((n_samps - n_hf)/cost_ratio))

    if mode == "sf" or mode == "random":
        indices_hf = diverse_set(X, init_sample, n_samps)

    else:
        indices_hf = diverse_set(X, init_sample, n_hf)

    indices_lf = rng.integers(0, X.size()[0], n_lf)
    
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

    steps = [0]*len(indices_hf)

    if mode == "mf":
        steps = [0] * len(indices_hf) + [0] * len(indices_lf)

    step = 0

    # here set up budget after discounting HF 
    if mode == "mf":

        # compute lf cost and hf cost
        budget = n_lf*cost_ratio + n_hf

    else:
        budget = n_samps

    while budget < total_budget:

        #check highest fidelity
        max_fid = X_init[..., -1].max().item()

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
                sampler = torch.quasirandom.SobolEngine(dimension=X.size(1))

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
                
                best_y = (y_init[X_init[..., -1] == max_fid]).max().item()

                af = CostMultiFidelityEI(model, best_f=best_y, cost_model=cost_model)

        elif mode == "sf":

            model = SingleTaskGP(X_init, y_init, outcome_transform=Standardize(m=1))

            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_mll(mll)

            if af_name == "MES":

                sampler = torch.quasirandom.SobolEngine(dimension=X.size(1))

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
        X_rest = torch.cat((X_rest[:best], X_rest[best + 1:]))
        y_rest = torch.cat((y_rest[:best], y_rest[best + 1:]))

        if mode == "mf":
            cost = cost_model(new_x).item()

        else:
            cost = 1.0

        budget += cost

    xs = pd.DataFrame(X_init.detach().numpy())
    ys = pd.DataFrame(y_init.detach().numpy())

    results = pd.concat((xs, ys), axis=1)

    num_features = X.size(1)

    results.columns = list(range(num_features)) + ["fidelity", "output"]

    results["step"] = steps

    results["fidelity"] = results["fidelity"].apply(lambda x: 1 if x == max_fid else 0)

    results["cost"] = results["fidelity"].apply(lambda x: 1 if x == 1 else cost_ratio)

    return results


@hydra.main(version_base=None, config_path="../../../config_bench", config_name="cofs")
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
                        sampling_budget=cfg.sampling_budget,
                        noise=cfg.noise,
                        lowfid=cfg.lowfid,
                        data_type=cfg.data_type,
                        file_path=cfg.file_path,
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
                        sampling_budget=cfg.sampling_budget,
                        noise=cfg.noise,
                        lowfid=cfg.lowfid,
                        highfid=cfg.highfid,
                        data_type=cfg.data_type,
                        file_path=cfg.file_path,
                    )

                    results.to_csv(f"{run_dir}/{seed}.csv")


if __name__ == "__main__":

    main()
