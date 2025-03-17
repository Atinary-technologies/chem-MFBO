"""Do simple regression task on chemistry benchmarks to assess MF vs SF
GP prediction.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import r2_score
from chem_mfbo.benchmark.real_problems import load_and_preprocess_data
import random
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior


def train_predict_model(
    X_train: torch.Tensor, 
    y_train: torch.Tensor, 
    X_test: torch.Tensor, 
    scale: bool = False, 
    mode: str = "mf"
) -> torch.Tensor:
    """Train models and predict on test set. If mf=True, use
    SingleTaskMultiFidelityGP, otherwise SingleTaskGP.
    """

    if mode == "sf":
        X_train = X_train[:, :-1]
        X_test = X_test[:, :-1]

    if mode == "mf":
        model = SingleTaskMultiFidelityGP(
            X_train,
            y_train,
            linear_truncated=False,
            outcome_transform=Standardize(m=1),
            data_fidelities=[-1],
        )

    else:
        model = SingleTaskGP(
            X_train,
            y_train,
            covar_module=ScaleKernel(
            MaternKernel(nu=2.5, lengthscale_prior=GammaPrior(2.0, 0.5))
            ),
            outcome_transform=Standardize(m=1)
        )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_mll(mll)

    # predict
    y_pred = model.posterior(X_test).mean
    y_var = model.posterior(X_test).variance

    return y_pred, y_var

DATASETS = ["cofs", 
            "bandgap", 
            "polarizability", 
            "freesolv"]

N_SAMPLES = [10, 20, 30, 40]

HIGHFID = 0.666666
LOWFID = 0.333333

SAVE_PATH = "plots/regression"

if __name__ == "__main__":
    
    torch.manual_seed(33)
    rng = np.random.default_rng(33)
    random.seed(33)

    # load data
    for dataset in DATASETS:

        file_path = f"/home/sabanza/Documents/chem-MFBO/data/clean/{dataset}.csv"

        if dataset == "cofs" or dataset == "bandgap":
            data_type = "material"
        
        else:
            data_type = "molecular"

        n_rows = int(np.ceil(len(N_SAMPLES) ** 0.5))  # approximate square layout
        n_cols = int(np.ceil(len(N_SAMPLES) / n_rows))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True
        )

        axes = axes.flatten()

        X, y_hf, y_lf = load_and_preprocess_data(file_path, 
                                data_type)

        # Select test indices based on uniform coverage of the y_hf space
        sorted_indices = torch.argsort(y_hf.flatten())
        step = len(sorted_indices) // 8
        test_indices = [sorted_indices[0].item()] + sorted_indices[step::step].tolist() + [sorted_indices[-1].item()]


        for idx, samp in enumerate(N_SAMPLES):

            ax = axes[idx]
            title = f"{samp} training samples"

            X, y_hf, y_lf = load_and_preprocess_data(file_path, 
                        data_type)
   
            # add fidelities
            X_hf = torch.cat((X, torch.ones(X.size()[0], 1) * HIGHFID), dim=1)
            X_lf = torch.cat((X, torch.ones(X.size()[0], 1) * LOWFID), dim=1)

            # this is used to select HF and LF samples (in the MF case we take 50% HF and 50% LF based on 0.05 cost)
            n_hf = int(np.ceil(samp/2))
            n_lf = int(np.floor((samp - n_hf)/0.05))

            # Select X_test and y_test
            X_test = X_hf[test_indices]
            y_test = y_hf[test_indices]
            
            # Remove test data from X_hf and y_hf
            X_hf = torch.cat((X_hf[:test_indices[0]], X_hf[test_indices[-1]+1:]))
            y_hf = torch.cat((y_hf[:test_indices[0]], y_hf[test_indices[-1]+1:]))
         
            for mode in ["mf", "sf"]:
                
                if mode == "sf":
                    indices_hf = rng.integers(0, 
                                        X_hf.size()[0], 
                                        samp)

                else:
                    indices_hf = rng.integers(0, 
                                        X_hf.size()[0], 
                                        n_hf)

                indices_lf = rng.integers(0, X.size()[0], n_lf)
                
                X_hf_init = X_hf[indices_hf]
                y_mf_init = y_hf[indices_hf]
                X_lf_init = X_lf[indices_lf]
                y_sf_init = y_lf[indices_lf]

                if mode == "sf":
                    X_init = X_hf_init
                    y_init = y_mf_init
                
                else:
                    X_init = torch.cat((X_lf_init, X_hf_init))
                    y_init = torch.cat((y_sf_init, y_mf_init))


                # Train and predict
                y_pred, y_var = train_predict_model(X_init, 
                                                    y_init, 
                                                    X_test, 
                                                    mode=mode)

                y_pred = y_pred.detach().numpy().flatten()
                y_var = y_var.detach().numpy().flatten()

                y_true = y_test.detach().numpy().flatten()

                std = np.sqrt(y_var).flatten()

                # Calculate R2 score (or any other metric that we want)
                r2 = r2_score(y_true, 
                            y_pred)

                if mode == "mf":
                    # Create a scatter plot with y_true on the x-axis
                    ax.errorbar(
                        y_true, 
                        y_pred, 
                        yerr=std, 
                        fmt='o', 
                        alpha=0.7, 
                        label='MF', 
                        capsize=3
                    )
                else:
                    # Create a scatter plot with y_true on the x-axis
                    ax.errorbar(
                        y_true, 
                        y_pred, 
                        yerr=std, 
                        fmt='o', 
                        alpha=0.7, 
                        label='SF', 
                        capsize=3
                    )

                # Plot the y = x line
                min_val = min(np.min(y_true), np.min(y_pred), np.min(y_pred))
                max_val = max(np.max(y_true), np.max(y_pred), np.max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')

                # Add labels and title
                # ax.set_xlabel("y_true")
                # ax.set_ylabel("y_predicted")
                ax.set_title(title)

                # Display R^2 scores as text on the plot
                if mode == "mf":
                    ax.text(
                        0.05,
                        0.85,
                        f'R² {mode}: {r2:.2f}',
                        ha='left',
                        va='top',
                        transform=ax.transAxes,
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7),
                    )
                else:
                    ax.text(
                        0.05,
                        0.95,
                        f'R² {mode}: {r2:.2f}',
                        ha='left',
                        va='top',
                        transform=ax.transAxes,
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7),
                    )

                # Add legend and grid
                ax.legend(loc='lower right')
                ax.grid(True)

        fig.supxlabel("y true")
        fig.supylabel("y predicted")
        fig.suptitle(f"{dataset}")
        plt.savefig(os.path.join(f"{SAVE_PATH}/{dataset}.png"), dpi=400)


            
