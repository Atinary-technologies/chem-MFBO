"""Script to run catalytic regression. Read BH dataset and run regression.
"""
import random
import matplotlib.pyplot as plt
import torch
import ast
import os
import numpy as np
import pandas as pd
from botorch import fit_gpytorch_mll
from chem_mfbo.regression.chemistry_regression import train_predict_model
from sklearn.preprocessing import MinMaxScaler
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


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
            outcome_transform=Standardize(m=1)
        )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_mll(mll)

    # predict
    y_pred = model.posterior(X_test).mean
    y_var = model.posterior(X_test).variance

    return y_pred, y_var


BH_PATH = "data/clean/BH_dataset.csv"
SAVE_PLOTS = "plots/regression"

N_SAMPLES = [10, 20, 30, 40]
HIGHFID = 0.8
MEDFID = 0.5
LOWFID = 0.2

if __name__ == "__main__":

    torch.manual_seed(33)
    rng = np.random.default_rng(33)
    random.seed(33)

    # plot stuff
    n_rows = int(np.ceil(len(N_SAMPLES) ** 0.5))  # approximate square layout
    n_cols = int(np.ceil(len(N_SAMPLES) / n_rows))

    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True
        )

    axes = axes.flatten()

    # do catalytic regression
    df = pd.read_csv(BH_PATH)

    # Convert string representations of numpy arrays back to numpy arrays
    df["fp"] = df["fp"].apply(lambda x: np.array(ast.literal_eval(x)))
    
    # load X and ys
    X = torch.tensor(np.stack(df["fp"].values))

    # min max scale X
    scaler = MinMaxScaler()
    X = torch.tensor(scaler.fit_transform(X))

    y_hf = torch.tensor(scaler.fit_transform(df["HF"].values.reshape(-1, 1)))
    y_medf = torch.tensor(scaler.fit_transform(df["MF"].values.reshape(-1, 1)))
    y_lf = torch.tensor(scaler.fit_transform(df["LF"].values.reshape(-1, 1)))

    # Select test indices based on uniform coverage of the y_hf space
    sorted_indices = torch.argsort(y_hf.flatten())
    step = len(sorted_indices) // 8
    test_indices = [sorted_indices[0].item()] + sorted_indices[step::step].tolist() + [sorted_indices[-1].item()]


    for idx, samp in enumerate(N_SAMPLES):

            y_hf = torch.tensor(df["HF"]).unsqueeze(1)
            y_medf = torch.tensor(df["MF"]).unsqueeze(1)
            y_lf = torch.tensor(df["LF"]).unsqueeze(1)

            ax = axes[idx]
            title = f"{samp} training samples"
   
            # add fidelities
            X_hf = torch.cat((X, torch.ones(X.size()[0], 1) * HIGHFID), dim=1)
            X_medf = torch.cat((X, torch.ones(X.size()[0], 1) * MEDFID), dim=1)
            X_lf = torch.cat((X, torch.ones(X.size()[0], 1) * LOWFID), dim=1)

            # this is used to select HF and LF samples (in the MF case we take 60% HF and 30% MF 10% LF based on 0.1 MF and 0.01 LF cost)
            n_hf = int(samp * 0.6)
            n_medf = int(samp * 0.3 / 0.1)
            n_lf = int(samp * 0.1 / 0.01)
        
            # Select X_test and y_test
            X_test = X_hf[test_indices]
            y_test = y_hf[test_indices]

            # Remove test data from X_hf and y_hf
            mask = torch.ones(X_hf.size(0), dtype=bool)
            mask[test_indices] = False
            X_hf = X_hf[mask]
            y_hf = y_hf[mask]

            for mode in ["mf", "sf"]:
                
                # select n = samp points if sf, otherwise split the budget into HF, MEDF and LF
                if mode == "sf":
                    indices_hf = rng.integers(0, 
                                        X_hf.size()[0], 
                                        samp)

                else:
                    indices_hf = rng.integers(0, 
                                        X_hf.size()[0], 
                                        n_hf)

                indices_medf = rng.integers(0, 
                                            X_medf.size()[0], 
                                            n_medf)
                
                indices_lf = rng.integers(0, 
                                          X_lf.size()[0], 
                                          n_lf)
                
                # here we select the samples from the remaining tensors 
                X_hf_init = X_hf[indices_hf]
                y_hf_init = y_hf[indices_hf]
                X_medf_init = X_medf[indices_medf]
                y_medf_init = y_medf[indices_medf]
                X_lf_init = X_lf[indices_lf]
                y_sf_init = y_lf[indices_lf]

                if mode == "sf":
                    X_init = X_hf_init
                    y_init = y_hf_init
                
                # if MF, we concatenate all the 
                else:
                    X_init = torch.cat((X_lf_init, X_medf_init, X_hf_init))
                    y_init = torch.cat((y_sf_init, y_medf_init, y_hf_init))


                # Train and predict
                y_pred, y_var = train_predict_model(X_init, 
                                                    y_init, 
                                                    X_test, 
                                                    scale=True,
                                                    mode=mode)

                y_pred = y_pred.detach().numpy().flatten()
                y_var = y_var.detach().numpy().flatten()

                y_true = y_test.detach().numpy().flatten()

                std = np.sqrt(y_var).flatten()

                # Calculate Root Mean Squared Error (RMSE)
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

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


                ax.set_title(title)

                # Display R^2 scores as text on the plot
                if mode == "mf":
                    ax.text(
                        0.05,
                        0.85,
                        f'RMSE {mode}: {rmse:.2f}',
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
                        f'RMSE {mode}: {rmse:.2f}',
                        ha='left',
                        va='top',
                        transform=ax.transAxes,
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7),
                    )

                # Add legend and grid
                ax.legend(loc='lower right')
                ax.grid(True)

    fig.supxlabel("true yield")
    fig.supylabel("predicted yield")
    fig.suptitle(f"BH catalytic 3 fidelities")
    plt.savefig(os.path.join(f"{SAVE_PLOTS}/BH.png"), dpi=400)


            




