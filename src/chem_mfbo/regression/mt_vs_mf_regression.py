"""Compare SingleTaskMultiFidelityGP vs MultiTaskGP for BRANIN regression"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions.multi_fidelity import AugmentedBranin
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import Prior
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from sklearn.metrics import r2_score
from torch.distributions import Exponential


class ExponentialPrior(Prior):
    def __init__(self, rate):
        super().__init__()
        self.dist = Exponential(rate)

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def rsample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape)


torch.manual_seed(33)

SAVE_PATH = "plots/regression"
N_SAMPLES = [10, 20, 30, 40]
BRANIN = AugmentedBranin(noise_std=0, negate=True)
BOUNDS = torch.tensor(BRANIN._bounds)


def sample_branin_train(n_samps: int = 30, bias: float = 0.2) -> tuple[torch.Tensor]:
    """Sample branin tensors and return data to train and test
    GP models
    """

    # number of high fid, it's used as index
    n_hf = round(n_samps / 2)

    # get samples
    sampler = torch.quasirandom.SobolEngine(dimension=2)
    X = sampler.draw(n_samps, dtype=torch.float64)
    X = X * (BOUNDS[:-1, 1] - BOUNDS[:-1, 0]) + BOUNDS[:-1, 0]

    # create tensors to encode fidelities (kernel c value, and bias)
    high_fid = torch.ones((n_hf, 1))
    low_fid = torch.zeros((n_hf, 1))
    bias_fid = torch.ones((n_hf, 1)) * bias

    fids = torch.cat((high_fid, low_fid))
    fids_bias = torch.cat((high_fid, bias_fid))

    X_data = torch.cat((X, fids), dim=1)
    X_simulation = torch.cat((X, fids_bias), dim=1)

    y = BRANIN(X_simulation).unsqueeze(1)

    # take final train data
    X_train_hf = X_data[:n_hf]
    X_train_lf = X_data[n_hf:]
    y_train_hf = y[:n_hf]
    y_train_lf = y[n_hf:]

    X_train = torch.cat((X_train_hf, X_train_lf))
    y_train = torch.cat((y_train_hf, y_train_lf))

    return X_train, y_train


def sample_test(n_samples: int) -> tuple[torch.Tensor]:
    """Sample quasirandom tensor for testing models at high
    fidelity data
    """

    # take sobol samples
    sampler = torch.quasirandom.SobolEngine(dimension=2)
    test_feats = sampler.draw(n_samples)
    test_feats = test_feats * (BOUNDS[:-1, 1] - BOUNDS[:-1, 0]) + BOUNDS[:-1, 0]

    # fidelities and concatenate
    fids = torch.ones(n_samples, 1)

    X_test = torch.cat((test_feats, fids), dim=1)

    # get y data and
    y_test = BRANIN(X_test).unsqueeze(1)

    return X_test, y_test


def train_predict_model(
    X_train: torch.Tensor, 
    y_train: torch.Tensor, 
    X_test: torch.Tensor, 
    mt: bool = False
) -> torch.Tensor:
    """Train models and predict on test set. If mt = True, use MultiTask, use
    SingleTaskMultiFidelityGP otherwise.
    """

    # min max normalize X_train
    X_train = (X_train - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    X_test = (X_test - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])

    if mt:

        model = MultiTaskGP(
            train_X=X_train,
            train_Y=y_train,
            rank=1,
            task_feature=-1,
            outcome_transform=Standardize(m=1),
            task_covar_prior=LKJCovariancePrior(
                n=2,
                eta=torch.tensor(1.5).to(X_train),
                sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05),
            ),
        )

    else:
        model = SingleTaskMultiFidelityGP(
            X_train,
            y_train,
            linear_truncated=False,
            outcome_transform=Standardize(m=1),
            data_fidelities=[-1],
        )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_mll(mll)

    # predict
    y_pred = model.posterior(X_test).mean
    y_var = model.posterior(X_test).variance

    return y_pred, y_var


if __name__ == "__main__":

    # create plots directory
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    X_test, y_test = sample_test(8)

    x = 0

    n_rows = int(np.ceil(len(N_SAMPLES) ** 0.5))  # approximate square layout
    n_cols = int(np.ceil(len(N_SAMPLES) / n_rows))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True
    )

    axes = axes.flatten()

    # for i in iterations, train models, get plots and save them
    for idx, samp in enumerate(N_SAMPLES):

        ax = axes[idx]
        title = f"{samp}_training_samples"

        X_train, y_train = sample_branin_train(n_samps=samp)
        y_pred_mf, y_var_mf = train_predict_model(X_train, y_train, X_test)
        y_pred_mt, y_var_mt = train_predict_model(X_train, y_train, X_test, mt=True)

        # to numpy
        y_pred_mf = y_pred_mf.detach().numpy().flatten()
        y_pred_mt = y_pred_mt.detach().numpy().flatten()
        y_true = y_test.detach().numpy().flatten()

        std_mf = np.sqrt(y_var_mf.detach().numpy()).flatten()
        std_mt = np.sqrt(y_var_mt.detach().numpy()).flatten()

        # compute r2 for each
        r2_mf = r2_score(y_true, y_pred_mf)
        r2_mt = r2_score(y_true, y_pred_mt)

        # Print R^2 values for reference
        print("R^2 score for pred_mf:", r2_mf)
        print("R^2 score for pred_mt:", r2_mt)

        # Create a scatter plot with y_true on the x-axis
        ax.errorbar(
            y_true, y_pred_mf, yerr=std_mf, fmt='o', alpha=0.7, label='MF', capsize=3
        )
        ax.errorbar(
            y_true, y_pred_mt, yerr=std_mt, fmt='o', alpha=0.7, label='MT', capsize=3
        )

        # Plot the y = x line
        min_val = min(np.min(y_true), np.min(y_pred_mf), np.min(y_pred_mt))
        max_val = max(np.max(y_true), np.max(y_pred_mf), np.max(y_pred_mt))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Add labels and title
        # ax.set_xlabel("y_true")
        # ax.set_ylabel("y_predicted")
        ax.set_title(title)

        # Display R^2 scores as text on the plot
        ax.text(
            0.05,
            0.85,
            f'R² MF: {r2_mf:.2f}\nR² MT: {r2_mt:.2f}',
            ha='left',
            va='top',
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7),
        )

        # Add legend and grid
        ax.legend()
        ax.grid(True)

    fig.supxlabel("y_true")
    fig.supylabel("y_predicted")
    plt.savefig(os.path.join(SAVE_PATH, "mf_vs_mt_plots" + ".png"), dpi=400)
