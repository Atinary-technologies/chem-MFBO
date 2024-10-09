import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions import AugmentedBranin, AugmentedHartmann
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mf_kmc.simulations.implementations.park.park import Park

torch.manual_seed(33)

N_SAMPS = 100

simulations = {
    "branin": [AugmentedBranin(), 2],
    "park": [Park(), 4],
    "hartmann": [AugmentedHartmann(), 6],
}


l_levels = np.linspace(0, 0.9, 10)


if __name__ == "__main__":

    for name, sim in simulations.items():

        problem = sim[0]

        samples = torch.rand(size=(N_SAMPS, sim[1]))

        if name == "branin":
            bounds = torch.tensor(problem._bounds)[:2, :]

            samples = samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        fig, axes = plt.subplots(2, 5, figsize=(10, 8))

        for i, l in enumerate(l_levels):

            s_lf = torch.cat(
                (samples, torch.ones(samples.size()[0]).unsqueeze(1) * l), dim=1
            )
            s_hf = torch.cat(
                (samples, torch.ones(samples.size()[0]).unsqueeze(1) * 1.0), dim=1
            )
            s = problem(s_lf).detach().numpy()
            s_true = problem(s_hf).detach().numpy()

            r2_raw = r2_score(s_true, s)
            corr = np.corrcoef(s, s_true)[0][1]
            X_train, X_test, y_train, y_test = train_test_split(
                s, s_true, test_size=0.5
            )

            linear = LinearRegression()

            linear.fit(X_train.reshape(-1, 1), y_train)

            y_pred = linear.predict(X_test.reshape(-1, 1))

            r2 = r2_score(y_test, y_pred)

            n_row = i // 5
            n_col = i % 5

            axes[n_row, n_col].scatter(s, s_true)

            # axes[n_row, n_col].text(0.1, 0.9, f"r2_raw: {round(r2_raw, 3)}")
            # axes[n_row, n_col].text(0.8, 0.9, f"corr: {round(corr, 3)}")
            axes[n_row, n_col].text(0.9, 2, f"r2: {round(r2, 3)}")

            axes[n_row, n_col].set_title(f"alpha: {round(l, 3)}")

        plt.suptitle(name)

        fig.supxlabel("low fidelity value")
        fig.supylabel("high fidelity value")
        plt.tight_layout()

        path_plot = Path("plots/r2/")

        if not path_plot.exists():
            os.makedirs(path_plot)

        plt.savefig(path_plot.joinpath(f"{name}_r2.png"))
