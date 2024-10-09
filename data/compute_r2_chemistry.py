"""Compute r2 for each problem and generate a plot."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from rdkit import Chem
from sklearn.metrics import r2_score
from summit.benchmarks import MIT_case2
from summit.domain import *
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



def load_ys(problem: str, n_samps: int = 100, seed: int = 33):

    """Sample y values at low and high fidelity levels using random technique."""

    rng = default_rng(seed=seed)

    if problem == "cofs":
        data = pd.read_csv("data/converted_data_raw.csv")
        y_true = data["gcmc_y"].values
        y_pred = data["henry_y"].values

    elif problem == "polarizability" or problem == "polarizability_neg":
        data = pd.read_csv(
            "data/polarizability.csv",
            delimiter='|',
            comment='#',
            skiprows=9,
            header=None,
        )

        data = data[[9, 17, 20]]
        data = data.dropna(subset=17)

        # transform SMILES into Inchi
        data["smiles"] = data[9].apply(inchi_to_smiles)

        data = data[data['smiles'] != False]

        # take only those who have an alkyl chain (organic molecules)
        data = data[data['smiles'].apply(has_carbon)]

        # do something
        y_true = data[17].values
        y_pred = data[20].values

    elif problem == "kinetic":
        experiment = MIT_case2(noise_level=0)

        # Sample the input space
        samples = SimpleSampler(experiment.domain, seed=42).sample(n_samps)

        # Run the experiment at low time
        samples_low_time = samples.copy()
        samples_low_time['t'] = 60
        results_low_time = [
            experiment._run(conditions=spl)
            for spl in samples_low_time.to_dict("records")
        ]
        y_pred = np.array([result[0][('y', 'DATA')] for result in results_low_time])

        samples_high_time = samples.copy()
        samples_high_time['t'] = 600
        results_high_time = [
            experiment._run(conditions)
            for conditions in samples_high_time.to_dict("records")
        ]
        y_true = np.array([result[0][('y', 'DATA')] for result in results_high_time])

    else:
        data = pd.read_csv("data/freesolv.csv")
        y_true = data["expt"].values
        y_pred = data["calc"].values

    assert len(y_true) >= n_samps, f"Number of examples lower than n samples"

    random_indices = rng.integers(0, len(y_true), size=n_samps)

    y_hf = y_true[random_indices]
    y_lf = y_pred[random_indices]

    return y_hf, y_lf


def inchi_to_smiles(inchi: str) -> str:
    """Transform Inchi into SMILES"""
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
        return smiles
    except:
        return False


def has_carbon(smiles: str) -> bool:
    """Check if molecule contains the alkyl motif"""
    mol = Chem.MolFromSmiles(smiles)
    carb = mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][#6]"))

    return carb


class SimpleSampler:
    def __init__(self, domain, seed=None):
        self.domain = domain
        self.rng = np.random.default_rng(seed)

    def sample(self, n_samples):
        samples = {}
        for var in self.domain.input_variables:
            if isinstance(var, ContinuousVariable):
                samples[var.name] = self.rng.uniform(
                    var.bounds[0], var.bounds[1], n_samples
                )
            elif isinstance(var, CategoricalVariable):
                samples[var.name] = self.rng.choice(var.levels, n_samples)
            else:
                raise ValueError(f"Unsupported variable type: {type(var)}")
        return pd.DataFrame(samples)


DATASETS = ["cofs", "freesolv", "polarizability", "polarizability_neg"]

if __name__ == "__main__":

    os.makedirs("plots/r2/", exist_ok=True)

    # for the simple datasets, take 100 random ys and compute r2
    for dataset in DATASETS:
        y_hf, y_lf = load_ys(dataset)

        if dataset == "polarizability_neg":
            rng = np.random.default_rng(seed=33)
            y_lf = y_lf + rng.normal(loc=0, scale=1, size=(len(y_lf)))*3

        X_train, X_test, y_train, y_test = train_test_split(y_lf, y_hf, test_size=0.5)

        linear = LinearRegression()

        linear.fit(X_train.reshape(-1, 1), y_train)

        y_pred = linear.predict(X_test.reshape(-1, 1))

        r2 = round(r2_score(y_test, y_pred), 2)

        corr = round(np.corrcoef(y_hf, y_lf)[0][1], 2)

        fig, ax = plt.subplots()
        ax.scatter(y_lf, y_hf)
        plt.text(
            0.05,
            0.95,
            transform=ax.transAxes,
            s=f"$R^2$: {r2}",
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5),
        )
        plt.text(
            0.05,
            0.85,
            transform=ax.transAxes,
            s=f"Correlation: {corr}",
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5),
        )
        plt.grid()
        plt.xlabel("low fidelity value")
        plt.ylabel("high fidelity value")
        plt.title(f"{dataset}")

        path_plot = Path("plots/r2/")

        if not path_plot.exists():
            os.makedirs(path_plot)

        plt.savefig(path_plot.joinpath(f"r2_{dataset}.png"))
        plt.savefig(path_plot.joinpath(f"r2_{dataset}.svg"))
