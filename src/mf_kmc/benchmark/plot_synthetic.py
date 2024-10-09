"""Code to make plots for the benchmarking framework."""
import json
import os
import re
from pathlib import Path
from typing import Dict

import hydra
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

# some parameters to plot (taken from Takeno repo on MF-MES)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams["font.size"] = 30
plt.rcParams["axes.titlesize"] = 21
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30

plt.rcParams['figure.figsize'] = (6.5, 6.5)
plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams['errorbar.capsize'] = 4.0
plt.rcParams['lines.markersize'] = 12

plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markeredgewidth'] = 1.5

plt.rcParams['legend.borderaxespad'] = 0.15
plt.rcParams['legend.borderpad'] = 0.2
plt.rcParams['legend.columnspacing'] = 0.5
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['legend.handleheight'] = 0.5

# colors to use in the plotting
COLORS_DICT = {
    "mf_ei": cm.plasma(0.1),
    "mf_mes": cm.plasma(0.2),
    "mf_kg": cm.plasma(0.5),
    "sf_kg": cm.plasma(0.6),
    "sf_ei": cm.plasma(0.75),
    "sf_mes": cm.plasma(0.80),
    "random": "k",
}

MARKERS = {
    "mf_mes": "o",
    "sf_mes": "o",
    "sf_ei": "D",
    "mf_ei": "D",
    "random": "x",
}

LINESTYLE = {
    "mf_mes": "--",
    "sf_mes": "-",
    "sf_ei": "-",
    "sf_kg": "-",
    "mf_ei": "--",
    "mf_kg": "--",
    "random": ":",
}


OPTIM = {"branin": -0.397887, "park": 25.5893, "hartmann": 1}


def plot_benchmark_synthetic_functions(
    results_path: str, plot_path: str, simulations: list, optimizers: list
) -> None:
    """Plot benchmark results for synthetic functions.

    This script takes the folders corresponding to some given runs and returns
    one figure per simulation plotting simple regret vs budget.

    Args:
        results_path: str, path to the results folder
        plot_path: str, path to the plot folder
        simulations: list, simulations to plot
        optimizers: list, optimizer runs to plot

    """

    results_path = Path(results_path)

    plots_path = Path(plot_path).joinpath(results_path.name)

    if not plots_path.exists():
        plots_path.mkdir(parents=True, exist_ok=True)

    files = sorted(os.listdir(results_path))

    regrets = {}

    for simulation in simulations:

        plt.figure(figsize=(12, 12))

        # filter simulation
        sim_files = [file for file in files if simulation in file]

        # filter af
        sim_files = [
            file for file in sim_files if any(opt in file for opt in optimizers)
        ]

        # join benchmark path
        sim_files = [results_path.joinpath(file) for file in sim_files]

        # get number of steps in single fidelity run. This will be the absolute x value for budget
        # for this we load one of the single fidelity files
        sf_runs = [file for file in sim_files if "single_fidelity" in file.name][0]

        sf_example = [file for file in os.listdir(sf_runs) if ".csv" in file][0]

        df_example = pd.read_csv(sf_runs.joinpath(sf_example))

        # cumulative cost
        cumcost = df_example.groupby('step')['cost'].sum().cumsum()

        # real scale (cumulative cost referred to the single fidelity dataframe)
        budget_steps = cumcost.values

        # # initialization cost
        # init_cost = cumcost.min()

        sim_files = sorted(sim_files)

        # this is to plot query frequency
        mean_queries = []
        std_queries = []
        name_queries = []

        # compute results for all different runs in an optimizer
        for optim_run in sim_files:

            # this block is used to extract the name of the simulation, which will be used to name stuff
            name_dict = extract_name_dict(optim_run.name)

            name = name_dict["optimizer"]

            # ugly way of replacing stuff
            name = name.replace("multi_fidelity", "mf")
            name = name.replace("single_fidelity", "sf")

            # get all the .csv files inside the optimizer (each one corresponds to a seed)
            optim_run_files = [file for file in os.listdir(optim_run) if ".csv" in file]

            simple_regret = []

            queries = []

            # compute results for each individual file
            for f_run in optim_run_files:

                # I need a function that computes the simple regret for each budget measure in
                # the input

                df = pd.read_csv(optim_run.joinpath(f_run))

                # extract simple regret
                # Here we need to compute simple regret if synthetic functions, else
                # absolute value of the target magnitude (for real problems)
                sr = extract_sr(
                    df, budget_steps, "result", "fidelity", best=OPTIM[simulation]
                )

                simple_regret.append(sr)

                # if multi_fidelity, compute extra stuff
                if "mf" in name:
                    norm_queries = (
                        df[df['step'] != 0]["fidelity"]
                        .value_counts(normalize=True)
                        .values
                    )
                    if len(norm_queries) == 1:
                        norm_queries = np.array([norm_queries[0], 0.0])
                    queries.append(norm_queries)

            simple_regret = np.array(simple_regret)

            budgets = budget_steps

            # compute mean and std dev for plotting
            mean = np.mean(simple_regret, axis=0)

            std_dev = np.std(simple_regret, axis=0) / np.sqrt(len(simple_regret))

            regrets[name] = mean

            # plot
            plt.plot(
                budgets,
                mean,
                label=name.replace("_", " ").upper(),
                color=COLORS_DICT[name],
                linestyle=LINESTYLE[name],
                linewidth=4,
                marker=MARKERS[name],
                markersize=10,
            )

            plt.fill_between(
                budgets,
                mean - std_dev,
                mean + std_dev,
                alpha=0.2,
                color=COLORS_DICT[name],
            )

            if "mf" in name:
                mean_queries.extend(list(np.mean(queries, axis=0)))
                std_queries.extend(list(np.std(queries, axis=0)))
                name_queries.extend([name] * 2)

        # compute discount for each AF
        discount_mes = compute_delta_cost(
            mf_array=regrets["mf_mes"], sf_array=regrets["sf_mes"], budgets=budget_steps
        )

        discount_ei = compute_delta_cost(
            mf_array=regrets["mf_ei"], sf_array=regrets["sf_ei"], budgets=budget_steps
        )

        plt.xlim(0, max(budgets))
        plt.grid(0.8)
        plt.legend(loc="best")
        plt.title(f"{simulation}".capitalize(), fontsize=30)

        # Remove upper and right spines
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(f"{plots_path}/plot_{simulation}_normal.png")
        plt.yscale("log")
        plt.savefig(f"{plots_path}/plot_{simulation}.svg")

        plt.text(
            0.5,
            0.85,
            f'discount mes: {discount_mes}',
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
        )
        plt.text(
            0.5,
            0.7,
            f'discount ei: {discount_ei}',
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
        )

        plt.savefig(f"{plots_path}/plot_{simulation}.png")

        # plot fidelity query distribution
        plot_data = pd.DataFrame(
            {
                "mean": mean_queries,
                "std": std_queries,
                "af": name_queries,
                "fidelity": [0, 1] * int(len(mean_queries) / 2),
            }
        )
        plot_data["fidelity"].replace({0: "low", 1: "high"}, inplace=True)
        plt.figure()
        sns.barplot(
            data=plot_data, x="fidelity", y="mean", hue="af", palette=COLORS_DICT
        )
        plt.grid(0.8)
        plt.ylim(0, 1)
        # Remove upper and right spines
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(f"{plots_path}/plot_queries_{simulation}.png")
        plt.savefig(f"{plots_path}/plot_queries_{simulation}.svg")


def compute_delta_cost(
    sf_array: np.array, mf_array: np.array, budgets: np.array
) -> np.array:

    """Compute average gain of MF over SF (or the loss) as a cost delta.
    We normalize to the single fidelity cost."""

    def _find_closest(arr, value):
        arr = np.array(arr)
        index = (np.abs(arr - value)).argmin()
        return index

    target = min(mf_array) * 2

    mf_ind = _find_closest(mf_array, target)
    sf_ind = _find_closest(sf_array, target)

    mf_cost = budgets[mf_ind]
    sf_cost = budgets[sf_ind]

    delta = (sf_cost - mf_cost) / sf_cost

    return delta


def extract_sr(
    df: pd.DataFrame,
    budgets: list,
    result_col_name: str,
    fidelity_name: str,
    best: float = None,
):
    """Function to extract simple regret from a dataframe
    referred to budgets. It is necessary to refer the multi fidelity results to the
    single fidelity cost scale, so we can compare them and compute averages for all the
    runs (multi-fidelity runs may have different lenghts).

    Args
        df: pd.Dataframe, dataframe where you extract the results from
        budgets: list, list containing the steps (budgets used for the single
                 fidelity optimization, each step corresponds to an increase equivalent to
                 the highest fidelity cost)
        best: float, optimum value for the simulation
        result_col_name: str, name of the column storing the target variable "y"
        fidelity_name: str, name of the column storing the fidelity parameter
    """

    # compute cumulative cost for all fidelities (along the optimization)
    df['cumcost'] = df['cost'].cumsum()

    # get the indices of the highest fidelity function (this is the only level where we compute results)
    ind = df[df[fidelity_name] == 1].index

    # get a "good" dataframe containing only high fidelity
    good = df.iloc[ind]

    # compute the simple regret at each step by substracting the result at each step to the best result
    if best:
        good["sr"] = best - df.iloc[ind][result_col_name]

        # compute the simple regret along the optimization as a cumulative minimum
        good["sr"] = good["sr"].cummin()

        # this series stores the simple regret at each step of the optimization
        sr_init = good.groupby('step')['sr'].min().reset_index()

    else:
        good["sr"] = df.iloc[ind][result_col_name]

        good["sr"] = good["sr"].cummax()

        sr_init = good.groupby('step')['sr'].max().reset_index()

    # store the cost at each step of the optimization
    cost_init = good.groupby('step')['cumcost'].max().reset_index()

    # here is the list with the real regret that we will plot: it will be referred to
    # the cost of the single fidelity
    sr_real = []

    # now the tricky part, for the cumulated cost at each single fidelity step, compute the
    # associated regret at the multi fidelity level (this way we can compare methods)
    for b in budgets:
        # if the cost is less than the cost of the sampling points for the MF, append nan
        # it does not make sense to compare them as the MF starts with a higher cost
        if all(b < cost_init['cumcost']):
            sr_real.append(np.nan)

        # else, the regret for this cost step is the one associated to the values that
        # are lower than this
        else:
            val = sr_init[b >= cost_init['cumcost']]["sr"].values[-1]
            sr_real.append(val)

    return np.array(sr_real)


def extract_name_dict(folder_path: str) -> Dict[str, str]:
    """Extract folder name to a dictionary with all the different parameters."""
    regex = r'(\w+)=([\w\._-]+)'

    matches = re.findall(regex, folder_path)

    # capture values in a dictionary
    d = {term[0]: term[1] for term in matches}

    return d


@hydra.main(version_base=None, config_path="config_plots", config_name="synthetic")
def main(cfg: DictConfig):

    for path in cfg.path:
        plot_benchmark_synthetic_functions(
            path, cfg.plot_path, cfg.simulations, cfg.optimizers
        )


if __name__ == "__main__":

    main()
