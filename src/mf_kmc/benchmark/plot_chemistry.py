"""Code to make plots for the benchmarking framework."""
import os
from pathlib import Path

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
colors = cm.plasma(np.linspace(0.1, 0.75, 6))

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

# Problem-specific configurations
PROBLEM_CONFIG = {
    "cofs": {
        "optimum": 18.56,  # Example optimum for cofs
        "result_col_name": "selectivity",
        "fidelity_name": "fidelity",
    },
    "polarizability": {
        "optimum": 1.0,  # Example optimum for polarizability
        "result_col_name": "Polarizability",
        "fidelity_name": "fidelity",
    },
    "kinetic": {
        "optimum": 1,  # Example optimum for kinetic
        "result_col_name": "result",
        "fidelity_name": "fidelity",
    },
    "freesolv": {
        "optimum": 25.47,  # Example optimum for freesolv
        "result_col_name": "solvation",
        "fidelity_name": "fidelity",
    },
}


def plot_benchmark_real_problems(
    results_path: str, plots_path: str, simulation: str, optimizers: list
) -> None:
    """Plot benchmark results for real problems.

    This script takes the folders corresponding to some given runs and returns
    one figure per simulation plotting best result found vs budget"""

    results_path = Path(results_path).joinpath(f"results_{simulation}")

    plots_path = Path(plots_path).joinpath(simulation)

    if not plots_path.exists():
        plots_path.mkdir(parents=True, exist_ok=True)

    files = os.listdir(results_path)

    # get only the paths to plot
    files = [file for file in files if "=" in file]

    results_name = PROBLEM_CONFIG[simulation]["result_col_name"]
    fidelity_name = PROBLEM_CONFIG[simulation]["fidelity_name"]
    best = PROBLEM_CONFIG[simulation]["optimum"]

    for file in files:

        path = results_path.joinpath(file)

        run_files = os.listdir(path)

        run_files = [file for file in run_files if file in optimizers]

        plt.figure(figsize=(12, 12))

        # get number of steps in single fidelity run. This will be the absolute x value for budget
        sf_runs = [file for file in run_files if "sf" in file][0]

        sf_example = [
            file for file in os.listdir(path.joinpath(sf_runs)) if ".csv" in file
        ][0]

        df_example = pd.read_csv(path.joinpath(sf_runs, sf_example))

        # cumulative cost
        cumcost = df_example.groupby('step')['cost'].sum().cumsum()

        # real scale for plotting the x axis
        budget_steps = cumcost.values[:36]

        # files = [file for file in files]

        # sim_files = sorted(files)

        # this is to plot query frequency
        mean_queries = []
        std_queries = []
        name_queries = []

        regrets = {}

        run_files = sorted(run_files)

        # swap random for legend style
        if "random" in run_files:
            run_files.remove("random")
            run_files.append("random")

        # compute results for all different runs in an optimizer
        for run in run_files:

            # ugly way of replacing stuff
            name = run.replace("single_fidelity", "sf")
            name = run.replace("multi_fidelity", "mf")

            name = name.lower()

            optim_run_dir = path.joinpath(run)

            # get all the .csv files inside the optimizer (each one corresponds to a seed)
            optim_run_files = [
                file for file in os.listdir(optim_run_dir) if ".csv" in file
            ]

            simple_regret = []

            queries = []

            # compute results for each individual file
            for f_run in optim_run_files:

                # I need a function that computes the simple regret for each budget measure in
                # the input

                df = pd.read_csv(optim_run_dir.joinpath(f_run))

                # extract simple regret
                sr = extract_sr(df, budget_steps, results_name, fidelity_name, best=best)

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
                marker=MARKERS[name],
                label=name.replace("_", " ").upper().replace("RANDOM", "random"),
                color=COLORS_DICT[name],
                linestyle=LINESTYLE[name],
                markersize=10,
                linewidth=4
            )

            if name == "random":
                plt.fill_between(
                    budgets,
                    mean - std_dev,
                    mean + std_dev,
                    alpha=0.075,
                    color=COLORS_DICT[name],
                )
            else:
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
        discount_mes = compute_delta_cost(mf_array=regrets["mf_mes"],
                                          sf_array=regrets["sf_mes"],
                                          budgets=budgets)

        discount_ei = compute_delta_cost(mf_array=regrets["mf_ei"],
                                         sf_array=regrets["sf_ei"],
                                         budgets=budgets)

        plt.grid(0.8, which="both")
        plt.xlim(0, max(budgets))
        plt.legend(loc='lower right')
        plt.title(f"{simulation}", fontsize=30)

        # Remove upper and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(f"{plots_path}/{file}.svg")
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
        plt.savefig(f"{plots_path}/{file}.png")
        
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
        plt.savefig(f"{plots_path}/plot_queries_{file}.png")
        plt.savefig(f"{plots_path}/plot_queries_{file}.svg")


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

    if all(sf_array) > target:
        delta = 1
        return delta
       
    mf_ind = _find_closest(mf_array, target)
    sf_ind = _find_closest(sf_array, target)
    mf_cost = budgets[mf_ind]
    sf_cost = budgets[sf_ind]

    mf_cost = np.mean(mf_cost)
    sf_cost = np.mean(sf_cost)

    delta = (sf_cost - mf_cost) / sf_cost

    if delta <= 0:
        delta = 0

    return delta


def extract_best(df, budgets, result_col_name, fidelity_name):
    """Function to extract best results from a dataframe referred to budgets"""

    df['cumcost'] = df['cost'].cumsum()

    ind = df[df[fidelity_name] == 1].index

    good = df.iloc[ind]

    good["sr"] = df.iloc[ind][result_col_name]

    good["sr"] = good["sr"].cummax()

    sr_init = good.groupby('step')['sr'].max().reset_index()

    cost_init = good.groupby('step')['cumcost'].max().reset_index()

    # here is the list with the sr real
    sr_real = []

    for b in budgets:
        if all(b < cost_init['cumcost']):
            sr_real.append(np.nan)

        else:
            val = sr_init[b >= cost_init['cumcost']]["sr"].values[-1]
            sr_real.append(val)

    return np.array(sr_real)


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

    # get a "hf_df" dataframe containing only high fidelity
    hf_df = df.iloc[ind]

    # compute the simple regret at each step by substracting the result at each step to the best result
    if best:
        hf_df["sr"] = best - hf_df[result_col_name]

        # compute the simple regret along the optimization as a cumulative minimum
        hf_df["sr"] = hf_df["sr"].cummin()

        # this series stores the simple regret at each step of the optimization
        sr_init = hf_df.groupby('step')['sr'].min().reset_index()

    else:
        hf_df["sr"] = hf_df[result_col_name]

        hf_df["sr"] = hf_df["sr"].cummax()

        sr_init = hf_df.groupby('step')['sr'].max().reset_index()

    # store the cost at each step of the optimization
    cost_init = hf_df.groupby('step')['cumcost'].max().reset_index()

    # here is the list with the real regret that we will plot: it will be referred to
    # the cost of the single fidelity
    sr_real = []

    # now the tricky part, for the cumulated cost at each single fidelity step, compute the
    # associated regret at the multi fidelity level (this way we can compare methods)
    for b in budgets:
        # if the cost is less than the cost of the sampling points for the MF, append nan
        # it does not make sense to compare them as the MF starts with a higher cost
        if all(b < cost_init['cumcost']):

            if best:
                sr_real.append(sr_init['sr'].min())
            else:
                sr_real.append(sr_init['sr'].min())

        # else, the regret for this cost step is the one associated to the values that
        # are lower than this
        else:
            val = sr_init[b >= cost_init['cumcost']]["sr"].values[-1]
            sr_real.append(val)

    return np.array(sr_real)


@hydra.main(version_base=None, config_path="config_plots", config_name="chemistry")
def main(cfg: DictConfig):

    for simulation in cfg.simulations:
        plot_benchmark_real_problems(
            cfg.path, cfg.plots_path, simulation, cfg.optimizers
        )


if __name__ == "__main__":

    main()
