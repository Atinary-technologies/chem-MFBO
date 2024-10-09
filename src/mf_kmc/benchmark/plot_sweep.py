"""Extract area under the curve from a sweep to a csv file storing the metric
for the given parameter. Then plot this file."""

import os
import re
from pathlib import Path
from typing import Dict

import hydra
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from omegaconf import DictConfig
from sklearn.metrics import auc


# colors to use in the plotting
COLORS_DICT = {
    "sf_ei": cm.plasma(0.1),
    "sf_mes": cm.plasma(0.2),
    "mf_kg": cm.plasma(0.5),
    "sf_kg": cm.plasma(0.6),
    "mf_ei": cm.plasma(0.75),
    "mf_mes": cm.plasma(0.80),
    "mf_gibbon": cm.plasma(0.85),
    "sf_gibbon": cm.plasma(0.88),
}

MARKERS = {
    "mf_mes": "x",
    "sf_mes": "s",
    "sf_ei": "D",
    "sf_kg": "^",
    "mf_ei": "v",
    "mf_kg": "o",
    "mf_gibbon": "o",
    "sf_gibbon": "v",
}

LINESTYLE = {
    "mf_mes": "--",
    "sf_mes": "-",
    "sf_ei": "-",
    "sf_kg": "-",
    "mf_ei": "--",
    "mf_kg": "--",
    "mf_gibbon": "--",
    "sf_gibbon": "-",
}

optima = {"park": 25.5893, "branin": -0.397887, "hartmann": 1.0}


def extract_performance_data(
    results_path: str, simulations: list, optimizers: list
) -> None:
    """Extract performance data to plot by directly computing stuff from .csv files."""

    for sim in simulations:

        results_path = Path(results_path)

        plots_path = Path(f"plots/sweep_final/{sim}")

        if not plots_path.exists():
            plots_path.mkdir(parents=True, exist_ok=True)

        benchmark_files = os.listdir(results_path)

        for af in optimizers:

            folders_to_plot = [
                file for file in benchmark_files if sim in file and af in file
            ]

            if folders_to_plot == []:
                continue

            # folder containing the SF run (we need it) to get the reference trace
            sf_folder = [
                file
                for file in folders_to_plot
                if "single_fidelity" in file and sim in file
            ][0]

            # get budgets from cumulative cost (reference budgets to compute regret)
            sf_example = [
                file
                for file in os.listdir(results_path.joinpath(sf_folder))
                if ".csv" in file
            ][0]

            df_example = pd.read_csv(results_path.joinpath(sf_folder, sf_example))

            # cumulative cost
            cumcost = df_example.groupby('step')['cost'].sum().cumsum()

            # real scale (cumulative cost referred to the single fidelity dataframe)
            budget_steps = cumcost.values

            optimum = optima[sim]

            sr_reference = extract_sr_average(
                sf_folder, results_path, budget_steps, best=optimum
            )

            # get MF folders
            mf_folders = [file for file in folders_to_plot if "multi_fidelity" in file]

            # this is used to store parameters and results
            results = []

            # loop over each mf_folder and compute area
            for folder in mf_folders:

                plt.figure(figsize=(12, 12))

                # get the associated parameters in the optimization
                params = extract_name_dict(folder)

                # mf_reference = extract_sr_average(folder, results_path, budget_steps)
                mf_reference = extract_sr_average(
                    folder, results_path, budget_steps, best=optimum
                )

                # compute area
                area = compute_area_difference(sr_reference, mf_reference, budget_steps)

                # compute delta cost
                delta_cost = compute_delta_cost(
                    sr_reference, mf_reference, budget_steps
                )

                save_name = f"{af}_{params.get('bias_lowfid')}_{params['cost_ratio']}"

                plt.plot(
                    budget_steps,
                    sr_reference,
                    marker='x',
                    label="sf",
                    markersize=12,
                    linewidth=4,
                )
                plt.plot(
                    budget_steps,
                    mf_reference,
                    marker='x',
                    label="mf",
                    markersize=12,
                    linewidth=4,
                )
                plt.text(
                    0.5,
                    0.9,
                    f'Area: {area}',
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    verticalalignment='top',
                )
                plt.text(
                    0.5,
                    0.6,
                    f'delta cost: {delta_cost}',
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    verticalalignment='top',
                )
                plt.xlabel("Cost")
                plt.ylabel("Simple regret")
                plt.grid(0.8)
                plt.legend(loc="best")

                plt.savefig(plots_path.joinpath(f"{save_name}.png"))
                plt.savefig(plots_path.joinpath(f"{save_name}.svg"))

                data = {
                    "bias_lowfid": params.get("bias_lowfid"),
                    "cost_ratio": params["cost_ratio"],
                    "area": area,
                    "discount": delta_cost,
                }

                results.append(data)

            # save dataframe with results
            df = pd.DataFrame(results)
            df.to_csv(f"plots/sweep_final/{sim}_{af}.csv")


def extract_name_dict(folder_path: str) -> Dict[str, str]:
    """Extract folder name to a dictionary with all the different parameters."""
    regex = r'(\w+)=([\w\._-]+)'

    matches = re.findall(regex, folder_path)

    # capture values in a dictionary
    d = {term[0]: term[1] for term in matches}

    return d


def compute_delta_cost(
    sf_array: np.array, mf_array: np.array, budgets: np.array
) -> np.array:

    """Compute average gain of MF over SF (or the loss) as a cost delta.
    We normalize to the single fidelity cost."""

    def _find_closest(arr, value):
        arr = np.array(arr)
        index = (np.abs(arr - value)).argmin()
        return index

    target = min(mf_array)*2

    mf_ind = _find_closest(mf_array, target)
    sf_ind = _find_closest(sf_array, target)

    mf_cost = budgets[mf_ind]
    sf_cost = budgets[sf_ind]

    delta = (sf_cost - mf_cost) / sf_cost

    return delta


def compute_area_difference(
    sf_array: np.array, mf_array: np.array, budgets: np.array
) -> np.array:
    """Compute area difference for a set of arrays. We add the steps that are missing to the mf array (filling
    the nans) and then complete the arrays to match them to bugdet steps starting from 0 (so we also considering
    sampling cost).

    """

    assert len(sf_array) == len(mf_array)

    area_sf = auc(budgets, sf_array)

    area_hf = auc(budgets, mf_array)

    return area_sf - area_hf


def extract_sr_average(
    folder: str, results_path: Path, budgets: np.array, best: float = None
) -> np.array:
    """Get average trace from single fidelity folder"""

    optim_run_dir = results_path.joinpath(folder)

    # get all the .csv files inside the run folder (each one corresponds to a seed)
    optim_run_files = [file for file in os.listdir(optim_run_dir) if ".csv" in file]

    simple_regret = []

    for f_run in optim_run_files:

        df = pd.read_csv(optim_run_dir.joinpath(f_run))

        sr = extract_sr(df, budgets, "result", "fidelity", best=best)

        simple_regret.append(sr)

    regret = np.mean(simple_regret, axis=0)

    return regret


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
        hf_df["sr"] = hf_df[ind][result_col_name]

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
            sr_real.append(sr_init['sr'].max())

        # else, the regret for this cost step is the one associated to the values that
        # are lower than this
        else:
            val = sr_init[b >= cost_init['cumcost']]["sr"].values[-1]
            sr_real.append(val)

    return np.array(sr_real)


def plot_heatmap(simulations: list, optimizers: list):

    for sim in simulations:

        for optim in optimizers:
            df = pd.read_csv(f"plots/sweep_final/{sim}_{optim}.csv")
            df_heat = df[["bias_lowfid", "cost_ratio", "discount"]]

            df_heat.rename(columns={"delta_cost": "discount"}, inplace=True)

            df_heat["discount"] = df_heat["discount"].apply(lambda x: round(x, 2))
            df_heat["cost_ratio"] = df_heat["cost_ratio"].apply(lambda x: round(x, 2))
            df_heat = df_heat.pivot(
                index="cost_ratio", columns="bias_lowfid", values="discount"
            )
            plt.figure(figsize=(5, 5))
            sns.heatmap(df_heat, cmap='coolwarm', center=0, cbar=False)

            # Remove axis labels
            plt.xlabel('')
            plt.ylabel('')


            for ax in plt.gcf().axes:
                ax.tick_params(
                    axis='both', which='both',
                    bottom=False, top=False, left=False, right=False,

                )
                
            plt.savefig(f"plots/sweep_final/{sim}_{optim}_heatmap.png")
            plt.savefig(f"plots/sweep_final/{sim}_{optim}_heatmap.svg")
            plt.close()

            plt.figure(figsize=(6, 3))
            if sim == "branin":
                r2 = [0.6, 0.7, 0.76, 0.77, 0.80, 0.86, 0.91, 0.94, 0.98, 0.99]

            elif sim == "park":
                r2 = [0.09, -0.01, 0.2, 0.46, 0.73, 0.82, 0.92, 0.97, 0.99, 0.99]
    
            plt.scatter(np.linspace(0, 0.9, 10), r2, s=175, c='g')
            
            plt.xticks([])
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.savefig(f"plots/sweep_final/{sim}_plot.png")
            plt.savefig(f"plots/sweep_final/{sim}_plot.svg")


@hydra.main(version_base=None, config_path="config_plots", config_name="sweep")
def main(cfg: DictConfig):

    if cfg.extract:
        extract_performance_data(cfg.path, cfg.simulations, cfg.optimizers)

    if cfg.plot:
        plot_heatmap(cfg.simulations, cfg.optimizers)


if __name__ == "__main__":

    main()
