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
from chem_mfbo.metrics.plot_config import COLORS_DICT, MARKERS, LINESTYLE, OPTIM, configure_plotting
from chem_mfbo.metrics.utils import extract_sr, compute_discount, extract_name_dict
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

configure_plotting()


def plot_benchmark_synthetic_functions(
    results_path: str, 
    plot_path: str, 
    simulations: list, 
    optimizers: list,
    discount: bool = True,
) -> None:
    """Plot benchmark results for synthetic functions.

    This script takes the folders corresponding to some given runs and returns
    one figure per simulation plotting simple regret vs budget.

    Args:
        results_path: str, path to the results folder
        plot_path: str, path to the plot folder
        simulations: list, simulations to plot
        optimizers: list, optimizer runs to plot
        discount: bool, compute discount for several tau values

    """

    results_path = Path(results_path)

    plots_path = Path(plot_path).joinpath(results_path.name)

    if not plots_path.exists():
        plots_path.mkdir(parents=True, exist_ok=True)

    files = sorted(os.listdir(results_path))

    regrets = {}

    for simulation in simulations:

        plt.figure()

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

        # sort files
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

        # compute discount if necessary
        if discount:

            taus = np.linspace(0.1, 1, 10)

            # compute discount for each AF
            if "mf_mes" in regrets: 
                
                discount_mes = [compute_discount(
                    mf_array=regrets["mf_mes"], 
                    sf_array=regrets["sf_mes"], 
                    budgets=budget_steps, 
                    tau=tau) for tau in taus]

                mes_discount_df = pd.DataFrame({"tau": taus,
                                               "discount": discount_mes})

                mes_discount_df.to_csv(f"{plots_path}/mes_discount_{simulation}.csv")    

            if "mf_ei" in regrets:

                discount_ei = [compute_discount(
                    mf_array=regrets["mf_ei"], 
                    sf_array=regrets["sf_ei"], 
                    budgets=budget_steps,
                    tau=tau) for tau in taus]
                
                ei_discount_df = pd.DataFrame({"tau": taus,
                                               "discount": discount_ei})
                
                ei_discount_df.to_csv(f"{plots_path}/ei_discount_{simulation}.csv")

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


@hydra.main(version_base=None, config_path="../../../config_plots", config_name="synthetic")
def main(cfg: DictConfig):
    print("Plotting synthetic results...")
    for path in cfg.path:
        plot_benchmark_synthetic_functions(
            path, cfg.plot_path, cfg.simulations, cfg.optimizers
        )


if __name__ == "__main__":

    main()
