"""Code to make plots for the benchmarking framework."""
import os
from pathlib import Path

import hydra
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
from omegaconf import DictConfig
from typing import Dict
from chem_mfbo.metrics.plot_config import COLORS_DICT, MARKERS, LINESTYLE, PROBLEM_CONFIG, configure_plotting
from chem_mfbo.metrics.utils import extract_sr, compute_discount
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


configure_plotting()


def plot_benchmark_real_problems(
    results_path: str, 
    plots_path: str, 
    simulation: str, 
    optimizers: list,
    discount: bool = True
) -> None:
    """Plot benchmark results for real problems.

    This script takes the folders corresponding to some given runs and returns
    one figure per simulation plotting best result found vs budget.
    
    Simulations to be plotted and runs can be modified from the config_plots file
    in config_plots.
    """

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

        plt.figure()

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
                sr = extract_sr(
                    df, budget_steps, results_name, fidelity_name, best=best
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
                marker=MARKERS[name],
                label=name.replace("_", " ").upper().replace("RANDOM", "random"),
                color=COLORS_DICT[name],
                linestyle=LINESTYLE[name],
                markersize=10,
                linewidth=4,
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

        plt.grid(0.8, which="both")
        plt.xlim(0, max(budgets))
        plt.legend(loc='lower right')
        plt.title(f"{simulation}", fontsize=30)

        # Remove upper and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(f"{plots_path}/{file}.svg")
        plt.savefig(f"{plots_path}/{file}.png")


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

                mes_discount_df.to_csv(f"{plots_path}/mes_discount_{simulation}_{file}.csv")    

            if "mf_ei" in regrets:

                discount_ei = [compute_discount(
                    mf_array=regrets["mf_ei"], 
                    sf_array=regrets["sf_ei"], 
                    budgets=budget_steps,
                    tau=tau) for tau in taus]
                
                ei_discount_df = pd.DataFrame({"tau": taus,
                                               "discount": discount_ei})
                
                ei_discount_df.to_csv(f"{plots_path}/ei_discount_{simulation}_{file}.csv")


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


@hydra.main(version_base=None, config_path="../../../config_plots", config_name="chemistry")
def main(cfg: DictConfig):

    print("Plotting chemistry experiments...")
    for simulation in cfg.simulations:
        plot_benchmark_real_problems(
            cfg.path, cfg.plots_path, simulation, cfg.optimizers
        )


if __name__ == "__main__":

    main()
