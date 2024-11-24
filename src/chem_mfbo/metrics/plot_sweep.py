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
from chem_mfbo.metrics.utils import compute_discount, extract_sr_average, extract_name_dict, plot_heatmap, compute_area_difference, plot_3d_heatmap
from omegaconf import DictConfig


#configure_plotting()

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
                
                # compute discount for several taus
                taus = list(np.linspace(0.1, 1, 10))

                # compute delta cost
                discount = [compute_discount(
                    sr_reference, 
                    mf_reference, 
                    budget_steps,
                    tau
                ) for tau in taus]

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
                    "discount": discount,
                    "taus": taus
                }

                results.append(data)

            # save dataframe with results
            df = pd.DataFrame(results)
            df.to_csv(f"plots/sweep_final/{sim}_{af}.csv")



@hydra.main(version_base=None, config_path="../../../config_plots", config_name="sweep")
def main(cfg: DictConfig):

    if cfg.extract:
        extract_performance_data(cfg.path, 
                                 cfg.simulations, 
                                 cfg.optimizers)

    if cfg.plot:
        plot_heatmap(cfg.simulations, cfg.optimizers)
        plot_3d_heatmap(cfg.simulations, cfg.optimizers)


if __name__ == "__main__":

    main()
