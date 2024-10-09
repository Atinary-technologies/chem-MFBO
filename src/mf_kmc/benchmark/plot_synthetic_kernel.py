"""Code to make plots for the synthetic functions traces.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, Union

import hydra
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

# some parameters to plot
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams["font.size"] = 30
plt.rcParams["axes.titlesize"] = 21
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30

plt.rcParams['figure.figsize'] = (20, 12)
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
    "mf_mes": cm.plasma(0.1),
    "sf_mes": cm.plasma(0.2),
    "mf_kg": cm.plasma(0.5),
    "sf_kg": cm.plasma(0.6),
    "mf_ei": cm.plasma(0.75),
    "sf_ei": cm.plasma(0.80),
}

MARKERS = {
    "mf_mes": "x",
    "sf_mes": "s",
    "sf_ei": "D",
    "sf_kg": "^",
    "mf_ei": "v",
    "mf_kg": "o",
}

LINESTYLE = {
    "mf_mes": "--",
    "sf_mes": "-",
    "sf_ei": "-",
    "sf_kg": "-",
    "mf_ei": "--",
    "mf_kg": "--",
}

SIMULATION_OPTIM = {"branin": -0.397887}


def get_paths_to_plot(
    path: str, simulations: list[str], optims: list[str], constraints: list[str]
) -> Dict:

    """Get files to plot and the corresponding simulation"""
    files = os.listdir(path)

    sims_to_plot = {}

    for surface in simulations:

        runs = [file for file in files if surface in file]

        if runs == []:
            print(f"No results for {surface}")
            continue

        dict_runs = {}

        for optim in optims:

            optim_files = [f"{path}/{file}" for file in runs if optim in file]

            dict_runs[optim] = optim_files

        sims_to_plot[surface] = dict_runs

    return sims_to_plot


def get_paths_chemistry(path: str):
    """Get paths to plot sweep of real problems"""
    folders = [folder for folder in os.listdir(path) if "lowfid" in folder]

    mes = [f"{path}/{folder}/mf_MES" for folder in folders]

    ei = [f"{path}/{folder}/mf_EI" for folder in folders]

    # I hard coded this, it can be changed later
    mes = mes + [f"{path}/lowfid=0.0/sf_MES"]

    ei = ei + [f"{path}/lowfid=0.0/sf_EI"]

    to_plot = {"cofs": {"mes": mes, "ei": ei}}

    return to_plot


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


def extract_avg_data(
    name: str, budget: int, result: str, best: float
) -> Union[np.array, np.array]:

    simple_regret = []

    optim_run_files = [file for file in os.listdir(name) if ".csv" in file]

    for f_run in optim_run_files:

        df = pd.read_csv(f"{name}/{f_run}")

        sr = extract_sr(df, budget, result, "fidelity", best=best)

        simple_regret.append(sr)

    simple_regret = np.array(simple_regret)

    # compute mean and std dev for plotting
    mean = np.mean(simple_regret, axis=0)

    std_dev = np.std(simple_regret, axis=0) / np.sqrt(len(simple_regret))

    return mean, std_dev


@hydra.main(version_base=None, config_path="config_plots", config_name="synthetic.yaml")
def main(cfg: DictConfig):

    savepath = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if "cofs" in cfg.path:
        dicts = get_paths_chemistry(cfg.path)

    else:
        dicts = get_paths_to_plot(
            cfg.path, cfg.simulations, cfg.optimizers, cfg.constraints
        )

    for key in dicts.keys():

        os.makedirs(f"{savepath}/{key}", exist_ok=True)

        runs = dicts[key]

        optim = SIMULATION_OPTIM.get(key)

        for af in runs.keys():

            to_plot = runs[af]

            n_rows = round(len(to_plot) / 5)
            n_cols = 5

            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)

            # get sf filename
            sf = [
                file for file in to_plot if "single_fidelity" in file or "sf" in file
            ][0]

            to_plot.remove(sf)

            to_plot = sorted(to_plot)

            budgets = list(range(cfg.initial_cost, cfg.budget_steps + 1))

            if key == "cofs":
                budgets = list(np.linspace(690, 6440, 26))

            # get info from sf
            if key == "cofs":
                mean_sf, std_sf = extract_avg_data(sf, budgets, cfg.result, None)

            else:
                mean_sf, std_sf = extract_avg_data(sf, budgets, cfg.result, optim)

            name_mf = f"mf_{af}"
            name_sf = f"sf_{af}"

            # for each mf, plot it in a different subplot
            for i, mf_name in enumerate(to_plot):

                if key == "cofs":
                    kernel = re.search(r'lowfid=([0-9]*\.?[0-9]+)', mf_name).group(1)

                else:
                    kernel = re.search(
                        r'lowfid_kernel=([0-9]*\.?[0-9]+)', mf_name
                    ).group(1)

                if i > 4:
                    row = 1
                    col = i - 5

                else:
                    row = 0
                    col = i

                if key == "cofs":
                    mean_mf, std_mf = extract_avg_data(
                        mf_name, budgets, cfg.result, None
                    )

                else:
                    mean_mf, std_mf = extract_avg_data(
                        mf_name, budgets, cfg.result, optim
                    )

                # plot mf
                axes[row, col].plot(
                    budgets,
                    mean_mf,
                    marker=MARKERS[name_mf],
                    label=name_mf.replace("_", " ").upper(),
                    color=COLORS_DICT[name_mf],
                    markersize=7,
                    linestyle=LINESTYLE[name_mf],
                )

                # plot sf
                axes[row, col].plot(
                    budgets,
                    mean_sf,
                    marker=MARKERS[name_sf],
                    label=name_sf.replace("_", " ").upper(),
                    color=COLORS_DICT[name_sf],
                    markersize=7,
                    linestyle=LINESTYLE[name_sf],
                )

                axes[row, col].set_title(f"kernel={kernel}")
                # more stuff
            fig.text(0.5, 0.04, 'Cost', ha='center')
            fig.text(0.04, 0.5, 'Simple Regret', va='center', rotation='vertical')
            fig.legend(
                labels=[f"mf {af}".upper(), f"sf {af}".upper()],
                loc='upper center',
                ncol=2,
            )
            plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])

            fig.savefig(f"{savepath}/{key}/{af}.png")
            fig.savefig(f"{savepath}/{key}/{af}.svg")


if __name__ == "__main__":
    main()
