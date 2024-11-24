"""Utils for plotting and metric computation
"""
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import re
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import seaborn as sns
import ast
from copy import copy
from matplotlib.colors import TwoSlopeNorm

def extract_name_dict(folder_path: str) -> Dict[str, str]:
    """Extract folder name to a dictionary with all the different parameters."""
    regex = r'(\w+)=([\w\._-]+)'

    matches = re.findall(regex, folder_path)

    # capture values in a dictionary
    d = {term[0]: term[1] for term in matches}

    return d


def extract_best(df: pd.DataFrame, 
                 budgets: np.array, 
                 result_col_name: str, 
                 fidelity_name: str):
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
) -> np.array:
    
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

def compute_discount(
    sf_array: np.array, 
    mf_array: np.array, 
    budgets: np.array,
    tau: float = 0.9,
) -> np.array:

    """Compute discount. Gain of MF over SF (or the loss) as a cost delta.
    We normalize to the single fidelity cost. Tau represents the amount """

    def _find_closest(arr, value):
        arr = np.array(arr)
        index = (np.abs(arr - value)).argmin()

        return index

    # compute the absolute regret difference in the SF scale scaled by tau
    # this is the amount of regret we want to minimize
    regret_reduced = (max(sf_array) - min(sf_array)) * tau

    # target regret is the 
    target = max(sf_array) - regret_reduced

    if all(mf_array) > target:
        delta = -1
        return delta

    mf_ind = _find_closest(mf_array, target)
    sf_ind = _find_closest(sf_array, target)
    mf_cost = budgets[mf_ind]
    sf_cost = budgets[sf_ind]

    mf_cost = np.mean(mf_cost)
    sf_cost = np.mean(sf_cost)
    
    delta = (sf_cost - mf_cost) / sf_cost

    return delta



def plot_3d_heatmap(simulations: list, 
                 optimizers: list,
                 tau: float = 0.9):

    for sim in simulations:

        for optim in optimizers:
            
            df = pd.read_csv(f"plots/sweep_final/{sim}_{optim}.csv")

            # reformat dataframe
            df["discount"] = df["discount"].apply(lambda x: np.array(ast.literal_eval(x)))
            df["taus"] = df["taus"].apply(lambda x: np.array(ast.literal_eval(x)))
            # expand dataframe
            expanded_data = []

            for _, row in df.iterrows():
                for tau, discount in zip(row['taus'], row['discount']):
                    expanded_data.append({
                        'bias_lowfid': row['bias_lowfid'],
                        'cost_ratio': row['cost_ratio'],
                        'tau': tau,
                        'discount': discount
                    })

            expanded_df = pd.DataFrame(expanded_data)

            # Diverging colormap normalization centered around 0
            # TwoSlopeNorm for centering colormap at 0
            norm = TwoSlopeNorm(vmin=-1, vmax=0.7, vcenter=0)

            # Plotting
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot
            sc = ax.scatter(
                expanded_df['bias_lowfid'],
                expanded_df['cost_ratio'],
                expanded_df['tau'],
                c=expanded_df['discount'],
                cmap='coolwarm',
                norm=norm
            )

            # Adding labels
            ax.set_xlabel(r'$\alpha$')
            ax.set_ylabel(r'$\rho$')
            ax.set_zlabel(r'$\tau$')
            plt.colorbar(sc, label=r'$\Delta$')
            plt.tight_layout()
            plt.savefig(f"plots/sweep_final/3d_{sim}_{optim}_plot.png")
            plt.savefig(f"plots/sweep_final/3d_{sim}_{optim}_plot.svg")


def plot_heatmap(simulations: list, 
                 optimizers: list,
                 tau: float = 0.9):

    for sim in simulations:

        for optim in optimizers:

            df = pd.read_csv(f"plots/sweep_final/{sim}_{optim}.csv")

            df["discount"] = df["discount"].apply(lambda x: np.array(ast.literal_eval(x)))
            df["taus"] = df["taus"].apply(lambda x: np.array(ast.literal_eval(x)))

            # get target discount
            target_id = np.where(df["taus"].values[0] == tau)[0][0]

            df["discount"] = df["discount"].apply(lambda x: float(x[target_id]))

            df_heat = copy(df)

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
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
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
