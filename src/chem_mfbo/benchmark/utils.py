"""Utils for plotting and metric computation
"""
import pandas as pd
import numpy as np



def extract_best(df: pd.DataFrame, 
                 budgets: np.array, 
                 result_col_name: str, 
                 fidelity_name: str) -> :
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


class Scorer():
    """Implement regret computation for a run. We always need a single-fidelity reference.
    """

    def __init__(self) -> None:
        pass


    def _compute_regret():
        pass

    def _score():
        pass