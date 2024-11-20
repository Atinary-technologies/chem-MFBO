"""Functions to do initial sampling"""

import random
from typing import Dict, List, Union

import numpy as np
from scipy.stats.qmc import LatinHypercube

from chem_mfbo.simulations.config import SimulationConfig


def sample_points(
    config: SimulationConfig,
    optim_name: str,
    budget: int = 50,
    init_samples_budget: float = 0.1,
    low_fid_samps_ratio: float = 0.8,
    cost_ratio: float = 0.1,
    strategy: str = "latinHC",
    seed: int = 33,
) -> Union[List[Dict], List[Dict]]:

    """Sample initial points. We stick to LatinHypercube sampling, although more strategies are
    possible.

    We define the number of high fidelity points and low fidelity points based on a proportion of the
    total budget and a proportion of low fidelity points from that initial sampling budget. The high
    fidelity cost is always 1, so the budget must be an integer"""

    # HERE COMPUTE NEW FRACTIONS OF POINTS
    n_inp_dim = len(config.input_parameters)

    init_budget = init_samples_budget * budget

    if "multi_fidelity" in optim_name:
        n_high_fid = round(init_budget * (1 - low_fid_samps_ratio))

        n_low_fid = round(init_budget * low_fid_samps_ratio / cost_ratio)

    elif "single_fidelity" in optim_name:
        n_high_fid = int(init_budget)
        n_low_fid = 0

    n_total = n_high_fid + n_low_fid

    high_bounds = np.array([inp.high_value for inp in config.input_parameters])
    low_bounds = np.array([inp.low_value for inp in config.input_parameters])

    # build list of dictionaries of inputs and init samples depending on strategies
    # we select input parameters and fidelities. We just have one fidelity level
    if strategy == 'random':

        samples = []

        for _ in range(n_total):

            input_sample = {}

            for inp in config.input_parameters:
                high = inp.high_value
                low = inp.low_value
                value = random.uniform(low, high)
                input_sample[inp.name] = value

            if "multi_fidelity" in optim_name:

                fid_value = random.choice(config.normalized_fidelities)
                fidelity = {"fidelity": fid_value}

            elif "single_fidelity" in optim_name:
                fidelity = {config.fidelity.name: 1.0}

            samples.append((input_sample, fidelity))

    elif strategy == "latinHC":

        final_samples = []

        # sample n_total latin Hypercube and then transform back to dimensions
        sampler = LatinHypercube(d=n_inp_dim, seed=seed)

        samples = sampler.random(n=n_high_fid)

        # this has to change if we use more than 1 fidelity levels
        samples = np.concatenate((samples, sampler.random(n=n_low_fid)), axis=0)

        scaled = (high_bounds - low_bounds) * samples + low_bounds

        # list with all the values for the fidelities (ugly but for the moment it's fine)
        fids = [1.0] * n_high_fid + [config.normalized_fidelities[0]] * n_low_fid

        for sample, fid in zip(scaled, fids):

            input_sample = {}

            for i, inp in enumerate(config.input_parameters):
                input_sample[inp.name] = sample[i]

            fidelity = {config.fidelity.name: fid}

            final_samples.append((input_sample, fidelity))

    return final_samples
