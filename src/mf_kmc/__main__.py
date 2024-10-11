"""Run simulation"""
import logging
import random
from typing import Any, Dict

import numpy as np
import torch

from mf_kmc.optimization.factory import optim_factory
from mf_kmc.optimization.sampling import sample_points
from mf_kmc.simulations import config
from mf_kmc.simulations.implementations.factory import sim_factory

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def run(
    simulation_module_name: str,
    optimizer_name: str,
    budget: int = 50,
    cost_ratio: float = 0.2,
    init_samples_budget: float = 0.1,
    low_fid_samps_ratio: float = 0.6,
    lowfid_kernel: float = 0,
    bias_lowfid: float = None,
    multitask: bool = False,
    batch_size: int = 1,
    seed: int = 33,
) -> Dict[str, Any]:
    """Run simulation based on required module name and optimizer.

    Args:
        simulation_module_name: str, name of the simulation you want to run.
                                (it must be included in the sim_config.yml).

        optimizer_name: str, name of the optimizer you want to use (it must be
                        included in the optimize_config.yml).

        budget: int, total budget you want to allocate for the simulation. It is
                then computed as budget*hf_cost, where hf_cost is the cost of a
                high fidelity sample (by default, we set it to 1).

        cost_ratio: float, cost_ratio between low and high fidelity (low/high).

        init_samples_budget: float, ratio of the total budget that you want to
                            allocate for the initial sampling.

        low_fid_samps_ratio: float, fraction of the initial budget you want to spend on
                            low fidelity points.

        low_fid: float, optional parameter to modify the low fidelity value included on the sim_config file.
                  It overwrites this default parameter if present.

        bias_lowfid: float, optional parameter to modify the low fidelity simulation bias.

        multitask: bool, whether to use a MultiTaskGP or the SingleFidelityGP 

        batch_size: int, batch size for the new recommendations, defaults to 1
                    (we have not tested more values for the moment).

        seed: int, random seed for reproducibility

    """

    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    config.init_config()

    # Get SimulationConfig that corresponds to the module name
    sim_config = config.get_simulation_config(simulation_module_name)

    # Get optimizer config corresponding to the one we want to use
    optim_config = config.get_optimizer_config(optimizer_name)

    # modify simulation bias if necessary (it will be in the config file)
    if bias_lowfid:
        sim_config.bias_lowfid = bias_lowfid

    # modify multitask kernel if necessary
    if multitask:
        optim_config.gp = "multitask"

    # accesing singleton, generate
    simulation = sim_factory(sim_config)

    # HERE WE CAN EDIT THE FIDELITY OPTIONS DIRECTLY ON THE OPTIM CONFIG FILE
    if lowfid_kernel != None:
        sim_config.fidelity.options = [lowfid_kernel, 1.0]

    # use a config factory (we need sim config because some things as target
    # fidelities are specific for the type of simulation)
    optimizer = optim_factory(optim_config, sim_config, cost_ratio)

    # this list of dictionaries stores each experiment with the associated data (inputs, fidelity,
    # target variables, step) and more things could be added)
    experiments = []

    # initial sampling
    samples = sample_points(
        sim_config,
        optimizer_name,
        budget=budget,
        init_samples_budget=init_samples_budget,
        low_fid_samps_ratio=low_fid_samps_ratio,
        cost_ratio=cost_ratio,
        seed=seed,
    )

    # sampling cost (from here we get lambda 1)
    initial_costs = optimizer.compute_cost(samples)

    lambda_m = max(initial_costs)

    # compute results for each sample (maybe have to refactor if allowing precomputed stuff)
    for (inp, fid), cost in zip(samples, initial_costs):

        name = sim_config.fidelity.name

        # map fidelities in [0,1] scale to the real value so the simulator understands it
        fid_sim = {name: sim_config.fidelities_map[fid[name]]}

        # run simulation
        res = simulation.simulate(inp, fid_sim)

        experiments.append({**inp, **fid, **res, 'step': 0, 'cost': cost})

    budgetleft = budget - budget * init_samples_budget

    cumulative_cost = 0.0

    step = 1

    # while BUDGET
    while np.floor(budgetleft / lambda_m) >= 1:

        LOGGER.info(f"BO step: {step}")

        new_points = optimizer.recommend(
            inputs=experiments,
            batch_size=batch_size
        )

        # Here compute costs
        costs = optimizer.compute_cost(new_points)

        # compute selected points from the simulation engine (we have to check how to handle
        # hard or costly computations
        for (inp, fid), cost in zip(new_points, costs):

            name = sim_config.fidelity.name

            # map fidelities in [0,1] scale to the real value so the simulator understands it
            fid_sim = {name: sim_config.fidelities_map[fid[name]]}

            res = simulation.simulate(inp, fid_sim)
            experiments.append({**inp, **fid, **res, 'step': step, 'cost': cost})

        step_cost = sum(costs)

        # update budget
        cumulative_cost += step_cost

        step += 1

        budgetleft -= step_cost

    # we can log the total cost at the end of the optimization
    LOGGER.info(f'Total cost of the optimization: {cumulative_cost}')

    return experiments


def main():
    logging.info('Running simulation')
    run(
        'park',
        'multi_fidelity_mes',
        budget=30,
        batch_size=1,
        cost_ratio=0.05,
        init_samples_budget=0.1,
        low_fid_samps_ratio=0.66,
        lowfid_kernel=0.0,
        bias_lowfid=0.3,
        multitask=True,
        seed=0,
    )


if __name__ == '__main__':
    main()
