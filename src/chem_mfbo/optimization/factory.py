"""Import optimizer dynamically from config.py."""
from typing import Union

from chem_mfbo.optimization.acquisition import get_cost_function
from chem_mfbo.optimization.optimizer_config import OptimizerConfig
from chem_mfbo.optimization.optimizer_mf import MultiFidelityOptimizer
from chem_mfbo.optimization.optimizer_sf import SingleFidelityOptimizer
from chem_mfbo.simulations.config import SimulationConfig


def optim_factory(
    optim_config: OptimizerConfig,
    sim_config: SimulationConfig,
    cost_ratio: float = None,
) -> Union[MultiFidelityOptimizer, SingleFidelityOptimizer]:
    """Reading optim_config file and sim_config file to load optimizer.
    We pass cost_ratio for the cost calculator, although it could be also
    read from the config file.

    """

    # define fidelity dimension
    fidelity_dimension = len(sim_config.input_parameters)

    # define target fidelity
    target_fidelities = {fidelity_dimension: 1.0}

    # create cost utility
    cost_model, cost_aware_utility = get_cost_function(
        sim_config.cost_model.value, target_fidelities, cost_ratio=cost_ratio
    )

    if "multi_fidelity" in optim_config.name:

        optimizer = MultiFidelityOptimizer(
            sim_config=sim_config,
            optim_config=optim_config,
            target_fids=target_fidelities,
            cost_model=cost_model,
            cost_aware_utility=cost_aware_utility,
        )

    elif "single_fidelity" in optim_config.name:

        optimizer = SingleFidelityOptimizer(
            sim_config=sim_config, optim_config=optim_config, cost_model=cost_model
        )

    return optimizer
