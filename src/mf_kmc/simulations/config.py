import yaml

from mf_kmc.optimization.optimizer_config import OptimizerConfig
from mf_kmc.simulations.models.simulation_config import SimulationConfig

CFG = None

# here the simulation paths
SIM_CONFIG_PATH = "config/sim_config.yml"

# results path to save config
OPTIM_CONFIG_PATH = "config/optimizer_config.yml"


class GlobalConfig:
    """Global config object providing SimulationObjects of each available simulation."""

    def __init__(
        self,
        simulation_config_path=SIM_CONFIG_PATH,
        optimizer_config_path=OPTIM_CONFIG_PATH,
    ):

        with open(simulation_config_path, mode="r") as cfg:
            sim_config = yaml.safe_load(cfg)

        self.simulations = [SimulationConfig(**config) for config in sim_config]

        with open(optimizer_config_path, mode="r") as cfg:
            optim_config = yaml.safe_load(cfg)

        self.optimizers = [OptimizerConfig(**config) for config in optim_config]


def init_config(
    simulation_config_path=SIM_CONFIG_PATH, optimizer_config_path=OPTIM_CONFIG_PATH
):
    """Create CFG if not created"""
    global CFG

    if not CFG:
        CFG = GlobalConfig(
            simulation_config_path=simulation_config_path,
            optimizer_config_path=optimizer_config_path,
        )

    return CFG


def get_simulation_config(name: str) -> SimulationConfig:
    """Get SimulationConfig object"""

    sim_config = [sim for sim in CFG.simulations if sim.module_name == name]

    if sim_config:
        return sim_config[0]

    else:
        raise ValueError(
            f"Cannot run simulation: module '{name}' is not specified or does not exist."
        )


def get_optimizer_config(name: str) -> OptimizerConfig:
    """Get OptimizerConfig object"""

    optim_config = [opt for opt in CFG.optimizers if opt.name == name]

    if optim_config:
        return optim_config[0]

    else:
        raise ValueError(
            f"Cannot use optimizer: optimizer '{name}' is not specified or does not exist."
        )
