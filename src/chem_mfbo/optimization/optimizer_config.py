# Shows list in simulation_config.yml using Pydantic
from enum import Enum

from pydantic import BaseModel


class NameType(str, Enum):
    single_fidelity_ei = "single_fidelity_ei"
    single_fidelity_mes = "single_fidelity_mes"
    single_fidelity_kg = "single_fidelity_kg"
    multi_fidelity_kg = "multi_fidelity_kg"
    multi_fidelity_mes = "multi_fidelity_mes"
    multi_fidelity_ei = "multi_fidelity_ei"
    multi_fidelity_gibbon = "multi_fidelity_gibbon"
    single_fidelity_gibbon = "single_fidelity_gibbon"


class OptimizerConfig(BaseModel):
    name: NameType
    gp: str
    acquisition: str
    n_fantasies: int
    n_raw_samples: int
    n_restarts: int
    batch_limit: int
    max_iter: int
