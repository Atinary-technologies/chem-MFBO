# HERE ANY MODEL WE NEED FOR THE MF OPTIMIZATION
from typing import Tuple

from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


def initialize_model(
    train_x: Tensor, train_obj: Tensor, fidelity_dim: int
) -> Tuple[ExactMarginalLogLikelihood, SingleTaskMultiFidelityGP]:
    """define a surrogate model suited for a training data-like fidelity parameter
    in dimension 6, as in [2]"""

    model = SingleTaskMultiFidelityGP(
        train_x,
        train_obj,
        linear_truncated=False,
        outcome_transform=Standardize(m=1),
        data_fidelities=[fidelity_dim],
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return mll, model
