# HERE ANY MODEL WE NEED FOR THE MF OPTIMIZATION
import math
from typing import Tuple
import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor


def initialize_model(
    train_x: Tensor,
    train_obj: Tensor,
    fidelity_dim: int,
    multitask: bool,
) -> Tuple[ExactMarginalLogLikelihood, SingleTaskMultiFidelityGP]:
    """Return a surrogate model and MLL for training. If multitask,
    use an index kernel to model fidelities."""

    if multitask:

        model = MultiTaskGP(
            train_X=train_x,
            train_Y=train_obj,
            rank=1,
            task_feature=fidelity_dim,
            outcome_transform=Standardize(m=1),
            task_covar_prior=LKJCovariancePrior(
                n=2,
                eta=torch.tensor(1.5).to(train_x),
                sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05),
            ),
        )

    else:
        model = SingleTaskMultiFidelityGP(
            train_x,
            train_obj,
            linear_truncated=False,
            outcome_transform=Standardize(m=1),
            data_fidelities=[fidelity_dim],
        )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return mll, model
