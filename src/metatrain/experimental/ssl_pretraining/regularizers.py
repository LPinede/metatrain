"""SIGReg (sliced isotropic Gaussian regularization), shared by all SSL methods.

Ported from TriForces (``triforces/losses/lejepa.py``, ``EppsPulley`` and
``SlicedGaussianRegularizer``) without the distributed reductions. Importing this
module registers the ``sigreg`` loss term.
"""

from typing import Dict, Optional, Tuple

import torch

from .objective import StepData
from .terms import register_term


def sigreg(
    x: torch.Tensor,
    n_slices: int,
    t_max: float = 3.0,
    n_points: int = 17,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Mean Epps-Pulley statistic of ``n_slices`` random unit projections of ``x``.

    The statistic of a projection is ``n`` times the weighted squared distance
    between its empirical characteristic function and that of a standard normal,
    integrated over ``[-t_max, t_max]`` (trapezoidal rule on ``n_points`` points of
    ``[0, t_max]``, using the symmetry of the integrand). It vanishes in
    expectation only for a standard normal projection.

    .. code-block:: python

        loss = sigreg(features, n_slices=256, generator=torch.Generator())

    :param x: Samples ``(n, D)``.
    :param n_slices: Number of random directions.
    :param t_max: Upper end of the integration interval.
    :param n_points: Number of quadrature points (odd).
    :param generator: CPU generator of the directions.
    :return: The scalar statistic; zero (in the graph) for fewer than two rows.
    """
    n, d = x.shape
    if n < 2:
        return x.sum() * 0.0
    with torch.no_grad():
        # directions are drawn on the CPU, so the stream does not depend on device
        a = torch.randn(d, n_slices, generator=generator, dtype=torch.float64)
        a = (a / a.norm(dim=0, keepdim=True)).to(x)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float64)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float64)
        weights[0] = weights[-1] = dt
        phi = torch.exp(-0.5 * t**2)
        t, phi, weights = t.to(x), phi.to(x), (weights * phi).to(x)
    xt = (x @ a)[..., None] * t  # (n, n_slices, n_points)
    error = (xt.cos().mean(0) - phi) ** 2 + xt.sin().mean(0) ** 2
    return (error @ weights).mean() * n


@register_term
class SigregTerm:
    """``weight * SIGReg`` of the context student features (the rows of
    ``vicreg``), a drop-in alternative to the ``vicreg`` term.

    .. code-block:: python

        term = SigregTerm(weight=0.05)
        loss, diagnostics = term(step)

    :param weight: Weight of the term.
    :param n_slices: Number of random directions per step.
    :param t_max: Upper end of the integration interval.
    :param n_points: Number of quadrature points (odd).
    """

    name = "sigreg"
    needs_gradients = False
    uses_prediction = False

    def __init__(
        self,
        weight: float,
        n_slices: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
    ) -> None:
        self.weight, self.n_slices = weight, n_slices
        self.t_max, self.n_points = t_max, n_points
        self.required_targets: dict = {}

    def __call__(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = step.student_features[~step.views.masked]
        loss = sigreg(
            features,
            self.n_slices,
            self.t_max,
            self.n_points,
            step.extras.get("generator"),
        )
        return self.weight * loss, {"sigreg/loss": loss.detach()}
