"""Training monitors of the high-order track.

Diagnostics only: nothing here changes the training (no automatic action).
"""

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence

import torch
from metatomic.torch import NeighborListOptions, System

from ..backbones import Backbone
from ..views import build_view, random_rotations


class FlatteningMonitor:
    """Flags "winning by flattening": the relative response of the teacher falls by
    more than ``factor`` over the last ``window`` validation epochs while the
    response loss also falls.

    :param window: Number of validation epochs compared (first against last).
    :param factor: Minimum fall factor of ``rho_target``.
    """

    def __init__(self, window: int = 5, factor: float = 3.0) -> None:
        self.rho: Deque[float] = deque(maxlen=window)
        self.loss: Deque[float] = deque(maxlen=window)
        self.factor = factor

    def update(self, rho_target: float, d5_loss: float) -> bool:
        """Record one validation epoch.

        :param rho_target: The epoch's ``action_response/rho_target``.
        :param d5_loss: The epoch's ``action_response/loss``.
        :return: Whether the monitor trips (a full window with ``rho_target``
            down by more than ``factor`` and a lower loss).
        """
        self.rho.append(rho_target)
        self.loss.append(d5_loss)
        if len(self.rho) < self.rho.maxlen:  # type: ignore[operator]
            return False
        return bool(
            self.rho[0] > self.factor * self.rho[-1] and self.loss[-1] < self.loss[0]
        )


def gradient_cosines(
    losses: Dict[str, torch.Tensor], parameters: Sequence[torch.nn.Parameter]
) -> Dict[str, float]:
    """Pairwise cosines of the gradients of the per-term losses.

    The graph is kept, so the losses can still be back-propagated afterwards. A
    loss that does not depend on the parameters (or on anything trainable) has a
    zero gradient and a cosine of 0 with the others.

    :param losses: The loss of each term, by name.
    :param parameters: The parameters to take the gradients on.
    :return: The cosine of every pair, under the key ``"{a}__{b}"`` (names sorted).
    """
    parameters = list(parameters)
    gradients: Dict[str, torch.Tensor] = {}
    for name, loss in losses.items():
        if loss.requires_grad:
            grads = torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True
            )
        else:
            grads = tuple(None for _ in parameters)
        gradients[name] = torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).flatten()
                for g, p in zip(grads, parameters, strict=True)
            ]
        )
    names = sorted(gradients)
    cosines: Dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            cosines[f"{a}__{b}"] = float(
                torch.nn.functional.cosine_similarity(gradients[a], gradients[b], dim=0)
            )
    return cosines


def _small_rotations(
    n: int, angle: float, dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    """``n`` rotations by ``angle`` (radians) about uniformly random axes, from the
    shared sampler: the axis is the image of ``z`` under a random rotation."""
    axes = random_rotations(n, "SO3", dtype, generator)[:, :, 2]
    zero = torch.zeros_like(axes[:, 0])
    cross = torch.stack(
        [
            torch.stack([zero, -axes[:, 2], axes[:, 1]], dim=-1),
            torch.stack([axes[:, 2], zero, -axes[:, 0]], dim=-1),
            torch.stack([-axes[:, 1], axes[:, 0], zero], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=dtype).expand(n, 3, 3)
    return eye + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


@torch.no_grad()
def finite_rotation_invariance(
    backbone: Backbone,
    core: torch.nn.Module,
    systems: List[System],
    options: NeighborListOptions,
    angle_deg: float = 5.0,
    n_rotations: int = 8,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Forward-only check of the rotation invariance of the summed features.

    A fixed random probe ``v`` of the per-structure sum of the features is compared
    between the systems and the same systems rotated (positions and cell, through
    :func:`build_view`) by ``angle_deg`` about random axes: the result is the mean
    over ``n_rotations`` draws of ``||s(R x) - s(x)|| / ||s(x)||``, with ``s`` the
    probe of the per-structure summed features, over the structures. The core is
    put in eval mode and restored afterwards; the features are not required to be
    invariant (a PET feature is not), so this measures how much the backbone
    content moves under a small rotation, and is 0 for an invariant model.

    :param backbone: The backbone adapter.
    :param core: The backbone core.
    :param systems: Systems holding the neighbor list ``options``.
    :param options: The neighbor list options of the backbone.
    :param angle_deg: Rotation angle in degrees.
    :param n_rotations: Number of random rotation draws.
    :param generator: CPU random generator (seeded to 0 by default).
    :return: The mean relative change of the probe.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    was_training = core.training
    core.eval()
    try:

        def probed(views: List[System], probe: torch.Tensor) -> torch.Tensor:
            features = backbone.features(core, views)
            sizes = [len(view) for view in views]
            return torch.stack([f.sum(0) for f in features.split(sizes)]) @ probe

        reference_features = backbone.features(core, systems)
        probe = torch.randn(
            reference_features.shape[1], dtype=torch.float64, generator=generator
        ).to(device=reference_features.device, dtype=reference_features.dtype)
        reference = probed(systems, probe)
        dtype = reference_features.dtype
        changes = []
        for _ in range(n_rotations):
            rotations = _small_rotations(
                len(systems), angle_deg * torch.pi / 180.0, dtype, generator
            ).to(reference_features.device)
            views = [
                build_view(
                    system,
                    options,
                    system.positions.detach() @ rotation.T,
                    system.cell.detach() @ rotation.T,
                )
                for system, rotation in zip(systems, rotations, strict=True)
            ]
            changes.append(
                (probed(views, probe) - reference).norm()
                / reference.norm().clamp(min=1e-12)
            )
        return float(torch.stack(changes).mean())
    finally:
        core.train(was_training)
