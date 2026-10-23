"""Registry of the predictor conditioners of Grad-JEPA (empty in v1).

A conditioner feeds extra per-edge information to the predictor: ``scalar_dim``
invariant scalars, appended to the message input, and ``vector_dim`` vectors in the
teacher frame. Each vector channel gets a probe-linear coefficient in the gradient
head and its invariants (norm and projections on the pair direction and on the
relative displacement) enter the messages, so the predictor stays O(3)-covariant and
linear in the probes for any conditioner, by construction.

Contract for ``edge_inputs``: the edge centers include the masked target atoms, so
a conditioner must not read teacher or target data at masked atoms (only information
the student may know there); and the vectors must be polar (they flip sign under
inversion, like positions): pseudovectors such as cross products break
O(3)-covariance for improper transformations.

A conditioner is a class with the attributes of :class:`Conditioner`, constructed
from its hypers (``cls(**hypers)``, the entry of ``predictor.conditioners`` under its
name). Add one with :func:`register_conditioner`, together with its hypers
``TypedDict`` as a ``NotRequired[Optional[...]]`` entry of ``ConditionersHypers``
(``documentation.py``); the module registering it must be imported before
:func:`build_conditioners` runs. Conditioners that are ``torch.nn.Module`` have their
parameters registered in the predictor.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

import torch


if TYPE_CHECKING:
    from .objective import StepData


class Conditioner(Protocol):
    """A source of extra edge inputs of the predictor."""

    name: str
    required_targets: dict  # e.g. {"energy": {"gradients": ["positions"]}}
    scalar_dim: int  # extra invariant edge-scalar channels
    vector_dim: int  # extra edge-vector channels

    def edge_inputs(
        self,
        step: "StepData",
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the inputs of the edges ``(centers, neighbors)``.

        :param step: The step data (``prediction`` is not yet set).
        :param centers: Center atom of each edge, ``(E,)``.
        :param neighbors: Context neighbor of each edge, ``(E,)``.
        :param directions: Unit pair directions in the teacher frame, ``(E, 3)``.
        :return: Invariant scalars ``(E, scalar_dim)`` and vectors in the teacher
            frame ``(E, vector_dim, 3)``.
        """
        ...


CONDITIONERS: Dict[str, Callable[..., Conditioner]] = {}
"""The registered conditioner classes, by name."""

T = TypeVar("T", bound=type)


def register_conditioner(cls: T) -> T:
    """Register a conditioner class under its ``name`` (usable as a decorator).

    :param cls: The conditioner class.
    :return: The class, unchanged.
    :raises ValueError: If a conditioner with the same name is already registered.
    """
    name = cls.name  # type: ignore[attr-defined]
    if name in CONDITIONERS:
        raise ValueError(f"conditioner '{name}' is already registered")
    CONDITIONERS[name] = cls
    return cls


def build_conditioners(
    conditioners_hypers: Dict[str, Optional[Dict[str, Any]]],
) -> List[Conditioner]:
    """Build the active conditioners, in registration order.

    :param conditioners_hypers: ``predictor.conditioners``: hypers by name; absent or
        ``None`` conditioners are disabled.
    :return: The active conditioners (none by default).
    :raises ValueError: On an unknown conditioner name.
    """
    for name in conditioners_hypers:
        if name not in CONDITIONERS:
            raise ValueError(
                f"unknown conditioner '{name}' "
                f"(registered: {', '.join(CONDITIONERS) or 'none'})"
            )
    active = []
    for name, cls in CONDITIONERS.items():
        hypers = conditioners_hypers.get(name)
        if hypers is not None:
            active.append(cls(**hypers))
    return active
