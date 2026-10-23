"""Registry of the named loss terms of the Grad-JEPA objective.

A term is a class with the attributes of :class:`Term`, constructed from its hypers
(``cls(**hypers)``, the entry of ``objective.terms`` under its name) and called on
the :class:`StepData` of every batch. It returns its already weighted loss and its
diagnostics. New terms are added with :func:`register_term`, together with their
hypers ``TypedDict`` as a ``NotRequired[Optional[...]]`` entry of ``TermsHypers``
(``documentation.py``), and the module registering them must be imported before
:func:`build_terms` runs. Under ``mtt train`` the options are merged with the
defaults, so a default term omitted from the options stays enabled: disable it
with ``null``.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

import torch

from .objective import (
    StepData,
    latent_loss,
    normalized_gradient_loss,
    relative_gradient_spread,
    vicreg_terms,
)


class Term(Protocol):
    """A loss term of the objective."""

    name: str
    needs_gradients: bool  # requires the probe gradients (double backward)
    required_targets: dict  # labels it consumes (empty for label-free terms)

    def __call__(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the (already weighted) loss and the diagnostics."""
        ...


TERMS: Dict[str, Callable[..., Term]] = {}
"""The registered term classes, by name."""

T = TypeVar("T", bound=type)


def register_term(cls: T) -> T:
    """Register a term class under its ``name`` (usable as a class decorator).

    :param cls: The term class.
    :return: The class, unchanged.
    :raises ValueError: If a term with the same name is already registered.
    """
    name = cls.name  # type: ignore[attr-defined]
    if name in TERMS:
        raise ValueError(f"loss term '{name}' is already registered")
    TERMS[name] = cls
    return cls


def build_terms(terms_hypers: Dict[str, Optional[Dict[str, Any]]]) -> List[Term]:
    """Build the active terms, in registration order.

    :param terms_hypers: ``objective.terms``: hypers by term name; absent or
        ``None`` terms are disabled.
    :return: The active terms.
    :raises ValueError: On an unknown term name or if no term is active.
    """
    for name in terms_hypers:
        if name not in TERMS:
            raise ValueError(
                f"unknown loss term '{name}' (registered: {', '.join(TERMS)})"
            )
    terms = []
    for name, cls in TERMS.items():
        hypers = terms_hypers.get(name)
        if hypers is not None:
            terms.append(cls(**hypers))
    if len(terms) == 0:
        raise ValueError("`objective.terms` must enable at least one loss term")
    return terms


@register_term
class LatentTerm:
    """``weight * (L_lat^m + unmasked_scale * L_lat^u)``.

    :param weight: Weight of the term.
    """

    name = "latent"
    needs_gradients = False

    def __init__(self, weight: float) -> None:
        self.weight = weight
        self.required_targets: dict = {}

    def __call__(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        views = step.views
        prediction = step.prediction["latent"]
        masked = views.latent_targets & views.in_masked_structure
        unmasked = views.latent_targets & ~views.in_masked_structure
        loss_masked = latent_loss(prediction, step.teacher_features, masked)
        loss_unmasked = latent_loss(prediction, step.teacher_features, unmasked)
        scale = step.hypers["objective"]["unmasked_scale"]
        loss = self.weight * (loss_masked + scale * loss_unmasked)
        return loss, {"latent_masked": loss_masked, "latent_unmasked": loss_unmasked}


@register_term
class GradientTerm:
    """``weight * (L_grad^m + unmasked_scale * L_grad^u)``.

    :param weight: Weight of the term.
    """

    name = "gradient"
    needs_gradients = True

    def __init__(self, weight: float) -> None:
        self.weight = weight
        self.required_targets: dict = {}

    def __call__(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        views = step.views
        prediction = step.prediction["gradient"]
        target = step.teacher_gradients
        assert target is not None
        masked = views.gradient_targets & views.in_masked_structure
        unmasked = views.gradient_targets & ~views.in_masked_structure
        loss_masked = normalized_gradient_loss(prediction, target, masked)
        loss_unmasked = normalized_gradient_loss(prediction, target, unmasked)
        scale = step.hypers["objective"]["unmasked_scale"]
        loss = self.weight * (loss_masked + scale * loss_unmasked)
        if bool(views.gradient_targets.any()):
            cosine = torch.nn.functional.cosine_similarity(
                prediction[:, views.gradient_targets].reshape(-1, 3),
                target[:, views.gradient_targets].reshape(-1, 3),
                dim=-1,
            ).mean()
        else:
            cosine = torch.zeros(())
        return loss, {
            "gradient_masked": loss_masked,
            "gradient_unmasked": loss_unmasked,
            "gradient_cosine": cosine,
            "teacher_gradient_rms": target.pow(2).sum(-1).mean().sqrt(),
        }


@register_term
class VICRegTerm:
    """``variance * L_var + covariance * L_cov`` on the context student features.

    :param variance: Weight of the variance hinge.
    :param covariance: Weight of the covariance penalty.
    """

    name = "vicreg"
    needs_gradients = False

    def __init__(self, variance: float, covariance: float) -> None:
        self.variance = variance
        self.covariance = covariance
        self.required_targets: dict = {}

    def __call__(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        variance, covariance = vicreg_terms(step.student_features[~step.views.masked])
        loss = self.variance * variance + self.covariance * covariance
        return loss, {"variance": variance, "covariance": covariance}


@register_term
class GradientSpreadTerm:
    """``weight * relu(floor - spread)``, over the context gradient targets.

    :param weight: Weight of the hinge.
    :param floor: Floor of the relative gradient spread (inverse length units).
    """

    name = "gradient_spread"
    needs_gradients = True

    def __init__(self, weight: float, floor: float) -> None:
        self.weight = weight
        self.floor = floor
        self.required_targets: dict = {}

    def __call__(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gradients = step.student_gradients
        assert gradients is not None
        atoms = ~step.views.masked & step.views.gradient_targets
        if int(atoms.sum()) >= 2:
            spread = relative_gradient_spread(
                step.student_features, gradients, step.probes, atoms
            )
        else:
            # undefined spread: no penalty, but keep the graph connected
            spread = gradients.sum() * 0.0 + self.floor
        loss = self.weight * torch.relu(self.floor - spread)
        return loss, {"gradient_spread": spread}
