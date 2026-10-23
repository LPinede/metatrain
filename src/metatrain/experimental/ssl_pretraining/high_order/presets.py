"""Named configurations of the high-order SSL method.

A preset is a nested override of the trainer hypers (``views``, ``high_order``,
``objective.terms`` and ``rotation_group``), applied with :func:`apply_preset`.
There is no ``preset`` hyperparameter: write the result of :func:`apply_preset` into
the ``architecture.training`` section of the options file.

Every preset writes every term of ``objective.terms`` explicitly, ``None`` for the
disabled ones: ``mtt train`` merges the options with the defaults, so an omitted
default term (``latent``, ``gradient``, ``vicreg``, ``gradient_spread``) would come
back enabled.

The term entries are the only settings of the terms. The two family weights of the
method are preset-level helpers that write the ``weight`` of each term:
``analytic_weight`` (:func:`analytic_terms`) sets the weights of
``residual_multipole`` and ``deformation_jets`` and, times ``chirality_weight``, of
``chirality``. ``teacher_weight`` will do the same for the teacher terms.

.. code-block:: python

    import copy

    from metatrain.experimental.ssl_pretraining.documentation import TrainerHypers
    from metatrain.utils.hypers import init_with_defaults

    base = copy.deepcopy(init_with_defaults(TrainerHypers))
    hypers = apply_preset("high_order_sigreg", base)
"""

import copy
from typing import Any, Dict, Optional

from metatrain.utils.hypers import init_with_defaults

from ..documentation import (
    SigregTermHypers,
    TermsHypers,
    VICRegTermHypers,
)
from .documentation import (
    ChiralityTermHypers,
    DeformationJetsTermHypers,
    EquivariantVicregTermHypers,
    ResidualMultipoleTermHypers,
    RotationAveragedVicregTermHypers,
)


def _defaults(hypers_class: Any) -> Dict[str, Any]:
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    return copy.deepcopy(init_with_defaults(hypers_class))


def analytic_terms(
    analytic_weight: float = 1.0,
    chirality_weight: float = 0.1,
    head: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """The entries of the analytic terms (A, B and chirality), with their defaults.

    .. code-block:: python

        terms = analytic_terms(analytic_weight=2.0, head={"use_edges": True})

    :param analytic_weight: Family weight: the weight of ``residual_multipole`` and
        ``deformation_jets``.
    :param chirality_weight: Weight of ``chirality`` relative to
        ``analytic_weight``.
    :param head: Overrides of the head form (``use_edges``, ``channel_fraction``),
        applied to the heads of the three terms.
    :return: The three term entries, by name.
    """
    terms = {
        "residual_multipole": _defaults(ResidualMultipoleTermHypers),
        "deformation_jets": _defaults(DeformationJetsTermHypers),
        "chirality": _defaults(ChiralityTermHypers),
    }
    terms["residual_multipole"]["weight"] = analytic_weight
    terms["deformation_jets"]["weight"] = analytic_weight
    terms["chirality"]["weight"] = chirality_weight * analytic_weight
    for entry in terms.values():
        entry["head"].update(head or {})
    return terms


def _preset(
    terms: Dict[str, Dict[str, Any]], second_rotation: bool = False
) -> Dict[str, Any]:
    """A preset on the high-order views with the teacher pair off and ``terms`` as
    the active terms; every other term is disabled."""
    all_terms: Dict[str, Optional[Dict[str, Any]]] = {
        name: None for name in TermsHypers.__annotations__
    }
    all_terms.update(terms)
    return {
        "views": {"builder": "high_order"},
        # no teacher term reads the antithetic teacher pair
        "high_order": {"teacher_pair": False, "second_rotation": second_rotation},
        "rotation_group": "O3",
        "objective": {"terms": all_terms},
    }


def _vicreg() -> Dict[str, Dict[str, Any]]:
    return {"vicreg": _defaults(VICRegTermHypers)}


PRESETS: Dict[str, Dict[str, Any]] = {
    # A, B and chirality with pooled per-block heads, plus VICReg
    "high_order_analytic_only": _preset({**analytic_terms(), **_vicreg()}),
    # Abbott et al.'s full per-block heads (node part plus per-edge part)
    "high_order_edge_heads": _preset(
        {**analytic_terms(head={"use_edges": True}), **_vicreg()}
    ),
    # the analytic heads read only the first half of the token channels
    "high_order_dedicated_channels": _preset(
        {**analytic_terms(head={"channel_fraction": 0.5}), **_vicreg()}
    ),
    # SIGReg instead of VICReg
    "high_order_sigreg": _preset(
        {**analytic_terms(), "sigreg": _defaults(SigregTermHypers)}
    ),
    # VICReg on the features averaged over two student rotations
    "high_order_vicreg_averaged": _preset(
        {
            **analytic_terms(),
            "vicreg_averaged": _defaults(RotationAveragedVicregTermHypers),
        },
        second_rotation=True,
    ),
    # adds the equivariant VICReg between two student rotations
    "high_order_equivariant_vicreg": _preset(
        {
            **analytic_terms(),
            **_vicreg(),
            "equivariant_vicreg": _defaults(EquivariantVicregTermHypers),
        },
        second_rotation=True,
    ),
}
"""The implemented presets, by name."""

PENDING_PRESETS = (
    "high_order",
    "high_order_truncated_l1",
    "high_order_no_response",
    "high_order_shuffled_response",
    "high_order_so3",
)
"""The presets that need the teacher terms (rotation-conditioned latent and action
response, tasks H8 and H9); :func:`apply_preset` rejects them until they exist."""


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Merge ``override`` into ``base`` in place; ``None`` and non-dict values
    replace."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def apply_preset(name: str, base_hypers: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base_hypers`` with the preset ``name`` merged in.

    ``base_hypers`` may be the full trainer hypers or a partial ``training``
    section of an options file; it is not modified. The term entries of the
    preset replace those of ``base_hypers`` (every term is written).

    .. code-block:: python

        training = apply_preset("high_order_analytic_only", {"num_epochs": 10})

    :param name: The preset name (a key of :data:`PRESETS`).
    :param base_hypers: The trainer hypers to start from.
    :return: The trainer hypers with the preset applied.
    :raises NotImplementedError: For a preset that needs the teacher terms.
    :raises ValueError: For an unknown preset.
    """
    if name in PENDING_PRESETS:
        raise NotImplementedError(
            f"preset '{name}' needs the teacher terms (tasks H8 and H9), which do "
            "not exist yet"
        )
    if name not in PRESETS:
        raise ValueError(
            f"unknown preset '{name}' (available: {', '.join(sorted(PRESETS))})"
        )
    hypers = copy.deepcopy(base_hypers)
    preset = PRESETS[name]
    _merge(hypers, {k: v for k, v in preset.items() if k != "objective"})
    objective = hypers.setdefault("objective", {})
    objective["terms"] = copy.deepcopy(preset["objective"]["terms"])
    return hypers
