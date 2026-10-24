"""
SSL pretraining (Experimental)
==============================

Label-free self-supervised pretraining of a :ref:`PET <architecture-pet>` backbone
(selected with the ``backbone`` model hyperparameter). The objective is Grad-JEPA:
a joint-embedding predictive objective on the *gradients* of the representation.

At every step, a batch of structures is turned into two views:

- the **teacher** (an exponential moving average of the student) sees the full
  structure, optionally displaced by small per-atom displacements ``delta``;
- the **student** sees the clean structure, randomly rotated, with some atoms
  masked out (all their neighbor pairs are removed).

Both encoders produce per-atom PET features ``h_i``. Random probe vectors ``v``
define random scalars ``s_v = sum_i v . h_i``, whose gradients ``g = ds_v/dx``
w.r.t. the positions are conservative, force-like vector fields. A local predictor
(rotation-covariant by construction, linear in ``v``, conditioned on ``delta`` and
on the positions of the masked atoms) predicts, from the student features and
gradients, the teacher latents at the target atoms and the teacher gradients at
the target and boundary atoms.

The final checkpoint is a regular PET checkpoint without targets, holding by default
the EMA teacher weights (``export_weights``): fine-tune it with
the ``pet`` architecture and ``training.finetune.read_from``. Use the same
``model`` hyperparameters as for the pretraining. Interrupted runs are resumed
with the ``restart_from`` trainer option (``mtt train --restart`` is not supported).

The training datasets must not define targets (use ``targets: {}``).

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

{{SECTION_MODEL_HYPERS}}
"""

from typing import Annotated, Literal, Optional

from pydantic import AfterValidator
from typing_extensions import NotRequired, TypedDict

from metatrain.pet.documentation import ModelHypers as PETModelHypers
from metatrain.utils.hypers import init_with_defaults

from .high_order.documentation import (
    ActionResponseTermHypers,
    ChiralityTermHypers,
    DeformationJetsTermHypers,
    EquivariantVicregTermHypers,
    HighOrderHypers,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    ResidualMultipoleTermHypers,
    RotationAveragedVicregTermHypers,
    RotationConditionedLatentTermHypers,
)


class ModelHypers(PETModelHypers):
    """The backbone choice; the other model hyperparameters are the backbone's."""

    backbone: Literal["pet"] = "pet"
    """The backbone to pretrain (through its adapter in ``backbones.py``). It is
    not stored in the checkpoint, which stays a stock backbone checkpoint."""


class NoiseHypers(TypedDict):
    """Displacements applied to the teacher view."""

    probability: float = 0.5
    """Probability that a structure is displaced in the teacher view."""
    sigma: float = 0.05
    """Standard deviation (per Cartesian component, in length units) of the
    displacements."""
    skin: float = 0.5
    """Neighbor-list skin, in length units. Displacement norms are clipped to
    ``skin / 2`` so that the skin neighbor list stays exact."""


class MaskHypers(TypedDict):
    """Masking of the student view."""

    probability: float = 1.0
    """Probability that a structure is masked in the student view. As in
    Atom-JEPA, every structure is masked by default; unmasked structures only add
    rotation/displacement consistency terms (see ``objective.unmasked_scale``)."""
    generator: Literal["nl_ball", "k_hop_ego"] = "nl_ball"
    """How the region is sampled: ``nl_ball`` (union of balls around random
    centers, until ``fraction_range`` is reached) or ``k_hop_ego`` (k-hop ego
    network on the bond graph, as in Atom-JEPA)."""
    fraction_range: list[float] = [0.15, 0.4]
    """``nl_ball`` only: range of the target fraction of atoms in the region."""
    ball_radius_range: list[float] = [1.0, 3.0]
    """``nl_ball`` only: range of the ball radii (must not exceed the PET cutoff)."""
    ego_hops: list[int] = [1, 2, 3]
    """``k_hop_ego`` only: candidate hop counts."""
    topology_cutoff: float = 1.8
    """``k_hop_ego`` only: distance defining the bond graph (must not exceed the
    PET cutoff)."""
    swap_probability: float = 0.0
    """Probability of masking the complement of the region instead of the region
    (``1.0`` with ``k_hop_ego`` reproduces Atom-JEPA: the ego network is the
    context)."""
    min_atoms: int = 3
    """Structures with fewer atoms are never masked."""


class LatentTermHypers(TypedDict):
    """``latent`` term: ``weight * (L_lat^m + unmasked_scale * L_lat^u)``."""

    weight: float = 1.0
    """Weight of the latent prediction (layer-normalized teacher features at the
    target atoms)."""


class GradientTermHypers(TypedDict):
    """``gradient`` term: ``weight * (L_grad^m + unmasked_scale * L_grad^u)``."""

    weight: float = 1.0
    """Weight of the gradient prediction (teacher random-scalar gradients at the
    target and boundary atoms), normalized by the teacher gradient scale."""


class VICRegTermHypers(TypedDict):
    """``vicreg`` term: ``variance * L_var + covariance * L_cov``."""

    variance: float = 1.0
    """Weight of the VICReg variance hinge on the layer-normalized student
    features."""
    covariance: float = 0.04
    """Weight of the VICReg covariance penalty on the correlation matrix of the
    student features (scale-free)."""


class GradientSpreadTermHypers(TypedDict):
    """``gradient_spread`` term: ``weight * relu(floor - spread)``.

    It keeps the spread of the student gradients, relative to the spread of the
    per-atom random scalars ``v . h_i``, above ``floor`` (prevents the
    representation becoming flat in the positions)."""

    weight: float = 0.1
    """Weight of the hinge."""
    floor: float = 0.01
    """Floor of the relative gradient spread, in inverse length units
    (scale-free)."""


def _odd(value: int) -> int:
    if value < 3 or value % 2 != 1:
        raise ValueError("must be an odd integer of at least 3")
    return value


class SigregTermHypers(TypedDict):
    """``sigreg`` term: ``weight * SIGReg`` of the context student features (the
    rows of ``vicreg``), an alternative to ``vicreg``. Random one-dimensional
    projections of the features are pushed towards a standard normal with the
    Epps-Pulley test (as in LeJEPA)."""

    weight: NonNegativeFloat = 0.05
    """Weight of the term."""
    n_slices: PositiveInt = 256
    """Number of random projection directions per step."""
    t_max: PositiveFloat = 3.0
    """Upper end of the integration interval of the Epps-Pulley test."""
    n_points: Annotated[int, AfterValidator(_odd)] = 17
    """Number of quadrature points of the Epps-Pulley test (odd)."""


class TermsHypers(TypedDict):
    """The loss terms, by name. A term that is absent or ``null`` is disabled.

    ``mtt train`` merges the options with the defaults, so a default term left out
    of the options file stays enabled: set it to ``null`` to disable it. The
    latent-only baseline (no probe gradients are computed) is
    ``{latent: {weight: 1.0}, vicreg: {variance: 1.0, covariance: 0.04},
    gradient: null, gradient_spread: null}``."""

    latent: NotRequired[Optional[LatentTermHypers]] = init_with_defaults(
        LatentTermHypers
    )
    """Latent prediction."""
    gradient: NotRequired[Optional[GradientTermHypers]] = init_with_defaults(
        GradientTermHypers
    )
    """Gradient prediction (needs the probe gradients)."""
    vicreg: NotRequired[Optional[VICRegTermHypers]] = init_with_defaults(
        VICRegTermHypers
    )
    """VICReg variance and covariance regularization."""
    gradient_spread: NotRequired[Optional[GradientSpreadTermHypers]] = (
        init_with_defaults(GradientSpreadTermHypers)
    )
    """Gradient-spread hinge (needs the probe gradients)."""
    residual_multipole: NotRequired[Optional[ResidualMultipoleTermHypers]] = None
    """High-order method, term A: residual multipoles of the boundary atoms."""
    deformation_jets: NotRequired[Optional[DeformationJetsTermHypers]] = None
    """High-order method, term B: displacement and smooth-field gradient of the
    deformed structures."""
    chirality: NotRequired[Optional[ChiralityTermHypers]] = None
    """High-order method: chirality of the visible atoms."""
    sigreg: NotRequired[Optional[SigregTermHypers]] = None
    """SIGReg regularization (an alternative to ``vicreg``)."""
    equivariant_vicreg: NotRequired[Optional[EquivariantVicregTermHypers]] = None
    """High-order method: VICReg between two rotations of the student input, per
    irrep block of a linear projector (needs ``high_order.second_rotation``)."""
    vicreg_averaged: NotRequired[Optional[RotationAveragedVicregTermHypers]] = None
    """High-order method: VICReg on the features averaged over two rotations of the
    student input (needs ``high_order.second_rotation``)."""
    rotation_conditioned_latent: NotRequired[
        Optional[RotationConditionedLatentTermHypers]
    ] = None
    """High-order method, term C: rotation-conditioned latent prediction (``l_max:
    0`` is the invariant JEPA)."""
    action_response: NotRequired[Optional[ActionResponseTermHypers]] = None
    """High-order method, term D5: response of the teacher to the teacher modes at
    the mode centres, predicted from the student tokens and the mode amplitudes."""


def _at_least_one_term(terms: dict) -> dict:
    if all(value is None for value in terms.values()):
        raise ValueError("`objective.terms` must enable at least one loss term")
    return terms


class ObjectiveHypers(TypedDict):
    """The loss: a sum of named terms."""

    unmasked_scale: float = 0.25
    """Relative weight of the latent and gradient terms of unmasked structures
    (each term is averaged over its own targets, masked and unmasked separately)."""
    terms: Annotated[TermsHypers, AfterValidator(_at_least_one_term)] = (
        init_with_defaults(TermsHypers)
    )
    """The active loss terms and their hypers."""


class ConditionersHypers(TypedDict):
    """The predictor conditioners, by name (none are registered in v1). A conditioner
    that is absent or ``null`` is disabled. Each registered conditioner has an
    optional ``NotRequired[Optional[...]]`` entry here, as the loss terms do in
    ``TermsHypers``, so that unknown names and misspelled keys are rejected."""


def _at_least_one_layer(num_layers: int) -> int:
    if num_layers < 1:
        raise ValueError("`predictor.num_layers` must be at least 1")
    return num_layers


class PredictorHypers(TypedDict):
    """Hyperparameters of the predictor."""

    num_layers: Annotated[int, AfterValidator(_at_least_one_layer)] = 1
    """Number of predictor layers. The first ``num_layers - 1`` are residual
    message-passing updates among the context atoms; the last one is the readout at
    the target atoms. The default, 1, is a single readout layer."""
    conditioners: ConditionersHypers = init_with_defaults(ConditionersHypers)
    """Extra edge inputs of the predictor, by name and with their hypers."""
    hidden_dim: int = 128
    """Width of the predictor messages."""
    probe_dim: int = 32
    """Rank of the learned projection of the probe vectors."""
    num_rbf: int = 16
    """Number of radial basis functions."""
    species_query: bool = False
    """Whether the predictor knows the species of the masked atoms."""


class ViewsHypers(TypedDict):
    """Construction of the student and teacher views."""

    builder: str = "grad_jepa"
    """Registered view builder (``objective.VIEW_BUILDERS``). ``grad_jepa``: a
    rotated, masked student view and a displaced teacher view. ``high_order``: a
    rotated student view, masked or deformed, and a teacher view deformed by
    windowed l=1 modes (see ``high_order``)."""


class TrainerHypers(TypedDict):
    """Hyperparameters for the Grad-JEPA trainer."""

    batch_size: int = 16
    """Batch size."""
    max_atoms_per_batch: Optional[int] = None
    """If set, use greedy atom-count packing instead of a fixed ``batch_size``."""
    min_atoms_per_batch: int = 0
    """Minimum number of atoms of a packed batch (with ``max_atoms_per_batch``)."""
    num_epochs: int = 100
    """Number of epochs."""
    warmup_fraction: float = 0.01
    """Fraction of the steps used for the linear learning-rate warmup, followed by a
    cosine decay."""
    learning_rate: float = 1e-4
    """Peak learning rate."""
    weight_decay: float = 0.0
    """AdamW weight decay."""
    grad_clip_norm: float = 1.0
    """Gradient clipping norm."""
    num_workers: Optional[int] = None
    """Number of data-loading workers (chosen automatically if ``None``)."""
    log_interval: int = 1
    """Interval (in epochs) between logs."""
    monitor_every: int = 200
    """Interval (in steps) between the gradient cosines of the per-term losses on
    the last backbone layer (logged as ``monitor/cos/<a>__<b>``); 0 turns them
    off. Each one costs one backward pass per term."""
    checkpoint_interval: int = 10
    """Interval (in epochs) between checkpoints (the last epoch is always
    saved)."""
    seed: int = 0
    """Seed of the random views and probes (validation views are fixed)."""
    export_weights: Literal["final_teacher", "best_rankme_student"] = "final_teacher"
    """Weights stored as the checkpoint's best model, i.e. the weights used by
    ``finetune.read_from`` and by the exported model: the EMA teacher at the last
    epoch (as in I-JEPA), or the student at the epoch with the highest validation
    RankMe."""
    restart_from: Optional[str] = None
    """Path to a Grad-JEPA checkpoint to resume from (same hyperparameters)."""
    ema_tau_range: list[float] = [0.996, 1.0]
    """EMA momentum of the teacher, linearly increased over training."""
    global_augmentation: bool = True
    """Randomly rotate (O(3)) every training structure as a whole, for both views,
    with metatrain's ``O3Augmenter`` (as in PET training). Validation is not
    augmented."""
    rotation_group: Literal["O3", "SO3", "none"] = "O3"
    """Random transformations applied to the student view only, relative to the
    teacher view."""
    num_probes: int = 2
    """Number of random probe vectors per step."""
    views: ViewsHypers = init_with_defaults(ViewsHypers)
    """Construction of the views."""
    noise: NoiseHypers = init_with_defaults(NoiseHypers)
    """Displacements of the teacher view."""
    mask: MaskHypers = init_with_defaults(MaskHypers)
    """Masking of the student view."""
    predictor: PredictorHypers = init_with_defaults(PredictorHypers)
    """Predictor hyperparameters."""
    objective: ObjectiveHypers = init_with_defaults(ObjectiveHypers)
    """The loss terms and their weights."""
    freeze_backbone: bool = False
    """Keep the backbone at its initial weights (head-capacity control of the design
    pilots: train the predictor and the loss terms on a frozen, randomly initialized
    backbone). The student backbone gets no gradient and is excluded from the
    optimizer and from gradient clipping. The EMA update is skipped: the teacher is
    the copy of the student made at the start, so it stays equal to the student, and
    the exported model is the frozen backbone. The predictor and the parametric loss
    terms are still trained."""
    allow_no_anti_collapse: bool = False
    """Whether ``objective.terms`` may hold no anti-collapse term. Exactly one of
    ``vicreg``, ``sigreg`` and ``vicreg_averaged`` must be active otherwise. Set it
    only for the design-pilot arms without anti-collapse term, which test whether
    the representation collapses."""
    high_order: HighOrderHypers = init_with_defaults(HighOrderHypers)
    """The views of the high-order SSL method (with ``views.builder: high_order``)."""
