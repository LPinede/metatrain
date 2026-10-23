"""
Grad-JEPA (Experimental)
========================

Label-free pretraining of a :ref:`PET <architecture-pet>` backbone with a
joint-embedding predictive objective on the *gradients* of the representation.

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

from metatrain.pet.documentation import ModelHypers  # noqa: F401
from metatrain.utils.hypers import init_with_defaults


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


class PredictorHypers(TypedDict):
    """Hyperparameters of the predictor."""

    hidden_dim: int = 128
    """Width of the predictor messages."""
    probe_dim: int = 32
    """Rank of the learned projection of the probe vectors."""
    num_rbf: int = 16
    """Number of radial basis functions."""
    species_query: bool = False
    """Whether the predictor knows the species of the masked atoms."""


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
    noise: NoiseHypers = init_with_defaults(NoiseHypers)
    """Displacements of the teacher view."""
    mask: MaskHypers = init_with_defaults(MaskHypers)
    """Masking of the student view."""
    predictor: PredictorHypers = init_with_defaults(PredictorHypers)
    """Predictor hyperparameters."""
    objective: ObjectiveHypers = init_with_defaults(ObjectiveHypers)
    """The loss terms and their weights."""
