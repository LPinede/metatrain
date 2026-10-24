"""Hyperparameters of the high-order views (``training.high_order``) and of the
high-order loss terms (entries of ``training.objective.terms``).

This module is the single source of the defaults. It must stay torch-free: the
shared ``documentation.py`` imports it.
"""

from typing import Annotated, Literal, Optional

from pydantic import AfterValidator
from typing_extensions import TypedDict

from metatrain.utils.hypers import init_with_defaults


def _probability(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("must lie in [0, 1]")
    return value


def _fraction(value: float) -> float:
    if not 0.0 < value <= 1.0:
        raise ValueError("must lie in (0, 1]")
    return value


def _positive(value: float) -> float:
    if not value > 0:
        raise ValueError("must be positive")
    return value


def _non_negative(value: float) -> float:
    if value < 0:
        raise ValueError("must be non-negative")
    return value


Probability = Annotated[float, AfterValidator(_probability)]
PositiveFloat = Annotated[float, AfterValidator(_positive)]
NonNegativeFloat = Annotated[float, AfterValidator(_non_negative)]
PositiveInt = Annotated[int, AfterValidator(_positive)]
NonNegativeInt = Annotated[int, AfterValidator(_non_negative)]


class HighOrderHypers(TypedDict):
    """Views of the high-order SSL method (used with ``views.builder: high_order``).

    The settings of the loss terms (orders, weights, head form) are the entries of
    ``objective.terms`` only; the presets of ``high_order/presets.py`` set them."""

    p_def: Probability = 0.5
    """Probability that a structure takes the deform-only student branch (else the
    mask-only branch, masked as in Grad-JEPA through ``mask``)."""
    field_amplitude: PositiveFloat = 0.05
    """Root-mean-square displacement (length units) of the smooth random field of
    the deform-only branch, per structure, over its atoms (after centring and, for
    non-periodic structures, removal of the rigid rotation). If the displacement
    would exceed ``noise.skin / 2`` on some atom, the whole field is scaled down
    for that structure; the factor is logged as ``deformation_jets/field_scale``."""
    wavelength: PositiveFloat = 8.0
    """Wavelength (length units) of the plane waves of the smooth field of
    non-periodic structures."""
    n_waves: PositiveInt = 6
    """Number of plane waves of the smooth field of non-periodic structures."""
    max_wave_index: PositiveInt = 2
    """Periodic structures: the wave vectors of the smooth field are reciprocal
    lattice vectors with integer indices in ``[-max_wave_index, max_wave_index]``."""
    iid_sigma: NonNegativeFloat = 0.03
    """Standard deviation (per Cartesian component, length units) of the iid
    per-atom noise added to the smooth field."""
    mode_amplitude: PositiveFloat = 0.05
    """Standard deviation of the 9 amplitudes (dilation, rotation, shear) of the
    windowed l=1 teacher modes at each centre."""
    mode_window: PositiveFloat = 3.0
    """Width (length units) of the Gaussian window of the teacher modes."""
    max_centers: PositiveInt = 4
    """Maximum number of teacher-mode centres per structure."""
    teacher_pair: bool = True
    """Whether the teacher runs on the antithetic pair ``x0 ± u_T`` (else on
    ``x0 + u_T`` only, with the clean ``x0`` as the reference)."""
    second_rotation: bool = False
    """Whether a second student rotation is encoded (equivariant or averaged
    VICReg)."""


def _three_distinct(value: list) -> list:
    if len(value) != 3 or len(set(value)) != 3:
        raise ValueError("must hold three distinct atomic types")
    return value


class AnchorHeadHypers(TypedDict):
    """Form of the per-block heads of a term (analytic or teacher)."""

    use_edges: bool = False
    """Abbott et al.'s full per-block head (a node part plus a nonlinear map of each
    edge token, summed over neighbours), instead of an MLP on the pooled per-atom
    feature."""
    channel_fraction: Annotated[float, AfterValidator(_fraction)] = 1.0
    """Fraction of the node and edge token channels read by the heads."""


class ResidualMultipoleTermHypers(TypedDict):
    """``residual_multipole`` term (A): the residual multipoles of the boundary atoms,
    one channel per element and radial function, orders 0 to ``l_max``."""

    weight: NonNegativeFloat = 1.0
    """Weight of the term."""
    l_max: NonNegativeInt = 4
    """Highest order of the targets."""
    n_radial: PositiveInt = 6
    """Number of radial functions."""
    hidden: PositiveInt = 64
    """Hidden width of the heads."""
    floor_gamma: NonNegativeFloat = 0.1
    """Weight of the per-batch variance floor of the l > 0 predictions (at 0.1
    target standard deviations)."""
    head: AnchorHeadHypers = init_with_defaults(AnchorHeadHypers)
    """Form of the heads."""


class DeformationJetsTermHypers(TypedDict):
    """``deformation_jets`` term (B): the student displacement and the dilation,
    local rotation and shear of the smooth field, at the atoms of deformed
    structures."""

    weight: NonNegativeFloat = 1.0
    """Weight of the term."""
    hidden: PositiveInt = 64
    """Hidden width of the heads."""
    l_max: Literal[0, 1, 2] = 2
    """Highest order of the blocks (1 drops the shear block, 0 keeps only the
    dilation)."""
    head: AnchorHeadHypers = init_with_defaults(AnchorHeadHypers)
    """Form of the heads."""


class ChiralityTermHypers(TypedDict):
    """``chirality`` term: the triple product of three element-channel dipoles, at
    the visible atoms with at least 3 neighbours within the cutoff."""

    weight: NonNegativeFloat = 0.1
    """Weight of the term."""
    hidden: PositiveInt = 64
    """Hidden width of the head."""
    channel_types: Annotated[list[int], AfterValidator(_three_distinct)] = [1, 6, 8]
    """Atomic types of the three dipole channels."""
    head: AnchorHeadHypers = init_with_defaults(AnchorHeadHypers)
    """Form of the head."""


class EquivariantVicregTermHypers(TypedDict):
    """``equivariant_vicreg`` term: for every irrep block ``(l, sigma)`` with
    ``1 <= l <= l_max``, a linear projector of the pooled student feature is applied
    to two rotations of the same student input and mapped back to the teacher frame;
    the term asks the two back-mapped outputs to agree, and each block to spread
    (variance hinge) and decorrelate (channel covariance). Needs
    ``high_order.second_rotation``."""

    weight: NonNegativeFloat = 0.1
    """Weight of the term."""
    l_max: PositiveInt = 2
    """Highest order of the projected blocks."""
    channels: PositiveInt = 16
    """Number of channels of each block."""
    rotation_group: Literal["O3", "SO3"] = "O3"
    """``O3``: proper and pseudo blocks of every order; ``SO3``: proper blocks only."""
    warmup_fraction: Probability = 0.05
    """The term is off for this fraction of the steps, then ramps linearly to its
    weight over the same fraction."""


class RotationAveragedVicregTermHypers(TypedDict):
    """``vicreg_averaged`` term: the shared VICReg on the student features averaged
    over two rotations of the same student input, at the rows of ``vicreg``. Needs
    ``high_order.second_rotation``."""

    variance: NonNegativeFloat = 1.0
    """Weight of the variance hinge."""
    covariance: NonNegativeFloat = 0.04
    """Weight of the covariance penalty."""


class RotationConditionedLatentTermHypers(TypedDict):
    """``rotation_conditioned_latent`` term (C): predicts the teacher latent of the
    clean structure (the mean over the antithetic teacher pair, else the clean
    teacher) at the latent targets, as
    ``sum_{(l, sigma)} <D^(l,sigma)(g_S), A^(l,sigma) W^(l,sigma)>`` with per-block
    heads ``A`` on the student tokens and low-rank maps ``W`` to the latent, and
    compares it with the teacher after the contraction of both with the step's
    probes. The pseudo blocks (``(0, -)`` included, active from order 1) exist only
    with ``rotation_group: O3``; with ``rotation_group: none`` the order stays 0.
    ``l_max: 0`` is the invariant JEPA.

    Order curriculum: ``L = 0`` while the training progress is below
    ``warmup_fraction``, then ``min(2, l_max)``; after the warm-up, ``L`` grows by
    one (up to ``l_max``) when the norm ratio of order ``L`` stays above
    ``promote_threshold`` for 3 consecutive validation epochs. There is no promotion
    during the warm-up."""

    weight: NonNegativeFloat = 1.0
    """Weight of the term."""
    l_max: NonNegativeInt = 4
    """Highest order of the curriculum."""
    hidden: PositiveInt = 64
    """Hidden width of the heads and of the invariant context of masked atoms."""
    rank: PositiveInt = 64
    """Rank of each block's map to the teacher latent."""
    promote_threshold: PositiveFloat = 0.2
    """Norm ratio of the highest active order above which it counts towards a
    promotion."""
    warmup_fraction: Probability = 0.05
    """Fraction of the steps with ``L = 0``."""
    floor_gamma: NonNegativeFloat = 0.1
    """Weight of the per-batch variance floor of each l > 0 block contribution (at
    0.1 target standard deviations)."""
    head: AnchorHeadHypers = init_with_defaults(AnchorHeadHypers)
    """Form of the heads."""


class ActionResponseTermHypers(TypedDict):
    """``action_response`` term (D5): at each teacher-mode centre, the response of
    the teacher latent to the windowed l=1 teacher modes, ``(h+ - h-) / 2`` with
    the antithetic teacher pair (else ``h+ - h(x0)``), predicted as
    ``sum_blocks <D(g_S) a, A W>`` from the mode amplitudes ``a`` (dilation
    (0, +), rotation (1, -), shear (2, +)) mapped to the student frame, per-block
    heads ``A`` on the student tokens at the centre and low-rank maps ``W`` to the
    latent; both sides are contracted with the step's probes. The rotation and shear
    contributions get a per-batch variance floor at 0.1 target standard
    deviations. The term is off during the first ``warmup_fraction`` of the
    steps."""

    weight: NonNegativeFloat = 1.0
    """Weight of the term."""
    hidden: PositiveInt = 64
    """Hidden width of the heads."""
    rank: PositiveInt = 64
    """Rank of each block's map to the teacher latent."""
    l_max: Literal[0, 1, 2] = 2
    """Highest order of the blocks (1 drops the shear block)."""
    mode_amplitude: Optional[PositiveFloat] = None
    """Reference amplitude of the relative-response monitors ``rho``
    (``action_response/rho_*``); ``null`` uses ``high_order.mode_amplitude``."""
    shuffle_amplitudes: bool = False
    """Control: permute the amplitudes across the centres of the batch before the
    contraction, which decorrelates them from the actual teacher modes."""
    warmup_fraction: Probability = 0.05
    """Fraction of the steps with the term off."""
    floor_gamma: NonNegativeFloat = 0.1
    """Weight of the per-batch variance floor of the rotation and shear
    contributions (at 0.1 target standard deviations)."""
    head: AnchorHeadHypers = init_with_defaults(AnchorHeadHypers)
    """Form of the heads."""
