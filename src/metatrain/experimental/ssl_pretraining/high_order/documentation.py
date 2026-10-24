"""Hyperparameters of the high-order views (``training.high_order``) and of the
high-order loss terms (entries of ``training.objective.terms``).

This module is the single source of the defaults. It must stay torch-free: the
shared ``documentation.py`` imports it.
"""

from typing import Annotated, Literal

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
    """Form of the per-block heads of an analytic term."""

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
