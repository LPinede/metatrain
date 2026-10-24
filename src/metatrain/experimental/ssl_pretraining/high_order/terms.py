"""Loss terms of the high-order method.

The analytic anchors:

- ``residual_multipole`` (A): the residual multipoles of the boundary atoms;
- ``deformation_jets`` (B): the student displacement and the irreducible parts of
  the smooth-field gradient, at the atoms of deformed structures;
- ``chirality``: the triple product of three element-channel dipoles;

the teacher terms:

- ``rotation_conditioned_latent`` (C): the teacher latent, predicted through the
  student rotation and compared after a probe contraction, with an order
  curriculum (order 0: the invariant JEPA);
- ``action_response`` (D5): the teacher's response to the windowed l=1 teacher
  modes at the mode centres, predicted from the mode amplitudes mapped to the
  student frame and compared after a probe contraction;

and the collapse regularizers that need the second student rotation
(``high_order.second_rotation``):

- ``equivariant_vicreg``: VICReg between the back-mapped outputs of a linear
  projector on two rotations of the same student input, per irrep block;
- ``vicreg_averaged``: the shared VICReg on the features averaged over the two
  rotations.

Every anchor predicts its irrep blocks ``(l, sigma)`` with per-block heads on the
student tokens, maps them back to the teacher frame with ``D^(l,sigma)(g_S)^T``
(det(g_S) for pseudo blocks), and compares them with exact teacher-frame targets.
A term returns ``weight`` times the mean over its blocks of the normalized MSE
(:class:`BlockNormalizer`). Term C instead contracts ``D^(l,sigma)(g_S)`` with
its head outputs to predict the teacher latent, compared after the contraction of
both sides with the step's probes.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ..backbones import HeadInputs
from ..objective import StepData, vicreg_terms
from ..terms import TermContext
from .heads import BlockHead, BlockNormalizer, InvariantContext, batch_variance_floor
from .irreps import XYZ_TO_L1, back_map, irrep_matrix
from .targets import chirality, decompose_gradient, residual_multipoles
from .views import HighOrderExtras


MIN_CHIRALITY_NEIGHBORS = 3
"""Minimum number of neighbours within the cutoff of a chirality site."""

FLOOR_LEVEL = 0.1
"""Level of the per-batch variance floor of each component, in standard deviations
of that component's target."""

MIN_TARGET_VARIANCE = 1e-8
"""Batch variance of a block's targets (target units squared) below which the block
counts as having no targets: it costs nothing and leaves the EMA of its target
variance untouched (normalizing by a vanishing variance would blow up the loss)."""


def _block_loss(
    norm: BlockNormalizer,
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    diag: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Normalized MSE of one block; zero, but in the graph, for a block with fewer
    than two sites or without target variance (reported as ``{name}/skipped``)."""
    skipped = (
        target.shape[0] < 2
        or float(target.detach().var(unbiased=False)) < MIN_TARGET_VARIANCE
    )
    diag[f"{name}/skipped"] = pred.new_tensor(float(skipped)).detach()
    if skipped:
        return pred.sum() * 0.0
    loss = norm(name, pred, target)
    with torch.no_grad():
        diag[f"{name}/rmse"] = (pred - target).pow(2).mean().sqrt()
        diag[f"{name}/norm_ratio"] = pred.norm() / target.norm().clamp(min=1e-12)
    return loss


def _pairs_of(
    sites: torch.Tensor, ho: HighOrderExtras, within: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The clean pairs centred on ``sites`` (boolean over the atoms), restricted to
    ``within`` (boolean over the pairs) if given, with the index of their centre
    among the sites."""
    selected = sites[ho.pair_index[:, 0]]
    if within is not None:
        selected = selected & within
    local = torch.cumsum(sites.long(), 0) - 1
    return selected, local[ho.pair_index[selected, 0]]


def _head_kwargs(hypers: Dict[str, Any]) -> Dict[str, Any]:
    """Split the ``head`` entry off the term hypers."""
    hypers = dict(hypers)
    return {"head_kwargs": hypers.pop("head", None), **hypers}


class ResidualMultipoleTerm(torch.nn.Module):
    """Term A: the residual multipoles (orders ``0..l_max``, all proper) of the
    boundary atoms, one channel per element (``species_index``) and radial function.

    :param dims: ``(Dn, De)`` token dimensions.
    :param l_max: Highest order.
    :param n_radial: Number of radial functions.
    :param n_species: Number of elements (``len(atomic_types)``).
    :param cutoff: Cutoff of the radial functions.
    :param hidden: Hidden width of the heads.
    :param weight: Weight of the term.
    :param floor_gamma: Weight of the per-batch variance floor of the l > 0 blocks.
    :param head_kwargs: Keyword arguments of every :class:`BlockHead`.
    """

    name = "residual_multipole"
    needs_gradients = False
    uses_prediction = False

    def __init__(
        self,
        dims: Tuple[int, int],
        l_max: int,
        n_radial: int,
        n_species: int,
        cutoff: float,
        hidden: int,
        weight: float,
        floor_gamma: float,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.l_max, self.n_radial, self.n_species = l_max, n_radial, n_species
        self.cutoff, self.weight, self.floor_gamma = cutoff, weight, floor_gamma
        self.required_targets: dict = {}
        self.heads = torch.nn.ModuleDict(
            {
                f"l{l}": BlockHead(
                    dims, l, n_radial * n_species, hidden, **(head_kwargs or {})
                )
                for l in range(l_max + 1)
            }
        )
        self.norm = BlockNormalizer()

    @classmethod
    def from_hypers(
        cls, hypers: Dict[str, Any], context: TermContext
    ) -> "ResidualMultipoleTerm":
        """Build the term from its hypers and the run's context."""
        return cls(
            dims=context.dims,
            n_species=len(context.atomic_types),
            cutoff=context.cutoff,
            **_head_kwargs(hypers),
        )

    def teacher_frame_targets(self, step: StepData) -> Dict[str, torch.Tensor]:
        """The targets, ``(n_boundary, 2l+1, n_radial * n_species)`` per block."""
        ho: HighOrderExtras = step.extras["high_order"]
        n_sites = int(ho.boundary.sum())
        selected, centers = _pairs_of(ho.boundary, ho)
        species = ho.species_index[ho.pair_index[selected, 1]]
        weights = torch.nn.functional.one_hot(species, self.n_species)
        residuals = residual_multipoles(
            centers,
            ho.clean_vectors[selected],
            ho.observed_vectors[selected],
            ho.hidden_source[selected],
            weights.to(ho.clean_vectors),
            n_sites,
            self.l_max,
            self.n_radial,
            self.cutoff,
        )
        # (n, n_radial, n_species, 2l+1) -> (n, 2l+1, n_radial * n_species)
        return {
            f"l{l}": r.reshape(
                n_sites, self.n_radial * self.n_species, 2 * l + 1
            ).transpose(1, 2)
            for l, r in enumerate(residuals)
        }

    def predict(self, step: StepData) -> Dict[str, torch.Tensor]:
        """The predictions, back-mapped to the teacher frame."""
        ho: HighOrderExtras = step.extras["high_order"]
        tokens = step.extras["student_tokens"].select(ho.boundary)
        rotations = ho.rotations[ho.boundary]
        return {
            f"l{l}": back_map(
                self.heads[f"l{l}"](tokens), irrep_matrix(rotations, l, 1)
            )
            for l in range(self.l_max + 1)
        }

    def forward(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diag: Dict[str, torch.Tensor] = {}
        preds, targets = self.predict(step), self.teacher_frame_targets(step)
        names = {block: f"{self.name}/{block}" for block in preds}
        losses = [
            _block_loss(self.norm, names[block], preds[block], targets[block], diag)
            for block in preds
        ]
        loss = torch.stack(losses).mean()
        # per-batch floor on the l > 0 predictions, per component and relative to
        # the target's spread (components whose target never varies cost nothing),
        # in units of the block scale
        for l in range(1, self.l_max + 1):
            scale = self.norm.scale(names[f"l{l}"])
            pred, target = preds[f"l{l}"] / scale, targets[f"l{l}"] / scale
            if target.shape[0] >= 2:
                level = FLOOR_LEVEL * target.std(dim=0)
                loss = loss + self.floor_gamma * batch_variance_floor(pred, level)
        diag[f"{self.name}/loss"] = loss.detach()
        return self.weight * loss, diag


def jet_targets(u: torch.Tensor, grad: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The blocks of term B, ``(n, 2l+1, 1)``, in the real-harmonic bases.

    :param u: Displacements ``(n, 3)``, xyz.
    :param grad: Smooth-field gradients ``(n, 3, 3)``, ``G_ab = du_a / dx_b``.
    :return: ``displacement`` (1,+), ``dilation`` (0,+), ``rotation`` (1,-) and
        ``shear`` (2,+).
    """
    dilation, rotation, shear = decompose_gradient(grad)
    to_l1 = list(XYZ_TO_L1)  # the l=1 real basis is ordered (y, z, x)
    return {
        "displacement": u[:, to_l1, None],
        "dilation": dilation[:, :, None],
        "rotation": rotation[:, to_l1, None],
        "shear": shear[:, :, None],
    }


class DeformationJetTerm(torch.nn.Module):
    """Term B: at the atoms of deformed structures (all visible), the student
    displacement ``u_S`` and the dilation, local rotation and shear of the exact
    smooth-field gradient.

    :param dims: ``(Dn, De)`` token dimensions.
    :param hidden: Hidden width of the heads.
    :param weight: Weight of the term.
    :param head_kwargs: Keyword arguments of every :class:`BlockHead`.
    :param l_max: Highest order of the blocks (1 drops the shear block).
    """

    name = "deformation_jets"
    needs_gradients = False
    uses_prediction = False
    BLOCKS = {
        "displacement": (1, 1),
        "dilation": (0, 1),
        "rotation": (1, -1),
        "shear": (2, 1),
    }
    """``(l, sigma)`` of each block."""

    def __init__(
        self,
        dims: Tuple[int, int],
        hidden: int,
        weight: float,
        head_kwargs: Optional[Dict[str, Any]] = None,
        l_max: int = 2,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.required_targets: dict = {}
        self.blocks = {k: v for k, v in self.BLOCKS.items() if v[0] <= l_max}
        self.heads = torch.nn.ModuleDict(
            {
                block: BlockHead(dims, l, 1, hidden, **(head_kwargs or {}))
                for block, (l, _) in self.blocks.items()
            }
        )
        self.norm = BlockNormalizer()

    @classmethod
    def from_hypers(
        cls, hypers: Dict[str, Any], context: TermContext
    ) -> "DeformationJetTerm":
        """Build the term from its hypers and the run's context."""
        return cls(dims=context.dims, **_head_kwargs(hypers))

    @staticmethod
    def _sites(ho: HighOrderExtras) -> torch.Tensor:
        return ho.branch_deformed[ho.structure_index] & (ho.boundary | ho.context)

    def teacher_frame_targets(self, step: StepData) -> Dict[str, torch.Tensor]:
        """The targets, ``(n_sites, 2l+1, 1)`` per block."""
        ho: HighOrderExtras = step.extras["high_order"]
        sites = self._sites(ho)
        targets = jet_targets(ho.u_student[sites], ho.grad_smooth[sites])
        return {block: targets[block] for block in self.blocks}

    def predict(self, step: StepData) -> Dict[str, torch.Tensor]:
        """The predictions, back-mapped to the teacher frame."""
        ho: HighOrderExtras = step.extras["high_order"]
        sites = self._sites(ho)
        tokens = step.extras["student_tokens"].select(sites)
        rotations = ho.rotations[sites]
        return {
            block: back_map(
                self.heads[block](tokens), irrep_matrix(rotations, l, sigma)
            )
            for block, (l, sigma) in self.blocks.items()
        }

    def forward(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diag: Dict[str, torch.Tensor] = {}
        ho: HighOrderExtras = step.extras["high_order"]
        preds, targets = self.predict(step), self.teacher_frame_targets(step)
        losses = [
            _block_loss(
                self.norm, f"{self.name}/{block}", preds[block], targets[block], diag
            )
            for block in self.blocks
        ]
        loss = torch.stack(losses).mean()
        diag[f"{self.name}/loss"] = loss.detach()
        # mean safety factor of the smooth field over the deformed structures
        scales = ho.field_scale[ho.branch_deformed]
        diag[f"{self.name}/field_scale"] = (
            scales.mean().detach() if len(scales) else loss.new_ones(())
        )
        return self.weight * loss, diag


class ChiralityTerm(torch.nn.Module):
    """Chirality, a (0,-) block: ``kappa = mu_a . (mu_b x mu_c)`` with the clean
    channel dipoles of three element types, at the visible atoms with at least
    :data:`MIN_CHIRALITY_NEIGHBORS` clean neighbours within ``cutoff``.

    :param dims: ``(Dn, De)`` token dimensions.
    :param hidden: Hidden width of the head.
    :param weight: Weight of the term.
    :param channel_types: The three atomic types (atomic numbers) of the channels.
    :param cutoff: Cutoff of the dipoles and of the neighbour count.
    :param head_kwargs: Keyword arguments of the :class:`BlockHead`.
    """

    name = "chirality"
    needs_gradients = False
    uses_prediction = False

    def __init__(
        self,
        dims: Tuple[int, int],
        hidden: int,
        weight: float,
        channel_types: Sequence[int],
        cutoff: float,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if len(set(channel_types)) != 3 or len(channel_types) != 3:
            raise ValueError("chirality needs three distinct channel types")
        self.weight, self.cutoff = weight, cutoff
        self.channel_types = tuple(int(t) for t in channel_types)
        self.required_targets: dict = {}
        self.head = BlockHead(dims, 0, 1, hidden, **(head_kwargs or {}))
        self.norm = BlockNormalizer()

    @classmethod
    def from_hypers(
        cls, hypers: Dict[str, Any], context: TermContext
    ) -> "ChiralityTerm":
        """Build the term from its hypers and the run's context."""
        return cls(dims=context.dims, cutoff=context.cutoff, **_head_kwargs(hypers))

    def _sites(self, ho: HighOrderExtras) -> Tuple[torch.Tensor, torch.Tensor]:
        """The sites (boolean over the atoms) and the clean pairs within the
        cutoff (boolean over the pairs)."""
        close = ho.clean_vectors.norm(dim=-1) < self.cutoff
        counts = torch.bincount(ho.pair_index[close, 0], minlength=ho.boundary.numel())
        sites = (ho.boundary | ho.context) & (counts >= MIN_CHIRALITY_NEIGHBORS)
        return sites, close

    def teacher_frame_targets(self, step: StepData) -> Dict[str, torch.Tensor]:
        """The target, ``(n_sites, 1, 1)``."""
        ho: HighOrderExtras = step.extras["high_order"]
        sites, close = self._sites(ho)
        selected, centers = _pairs_of(sites, ho, within=close)
        neighbors = ho.species[ho.pair_index[selected, 1]]
        weights = torch.stack([neighbors == t for t in self.channel_types], dim=-1).to(
            ho.clean_vectors
        )
        kappa = chirality(
            centers,
            ho.clean_vectors[selected],
            weights,
            int(sites.sum()),
            self.cutoff,
        )
        return {"l0": kappa[:, None, None]}

    def predict(self, step: StepData) -> Dict[str, torch.Tensor]:
        """The prediction, back-mapped to the teacher frame (times det(g_S))."""
        ho: HighOrderExtras = step.extras["high_order"]
        sites, _ = self._sites(ho)
        pred = self.head(step.extras["student_tokens"].select(sites))
        return {"l0": back_map(pred, irrep_matrix(ho.rotations[sites], 0, -1))}

    def forward(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diag: Dict[str, torch.Tensor] = {}
        pred = self.predict(step)["l0"]
        target = self.teacher_frame_targets(step)["l0"]
        loss = _block_loss(self.norm, f"{self.name}/l0", pred, target, diag)
        diag[f"{self.name}/loss"] = loss.detach()
        return self.weight * loss, diag


AGREEMENT_EPS = 1e-8
"""Added to the variance that normalizes the agreement (avoids 0/0 for constant
outputs)."""

VARIANCE_LEVEL = 0.5
"""Level of the variance hinge of ``equivariant_vicreg``, on the standard deviation
of each RMS-normalized component."""

BLOCK_COVARIANCE_WEIGHT = 0.04
"""Weight of the channel covariance relative to the agreement and the variance
hinge in ``equivariant_vicreg``."""


def _pooled(x: HeadInputs) -> torch.Tensor:
    """The pooled per-atom feature: node tokens and the cutoff-weighted edge-token
    sum (equal to the backbone's per-atom ``"feature"`` output)."""
    return torch.cat([x.node, (x.edge * x.cutoff[..., None]).sum(1)], dim=-1)


def _second_rotation_tokens(step: StepData, name: str) -> HeadInputs:
    """The tokens of the second student rotation, or a clear error without it."""
    tokens = step.extras.get("student_tokens_extra", {})
    if "rot2" not in tokens:
        raise ValueError(f"the '{name}' term needs `high_order.second_rotation`")
    return tokens["rot2"]


def equivariant_vicreg_block(
    z1: torch.Tensor, z2: torch.Tensor, var_level: float = VARIANCE_LEVEL
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Agreement, variance hinge and channel covariance of one block of back-mapped
    outputs of two rotations, ``(n, 2l+1, channels)`` each.

    - agreement: ``mean |z1 - z2|^2`` over the atoms, divided by ``(2l+1)`` times
      the mean per-component variance of the two views; 2 for outputs that ignore
      the orientation (two independent random back-maps), 0 for equivariant ones;
    - variance hinge: both views are divided by one scalar, their joint RMS, then
      ``relu(var_level - std)`` of each component over the atoms (scale-free);
    - covariance: the squared off-diagonal of the channel Gram matrix of the
      centred, normalized outputs, contracted over the atoms and the ``2l+1``
      components (invariant under a rotation shared by all atoms), over channels.

    The hinge and the covariance are averaged over the two views.

    .. code-block:: python

        agreement, hinge, covariance = equivariant_vicreg_block(z1, z2)

    :param z1: Back-mapped outputs of the first rotation.
    :param z2: Back-mapped outputs of the second rotation.
    :param var_level: Level of the variance hinge.
    :return: The three scalars (zero, in the graph, for fewer than two atoms).
    """
    if z1.shape[0] < 2:
        zero = (z1.sum() + z2.sum()) * 0.0
        return zero, zero, zero
    n, m, c = z1.shape
    variance = 0.5 * (
        z1.var(0, unbiased=False).mean() + z2.var(0, unbiased=False).mean()
    )
    agreement = ((z1 - z2) ** 2).sum(1).mean() / (m * variance + AGREEMENT_EPS)
    scale = (0.5 * (z1.pow(2).mean() + z2.pow(2).mean())).sqrt() + AGREEMENT_EPS
    hinge, covariance = z1.new_zeros(()), z1.new_zeros(())
    for z in (z1 / scale, z2 / scale):
        hinge = hinge + torch.relu(var_level - z.std(0)).mean()
        centered = z - z.mean(0)
        gram = torch.einsum("nmc,nmd->cd", centered, centered) / (n * m)
        off_diagonal = gram - torch.diag(torch.diagonal(gram))
        covariance = covariance + off_diagonal.pow(2).sum() / c
    return agreement, 0.5 * hinge, 0.5 * covariance


class EquivariantVicregTerm(torch.nn.Module):
    """VICReg between two rotations of the same student input, per irrep block.

    For every block ``(l, sigma)``, ``1 <= l <= l_max`` (both parities with
    ``rotation_group="O3"``), a linear map (no bias) of the pooled student feature
    of each view is mapped back to the teacher frame with its own rotation (``g_S`` and
    ``g2 g_S``), over the visible atoms. The loss is ``weight * ramp *`` the mean
    over the blocks of ``agreement + hinge + 0.04 covariance``
    (:func:`equivariant_vicreg_block`). The projector is linear so that it cannot
    create angular orders: the criterion is met only by order-l content of the
    backbone features. Order 0 is left to the shared anti-collapse term.

    .. code-block:: python

        term = EquivariantVicregTerm(dims=(256, 256), l_max=2, channels=16)
        loss, diagnostics = term(step)

    :param dims: ``(Dn, De)`` token dimensions.
    :param l_max: Highest order of the blocks.
    :param channels: Number of channels of each block.
    :param weight: Weight of the term.
    :param rotation_group: ``"O3"`` (proper and pseudo blocks) or ``"SO3"``
        (proper blocks only).
    :param warmup_fraction: Off for this fraction of the steps, then ramped
        linearly to ``weight`` over the same fraction.
    """

    name = "equivariant_vicreg"
    needs_gradients = False
    uses_prediction = False

    def __init__(
        self,
        dims: Tuple[int, int],
        l_max: int = 2,
        channels: int = 16,
        weight: float = 0.1,
        rotation_group: str = "O3",
        warmup_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        if rotation_group not in ("O3", "SO3"):
            raise ValueError(f"unknown rotation group '{rotation_group}'")
        self.weight, self.warmup_fraction, self.channels = (
            weight,
            warmup_fraction,
            channels,
        )
        self.required_targets: dict = {}
        sigmas = (1, -1) if rotation_group == "O3" else (1,)
        self.blocks = [(l, sigma) for l in range(1, l_max + 1) for sigma in sigmas]
        self.proj = torch.nn.ModuleDict(
            {
                self._key(l, sigma): torch.nn.Linear(
                    dims[0] + dims[1], (2 * l + 1) * channels, bias=False
                )
                for l, sigma in self.blocks
            }
        )

    @staticmethod
    def _key(l: int, sigma: int) -> str:
        return f"l{l}{'p' if sigma == 1 else 'm'}"

    @classmethod
    def from_hypers(
        cls, hypers: Dict[str, Any], context: TermContext
    ) -> "EquivariantVicregTerm":
        """Build the term from its hypers and the run's context."""
        return cls(dims=context.dims, **hypers)

    def ramp(self, step_fraction: float) -> float:
        """The schedule factor: 0 during the warm-up, then linear to 1 over the
        next ``warmup_fraction`` of the steps."""
        w = self.warmup_fraction
        if w == 0:
            return 1.0
        return float(min(max((step_fraction - w) / w, 0.0), 1.0))

    def back_mapped(
        self, step: StepData, l: int, sigma: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The projected block of both views at the visible atoms, mapped back to
        the teacher frame, ``(n_visible, 2l+1, channels)`` each."""
        ho: HighOrderExtras = step.extras["high_order"]
        visible = ~step.views.masked
        second = _second_rotation_tokens(step, self.name)
        assert ho.rotations2 is not None
        proj = self.proj[self._key(l, sigma)]
        out = []
        for tokens, rotations in (
            (step.extras["student_tokens"], ho.rotations),
            (second, ho.rotations2),
        ):
            p = proj(_pooled(tokens.select(visible)))
            p = p.reshape(-1, 2 * l + 1, self.channels)
            out.append(back_map(p, irrep_matrix(rotations[visible], l, sigma)))
        return out[0], out[1]

    def forward(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diag: Dict[str, torch.Tensor] = {}
        losses = []
        for l, sigma in self.blocks:
            key = f"{self.name}/{self._key(l, sigma)}"
            agreement, hinge, covariance = equivariant_vicreg_block(
                *self.back_mapped(step, l, sigma)
            )
            losses.append(agreement + hinge + BLOCK_COVARIANCE_WEIGHT * covariance)
            diag[f"{key}/agreement"] = agreement.detach()
            diag[f"{key}/variance"] = hinge.detach()
            diag[f"{key}/covariance"] = covariance.detach()
        loss = torch.stack(losses).mean()
        ramp = self.ramp(step.extras["step_fraction"])
        diag[f"{self.name}/loss"] = loss.detach()
        diag[f"{self.name}/ramp"] = loss.new_tensor(ramp)
        return self.weight * ramp * loss, diag


def vicreg_on_average(
    f1: torch.Tensor, f2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The shared VICReg terms of the features averaged over two views."""
    return vicreg_terms(0.5 * (f1 + f2))


class RotationAveragedVicregTerm:
    """``variance * L_var + covariance * L_cov`` of the student features averaged
    over two rotations of the same student input, at the rows of ``vicreg``.

    Averaging halves, in expectation, the variance that features can get from the
    orientation of the input alone, so VICReg must find more of its variance in
    the structures.

    .. code-block:: python

        term = RotationAveragedVicregTerm(variance=1.0, covariance=0.04)
        loss, diagnostics = term(step)

    :param variance: Weight of the variance hinge.
    :param covariance: Weight of the covariance penalty.
    """

    name = "vicreg_averaged"
    needs_gradients = False
    uses_prediction = False

    def __init__(self, variance: float = 1.0, covariance: float = 0.04) -> None:
        self.variance, self.covariance = variance, covariance
        self.required_targets: dict = {}

    def __call__(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rows = ~step.views.masked
        second = _second_rotation_tokens(step, self.name)
        f1 = _pooled(step.extras["student_tokens"].select(rows))
        f2 = _pooled(second.select(rows))
        variance, covariance = vicreg_on_average(f1, f2)
        loss = self.variance * variance + self.covariance * covariance
        return loss, {
            f"{self.name}/variance": variance.detach(),
            f"{self.name}/covariance": covariance.detach(),
        }


def _spread(x: torch.Tensor) -> torch.Tensor:
    """The norm of ``x`` centred over the batch (per column)."""
    if x.shape[0] == 0:
        return x.new_zeros(())
    return (x - x.mean(dim=0)).norm()


CONTEXT_RBF = 16
"""Number of radial functions of the invariant context of masked atoms."""

PROMOTION_EPOCHS = 3
"""Consecutive validation epochs above the threshold that promote the order."""


class RotationConditionedLatentTerm(torch.nn.Module):
    """Term C: the teacher latent of the clean structure, predicted through the
    student rotation and compared after the probe contraction.

    The teacher latent at the latent targets (``Views.latent_targets``) is
    ``h = (h+ + h-) / 2`` with the antithetic teacher pair, or ``h = h(x0)`` with
    the clean teacher (one-sided views). The prediction of the latent itself is
    ``h_hat = sum_{active (l, sigma)} <D^(l,sigma)(g_S), Phi^(l,sigma)>`` with
    ``Phi^(l,sigma) = A^(l,sigma) W^(l,sigma)``: ``A`` is a per-block head output
    ``(n, 2l+1, 2l+1, rank)`` on the student tokens and ``W`` a learnable
    ``(rank, feature_dim)`` matrix (low rank). Both sides are contracted with the
    step's probes ``v_p`` (redrawn every step): the loss compares
    ``t_hat_p = v_p . h_hat`` with ``t_p = v_p . h``, and ``v_p`` is contracted into
    ``W`` first (cost ``rank * feature_dim * num_probes``). Because the heads predict
    ``h`` and not one output per probe, they can learn although the probes change
    at every step.

    The optimum of each block is the order-``(l, sigma)`` Fourier coefficient of
    the teacher latent over the orientations of the student input; a teacher
    without orientation content makes the zero output of every block but
    ``(0, +)`` optimal (C is neutral).

    Blocks: ``(0, +)`` is always active, so ``L = 0`` is the invariant JEPA. With
    ``rotation_group="O3"`` the pseudo blocks ``(l, -)`` exist too, ``(0, -)``
    included; ``(0, -)`` is active whenever the order is at least 1. With
    ``rotation_group="none"`` the student is not rotated, so the order is forced to
    0 (only the ``(0, +)`` head exists).

    At masked atoms, the node tokens are replaced by an :class:`InvariantContext`
    over the clean pairs from the masked atom to its visible neighbours (their
    distances are those of the student input, since masked structures are not
    deformed), and the cutoff factors are set to 0 (no edge token is read).

    Order curriculum (state saved with the term): ``L = 0`` while the training
    progress is below ``warmup_fraction``, then ``active_l_max``, which starts at
    ``min(2, l_max)``. :meth:`maybe_promote` (the validation-epoch hook) adds the
    next order after :data:`PROMOTION_EPOCHS` consecutive validation epochs with the
    norm ratio of order ``active_l_max`` above ``promote_threshold``; it never
    promotes during the warm-up. Evaluation uses the order of the last training
    step. The norm ratio of an order is the batch spread (per-probe standard
    deviations, combined) of its contribution over that of the target.

    The loss is ``weight * (L_C + floor_gamma * sum_blocks floor)``: ``L_C`` is the
    normalized MSE of ``t_hat`` (:func:`_block_loss`, skipped without target
    variance) and each active ``l > 0`` block contribution gets the per-batch
    variance floor at :data:`FLOOR_LEVEL` target standard deviations (per probe, in
    units of the target scale); ``(0, -)`` gets no floor.

    .. code-block:: python

        term = RotationConditionedLatentTerm((256, 256), 512, 4, 64, 4.5, 1.0)
        loss, diagnostics = term(step)
        term.maybe_promote(validation_diagnostics)

    :param dims: ``(Dn, De)`` token dimensions.
    :param feature_dim: Dimension of the teacher latent.
    :param l_max: Highest order of the curriculum (0: the invariant JEPA).
    :param hidden: Hidden width of the heads and of the invariant context.
    :param cutoff: Cutoff of the invariant context's radial functions.
    :param weight: Weight of the term.
    :param rank: Rank of each block's map to the teacher latent.
    :param promote_threshold: Norm ratio of the highest active order that counts
        towards a promotion.
    :param rotation_group: ``"O3"`` (proper and pseudo blocks), ``"SO3"`` (proper
        blocks only) or ``"none"`` (order 0 only).
    :param floor_gamma: Weight of the floors of the ``l > 0`` block contributions.
    :param warmup_fraction: Fraction of the steps with ``L = 0``.
    :param head_kwargs: Keyword arguments of every :class:`BlockHead`.
    """

    name = "rotation_conditioned_latent"
    needs_gradients = False
    uses_prediction = False

    def __init__(
        self,
        dims: Tuple[int, int],
        feature_dim: int,
        l_max: int,
        hidden: int,
        cutoff: float,
        weight: float,
        rank: int = 64,
        promote_threshold: float = 0.2,
        rotation_group: str = "O3",
        floor_gamma: float = 0.1,
        warmup_fraction: float = 0.05,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if rotation_group not in ("O3", "SO3", "none"):
            raise ValueError(f"unknown rotation group '{rotation_group}'")
        if rotation_group == "none":
            l_max = 0  # without student rotations, D(g_S) is the identity
        self.l_max, self.rank, self.weight = l_max, rank, weight
        self.promote_threshold, self.floor_gamma = promote_threshold, floor_gamma
        self.warmup_fraction = warmup_fraction
        self.required_targets: dict = {}
        sigmas = (1, -1) if rotation_group == "O3" else (1,)
        # (0, -) only if some order >= 1 can become active
        self.head_keys = [
            (l, sigma)
            for l in range(l_max + 1)
            for sigma in sigmas
            if not (l == 0 and sigma == -1 and l_max == 0)
        ]
        self.heads = torch.nn.ModuleDict(
            {
                self.key(l, sigma): BlockHead(
                    dims, l, (2 * l + 1) * rank, hidden, **(head_kwargs or {})
                )
                for l, sigma in self.head_keys
            }
        )
        # W^(l,sigma) as the weight (feature_dim, rank) of a bias-free linear map
        self.maps = torch.nn.ModuleDict(
            {
                self.key(l, sigma): torch.nn.Linear(rank, feature_dim, bias=False)
                for l, sigma in self.head_keys
            }
        )
        self.context = InvariantContext(dims[0], hidden, CONTEXT_RBF, cutoff)
        self.norm = BlockNormalizer()
        # curriculum state, saved with the term
        self.register_buffer("active_order", torch.tensor(min(2, l_max)))
        self.register_buffer("promotion_streak", torch.tensor(0))
        self.register_buffer("warmed_up", torch.tensor(warmup_fraction == 0))

    @staticmethod
    def key(l: int, sigma: int) -> str:
        """The name of the head of block ``(l, sigma)``."""
        return f"l{l}{'p' if sigma == 1 else 'm'}"

    @property
    def active_l_max(self) -> int:
        """The highest order after the warm-up."""
        return int(self.active_order)

    @classmethod
    def from_hypers(
        cls, hypers: Dict[str, Any], context: TermContext
    ) -> "RotationConditionedLatentTerm":
        """Build the term from its hypers and the run's context."""
        if context.feature_dim is None:
            raise ValueError(f"the '{cls.name}' term needs `context.feature_dim`")
        return cls(
            dims=context.dims,
            feature_dim=context.feature_dim,
            cutoff=context.cutoff,
            rotation_group=context.rotation_group,
            **_head_kwargs(hypers),
        )

    def context_pairs(self, step: StepData) -> Tuple[torch.Tensor, torch.Tensor]:
        """The clean pairs from a masked atom to a visible neighbour within the
        cutoff, ``(p, 2)``, and their distances ``(p,)``."""
        ho: HighOrderExtras = step.extras["high_order"]
        masked = step.views.masked
        selected = masked[ho.pair_index[:, 0]] & ~masked[ho.pair_index[:, 1]]
        return ho.pair_index[selected], ho.clean_vectors[selected].norm(dim=-1)

    def head_inputs(self, step: StepData) -> HeadInputs:
        """The student tokens, with the invariant context as the node tokens of the
        masked atoms and their cutoff factors set to 0."""
        tokens: HeadInputs = step.extras["student_tokens"]
        masked = step.views.masked
        if not bool(masked.any()):
            return tokens
        context = self.context(tokens.node, *self.context_pairs(step))
        node = torch.where(masked[:, None], context, tokens.node)
        cutoff = tokens.cutoff * (~masked)[:, None].to(tokens.cutoff)
        return HeadInputs(node, tokens.edge, cutoff)

    def teacher_frame_targets(self, step: StepData) -> torch.Tensor:
        """The probe-contracted teacher latents at the latent targets,
        ``(n_sites, num_probes)``."""
        extra = step.extras["teacher_features"]
        teacher = step.teacher_features
        if "minus" in extra:
            # ``Views.teacher`` is the plus view of the antithetic pair
            teacher = 0.5 * (teacher + extra["minus"])
        atoms = step.views.latent_targets
        return (teacher @ step.probes.T.to(teacher))[atoms]

    def order(self) -> int:
        """The highest order used now (0 during the warm-up)."""
        return self.active_l_max if bool(self.warmed_up) else 0

    def active_blocks(self) -> List[Tuple[int, int]]:
        """The blocks used now: ``(0, +)``, and every block up to the order (with
        ``(0, -)``) once the order is at least 1."""
        order = self.order()
        return [
            (l, sigma)
            for l, sigma in self.head_keys
            if l <= order and (sigma == 1 or order >= 1)
        ]

    def contributions(self, step: StepData) -> Dict[Tuple[int, int], torch.Tensor]:
        """The probe contraction ``v_p . <D^(l,sigma)(g_S), A W>`` of each active
        block, ``(n_sites, num_probes)``."""
        ho: HighOrderExtras = step.extras["high_order"]
        atoms = step.views.latent_targets
        inputs = self.head_inputs(step).select(atoms)
        rotations = ho.rotations[atoms]
        out = {}
        for l, sigma in self.active_blocks():
            name = self.key(l, sigma)
            a = self.heads[name](inputs).reshape(-1, 2 * l + 1, 2 * l + 1, self.rank)
            matrices = irrep_matrix(rotations, l, sigma).to(a)
            # <D, A> per rank channel, then W v_p (probes contracted into W first)
            rotated = torch.einsum("nmq,nmqk->nk", matrices, a)
            probed = step.probes.to(a) @ self.maps[name].weight  # (P, rank)
            out[(l, sigma)] = rotated @ probed.T
        return out

    def predict(self, step: StepData) -> torch.Tensor:
        """The probe-contracted prediction at the latent targets,
        ``(n_sites, num_probes)``."""
        return torch.stack(list(self.contributions(step).values())).sum(0)

    def forward(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.training:
            fraction = float(step.extras["step_fraction"])
            self.warmed_up.fill_(fraction >= self.warmup_fraction)
        diag: Dict[str, torch.Tensor] = {}
        target = self.teacher_frame_targets(step)
        parts = self.contributions(step)
        pred = torch.stack(list(parts.values())).sum(0)
        block = f"{self.name}/total"
        loss = _block_loss(self.norm, block, pred, target, diag)
        # per-batch floors of the l > 0 block contributions, per probe and relative
        # to the target's spread, in units of the target scale
        floor = pred.new_zeros(())
        scale = self.norm.scale(block)
        if target.shape[0] >= 2:
            level = FLOOR_LEVEL * target.std(dim=0) / scale
            for (l, _), part in parts.items():
                if l > 0:
                    floor = floor + batch_variance_floor(part / scale, level)
        loss = loss + self.floor_gamma * floor
        with torch.no_grad():
            # spread over the batch (per-probe centring) relative to the target's
            target_spread = _spread(target).clamp(min=1e-12)
            for l in range(self.order() + 1):
                # both parities, except (0, -), which is reported apart
                order_part = [
                    p for (k, s), p in parts.items() if k == l and (k, s) != (0, -1)
                ]
                diag[f"{self.name}/l{l}/norm_ratio"] = (
                    _spread(torch.stack(order_part).sum(0)) / target_spread
                )
            if (0, -1) in parts:
                diag[f"{self.name}/l0_pseudo/norm_ratio"] = (
                    _spread(parts[(0, -1)]) / target_spread
                )
        diag[f"{self.name}/floor"] = floor.detach()
        diag[f"{self.name}/loss"] = loss.detach()
        diag[f"{self.name}/active_l_max"] = loss.new_tensor(float(self.order()))
        return self.weight * loss, diag

    def maybe_promote(self, diagnostics: Mapping[str, Any]) -> None:
        """Count a validation epoch towards the promotion of the next order.

        The streak grows when the norm ratio of order ``active_l_max`` exceeds
        ``promote_threshold`` and resets otherwise; at :data:`PROMOTION_EPOCHS` the
        order is promoted (up to ``l_max``) and the streak restarts. Nothing
        happens during the warm-up, at ``l_max``, or without the norm ratio.

        :param diagnostics: The epoch-averaged validation diagnostics (floats or
            scalar tensors).
        """
        key = f"{self.name}/l{self.active_l_max}/norm_ratio"
        if not bool(self.warmed_up) or self.active_l_max >= self.l_max:
            self.promotion_streak.zero_()
            return
        if key not in diagnostics:
            return
        if float(diagnostics[key]) > self.promote_threshold:
            self.promotion_streak += 1
        else:
            self.promotion_streak.zero_()
        if int(self.promotion_streak) >= PROMOTION_EPOCHS:
            self.active_order += 1
            self.promotion_streak.zero_()

    def on_validation_epoch_end(self, diagnostics: Mapping[str, Any]) -> None:
        """The trainer hook: :meth:`maybe_promote`."""
        self.maybe_promote(diagnostics)


class ActionResponseTerm(torch.nn.Module):
    """Term D5: the response of the teacher to the windowed l=1 teacher modes at
    the mode centres, predicted from the student tokens and the mode amplitudes.

    At each centre ``c`` (``HighOrderExtras.centers``), the teacher sees ``x0 +- u_T``
    with ``u_T`` built from the amplitudes ``a_c`` (dilation (0, +), rotation
    (1, -), shear (2, +); teacher frame). The response is
    ``Delta = (h+ - h-) / 2`` with the antithetic teacher pair, or
    ``Delta = h+ - h(x0)`` with the one-sided teacher. The prediction of the response
    vector is ``Delta_hat = sum_blocks <a_S, A W>``: ``a_S = D^(l,sigma)(g_S) a`` are
    the amplitudes mapped to the student frame (``det(g_S)`` for the axial rotation
    block), ``A`` is a per-block head output ``(n_c, 2l+1, rank)`` on the student
    tokens at the centre and ``W`` a learnable ``(rank, feature_dim)`` matrix (low
    rank). Both sides are contracted with the step's probes ``v_p`` (redrawn every
    step), ``v_p`` into ``W`` first, as in term C.

    ``shuffle_amplitudes`` (the control): the amplitudes are permuted across the
    centres of the batch (step generator) before the contraction, so they are
    decorrelated from the actual ``u_T``.

    The loss is ``weight * (L_D5 + floor_gamma * floors)``: ``L_D5`` is the
    normalized MSE of the probe-contracted response (:func:`_block_loss`, skipped
    with fewer than two centres or without target variance) and the rotation and
    shear contributions get the per-batch variance floor at :data:`FLOOR_LEVEL`
    target standard deviations (per probe, in units of the target scale). The term
    is off (weight 0) while the training progress is below ``warmup_fraction``.

    Monitors: ``rho_target = ||Delta|| / (mode_amplitude ||h_c||)`` (vector norms
    over the centres of the batch, ``h_c`` the teacher latent of the clean
    structure: the pair mean, or ``h(x0)``), the same ratio for each block's
    predicted response (``rho_l{l}_{block}``), and the ``r2`` of the
    probe-contracted prediction. Also ``n_centers`` (centres of the step) and
    ``cap_fraction`` (mean over the structures with a centre of the fraction of
    atoms whose ``u_T`` the skin cap scaled down).

    .. code-block:: python

        term = ActionResponseTerm((256, 256), 512, 64, 1.0, 0.05)
        loss, diagnostics = term(step)

    :param dims: ``(Dn, De)`` token dimensions.
    :param feature_dim: Dimension of the teacher latent.
    :param hidden: Hidden width of the heads.
    :param weight: Weight of the term.
    :param mode_amplitude: Reference amplitude of ``rho``; ``None`` reads
        ``high_order.mode_amplitude`` from the step hypers.
    :param rank: Rank of each block's map to the teacher latent.
    :param l_max: Highest order of the blocks (2: all three).
    :param shuffle_amplitudes: Permute the amplitudes across the centres.
    :param floor_gamma: Weight of the floors of the rotation and shear contributions.
    :param warmup_fraction: Fraction of the steps with the term off.
    :param head_kwargs: Keyword arguments of every :class:`BlockHead`.
    """

    name = "action_response"
    needs_gradients = False
    uses_prediction = False
    BLOCKS = {"dilation": (0, 1), "rotation": (1, -1), "shear": (2, 1)}
    """``(l, sigma)`` of each block."""
    SLICES = {"dilation": slice(0, 1), "rotation": slice(1, 4), "shear": slice(4, 9)}
    """Columns of each block in ``HighOrderExtras.center_amplitudes``."""

    def __init__(
        self,
        dims: Tuple[int, int],
        feature_dim: int,
        hidden: int,
        weight: float,
        mode_amplitude: Optional[float] = None,
        rank: int = 64,
        l_max: int = 2,
        shuffle_amplitudes: bool = False,
        floor_gamma: float = 0.1,
        warmup_fraction: float = 0.05,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.weight, self.mode_amplitude, self.rank = weight, mode_amplitude, rank
        self.shuffle_amplitudes, self.floor_gamma = shuffle_amplitudes, floor_gamma
        self.warmup_fraction = warmup_fraction
        self.required_targets: dict = {}
        self.blocks = {k: v for k, v in self.BLOCKS.items() if v[0] <= l_max}
        self.heads = torch.nn.ModuleDict(
            {
                block: BlockHead(dims, l, rank, hidden, **(head_kwargs or {}))
                for block, (l, _) in self.blocks.items()
            }
        )
        # W of each block as the weight (feature_dim, rank) of a bias-free linear map
        self.maps = torch.nn.ModuleDict(
            {
                block: torch.nn.Linear(rank, feature_dim, bias=False)
                for block in self.blocks
            }
        )
        self.norm = BlockNormalizer()

    @classmethod
    def from_hypers(
        cls, hypers: Dict[str, Any], context: TermContext
    ) -> "ActionResponseTerm":
        """Build the term from its hypers and the run's context."""
        if context.feature_dim is None:
            raise ValueError(f"the '{cls.name}' term needs `context.feature_dim`")
        return cls(
            dims=context.dims, feature_dim=context.feature_dim, **_head_kwargs(hypers)
        )

    def responses(self, step: StepData) -> Tuple[torch.Tensor, torch.Tensor]:
        """The teacher response ``Delta`` and the clean teacher latent ``h_c`` at
        the centres, ``(n_c, feature_dim)`` each."""
        centers = step.extras["high_order"].centers
        extra = step.extras["teacher_features"]
        teacher = step.teacher_features
        if "minus" in extra:
            # ``Views.teacher`` is the plus view of the antithetic pair
            plus, minus = teacher[centers], extra["minus"][centers]
            return 0.5 * (plus - minus), 0.5 * (plus + minus)
        # one-sided: ``Views.teacher`` is the clean x0, the extra view is the plus
        clean = teacher[centers]
        return extra["plus"][centers] - clean, clean

    def teacher_frame_targets(self, step: StepData) -> torch.Tensor:
        """The probe-contracted responses at the centres, ``(n_c, num_probes)``."""
        delta, _ = self.responses(step)
        return delta @ step.probes.T.to(delta)

    def amplitudes(self, step: StepData) -> torch.Tensor:
        """The teacher-frame amplitudes of the centres, ``(n_c, 9)``, permuted
        across the centres with ``shuffle_amplitudes``."""
        amplitudes = step.extras["high_order"].center_amplitudes
        if self.shuffle_amplitudes:
            perm = torch.randperm(len(amplitudes), generator=step.extras["generator"])
            amplitudes = amplitudes[perm.to(amplitudes.device)]
        return amplitudes

    def student_amplitudes(
        self, step: Any, amplitudes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """``D^(l,sigma)(g_S) a`` of each block at the centres, ``(n_c, 2l+1)``.

        :param step: The step data (reads the centres and the student rotations).
        :param amplitudes: Teacher-frame amplitudes ``(n_c, 9)``.
        """
        ho = step.extras["high_order"]
        rotations = ho.rotations[ho.centers]
        out = {}
        for block, (l, sigma) in self.blocks.items():
            a = amplitudes[:, self.SLICES[block]]
            if block == "rotation":
                a = a[:, list(XYZ_TO_L1)]  # xyz -> the real l=1 order
            matrices = irrep_matrix(rotations.to(a), l, sigma)
            out[block] = torch.einsum("nmp,np->nm", matrices, a)
        return out

    def rank_contributions(
        self, step: StepData, amplitudes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """``<a_S, A>`` of each block, per rank channel, ``(n_c, rank)``."""
        ho = step.extras["high_order"]
        tokens = step.extras["student_tokens"].select(ho.centers)
        out = {}
        for block, a_student in self.student_amplitudes(step, amplitudes).items():
            heads = self.heads[block](tokens)  # (n_c, 2l+1, rank)
            out[block] = torch.einsum("nm,nmk->nk", a_student.to(heads), heads)
        return out

    def _probed(
        self, step: StepData, rank_parts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """The probe contraction ``v_p . (<a_S, A> W)`` of each block,
        ``(n_c, num_probes)``, with ``v_p`` contracted into ``W`` first."""
        out = {}
        for block, part in rank_parts.items():
            probed = step.probes.to(part) @ self.maps[block].weight  # (P, rank)
            out[block] = part @ probed.T
        return out

    def predict(self, step: StepData) -> torch.Tensor:
        """The probe-contracted prediction at the centres, ``(n_c, num_probes)``."""
        parts = self._probed(step, self.rank_contributions(step, self.amplitudes(step)))
        return torch.stack(list(parts.values())).sum(0)

    def forward(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        on = float(float(step.extras["step_fraction"]) >= self.warmup_fraction)
        tokens: HeadInputs = step.extras["student_tokens"]
        diag: Dict[str, torch.Tensor] = {}
        ho = step.extras["high_order"]
        n_centers = float(ho.centers.numel())
        diag[f"{self.name}/n_centers"] = tokens.node.new_tensor(n_centers).detach()
        if ho.centers.numel() == 0:
            zero = tokens.node.sum() * 0.0
            diag[f"{self.name}/skipped"] = zero.new_ones(()).detach()
            diag[f"{self.name}/loss"] = zero.detach()
            diag[f"{self.name}/active"] = zero.new_tensor(on)
            return zero, diag
        target = self.teacher_frame_targets(step)
        rank_parts = self.rank_contributions(step, self.amplitudes(step))
        parts = self._probed(step, rank_parts)
        pred = torch.stack(list(parts.values())).sum(0)
        block = f"{self.name}/total"
        loss = _block_loss(self.norm, block, pred, target, diag)
        skipped = bool(diag[f"{block}/skipped"])
        # per-batch floors of the rotation and shear contributions, per probe and
        # relative to the target's spread, in units of the target scale
        floor = pred.new_zeros(())
        if not skipped:
            scale = self.norm.scale(block)
            level = FLOOR_LEVEL * target.std(dim=0) / scale
            for name, part in parts.items():
                if self.blocks[name][0] > 0:
                    floor = floor + batch_variance_floor(part / scale, level)
        loss = loss + self.floor_gamma * floor
        if not skipped:
            with torch.no_grad():
                delta, clean = self.responses(step)
                amplitude = self.mode_amplitude
                if amplitude is None:
                    amplitude = step.hypers["high_order"]["mode_amplitude"]
                reference = amplitude * clean.norm().clamp(min=1e-12)
                diag[f"{self.name}/rho_target"] = delta.norm() / reference
                # each block's predicted response vector, relative likewise
                for name, part in rank_parts.items():
                    vector = part @ self.maps[name].weight.T
                    l = self.blocks[name][0]
                    diag[f"{self.name}/rho_l{l}_{name}"] = vector.norm() / reference
                residual = (pred - target).pow(2).mean()
                spread = (target - target.mean(dim=0)).pow(2).mean()
                diag[f"{self.name}/r2"] = 1 - residual / spread.clamp(min=1e-12)
        with torch.no_grad():
            with_centers = ho.structure_index[ho.centers].unique()
            diag[f"{self.name}/cap_fraction"] = ho.teacher_cap_fraction[
                with_centers
            ].mean()
        diag[f"{self.name}/floor"] = floor.detach()
        diag[f"{self.name}/loss"] = loss.detach()
        diag[f"{self.name}/active"] = loss.new_tensor(on)
        return self.weight * on * loss, diag
