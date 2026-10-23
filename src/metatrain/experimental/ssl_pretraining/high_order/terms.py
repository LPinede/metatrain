"""Loss terms of the high-order method: the analytic anchors.

- ``residual_multipole`` (A): the residual multipoles of the boundary atoms;
- ``deformation_jets`` (B): the student displacement and the irreducible parts of
  the smooth-field gradient, at the atoms of deformed structures;
- ``chirality``: the triple product of three element-channel dipoles;

and the collapse regularizers that need the second student rotation
(``high_order.second_rotation``):

- ``equivariant_vicreg``: VICReg between the back-mapped outputs of a linear
  projector on two rotations of the same student input, per irrep block;
- ``vicreg_averaged``: the shared VICReg on the features averaged over the two
  rotations.

Every term predicts its irrep blocks ``(l, sigma)`` with per-block heads on the
student tokens, maps them back to the teacher frame with ``D^(l,sigma)(g_S)^T``
(det(g_S) for pseudo blocks), and compares them with exact teacher-frame targets.
A term returns ``weight`` times the mean over its blocks of the normalized MSE
(:class:`BlockNormalizer`).
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from ..backbones import HeadInputs
from ..objective import StepData, vicreg_terms
from ..terms import TermContext
from .heads import BlockHead, BlockNormalizer, batch_variance_floor
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
    ) -> None:
        super().__init__()
        self.weight = weight
        self.required_targets: dict = {}
        self.heads = torch.nn.ModuleDict(
            {
                block: BlockHead(dims, l, 1, hidden, **(head_kwargs or {}))
                for block, (l, _) in self.BLOCKS.items()
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
        return jet_targets(ho.u_student[sites], ho.grad_smooth[sites])

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
            for block, (l, sigma) in self.BLOCKS.items()
        }

    def forward(self, step: StepData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diag: Dict[str, torch.Tensor] = {}
        preds, targets = self.predict(step), self.teacher_frame_targets(step)
        losses = [
            _block_loss(
                self.norm, f"{self.name}/{block}", preds[block], targets[block], diag
            )
            for block in self.BLOCKS
        ]
        loss = torch.stack(losses).mean()
        diag[f"{self.name}/loss"] = loss.detach()
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
    ``rotation_group="O3"``), a linear map of the pooled student feature of each
    view is mapped back to the teacher frame with its own rotation (``g_S`` and
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
                    dims[0] + dims[1], (2 * l + 1) * channels
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
