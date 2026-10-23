"""Loss terms of the high-order method: the analytic anchors.

- ``residual_multipole`` (A): the residual multipoles of the boundary atoms;
- ``deformation_jets`` (B): the student displacement and the irreducible parts of
  the smooth-field gradient, at the atoms of deformed structures;
- ``chirality``: the triple product of three element-channel dipoles.

Every term predicts its irrep blocks ``(l, sigma)`` with per-block heads on the
student tokens, maps them back to the teacher frame with ``D^(l,sigma)(g_S)^T``
(det(g_S) for pseudo blocks), and compares them with exact teacher-frame targets.
A term returns ``weight`` times the mean over its blocks of the normalized MSE
(:class:`BlockNormalizer`).
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from ..objective import StepData
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
