"""The Grad-JEPA objective: view construction, random-scalar gradients and losses."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from metatomic.torch import NeighborListOptions, System

from .backbones import BACKBONES, Backbone, HeadInputs
from .masking import boundary_atoms, k_hop_ego_region, nl_ball_region
from .views import bounded_noise, build_view, pair_vectors, random_rotations


if TYPE_CHECKING:
    from .terms import Term


def sample_mask(
    n_atoms: int,
    pairs: torch.Tensor,
    distances: torch.Tensor,
    mask_hypers: Dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample the masked atoms of one system, as configured by the ``mask`` hypers.

    A region is sampled (with probability ``mask.probability``, for systems of at
    least ``mask.min_atoms`` atoms) and assigned to the masked role, or to the
    context role with probability ``mask.swap_probability``. A degenerate region
    (all or no atoms masked) gives no mask.

    :param n_atoms: Number of atoms.
    :param pairs: Pair atom indices, shape ``(n_pairs, 2)``, on CPU.
    :param distances: Pair distances, shape ``(n_pairs,)``, on CPU.
    :param mask_hypers: The ``mask`` trainer hypers.
    :param generator: CPU random generator.
    :return: Boolean mask of the masked atoms, shape ``(n_atoms,)``, on CPU.
    """
    masked = torch.zeros(n_atoms, dtype=torch.bool)
    apply_mask = (
        n_atoms >= mask_hypers["min_atoms"]
        and float(torch.rand(1, generator=generator)) < mask_hypers["probability"]
    )
    if apply_mask:
        if mask_hypers["generator"] == "nl_ball":
            low, high = mask_hypers["fraction_range"]
            fraction = low + (high - low) * float(torch.rand(1, generator=generator))
            region = nl_ball_region(
                n_atoms,
                pairs,
                distances,
                mask_hypers["ball_radius_range"],
                fraction,
                generator,
            )
        elif mask_hypers["generator"] == "k_hop_ego":
            region = k_hop_ego_region(
                n_atoms,
                pairs,
                distances,
                mask_hypers["ego_hops"],
                mask_hypers["topology_cutoff"],
                generator,
            )
        else:
            raise ValueError(f"unknown mask generator '{mask_hypers['generator']}'")
        swap = (
            float(torch.rand(1, generator=generator)) < mask_hypers["swap_probability"]
        )
        masked = ~region if swap else region
        if bool(masked.all()) or not bool(masked.any()):
            masked = torch.zeros_like(masked)  # degenerate region: no mask
    return masked


@dataclass
class Views:
    """Student/teacher views of a batch and the per-atom bookkeeping.

    Every view builder fills the required fields, which the shared objective reads:
    the student and teacher systems are encoded, ``rotations`` maps the student
    gradients back to the teacher frame, and ``masked`` gives the context atoms
    (``~masked``) of the ``vicreg`` term and of the ``masked_fraction`` and
    ``rankme`` diagnostics. The optional fields are filled by the Grad-JEPA builder
    (:func:`build_views`) for its predictor (``displacements``, ``pairs`` and
    ``vectors``, besides the required ``masked``) and terms (``latent_targets`` and
    ``in_masked_structure`` for ``latent``; ``gradient_targets`` and
    ``in_masked_structure`` for ``gradient``; ``gradient_targets`` for
    ``gradient_spread``). ``extras`` holds builder-specific data; its key
    ``"teacher_views"`` (``Dict[str, List[System]]``) names extra teacher views,
    encoded without gradients by :func:`encode_extra_teacher_views`, and its key
    ``"student_views"`` (same type) extra student views, encoded with gradients by
    :func:`encode_extra_student_views`.
    """

    student: List[System]
    teacher: List[System]
    rotations: torch.Tensor  # (n_atoms, 3, 3), rotation of each atom's system
    masked: torch.Tensor  # (n_atoms,) bool
    displacements: Optional[torch.Tensor] = None  # (n_atoms, 3), teacher view
    # (n_atoms,) bool, atom's structure is masked
    in_masked_structure: Optional[torch.Tensor] = None
    latent_targets: Optional[torch.Tensor] = None  # (n_atoms,) bool
    gradient_targets: Optional[torch.Tensor] = None  # (n_atoms,) bool
    # (n_pairs, 2) clean pairs below the cutoff, batch-global
    pairs: Optional[torch.Tensor] = None
    vectors: Optional[torch.Tensor] = None  # (n_pairs, 3) clean, teacher frame
    extras: Dict[str, Any] = field(default_factory=dict)  # builder-specific data


def build_views(
    systems: List[System],
    options: NeighborListOptions,
    hypers: Dict[str, Any],
    generator: torch.Generator,
    position_gradients: bool = True,
) -> Views:
    """Sample a rotation, a displacement field and a mask for every system.

    The teacher sees the full structure, displaced by ``delta`` (with probability
    ``noise.probability``), in the original frame. The student sees the clean
    structure, rotated, with all pairs touching masked atoms removed (with
    probability ``mask.probability``).

    :param systems: Clean systems holding a (skin) neighbor list for ``options``.
    :param options: PET neighbor list options.
    :param hypers: The trainer hypers.
    :param generator: CPU random generator driving all sampling.
    :param position_gradients: Whether the positions of the views require grad.
    :return: The views.
    """
    dtype = systems[0].positions.dtype
    device = systems[0].positions.device
    cutoff = options.cutoff
    noise_hypers = hypers["noise"]
    mask_hypers = hypers["mask"]
    rotations = random_rotations(
        len(systems), hypers["rotation_group"], dtype, generator
    )

    student, teacher = [], []
    all_rotations, all_displacements, all_masked, all_in_masked = [], [], [], []
    all_latent, all_gradient, all_pairs, all_vectors = [], [], [], []
    offset = 0
    for index, system in enumerate(systems):
        n_atoms = len(system)
        samples = system.get_neighbor_list(options).samples.values
        clean_vectors = pair_vectors(system.positions, system.cell, samples)
        distances = clean_vectors.norm(dim=-1)
        pairs = samples[:, :2]
        # region sampling loops over atoms: run it on CPU to avoid device syncs
        pairs_cpu = pairs.cpu()
        distances_cpu = distances.detach().cpu()

        # teacher view: full structure, optionally displaced
        delta = torch.zeros(n_atoms, 3, dtype=dtype)
        if float(torch.rand(1, generator=generator)) < noise_hypers["probability"]:
            delta = bounded_noise(
                n_atoms,
                noise_hypers["sigma"],
                noise_hypers["skin"] / 2,
                dtype,
                generator,
            )
            # remove the global translation (PET is invariant to it, the predictor
            # sees absolute displacements); relative displacements, and hence the
            # exactness of the skin neighbor list, are unchanged
            delta = delta - delta.mean(dim=0)
        delta = delta.to(device)
        teacher.append(
            build_view(
                system,
                options,
                (system.positions + delta).detach().requires_grad_(position_gradients),
                system.cell,
            )
        )

        # mask: a region, assigned to the masked role (or to the context role
        # with probability ``swap_probability``)
        masked = sample_mask(n_atoms, pairs_cpu, distances_cpu, mask_hypers, generator)
        masked = masked.to(device)

        # student view: clean, rotated, masked pairs removed
        rotation = rotations[index].to(device)
        keep = ~(masked[samples[:, 0]] | masked[samples[:, 1]])
        student.append(
            build_view(
                system,
                options,
                (system.positions @ rotation.T)
                .detach()
                .requires_grad_(position_gradients),
                system.cell @ rotation.T,
                keep_pairs=keep,
            )
        )

        close = distances < cutoff
        # atoms without neighbors have an identically zero teacher gradient
        has_neighbors = torch.zeros_like(masked)
        has_neighbors[pairs[close, 0]] = True
        if bool(masked.any()):
            # masked atoms are only predictable from context neighbors (the
            # predictor does one round of message passing within the cutoff)
            reachable = torch.zeros_like(masked)
            reachable[pairs[close & ~masked[pairs[:, 1]], 0]] = True
            latent_targets = masked & reachable
            gradient_targets = (masked & reachable) | boundary_atoms(
                masked, pairs, distances, cutoff
            )
        else:
            latent_targets = torch.ones_like(masked)
            gradient_targets = torch.ones_like(masked)
        gradient_targets = gradient_targets & has_neighbors
        all_rotations.append(rotation.expand(n_atoms, 3, 3))
        all_displacements.append(delta)
        all_masked.append(masked)
        all_in_masked.append(masked.any().expand(n_atoms))
        all_latent.append(latent_targets)
        all_gradient.append(gradient_targets)
        all_pairs.append(pairs[close] + offset)
        all_vectors.append(clean_vectors[close])
        offset += n_atoms

    return Views(
        student=student,
        teacher=teacher,
        rotations=torch.cat(all_rotations),
        displacements=torch.cat(all_displacements),
        masked=torch.cat(all_masked),
        in_masked_structure=torch.cat(all_in_masked),
        latent_targets=torch.cat(all_latent),
        gradient_targets=torch.cat(all_gradient),
        pairs=torch.cat(all_pairs),
        vectors=torch.cat(all_vectors),
    )


ViewBuilder = Callable[
    [List[System], NeighborListOptions, Dict[str, Any], torch.Generator, bool], Views
]
"""``(systems, options, hypers, generator, position_gradients) -> Views``."""

VIEW_BUILDERS: Dict[str, ViewBuilder] = {"grad_jepa": build_views}
"""The view builders, by the ``views.builder`` trainer hyperparameter."""


def register_view_builder(name: str, builder: ViewBuilder) -> None:
    """Register a view builder under ``name`` (idempotent for the same callable).

    :param name: The ``views.builder`` name.
    :param builder: The builder, with the signature of :func:`build_views`.
    :raises ValueError: If another builder is registered under ``name``.
    """
    if name in VIEW_BUILDERS and VIEW_BUILDERS[name] is not builder:
        raise ValueError(f"view builder '{name}' is already registered")
    VIEW_BUILDERS[name] = builder


def encode_extra_teacher_views(
    teacher: torch.nn.Module,
    extras: Dict[str, Any],
    backbone: Backbone = BACKBONES["pet"],
) -> Dict[str, torch.Tensor]:
    """No-grad per-atom features of every system list in ``extras["teacher_views"]``.

    :param teacher: The EMA teacher core.
    :param extras: The :attr:`Views.extras` of a view builder.
    :param backbone: The adapter of the teacher backbone.
    :return: The detached features, by view name (empty without extra views).
    """
    out: Dict[str, torch.Tensor] = {}
    for name, systems in extras.get("teacher_views", {}).items():
        with torch.no_grad():
            out[name] = backbone.features(teacher, systems).detach()
    return out


def encode_extra_student_views(
    student: torch.nn.Module,
    extras: Dict[str, Any],
    backbone: Backbone = BACKBONES["pet"],
) -> Dict[str, HeadInputs]:
    """Tokens, with gradients, of every system list in ``extras["student_views"]``.

    :param student: The student core.
    :param extras: The :attr:`Views.extras` of a view builder.
    :param backbone: The adapter of the student backbone.
    :return: The tokens, by view name (empty without extra views).
    :raises ValueError: If the backbone does not expose its tokens.
    """
    out: Dict[str, HeadInputs] = {}
    for name, systems in extras.get("student_views", {}).items():
        _, tokens = backbone.forward_with_tokens(student, systems)
        if tokens is None:
            raise ValueError(
                f"backbone '{backbone.name}' does not expose the tokens of the "
                "extra student views"
            )
        out[name] = tokens
    return out


def sample_probes(
    num_probes: int, dim: int, dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    """Rademacher probes scaled by ``1/sqrt(dim)``, shape ``(num_probes, dim)``."""
    signs = torch.randint(0, 2, (num_probes, dim), generator=generator) * 2 - 1
    return signs.to(dtype) / dim**0.5


def random_scalar_gradients(
    features: torch.Tensor,
    positions: List[torch.Tensor],
    probes: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """Gradients of the random scalars ``s_p = sum_i v_p . h_i`` w.r.t. positions.

    :param features: Per-atom features ``(n_atoms, D)``.
    :param positions: Position tensors of every system (requiring grad).
    :param probes: Probe vectors ``(n_probes, D)``.
    :param create_graph: Whether to build the graph of the gradient (student).
    :return: Gradients ``(n_probes, n_atoms, 3)``.
    """
    scalars = (features @ probes.T).sum(dim=0)  # (n_probes,)
    gradients = []
    for p in range(probes.shape[0]):
        grads = torch.autograd.grad(
            scalars[p],
            positions,
            create_graph=create_graph,
            retain_graph=create_graph or p < probes.shape[0] - 1,
        )
        gradients.append(torch.cat(grads))
    return torch.stack(gradients)


def normalized_gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, select: torch.Tensor
) -> torch.Tensor:
    """Per-probe MSE normalized by the mean squared norm of the target.

    :param prediction: ``(n_probes, n_atoms, 3)``.
    :param target: ``(n_probes, n_atoms, 3)``.
    :param select: Boolean atom mask of the targets.
    :return: Scalar loss.
    """
    if not bool(select.any()):
        return prediction.sum() * 0.0
    error = ((prediction - target)[:, select] ** 2).sum(-1).mean(-1)
    scale = (target[:, select] ** 2).sum(-1).mean(-1).detach() + 1e-8
    return (error / scale).mean()


def latent_loss(
    prediction: torch.Tensor, target: torch.Tensor, select: torch.Tensor
) -> torch.Tensor:
    """MSE to layer-normalized (no affine) teacher features on selected atoms."""
    if not bool(select.any()):
        return prediction.sum() * 0.0
    target = torch.nn.functional.layer_norm(target, target.shape[-1:])
    return ((prediction[select] - target[select]) ** 2).mean()


def vicreg_terms(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale-free VICReg terms over rows (atoms).

    The variance hinge (at 1) acts on the per-dimension spread of the
    layer-normalized features, and the covariance penalty on the off-diagonal of
    their correlation matrix, so neither depends on the raw feature scale.

    :param features: Features ``(n_rows, D)``.
    :return: The variance and covariance terms.
    """
    if features.shape[0] < 2:
        zero = features.sum() * 0.0
        return zero, zero
    normalized = torch.nn.functional.layer_norm(features, features.shape[-1:])
    centered = normalized - normalized.mean(dim=0)
    std = torch.sqrt(centered.var(dim=0) + 1e-4)
    variance = torch.relu(1.0 - std).mean()
    n, d = centered.shape
    standardized = centered / std
    correlation = standardized.T @ standardized / (n - 1)
    off_diagonal = correlation - torch.diag(torch.diagonal(correlation))
    return variance, (off_diagonal**2).sum() / d


def relative_gradient_spread(
    features: torch.Tensor,
    gradients: torch.Tensor,
    probes: torch.Tensor,
    select: torch.Tensor,
) -> torch.Tensor:
    """Spread of the gradients relative to the spread of the per-atom scalars.

    ``mean_p std_i(g_pi) / std_i(v_p . h_i)``, in inverse length units, which is
    invariant to the scale of the features.

    :param features: Features ``(n_atoms, D)``.
    :param gradients: Gradients ``(n_probes, n_atoms, 3)``.
    :param probes: Probes ``(n_probes, D)``.
    :param select: Boolean mask of the atoms to use (at least 2).
    :return: Scalar spread.
    """
    scalars = (features[select] @ probes.T).T  # (n_probes, n_selected)
    spread = gradients[:, select].std(dim=1).mean(dim=-1)  # (n_probes,)
    return (spread / (scalars.std(dim=1) + 1e-8)).mean()


def rankme(features: torch.Tensor) -> torch.Tensor:
    """RankMe effective rank (exp of the spectral entropy) of the rows."""
    # computed in float64 on the CPU, since MPS has no float64
    singular = torch.linalg.svdvals(features.detach().cpu().double())
    p = singular / (singular.sum() + 1e-12) + 1e-12
    return torch.exp(-(p * torch.log(p)).sum())


@dataclass
class StepData:
    """Everything the loss terms of one batch can read."""

    views: Views
    probes: torch.Tensor  # (n_probes, D)
    student_features: torch.Tensor  # (n_atoms, D)
    teacher_features: torch.Tensor  # (n_atoms, D), detached
    # (n_probes, n_atoms, 3), teacher frame; None when no term needs gradients
    student_gradients: Optional[torch.Tensor]
    teacher_gradients: Optional[torch.Tensor]
    # "latent", and "gradient" if computed; empty when no term uses the predictor
    prediction: Dict[str, torch.Tensor]
    hypers: Dict[str, Any]
    # a copy of ``views.extras``, plus "teacher_features" (no-grad features of the
    # extra teacher views), "step_fraction" (training progress in [0, 1], 1 in
    # validation), "generator" (the step's random generator) and, with a view
    # builder other than Grad-JEPA's, "student_tokens" (``HeadInputs`` of the
    # student view, with gradients), and "student_tokens_extra" (``HeadInputs`` of
    # the extra student views, with gradients, by name)
    extras: Dict[str, Any] = field(default_factory=dict)


RESERVED_EXTRAS = (
    "teacher_features",
    "step_fraction",
    "generator",
    "student_tokens",
    "student_tokens_extra",
)
"""``StepData.extras`` keys set by the objective, not usable by view builders."""


def build_step_data(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    predictor: Optional[torch.nn.Module],
    systems: List[System],
    options: NeighborListOptions,
    hypers: Dict[str, Any],
    generator: torch.Generator,
    needs_gradients: bool,
    backbone: Backbone = BACKBONES["pet"],
    step_fraction: float = 0.0,
) -> StepData:
    """Build the views, encode them and run the predictor (if given).

    Without ``needs_gradients``, the positions do not require grad, the probe
    gradients are not computed, the teacher runs without autograd and the predictor
    returns only ``latent``. The random stream (views, then probes) is the same in
    both modes.

    :param student: Student PET core.
    :param teacher: EMA teacher PET core (in eval mode, no parameter gradients).
    :param predictor: The :class:`GradJEPAPredictor`, or ``None`` to skip it.
    :param systems: Clean systems with the (skin) neighbor list.
    :param options: PET neighbor list options.
    :param hypers: The trainer hypers.
    :param generator: CPU random generator.
    :param needs_gradients: Whether some active term needs the probe gradients.
    :param backbone: The adapter of the student and teacher backbone.
    :param step_fraction: Training progress in [0, 1] (1 in validation).
    :return: The step data.
    """
    builder = hypers["views"]["builder"]
    views = VIEW_BUILDERS[builder](systems, options, hypers, generator, needs_gradients)

    reserved = set(views.extras) & set(RESERVED_EXTRAS)
    if reserved:
        raise ValueError(
            f"view builder '{builder}' uses the reserved extras key(s) "
            f"{sorted(reserved)}"
        )
    extras: Dict[str, Any] = dict(views.extras)
    if builder == "grad_jepa":
        student_features = backbone.features(student, views.student)
    else:
        # heads of other SSL methods read the backbone tokens
        student_features, extras["student_tokens"] = backbone.forward_with_tokens(
            student, views.student
        )
    probes = sample_probes(
        hypers["num_probes"],
        student_features.shape[1],
        student_features.dtype,
        generator,
    ).to(student_features.device)

    student_gradients: Optional[torch.Tensor] = None
    teacher_gradients: Optional[torch.Tensor] = None
    if needs_gradients:
        # the graph of the gradients is only needed to train the student
        student_gradients = random_scalar_gradients(
            student_features,
            [s.positions for s in views.student],
            probes,
            create_graph=student.training,
        )
        # back to the teacher frame: x' = x R^T  =>  dL/dx = dL/dx' R
        student_gradients = torch.einsum(
            "pni,nij->pnj", student_gradients, views.rotations
        )
        teacher_features = backbone.features(teacher, views.teacher)
        teacher_gradients = random_scalar_gradients(
            teacher_features,
            [s.positions for s in views.teacher],
            probes,
            create_graph=False,
        )
    else:
        with torch.no_grad():
            teacher_features = backbone.features(teacher, views.teacher)

    extras["teacher_features"] = encode_extra_teacher_views(teacher, extras, backbone)
    extras["student_tokens_extra"] = encode_extra_student_views(
        student, extras, backbone
    )
    extras["step_fraction"] = step_fraction
    extras["generator"] = generator

    step = StepData(
        views=views,
        probes=probes,
        student_features=student_features,
        teacher_features=teacher_features.detach(),
        student_gradients=student_gradients,
        teacher_gradients=teacher_gradients,
        prediction={},  # set below: the conditioners read the rest of the step
        hypers=hypers,
        extras=extras,
    )
    if predictor is not None:
        types = torch.cat([s.types for s in systems])
        with torch.set_grad_enabled(student.training):
            step.prediction = predictor(
                student_features,
                student_gradients,
                probes,
                views.displacements,
                types,
                views.masked,
                views.pairs,
                views.vectors,
                step,
            )
    return step


RESERVED_DIAGNOSTICS = ("loss", "masked_fraction", "rankme")
"""Metric names set by the objective and the trainer, not usable by the terms."""


def grad_jepa_term_losses(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    predictor: Optional[torch.nn.Module],
    terms: Sequence["Term"],
    systems: List[System],
    options: NeighborListOptions,
    hypers: Dict[str, Any],
    generator: torch.Generator,
    with_rank: bool = False,
    backbone: Backbone = BACKBONES["pet"],
    step_fraction: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Compute the (weighted) loss of every active term on one batch.

    The probe gradients are computed only if some term needs them, and the
    predictor runs only if some term uses its prediction (a term without the
    ``uses_prediction`` attribute does not).

    :param student: Student PET core (in training mode).
    :param teacher: EMA teacher PET core (in eval mode, no parameter gradients).
    :param predictor: The :class:`GradJEPAPredictor` (``None`` if no term uses it).
    :param terms: The active loss terms (see :func:`terms.build_terms`).
    :param systems: Clean systems with the (skin) neighbor list.
    :param options: PET neighbor list options.
    :param hypers: The trainer hypers.
    :param generator: CPU random generator.
    :param with_rank: Also report the RankMe of (at most 4096) student features.
    :param backbone: The adapter of the student and teacher backbone.
    :param step_fraction: Training progress in [0, 1] (1 in validation).
    :return: The loss of each term, by name and in the order of ``terms``, and a
        dictionary of scalar diagnostics.
    :raises ValueError: Without terms, or without a predictor for a term using it.
    """
    if len(terms) == 0:
        raise ValueError("the objective needs at least one loss term")
    uses_prediction = any(getattr(term, "uses_prediction", False) for term in terms)
    if uses_prediction and predictor is None:
        raise ValueError("an active loss term uses the predictor, but none is given")
    step = build_step_data(
        student,
        teacher,
        predictor if uses_prediction else None,
        systems,
        options,
        hypers,
        generator,
        needs_gradients=any(term.needs_gradients for term in terms),
        backbone=backbone,
        step_fraction=step_fraction,
    )

    losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, torch.Tensor] = {}
    for term in terms:
        if term.name in losses:
            raise ValueError(f"two active loss terms are named '{term.name}'")
        term_loss, diagnostics = term(step)
        losses[term.name] = term_loss
        for name, value in diagnostics.items():
            if name in RESERVED_DIAGNOSTICS:
                raise ValueError(f"diagnostic name '{name}' is reserved")
            if name in metrics:
                raise ValueError(f"diagnostic '{name}' is reported by two terms")
            metrics[name] = value

    context = ~step.views.masked
    metrics["masked_fraction"] = step.views.masked.cpu().double().mean()
    if with_rank:
        metrics["rankme"] = rankme(step.student_features[context][:4096])
    return losses, {name: float(value.detach()) for name, value in metrics.items()}


def grad_jepa_loss(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    predictor: Optional[torch.nn.Module],
    terms: Sequence["Term"],
    systems: List[System],
    options: NeighborListOptions,
    hypers: Dict[str, Any],
    generator: torch.Generator,
    with_rank: bool = False,
    backbone: Backbone = BACKBONES["pet"],
    step_fraction: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the loss of one batch: the sum of the active terms.

    See :func:`grad_jepa_term_losses` for the parameters.

    :return: The loss and a dictionary of scalar diagnostics.
    """
    losses, metrics = grad_jepa_term_losses(
        student,
        teacher,
        predictor,
        terms,
        systems,
        options,
        hypers,
        generator,
        with_rank=with_rank,
        backbone=backbone,
        step_fraction=step_fraction,
    )
    values = list(losses.values())
    loss = values[0]
    for term_loss in values[1:]:
        loss = loss + term_loss
    return loss, metrics
