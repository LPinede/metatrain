"""The high-order view builder (``views.builder: high_order``).

- **Student:** per structure, a mask-only branch (probability ``1 - p_def``: the
  clean structure, masked as in Grad-JEPA) or a deform-only branch (the structure
  displaced by ``u_S`` = a smooth random field plus iid noise, unmasked), rotated by
  a random ``g_S`` (``rotation_group``). The smooth field of a non-periodic
  structure has its rigid rotation removed (it would be indistinguishable from a
  change of ``g_S``).
- **Teacher:** the unrotated structure displaced both ways, ``x0 ± u_T``, where
  ``u_T`` is the sum of windowed l=1 modes (dilation, rotation, shear) with Gaussian
  amplitudes at a few centres far from the masked atoms. ``Views.teacher`` is the
  ``plus`` view and ``extras["teacher_views"]`` holds ``minus`` (``teacher_pair``),
  or ``Views.teacher`` is the clean ``x0`` and ``extras["teacher_views"]`` holds
  ``plus`` (one-sided fallback).
- **Second rotation** (``high_order.second_rotation``): the same student input
  (same mask, same displacement) under an extra random rotation ``g2`` per
  structure, in ``extras["student_views"]["rot2"]``. Its rotations are drawn after
  every other random number, so the other views do not depend on the option.

Displacements are bounded by ``noise.skin / 2`` per atom, separately for ``u_S`` and
``u_T``, so the skin neighbor list stays exact: ``u_T`` is capped per atom, and
``u_S`` is bounded by scaling its smooth part per structure and capping its iid
part (so that ``grad_smooth`` stays the exact gradient of ``u_smooth``). All random
numbers are drawn on CPU from the step generator, then moved to the device and dtype
of the positions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from metatomic.torch import NeighborListOptions, System

from ..masking import boundary_atoms
from ..objective import Views, sample_mask
from ..views import build_view, pair_vectors, random_rotations
from .deformations import (
    cap_norm,
    mode_displacements,
    periodic_field_with_gradient,
    smooth_field_with_gradient,
)


MIN_CENTER_NEIGHBORS = 3
"""Minimum number of neighbours (within the cutoff) of a teacher-mode centre: the 9
l=1 amplitudes of a centre can only be identifiable from the displacements of at
least 3 neighbours, so smaller environments (dimers, for example) have no centre."""


@dataclass
class HighOrderExtras:
    """Per-atom bookkeeping of the high-order views (``Views.extras["high_order"]``).

    Atom and pair indices are batch-global. Vectors and displacements are in the
    teacher frame (the frame of the unrotated input), except where noted.
    """

    branch_masked: torch.Tensor  # (n_structures,) bool: mask-only branch
    branch_deformed: torch.Tensor  # (n_structures,) bool: deform-only branch
    rotations: torch.Tensor  # (n_atoms, 3, 3) student rotation of each atom
    u_student: torch.Tensor  # (n_atoms, 3) total student displacement u_S
    grad_smooth: torch.Tensor  # (n_atoms, 3, 3) d(u_smooth)_a / dx_b at each atom
    u_smooth: torch.Tensor  # (n_atoms, 3) smooth part of u_S (u_S - u_smooth: iid)
    pair_index: torch.Tensor  # (n_pairs, 2) clean pairs within the cutoff
    clean_vectors: torch.Tensor  # (n_pairs, 3) clean pair vectors
    observed_vectors: torch.Tensor  # (n_pairs, 3) clean + u_S[j] - u_S[i]
    hidden_source: torch.Tensor  # (n_pairs,) bool: source atom (2nd index) masked
    boundary: torch.Tensor  # (n_atoms,) bool: visible, with a masked neighbour
    context: torch.Tensor  # (n_atoms,) bool: visible, no masked neighbour
    centers: torch.Tensor  # (n_centers,) atom indices of the teacher-mode centres
    center_amplitudes: torch.Tensor  # (n_centers, 9) [dilation, rotation(3), shear(5)]
    species: torch.Tensor  # (n_atoms,) atomic types
    species_index: torch.Tensor  # (n_atoms,) index of the type in ``atomic_types``
    structure_index: torch.Tensor  # (n_atoms,) structure of each atom
    student_pairs: torch.Tensor  # (n_student_pairs, 2) kept student pairs < cutoff
    student_distances: torch.Tensor  # (n_student_pairs,) their student distances
    # (n_atoms, 3, 3) total rotation g2 g_S of the second student view of each
    # atom, with ``high_order.second_rotation`` (else None)
    rotations2: Optional[torch.Tensor] = None


def _min_image(
    vectors: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor
) -> torch.Tensor:
    """Minimum-image convention for ``vectors`` along the periodic directions.

    Rounding the fractional coordinates gives the nearest image for orthogonal and
    mildly skewed cells; for strongly skewed cells it is an approximation (a
    shorter image may exist).

    :param vectors: Vectors, shape ``(n, 3)``.
    :param cell: Cell matrix (rows are lattice vectors), full rank if periodic.
    :param pbc: Periodicity of each lattice direction, shape ``(3,)``.
    :return: The wrapped vectors (unchanged for a non-periodic system).
    """
    if not bool(pbc.any()):
        return vectors
    frac = vectors @ torch.linalg.inv(cell)
    frac = torch.where(pbc, frac - torch.round(frac), frac)
    return frac @ cell


MIN_RELATIVE_INERTIA = 1e-3
"""Principal moments of inertia below this fraction of the trace mark rotation axes
that are not fitted by :func:`_remove_rigid_rotation`."""


def _remove_rigid_rotation(
    positions: torch.Tensor, field: torch.Tensor, gradient: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subtract from a non-periodic field its least-squares rigid infinitesimal
    rotation ``w x (r - c)`` about the centroid ``c``, and the constant
    antisymmetric gradient ``[w]_x`` of that rotation from its gradient.

    A rigid rotation of the student structure cannot be told apart from a change
    of ``g_S``, so it is not a learnable displacement target. Rotations about
    principal axes of (nearly) vanishing inertia, below
    :data:`MIN_RELATIVE_INERTIA` of the trace, are not fitted: their angular
    velocity is unidentifiable and would blow up (the axial rotation of a nearly
    linear molecule; every rotation of a single atom).

    :param positions: Positions, shape ``(n_atoms, 3)``.
    :param field: Field at the atoms, shape ``(n_atoms, 3)``.
    :param gradient: Its exact gradient, shape ``(n_atoms, 3, 3)``.
    :return: The projected field and its exact gradient.
    """
    centered = positions - positions.mean(dim=0)
    # the 3x3 fit runs on CPU in float64 (MPS has no float64)
    r, u = centered.detach().cpu().double(), field.detach().cpu().double()
    # normal equations of min_w sum_i |u_i - w x r_i|^2: I w = sum_i r_i x u_i
    eye = torch.eye(3, dtype=torch.float64)
    inertia = (r * r).sum() * eye - r.T @ r
    scale = float(inertia.diagonal().sum().clamp(min=1e-30))
    eigenvalues, axes = torch.linalg.eigh(inertia / scale)
    # solve along the well-conditioned principal axes only
    kept = eigenvalues > MIN_RELATIVE_INERTIA
    torque = axes.T @ torch.cross(r, u, dim=-1).sum(dim=0) / scale
    omega = axes[:, kept] @ (torque[kept] / eigenvalues[kept])
    zero = torch.zeros((), dtype=torch.float64)
    wx, wy, wz = omega.unbind()
    # cross-product matrix: skew @ v = w x v
    skew = torch.stack(
        [
            torch.stack([zero, -wz, wy]),
            torch.stack([wz, zero, -wx]),
            torch.stack([-wy, wx, zero]),
        ]
    )
    skew = skew.to(field)
    return field - centered @ skew.T, gradient - skew


def _student_displacement(
    field: torch.Tensor,
    gradient: torch.Tensor,
    iid: torch.Tensor,
    iid_sigma: float,
    max_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounded student displacement ``u_S = u_smooth + iid``, with ``|u_S| <=
    max_norm`` at every atom and ``grad_smooth`` the exact gradient of ``u_smooth``.

    The iid part is centred and capped per atom at ``b = min(3 iid_sigma,
    max_norm / 2)``; the centred smooth field (and its gradient) is scaled by a
    single factor per structure so that its largest norm is at most
    ``max_norm - b``. No per-atom cap acts on the smooth part, so it stays a smooth
    field with a known gradient.

    :param field: Smooth field, shape ``(n_atoms, 3)``.
    :param gradient: Its gradient, shape ``(n_atoms, 3, 3)``.
    :param iid: Raw iid noise, shape ``(n_atoms, 3)``.
    :param iid_sigma: Standard deviation of the iid noise.
    :param max_norm: Maximum displacement norm per atom (``skin / 2``).
    :return: ``u_smooth``, ``grad_smooth`` and ``u_student``.
    """
    bound = min(3 * iid_sigma, max_norm / 2)
    iid = cap_norm(iid - iid.mean(dim=0), bound)
    smooth = field - field.mean(dim=0)
    largest = smooth.norm(dim=-1).max().clamp(min=1e-12)
    scale = torch.clamp((max_norm - bound) / largest, max=1.0)
    u_smooth = scale * smooth
    return u_smooth, scale * gradient, u_smooth + iid


def _select_centers(
    eligible: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    min_distance: float,
    max_centers: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Greedy centres: eligible atoms in random order, each kept if it is at least
    ``min_distance`` (minimum image) from the centres already kept.

    :param eligible: Boolean mask of the candidate atoms, on CPU.
    :param positions: Positions, on CPU.
    :param cell: Cell matrix, on CPU.
    :param pbc: Periodicity, on CPU.
    :param min_distance: Minimum distance between centres.
    :param max_centers: Maximum number of centres.
    :param generator: CPU random generator.
    :return: Local atom indices of the centres (long, on CPU).
    """
    order = torch.randperm(len(eligible), generator=generator)
    kept: List[int] = []
    for atom in order[eligible[order]].tolist():
        if len(kept) == max_centers:
            break
        if kept:
            gaps = _min_image(positions[kept] - positions[atom], cell, pbc)
            if bool((gaps.norm(dim=-1) < min_distance).any()):
                continue
        kept.append(atom)
    return torch.tensor(kept, dtype=torch.long)


def _teacher_displacement(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    centers: torch.Tensor,
    amplitudes: torch.Tensor,
    window: float,
) -> torch.Tensor:
    """Sum of the windowed l=1 modes of the centres, with minimum-image relative
    positions in periodic systems.

    :param positions: Positions, shape ``(n_atoms, 3)``.
    :param cell: Cell matrix.
    :param pbc: Periodicity.
    :param centers: Local atom indices of the centres.
    :param amplitudes: ``(n_centers, 9)``: dilation, rotation (3), shear (5).
    :param window: Width of the Gaussian window.
    :return: The displacement ``u_T``, shape ``(n_atoms, 3)`` (uncapped).
    """
    u = torch.zeros_like(positions)
    for center, a in zip(centers.tolist(), amplitudes, strict=True):
        origin = positions[center]
        rel = _min_image(positions - origin, cell, pbc)
        u = u + mode_displacements(origin + rel, origin, a[0], a[1:4], a[4:9], window)
    return u


def build_high_order_views(
    systems: List[System],
    options: NeighborListOptions,
    hypers: Dict[str, Any],
    generator: torch.Generator,
    position_gradients: bool = True,
) -> Views:
    """Build the high-order student and teacher views of a batch.

    The positions of the student views and of ``Views.teacher`` require grad when
    ``position_gradients`` is set; the extra teacher views never do (they are
    encoded without gradients).

    ``Views.latent_targets`` holds the masked atoms with a visible neighbour within
    the cutoff (reachable from context, as in Grad-JEPA) and every visible atom. The
    other Grad-JEPA-only fields of :class:`Views` are left at their defaults.

    :param systems: Clean systems holding a (skin) neighbor list for ``options``.
    :param options: Backbone neighbor list options.
    :param hypers: The trainer hypers (``rotation_group``, ``noise.skin``, ``mask``
        and ``high_order``), plus the runtime key ``atomic_types`` (the atomic types
        of the dataset, added by the trainer; not a hyperparameter), which maps the
        atomic types to ``HighOrderExtras.species_index``.
    :param generator: CPU random generator driving all sampling.
    :param position_gradients: Whether the student and teacher positions require
        grad.
    :return: The views, with ``extras["high_order"]`` (:class:`HighOrderExtras`),
        ``extras["teacher_views"]`` and, with ``high_order.second_rotation``,
        ``extras["student_views"]``.
    :raises ValueError: If an atomic type is missing from ``atomic_types``.
    """
    dtype = systems[0].positions.dtype
    device = systems[0].positions.device
    cutoff = options.cutoff
    max_norm = hypers["noise"]["skin"] / 2
    ho = hypers["high_order"]
    rotations = random_rotations(
        len(systems), hypers["rotation_group"], dtype, generator
    )

    student: List[System] = []
    # per structure: the displaced positions, cell and kept pairs of the student
    # view, before its rotation (for the second rotation)
    student_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    teacher: List[System] = []
    extra_teacher: List[System] = []
    per_structure: Dict[str, List[torch.Tensor]] = {}

    def collect(**values: torch.Tensor) -> None:
        for name, value in values.items():
            per_structure.setdefault(name, []).append(value)

    offset = 0
    for index, system in enumerate(systems):
        n_atoms = len(system)
        positions = system.positions.detach()
        cell = system.cell.detach()
        if bool(system.pbc.any()) and bool((cell.norm(dim=-1) == 0).any()):
            raise ValueError(
                "periodic systems need three non-zero cell vectors (partially "
                "periodic systems with zero cell rows are not supported)"
            )
        samples = system.get_neighbor_list(options).samples.values
        clean_vectors = pair_vectors(positions, cell, samples)
        distances = clean_vectors.norm(dim=-1)
        pairs = samples[:, :2].long()
        close = distances < cutoff

        # student branch: deform-only (smooth field + iid noise) or mask-only
        deformed = float(torch.rand(1, generator=generator)) < ho["p_def"]
        zeros = torch.zeros(n_atoms, 3, dtype=dtype, device=device)
        if deformed:
            masked = torch.zeros(n_atoms, dtype=torch.bool)
            if bool(system.pbc.any()):
                field = periodic_field_with_gradient(
                    positions,
                    cell,
                    ho["field_amplitude"],
                    ho["max_wave_index"],
                    generator,
                )
            else:
                field = _remove_rigid_rotation(
                    positions,
                    *smooth_field_with_gradient(
                        positions,
                        ho["field_amplitude"],
                        ho["wavelength"],
                        ho["n_waves"],
                        generator,
                    ),
                )
            iid = ho["iid_sigma"] * torch.randn(
                n_atoms, 3, generator=generator, dtype=torch.float64
            )
            u_smooth, grad_smooth, u_student = _student_displacement(
                field[0],
                field[1],
                iid.to(device=device, dtype=dtype),
                ho["iid_sigma"],
                max_norm,
            )
        else:
            # region sampling loops over atoms: run it on CPU
            masked = sample_mask(
                n_atoms,
                pairs.cpu(),
                distances.cpu(),
                hypers["mask"],
                generator,
            )
            u_smooth = zeros
            grad_smooth = torch.zeros(n_atoms, 3, 3, dtype=dtype, device=device)
            u_student = zeros
        masked = masked.to(device)

        # student view: displaced, rotated, pairs touching masked atoms removed
        rotation = rotations[index].to(device)
        keep = ~(masked[pairs[:, 0]] | masked[pairs[:, 1]])
        student_view = build_view(
            system,
            options,
            ((positions + u_student) @ rotation.T).requires_grad_(position_gradients),
            cell @ rotation.T,
            keep_pairs=keep,
        )
        student.append(student_view)
        student_inputs.append((positions + u_student, cell, keep))
        student_nl = student_view.get_neighbor_list(options)
        student_distances = student_nl.values.reshape(-1, 3).norm(dim=-1).detach()
        student_close = student_distances < cutoff

        # roles of the atoms (clean pairs within the cutoff)
        boundary = boundary_atoms(masked, pairs, distances, cutoff)
        context = ~masked & ~boundary
        reachable = torch.zeros_like(masked)
        reachable[pairs[close & ~masked[pairs[:, 1]], 0]] = True

        # teacher modes at centres: context atoms with enough neighbours, at least
        # two cutoffs apart
        n_neighbors = torch.bincount(pairs[close, 0], minlength=n_atoms)
        eligible = context & (n_neighbors >= MIN_CENTER_NEIGHBORS)
        centers = _select_centers(
            eligible.cpu(),
            positions.cpu(),
            cell.cpu(),
            system.pbc.cpu(),
            2 * cutoff,
            ho["max_centers"],
            generator,
        )
        amplitudes = ho["mode_amplitude"] * torch.randn(
            len(centers), 9, generator=generator, dtype=torch.float64
        )
        amplitudes = amplitudes.to(device=device, dtype=dtype)
        centers = centers.to(device)
        u_teacher = cap_norm(
            _teacher_displacement(
                positions, cell, system.pbc, centers, amplitudes, ho["mode_window"]
            ),
            max_norm,
        )

        # the main teacher view (the shared teacher pass) and the extra one
        if ho["teacher_pair"]:
            main, extra = u_teacher, -u_teacher
        else:
            main, extra = zeros, u_teacher
        teacher.append(
            build_view(
                system,
                options,
                (positions + main).requires_grad_(position_gradients),
                cell,
            )
        )
        extra_teacher.append(build_view(system, options, positions + extra, cell))

        observed_vectors = (
            clean_vectors[close]
            + u_student[pairs[close, 1]]
            - u_student[pairs[close, 0]]
        )
        collect(
            branch_deformed=torch.tensor([deformed], device=device),
            rotations=rotation.expand(n_atoms, 3, 3),
            u_student=u_student,
            grad_smooth=grad_smooth,
            u_smooth=u_smooth,
            pair_index=pairs[close] + offset,
            clean_vectors=clean_vectors[close],
            observed_vectors=observed_vectors,
            hidden_source=masked[pairs[close, 1]],
            boundary=boundary,
            context=context,
            centers=centers + offset,
            center_amplitudes=amplitudes,
            structure_index=torch.full(
                (n_atoms,), index, dtype=torch.long, device=device
            ),
            student_pairs=student_nl.samples.values[student_close, :2].long() + offset,
            student_distances=student_distances[student_close],
            masked=masked,
            latent_targets=(masked & reachable) | ~masked,
        )
        offset += n_atoms

    out = {name: torch.cat(values) for name, values in per_structure.items()}
    masked = out.pop("masked")
    latent_targets = out.pop("latent_targets")

    species = torch.cat([system.types for system in systems])
    atomic_types = torch.tensor(hypers["atomic_types"], device=device)
    matches = species[:, None] == atomic_types[None, :]
    if not bool(matches.any(dim=1).all()):
        missing = sorted(set(species.tolist()) - set(atomic_types.tolist()))
        raise ValueError(f"atomic types {missing} are not in `atomic_types`")

    extras = HighOrderExtras(
        branch_masked=~out["branch_deformed"],
        species=species,
        species_index=matches.long().argmax(dim=1),
        **out,
    )
    teacher_views = {"minus" if ho["teacher_pair"] else "plus": extra_teacher}
    view_extras: Dict[str, Any] = {"high_order": extras, "teacher_views": teacher_views}
    if ho["second_rotation"]:
        # drawn last: the other views do not depend on the option
        extra_rotations = random_rotations(
            len(systems), hypers["rotation_group"], dtype, generator
        ).to(device)
        second: List[System] = []
        rotations2 = []
        for index, (system, (x, cell, keep)) in enumerate(
            zip(systems, student_inputs, strict=True)
        ):
            rotation = extra_rotations[index] @ rotations[index].to(device)
            second.append(
                build_view(
                    system,
                    options,
                    (x @ rotation.T).requires_grad_(position_gradients),
                    cell @ rotation.T,
                    keep_pairs=keep,
                )
            )
            rotations2.append(rotation.expand(len(system), 3, 3))
        extras.rotations2 = torch.cat(rotations2)
        view_extras["student_views"] = {"rot2": second}
    return Views(
        student=student,
        teacher=teacher,
        rotations=extras.rotations,
        masked=masked,
        latent_targets=latent_targets,
        extras=view_extras,
    )
