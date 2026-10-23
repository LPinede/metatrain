"""Geometry helpers that build the student and teacher views of a batch on-device.

All views share the atoms of the original system, so atom indices correspond
trivially between views. Views only differ by (1) a global rotation, (2) per-atom
displacements and (3) which neighbor-list pairs are kept (masking).
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors
from metatomic.torch.o3 import random_transformations

from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def get_skin_neighbor_list_transform(
    options: NeighborListOptions, skin: float
) -> Callable:
    """Collate transform computing a neighbor list with ``options.cutoff + skin``
    and registering it under ``options``.

    This is valid because ``options`` is non-strict: the list may contain pairs
    beyond the cutoff, which PET filters out. With per-atom displacements bounded
    by ``skin / 2``, every pair within the cutoff after displacement is present.

    :param options: The (non-strict) neighbor list options requested by PET.
    :param skin: Extra distance added to the cutoff.
    :return: A transform with the ``CollateFn`` callable signature.
    """
    if options.strict:
        raise ValueError("the skin neighbor list requires non-strict options")
    skin_options = NeighborListOptions(
        cutoff=options.cutoff + skin, full_list=options.full_list, strict=False
    )

    def transform(
        systems: List[System],
        targets: Dict[str, TensorBlock],
        extra: Dict[str, TensorBlock],
    ) -> Tuple[List[System], Dict[str, TensorBlock], Dict[str, TensorBlock]]:
        new_systems = []
        for system in systems:
            neighbors = get_system_with_neighbor_lists(
                system, [skin_options]
            ).get_neighbor_list(skin_options)
            new_system = System(
                positions=system.positions,
                types=system.types,
                cell=system.cell,
                pbc=system.pbc,
            )
            new_system.add_neighbor_list(options, neighbors)
            new_systems.append(new_system)
        return new_systems, targets, extra

    return transform


def pair_vectors(
    positions: torch.Tensor, cell: torch.Tensor, samples: torch.Tensor
) -> torch.Tensor:
    """Pair vectors ``r_j - r_i + S @ cell`` for neighbor-list samples.

    :param positions: Positions, shape ``(n_atoms, 3)``.
    :param cell: Cell matrix (rows are lattice vectors), shape ``(3, 3)``.
    :param samples: Neighbor-list sample values with columns
        ``(first_atom, second_atom, cell_shift_a, cell_shift_b, cell_shift_c)``.
    :return: Pair vectors, shape ``(n_pairs, 3)``.
    """
    shifts = samples[:, 2:5].to(positions.dtype)
    return positions[samples[:, 1]] - positions[samples[:, 0]] + shifts @ cell


def build_view(
    system: System,
    options: NeighborListOptions,
    positions: torch.Tensor,
    cell: torch.Tensor,
    keep_pairs: Optional[torch.Tensor] = None,
) -> System:
    """Build a new system with the given positions/cell, re-using (a subset of)
    the neighbor list of ``system``.

    Pair vectors are recomputed from ``positions`` and ``cell``, and registered
    for autograd, so gradients w.r.t. ``positions`` flow through the pairs.

    :param system: Source system, which must hold a neighbor list for ``options``.
    :param options: Neighbor list options to read from and write to.
    :param positions: New positions, shape ``(n_atoms, 3)``.
    :param cell: New cell, shape ``(3, 3)``.
    :param keep_pairs: Optional boolean mask over the pairs to keep.
    :return: The new system.
    """
    nl = system.get_neighbor_list(options)
    samples = nl.samples.values
    if keep_pairs is not None:
        samples = samples[keep_pairs]
    # detached: ``register_autograd_neighbors`` adds the dependency on positions
    vectors = pair_vectors(positions, cell, samples).detach()
    new_nl = TensorBlock(
        values=vectors.reshape(-1, 3, 1),
        samples=Labels(nl.samples.names, samples),
        components=nl.components,
        properties=nl.properties,
    )
    new_system = System(
        positions=positions, types=system.types, cell=cell, pbc=system.pbc
    )
    register_autograd_neighbors(new_system, new_nl)
    new_system.add_neighbor_list(options, new_nl)
    return new_system


def random_rotations(
    n: int, group: str, dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    """Sample ``n`` random orthogonal matrices with metatomic's sampler (the one
    behind metatrain's ``O3Augmenter``): Haar-distributed rotations from random unit
    quaternions, each negated with probability 1/2 for ``"O3"``.

    :param n: Number of matrices.
    :param group: ``"SO3"`` (proper rotations), ``"O3"`` (rotations and improper
        rotations) or ``"none"`` (identities).
    :param dtype: Output dtype.
    :param generator: CPU random generator.
    :return: Tensor of shape ``(n, 3, 3)``, on CPU.
    """
    if group == "none":
        return torch.eye(3, dtype=dtype).expand(n, 3, 3).clone()
    if group not in ("SO3", "O3"):
        raise ValueError(f"unknown rotation group '{group}'")
    transformations = random_transformations(
        n,
        device=torch.device("cpu"),
        dtype=dtype,
        add_inversions=group == "O3",
        generator=generator,
    )
    return transformations.matrices


def bounded_noise(
    n_atoms: int,
    sigma: float,
    max_norm: float,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Gaussian per-atom displacements, with each atom's norm clipped to
    ``max_norm`` (keeping the skin neighbor list exact).

    :param n_atoms: Number of atoms.
    :param sigma: Standard deviation per Cartesian component.
    :param max_norm: Maximum displacement norm per atom.
    :param dtype: Output dtype.
    :param generator: CPU random generator.
    :return: Tensor of shape ``(n_atoms, 3)``, on CPU.
    """
    delta = sigma * torch.randn(n_atoms, 3, dtype=torch.float64, generator=generator)
    norms = delta.norm(dim=1, keepdim=True).clamp(min=1e-12)
    delta = delta * torch.clamp(max_norm / norms, max=1.0)
    return delta.to(dtype)
