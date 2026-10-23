"""Region sampling for masked views.

Both generators work on the (clean) pair list of a single system, restricted to
pairs closer than some radius, so periodic images and minimum-image conventions
are handled by the neighbor list itself. A "region" is a boolean mask over atoms.
"""

from typing import List

import torch


def nl_ball_region(
    n_atoms: int,
    pairs: torch.Tensor,
    distances: torch.Tensor,
    radius_range: List[float],
    fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Union of balls (center atom + its neighbors closer than ``R``) until at
    least ``fraction`` of the atoms are in the region.

    A ball is only added if at least one atom stays outside the region. If no
    ball can be added (e.g. a single-atom system), the region may be empty.

    :param n_atoms: Number of atoms.
    :param pairs: Pair atom indices, shape ``(n_pairs, 2)``.
    :param distances: Pair distances, shape ``(n_pairs,)``.
    :param radius_range: ``[R_min, R_max]``, ball radius drawn uniformly.
    :param fraction: Target fraction of atoms in the region.
    :param generator: CPU random generator.
    :return: Boolean region mask of shape ``(n_atoms,)``.
    """
    region = torch.zeros(n_atoms, dtype=torch.bool, device=pairs.device)
    target = max(1, int(round(fraction * n_atoms)))
    for _ in range(4 * n_atoms):  # bounded number of attempts
        if int(region.sum()) >= target:
            break
        outside = torch.nonzero(~region).squeeze(-1)
        center = int(
            outside[int(torch.randint(len(outside), (1,), generator=generator))]
        )
        r_min, r_max = radius_range
        radius = r_min + (r_max - r_min) * float(torch.rand(1, generator=generator))
        ball = torch.zeros_like(region)
        ball[center] = True
        selected = (pairs[:, 0] == center) & (distances < radius)
        ball[pairs[selected, 1]] = True
        candidate = region | ball
        if bool(candidate.all()):
            continue
        region = candidate
    return region


def k_hop_ego_region(
    n_atoms: int,
    pairs: torch.Tensor,
    distances: torch.Tensor,
    ego_hops: List[int],
    topology_cutoff: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """k-hop ego network around a random center on the bond-scale graph
    (pairs closer than ``topology_cutoff``), as in Atom-JEPA.

    ``k`` is drawn uniformly from ``ego_hops`` and reduced until at least one
    atom stays outside the region. Atoms unreachable from the center stay outside.

    :param n_atoms: Number of atoms.
    :param pairs: Pair atom indices, shape ``(n_pairs, 2)``.
    :param distances: Pair distances, shape ``(n_pairs,)``.
    :param ego_hops: Candidate hop counts.
    :param topology_cutoff: Bond-scale distance defining the topology graph.
    :param generator: CPU random generator.
    :return: Boolean region mask of shape ``(n_atoms,)``.
    """
    if n_atoms < 2:
        return torch.zeros(n_atoms, dtype=torch.bool, device=pairs.device)
    bonds = pairs[(distances < topology_cutoff) & (pairs[:, 0] != pairs[:, 1])]
    k = ego_hops[int(torch.randint(len(ego_hops), (1,), generator=generator))]
    center = int(torch.randint(n_atoms, (1,), generator=generator))
    frontier_sets = [torch.zeros(n_atoms, dtype=torch.bool, device=pairs.device)]
    frontier_sets[0][center] = True
    for _ in range(k):
        grown = frontier_sets[-1].clone()
        grown[bonds[frontier_sets[-1][bonds[:, 0]], 1]] = True
        frontier_sets.append(grown)
    # cap k so that the complement is non-empty
    for region in reversed(frontier_sets):
        if not bool(region.all()):
            return region
    raise AssertionError("unreachable: the 0-hop region has a single atom")


def boundary_atoms(
    masked: torch.Tensor, pairs: torch.Tensor, distances: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Unmasked atoms with at least one masked neighbor closer than ``cutoff``.

    :param masked: Boolean mask of masked atoms, shape ``(n_atoms,)``.
    :param pairs: Pair atom indices, shape ``(n_pairs, 2)``.
    :param distances: Pair distances, shape ``(n_pairs,)``.
    :param cutoff: Distance threshold.
    :return: Boolean mask of boundary atoms, shape ``(n_atoms,)``.
    """
    close = distances < cutoff
    touches = close & masked[pairs[:, 1]] & ~masked[pairs[:, 0]]
    boundary = torch.zeros_like(masked)
    boundary[pairs[touches, 0]] = True
    return boundary
