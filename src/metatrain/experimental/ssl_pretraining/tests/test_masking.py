import ase
import ase.io
import pytest
import torch
from metatomic.torch import NeighborListOptions

from metatrain.experimental.ssl_pretraining.masking import (
    boundary_atoms,
    k_hop_ego_region,
    nl_ball_region,
)
from metatrain.experimental.ssl_pretraining.views import pair_vectors
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import CARBON_PATH, to_systems


OPTIONS = NeighborListOptions(cutoff=4.0, full_list=True, strict=False)


def _pairs(frame: ase.Atoms):
    (system,) = to_systems([frame], torch.float64)
    system = get_system_with_neighbor_lists(system, [OPTIONS])
    samples = system.get_neighbor_list(OPTIONS).samples.values
    distances = pair_vectors(system.positions, system.cell, samples).norm(dim=-1)
    return samples[:, :2], distances


CHAIN = ase.Atoms("C10", positions=[[1.5 * i, 0.0, 0.0] for i in range(10)])


@pytest.mark.parametrize("seed", range(10))
def test_k_hop_ego_on_chain(seed):
    pairs, distances = _pairs(CHAIN)
    generator = torch.Generator().manual_seed(seed)
    region = k_hop_ego_region(10, pairs, distances, [1], 1.8, generator)
    indices = torch.nonzero(region).squeeze(-1)
    # one hop on a chain: the center and its 1 or 2 bonded neighbors, contiguous
    assert len(indices) in (2, 3)
    assert int(indices.max() - indices.min()) == len(indices) - 1


def test_k_hop_ego_is_capped():
    pairs, distances = _pairs(CHAIN)
    region = k_hop_ego_region(10, pairs, distances, [50], 1.8, torch.Generator())
    assert 0 < int(region.sum()) < 10


def test_k_hop_ego_single_atom():
    pairs, distances = _pairs(ase.Atoms("C"))
    region = k_hop_ego_region(1, pairs, distances, [1], 1.8, torch.Generator())
    assert not region.any()


@pytest.mark.parametrize("seed", range(10))
def test_nl_ball_region(seed):
    pairs, distances = _pairs(CHAIN)
    generator = torch.Generator().manual_seed(seed)
    region = nl_ball_region(10, pairs, distances, [1.0, 3.2], 0.3, generator)
    assert 3 <= int(region.sum()) < 10


def test_nl_ball_single_atom():
    pairs, distances = _pairs(ase.Atoms("C"))
    region = nl_ball_region(1, pairs, distances, [1.0, 3.0], 0.5, torch.Generator())
    assert not region.any()


def test_boundary_atoms_on_chain():
    pairs, distances = _pairs(CHAIN)
    masked = torch.zeros(10, dtype=torch.bool)
    masked[4] = True
    boundary = boundary_atoms(masked, pairs, distances, cutoff=1.8)
    assert torch.nonzero(boundary).squeeze(-1).tolist() == [3, 5]


@pytest.mark.parametrize("seed", range(5))
def test_periodic_small_cell(seed):
    """4-atom graphite cell: atoms are their own neighbors through images."""
    frame = ase.io.read(CARBON_PATH, 0)
    pairs, distances = _pairs(frame)
    assert torch.any(pairs[:, 0] == pairs[:, 1])
    generator = torch.Generator().manual_seed(seed)
    for region in [
        nl_ball_region(4, pairs, distances, [1.0, 2.0], 0.3, generator),
        k_hop_ego_region(4, pairs, distances, [1, 2], 1.8, generator),
    ]:
        assert 0 < int(region.sum()) < 4
        boundary = boundary_atoms(region, pairs, distances, cutoff=4.0)
        assert not torch.any(boundary & region)
