import copy

import ase
import ase.io
import pytest
import torch
from metatomic.torch import NeighborListOptions

from metatrain.experimental.ssl_pretraining.documentation import TrainerHypers
from metatrain.experimental.ssl_pretraining.views import (
    get_skin_neighbor_list_transform,
    pair_vectors,
)
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import QM9_PATH, small_pet, to_systems


D = 12
N = 10


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    options = NeighborListOptions(cutoff=4.0, full_list=True, strict=False)
    chain = ase.Atoms(
        "C5H5", positions=[[1.2 * i, 0.3 * (i % 2), 0.0] for i in range(N)]
    )
    (system,) = to_systems([chain], torch.float64)
    system = get_system_with_neighbor_lists(system, [options])
    samples = system.get_neighbor_list(options).samples.values
    vectors = pair_vectors(system.positions, system.cell, samples)
    close = vectors.norm(dim=-1) < 4.0
    masked = torch.zeros(N, dtype=torch.bool)
    masked[3:5] = True
    gradients = torch.randn(2, N, 3, dtype=torch.float64)
    gradients[:, masked] = 0.0
    return {
        "features": torch.randn(N, D, dtype=torch.float64),
        "gradients": gradients,
        "probes": torch.randn(2, D, dtype=torch.float64),
        "displacements": 0.05 * torch.randn(N, 3, dtype=torch.float64),
        "types": system.types,
        "masked": masked,
        "pairs": samples[close, :2],
        "vectors": vectors[close],
    }


@pytest.fixture
def pet_core():
    """A small PET core (float32, training mode) over H, C, N and O."""
    torch.manual_seed(0)
    return small_pet([1, 6, 7, 8]).train()


@pytest.fixture
def small_teacher(pet_core):
    """A copy of ``pet_core`` set up as the EMA teacher (eval, no gradients)."""
    teacher = copy.deepcopy(pet_core).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@pytest.fixture
def small_batch(pet_core):
    """``(systems, options, hypers)``: 3 QM9 structures with the skin neighbor list
    of ``pet_core``, its neighbor list options and the default trainer hypers."""
    options = pet_core.requested_neighbor_lists()[0]
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    hypers = copy.deepcopy(init_with_defaults(TrainerHypers))
    transform = get_skin_neighbor_list_transform(options, hypers["noise"]["skin"])
    frames = ase.io.read(QM9_PATH, ":3")
    systems, _, _ = transform(to_systems(frames, torch.float32), {}, {})
    return systems, options, hypers
