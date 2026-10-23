import ase
import pytest
import torch
from metatomic.torch import NeighborListOptions

from metatrain.experimental.ssl_pretraining.views import pair_vectors
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import to_systems


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
