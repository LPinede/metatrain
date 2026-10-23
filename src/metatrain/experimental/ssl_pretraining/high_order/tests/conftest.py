import copy
from typing import List

import ase
import ase.io
import numpy as np
import pytest
import torch
from metatomic.torch import NeighborListOptions

from metatrain.experimental.ssl_pretraining.documentation import TrainerHypers
from metatrain.experimental.ssl_pretraining.tests import (
    CARBON_PATH,
    QM9_PATH,
    to_systems,
)
from metatrain.experimental.ssl_pretraining.views import (
    get_skin_neighbor_list_transform,
)
from metatrain.utils.hypers import init_with_defaults


ATOMIC_TYPES = [1, 6, 7, 8]
# the neighbor list options of the small PET of ``ssl_pretraining/tests``
OPTIONS = NeighborListOptions(cutoff=4.0, full_list=True, strict=False)


def _batch(frames: List[ase.Atoms], dtype: torch.dtype):
    """``(systems, options, hypers)``: the frames with the skin neighbor list, the
    neighbor list options and the default trainer hypers (with the runtime
    ``atomic_types`` key that the trainer adds)."""
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    hypers = copy.deepcopy(init_with_defaults(TrainerHypers))
    hypers["atomic_types"] = list(ATOMIC_TYPES)
    transform = get_skin_neighbor_list_transform(OPTIONS, hypers["noise"]["skin"])
    systems, _, _ = transform(to_systems(frames, dtype), {}, {})
    return systems, OPTIONS, hypers


@pytest.fixture
def small_batch():
    """3 small QM9 molecules (5, 4 and 3 atoms), float32."""
    return _batch(ase.io.read(QM9_PATH, ":3"), torch.float32)


@pytest.fixture
def tiny_batch():
    """Two dimers, float32."""
    frames = [
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ase.Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]),
    ]
    return _batch(frames, torch.float32)


@pytest.fixture
def large_batch():
    """Structures large enough to have context atoms (at least one cutoff away from
    every masked atom) and several centres, float64: a jittered 7x4x4 grid of carbon
    and hydrogen atoms (a 9 Å long molecule) and a periodic 3x3x1 carbon supercell."""
    rng = np.random.default_rng(0)
    grid = np.stack(
        np.meshgrid(np.arange(7), np.arange(4), np.arange(4), indexing="ij"), -1
    ).reshape(-1, 3)
    positions = 1.5 * grid + 0.1 * rng.standard_normal(grid.shape)
    symbols = ["C" if i % 3 else "H" for i in range(len(grid))]
    molecule = ase.Atoms(symbols, positions=positions)
    crystal = ase.io.read(CARBON_PATH, 0).repeat((3, 3, 1))
    return _batch([molecule, crystal], torch.float64)
