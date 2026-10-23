import copy

import ase
import ase.io
import numpy as np
import pytest
import torch

from metatrain.experimental.ssl_pretraining.tests import (
    CARBON_PATH,
    QM9_PATH,
    small_pet,
)

from . import ATOMIC_TYPES, make_batch, make_step


@pytest.fixture
def small_batch():
    """3 small QM9 molecules (5, 4 and 3 atoms), float32."""
    return make_batch(ase.io.read(QM9_PATH, ":3"), torch.float32)


@pytest.fixture
def tiny_batch():
    """Two dimers, float32."""
    frames = [
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ase.Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]),
    ]
    return make_batch(frames, torch.float32)


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
    return make_batch([molecule, crystal], torch.float64)


@pytest.fixture
def medium_batch():
    """5 QM9 molecules of 8 to 14 atoms with H, C, N and O, float32."""
    return make_batch(ase.io.read(QM9_PATH, "17:22"), torch.float32)


@pytest.fixture
def pet_core():
    """A small PET core (float32, training mode) over H, C, N and O."""
    torch.manual_seed(0)
    return small_pet(ATOMIC_TYPES).train()


@pytest.fixture
def small_teacher(pet_core):
    """A copy of ``pet_core`` set up as the EMA teacher (eval, no gradients)."""
    teacher = copy.deepcopy(pet_core).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@pytest.fixture
def step_data(medium_batch, pet_core, small_teacher):
    """High-order step data of ``medium_batch``, with both student branches."""
    return make_step(medium_batch, pet_core, small_teacher, seed=1)


@pytest.fixture
def step_data_masked(medium_batch, pet_core, small_teacher):
    """High-order step data of ``medium_batch``, mask-only branch."""
    return make_step(medium_batch, pet_core, small_teacher, seed=0, p_def=0.0)


@pytest.fixture
def step_data_deformed(medium_batch, pet_core, small_teacher):
    """High-order step data of ``medium_batch``, deform-only branch."""
    return make_step(medium_batch, pet_core, small_teacher, seed=0, p_def=1.0)


@pytest.fixture
def tiny_step_data(tiny_batch, pet_core, small_teacher):
    """High-order step data of two dimers (no boundary atom, no centre)."""
    return make_step(tiny_batch, pet_core, small_teacher, seed=0)
