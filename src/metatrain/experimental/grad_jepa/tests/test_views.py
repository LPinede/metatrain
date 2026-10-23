import ase.io
import pytest
import torch
from metatomic.torch import ModelOutput, NeighborListOptions, System

from metatrain.experimental.grad_jepa.views import (
    bounded_noise,
    build_view,
    get_skin_neighbor_list_transform,
    random_rotations,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import CARBON_PATH, QM9_PATH, small_pet, to_systems


FEATURE = {"feature": ModelOutput(sample_kind="atom")}


def _features(model, systems):
    return model(systems, FEATURE)["feature"].block().values


@pytest.mark.parametrize("group", ["SO3", "O3"])
def test_random_rotations(group):
    generator = torch.Generator().manual_seed(0)
    q = random_rotations(256, group, torch.float64, generator)
    eye = torch.eye(3, dtype=torch.float64).expand(256, 3, 3)
    torch.testing.assert_close(q @ q.transpose(1, 2), eye)
    det = torch.linalg.det(q)
    if group == "SO3":
        assert torch.all(det > 0)
    else:
        assert torch.any(det > 0) and torch.any(det < 0)


def test_random_rotations_are_reproducible():
    first = random_rotations(4, "O3", torch.float64, torch.Generator().manual_seed(5))
    second = random_rotations(4, "O3", torch.float64, torch.Generator().manual_seed(5))
    torch.testing.assert_close(first, second)


def test_random_rotations_none():
    q = random_rotations(3, "none", torch.float32, torch.Generator())
    torch.testing.assert_close(q, torch.eye(3).expand(3, 3, 3))


def test_bounded_noise_norm():
    delta = bounded_noise(1000, 1.0, 0.25, torch.float64, torch.Generator())
    assert delta.norm(dim=1).max() <= 0.25 + 1e-12


def test_rotation_gradient_mapping():
    """For an O(3)-invariant function of the pairs, the gradient w.r.t. the rotated
    positions, mapped back with ``g' R``, equals the gradient in the original frame."""
    options = NeighborListOptions(cutoff=4.0, full_list=True, strict=False)
    frame = ase.io.read(CARBON_PATH, 0)
    (system,) = to_systems([frame], torch.float64)
    system = get_system_with_neighbor_lists(system, [options])
    rotation = random_rotations(1, "O3", torch.float64, torch.Generator())[0]

    def energy(view: System) -> torch.Tensor:
        return view.get_neighbor_list(options).values.norm(dim=1).pow(3).sum()

    positions = system.positions.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(
        energy(build_view(system, options, positions, system.cell)), [positions]
    )
    rotated = (system.positions @ rotation.T).requires_grad_(True)
    view = build_view(system, options, rotated, system.cell @ rotation.T)
    (grad_rotated,) = torch.autograd.grad(energy(view), [rotated])
    torch.testing.assert_close(grad_rotated @ rotation, grad)


@pytest.mark.parametrize("path", [QM9_PATH, CARBON_PATH])
def test_skin_neighbor_list_is_exact_under_bounded_noise(path):
    model = small_pet([1, 6, 7, 8], torch.float64).eval()
    options = model.requested_neighbor_lists()[0]
    frames = ase.io.read(path, ":3")
    transform = get_skin_neighbor_list_transform(options, skin=0.5)
    skin_systems, _, _ = transform(to_systems(frames, torch.float64), {}, {})
    for frame, skin_system in zip(frames, skin_systems, strict=True):
        delta = bounded_noise(len(frame), 0.2, 0.25, torch.float64, torch.Generator())
        displaced = build_view(
            skin_system, options, skin_system.positions + delta, skin_system.cell
        )
        (exact,) = to_systems([frame], torch.float64)
        exact = System(
            positions=exact.positions + delta,
            types=exact.types,
            cell=exact.cell,
            pbc=exact.pbc,
        )
        exact = get_system_with_neighbor_lists(exact, [options])
        torch.testing.assert_close(
            _features(model, [displaced]), _features(model, [exact])
        )


def test_skin_requires_non_strict_options():
    options = NeighborListOptions(cutoff=4.0, full_list=True, strict=True)
    with pytest.raises(ValueError, match="non-strict"):
        get_skin_neighbor_list_transform(options, skin=0.5)


def test_removed_pairs_isolate_atoms():
    model = small_pet([1, 6, 7, 8]).eval()
    options = model.requested_neighbor_lists()[0]
    (frame,) = ase.io.read(QM9_PATH, ":1")
    (system,) = to_systems([frame], torch.float32)
    system = get_system_with_neighbor_lists(system, [options])
    masked = torch.zeros(len(frame), dtype=torch.bool)
    masked[0] = True
    samples = system.get_neighbor_list(options).samples.values
    keep = ~(masked[samples[:, 0]] | masked[samples[:, 1]])
    positions = system.positions.clone().requires_grad_(True)
    view = build_view(system, options, positions, system.cell, keep_pairs=keep)
    (grad,) = torch.autograd.grad(_features(model, [view]).sum(), [positions])
    assert torch.all(grad[masked] == 0)
    assert torch.any(grad[~masked] != 0)
