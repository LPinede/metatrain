import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining.backbones import BACKBONES
from metatrain.experimental.ssl_pretraining.high_order.monitors import (
    FlatteningMonitor,
    _small_rotations,
    finite_rotation_invariance,
    gradient_cosines,
)
from metatrain.experimental.ssl_pretraining.tests import QM9_PATH

from . import make_batch


def test_flattening_detects_rho_collapse_with_falling_loss():
    m = FlatteningMonitor(window=3, factor=3.0)
    flags = [m.update(rho, loss) for rho, loss in [(1.0, 1.0), (0.6, 0.8), (0.2, 0.5)]]
    assert flags == [False, False, True]


def test_flattening_quiet_when_rho_stable():
    m = FlatteningMonitor(window=3, factor=3.0)
    assert not any(m.update(1.0, 1.0 - 0.1 * k) for k in range(5))


def test_flattening_quiet_when_the_loss_rises_or_the_fall_is_small():
    rising = FlatteningMonitor(window=3, factor=3.0)
    assert not any(rising.update(r, loss) for r, loss in [(1, 1), (0.5, 1.1), (0.1, 2)])
    mild = FlatteningMonitor(window=3, factor=3.0)
    assert not any(mild.update(r, loss) for r, loss in [(1, 1), (0.7, 0.8), (0.4, 0.5)])


def test_flattening_uses_a_sliding_window():
    m = FlatteningMonitor(window=2, factor=2.0)
    flags = [
        m.update(rho, 1.0 / (k + 1)) for k, rho in enumerate([1.0, 0.4, 0.39, 0.1])
    ]
    # (1.0, 0.4) trips; (0.4, 0.39) does not; (0.39, 0.1) does
    assert flags == [False, True, False, True]


def test_gradient_cosines_identical_losses_are_one():
    p = torch.nn.Parameter(torch.randn(10))
    loss = (p**2).sum()
    cos = gradient_cosines({"a": loss, "b": 2 * loss}, [p])
    assert set(cos) == {"a__b"}
    assert abs(cos["a__b"] - 1.0) < 1e-6


def test_gradient_cosines_opposite_orthogonal_and_independent():
    p = torch.nn.Parameter(torch.randn(6))
    q = torch.nn.Parameter(torch.randn(6))
    losses = {
        "c": (p**2).sum(),
        "a": (p * p.detach()).sum(),
        "b": -(p * p.detach()).sum(),
        "d": (q**2).sum(),  # does not depend on p
        "e": torch.zeros(()),  # no graph at all
    }
    cos = gradient_cosines(losses, [p])
    assert set(cos) == {f"{a}__{b}" for a in "abcd" for b in "bcde" if a < b}
    assert abs(cos["a__b"] + 1.0) < 1e-6
    assert abs(cos["a__c"] - 1.0) < 1e-6
    assert cos["a__d"] == 0.0 and cos["d__e"] == 0.0


def test_gradient_cosines_keep_the_graph():
    p = torch.nn.Parameter(torch.randn(4))
    loss = (p**2).sum()
    gradient_cosines({"a": loss, "b": loss * 3}, [p])
    loss.backward()
    assert p.grad is not None


class _ToyCore(torch.nn.Module):
    """Per-atom features from the neighbor-list vectors: distances only
    (``invariant``) or also the squared z component of the vectors."""

    def __init__(self, options, invariant: bool) -> None:
        super().__init__()
        self.options, self.invariant = options, invariant
        self.weight = torch.nn.Parameter(torch.ones(()))


class _ToyBackbone:
    name = "toy"

    def features(self, core, systems):
        out = []
        for system in systems:
            nl = system.get_neighbor_list(core.options)
            vectors = nl.values.reshape(-1, 3)
            first = nl.samples.values[:, 0].long()
            d = vectors.norm(dim=-1)
            columns = [torch.exp(-d), d**2]
            if not core.invariant:
                columns.append(vectors[:, 2] ** 2)
            values = torch.stack(columns, dim=-1) * core.weight
            out.append(
                torch.zeros(len(system), len(columns), dtype=d.dtype).index_add(
                    0, first, values
                )
            )
        return torch.cat(out)


@pytest.fixture
def batch64():
    return make_batch(ase.io.read(QM9_PATH, ":3"), torch.float64)


def test_finite_rotation_invariance_small_for_exactly_invariant_model(batch64):
    systems, options, _ = batch64
    core = _ToyCore(options, invariant=True)
    value = finite_rotation_invariance(_ToyBackbone(), core, systems, options)
    assert value < 1e-10


def test_finite_rotation_invariance_detects_orientation_dependence(batch64):
    systems, options, _ = batch64
    core = _ToyCore(options, invariant=False)
    backbone = _ToyBackbone()
    value = finite_rotation_invariance(backbone, core, systems, options)
    assert value > 1e-3
    # the change scales with the angle, and vanishes without rotation
    smaller = finite_rotation_invariance(backbone, core, systems, options, 1.0)
    assert smaller < value
    assert finite_rotation_invariance(backbone, core, systems, options, 0.0) < 1e-10


def test_finite_rotation_invariance_restores_the_mode(batch64):
    systems, options, _ = batch64
    core = _ToyCore(options, invariant=True)
    for mode in (True, False):
        core.train(mode)
        finite_rotation_invariance(_ToyBackbone(), core, systems, options)
        assert core.training is mode


def test_finite_rotation_invariance_of_pet(small_batch, pet_core):
    systems, options, _ = small_batch
    backbone = BACKBONES["pet"]
    value = finite_rotation_invariance(backbone, pet_core, systems, options)
    assert 0.0 < value < 1.0 and pet_core.training
    again = finite_rotation_invariance(backbone, pet_core, systems, options)
    assert value == again
    assert finite_rotation_invariance(backbone, pet_core, systems, options, 0.0) == 0.0
    # forward only: no graph, no gradient
    assert all(p.grad is None for p in pet_core.parameters())


def test_small_rotations_are_rotations_by_the_angle():
    angle = 0.3
    r = _small_rotations(16, angle, torch.float64, torch.Generator().manual_seed(0))
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(r @ r.transpose(1, 2), eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(r), torch.ones(16, dtype=torch.float64))
    trace = r.diagonal(dim1=1, dim2=2).sum(-1)
    assert torch.allclose(
        trace, torch.full((16,), 1 + 2 * torch.cos(torch.tensor(angle))).double()
    )
    assert not torch.allclose(r[0], r[1])
