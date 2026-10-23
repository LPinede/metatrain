import pytest
import torch

from metatrain.experimental.ssl_pretraining.high_order.deformations import (
    cap_norm,
    mode_displacements,
    periodic_field_with_gradient,
    project_l1_modes,
    smooth_field_with_gradient,
)
from metatrain.experimental.ssl_pretraining.high_order.irreps import wigner_d


torch.manual_seed(2)


def _amps():
    return (
        torch.tensor(0.03, dtype=torch.float64),
        0.02 * torch.randn(3, dtype=torch.float64),
        0.02 * torch.randn(5, dtype=torch.float64),
    )


def test_mode_amplitudes_are_recovered():
    pos = 2.5 * torch.randn(30, 3, dtype=torch.float64)
    d, r, s = _amps()
    u = mode_displacements(pos, pos[0], d, r, s, 3.0)
    d2, r2, s2 = project_l1_modes(pos, pos[0], u, 3.0)
    assert torch.allclose(d2, d) and torch.allclose(r2, r) and torch.allclose(s2, s)


def test_mode_synthesis_equivariant_proper_and_improper():
    pos = 2.5 * torch.randn(30, 3, dtype=torch.float64)
    d, r, s = _amps()
    u = mode_displacements(pos, pos[0], d, r, s, 3.0)
    for sign in (1.0, -1.0):
        g = torch.linalg.qr(torch.randn(1, 3, 3, dtype=torch.float64))[0]
        g = sign * g * torch.linalg.det(g).sign().reshape(-1, 1, 1)
        det = torch.linalg.det(g)
        ug = mode_displacements(
            pos @ g[0].T,
            pos[0] @ g[0].T,
            d,
            det * (g[0] @ r),
            wigner_d(g, 2)[0] @ s,
            3.0,
        )
        assert torch.allclose(ug, u @ g[0].T, atol=1e-14)


def _gen():
    return torch.Generator().manual_seed(0)


def test_smooth_field_gradient_matches_finite_differences():
    pos = 2.0 * torch.randn(5, 3, dtype=torch.float64)
    u, grad = smooth_field_with_gradient(pos, 0.2, 8.0, 6, _gen())
    assert torch.allclose(u.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-14)
    h, num = 1e-6, torch.zeros_like(grad)
    for b in range(3):
        e = torch.zeros(3, dtype=torch.float64)
        e[b] = h
        up, _ = smooth_field_with_gradient(pos + e, 0.2, 8.0, 6, _gen())
        um, _ = smooth_field_with_gradient(pos - e, 0.2, 8.0, 6, _gen())
        num[:, :, b] = (up - um) / (2 * h)
    # shifting every atom also shifts the subtracted mean
    assert torch.allclose(num, grad - grad.mean(0), atol=1e-8)


def _cell():
    return torch.diag(torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64))


def test_periodic_field_is_periodic():
    cell = _cell()
    pos = torch.rand(10, 3, dtype=torch.float64) @ cell
    u, _ = periodic_field_with_gradient(pos, cell, 0.1, 2, _gen())
    u_shift, _ = periodic_field_with_gradient(pos + cell[0], cell, 0.1, 2, _gen())
    assert torch.allclose(u, u_shift, atol=1e-12)


def test_periodic_field_periodic_for_triclinic_cell():
    cell = torch.tensor(
        [[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, -1.0, 6.0]], dtype=torch.float64
    )
    pos = torch.rand(10, 3, dtype=torch.float64) @ cell
    u, _ = periodic_field_with_gradient(pos, cell, 0.1, 2, _gen())
    for k in range(3):
        u_k, _ = periodic_field_with_gradient(pos + cell[k], cell, 0.1, 2, _gen())
        assert torch.allclose(u, u_k, atol=1e-12)


def test_periodic_field_gradient_matches_finite_differences():
    cell = _cell()
    pos = torch.rand(5, 3, dtype=torch.float64) @ cell
    u, grad = periodic_field_with_gradient(pos, cell, 0.1, 2, _gen())
    assert torch.allclose(u.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-14)
    h, num = 1e-6, torch.zeros_like(grad)
    for b in range(3):
        e = torch.zeros(3, dtype=torch.float64)
        e[b] = h
        up, _ = periodic_field_with_gradient(pos + e, cell, 0.1, 2, _gen())
        um, _ = periodic_field_with_gradient(pos - e, cell, 0.1, 2, _gen())
        num[:, :, b] = (up - um) / (2 * h)
    # shifting every atom also shifts the subtracted mean
    assert torch.allclose(num, grad - grad.mean(0), atol=1e-8)


def _devices():
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fields_follow_input_device_and_dtype(device, dtype):
    if device.type == "mps" and dtype == torch.float64:
        pytest.skip("float64 is not supported on mps")
    cell = _cell().to(device=device, dtype=dtype)
    pos = (torch.rand(10, 3, dtype=torch.float64) @ _cell()).to(device, dtype)
    for u, grad in (
        smooth_field_with_gradient(pos, 0.2, 8.0, 6, _gen()),
        periodic_field_with_gradient(pos, cell, 0.1, 2, _gen()),
    ):
        assert u.device.type == device.type and grad.device.type == device.type
        assert u.dtype == dtype and grad.dtype == dtype
    # same random draw on every device: results agree with the CPU reference
    ref_u, _ = smooth_field_with_gradient(pos.cpu(), 0.2, 8.0, 6, _gen())
    u, _ = smooth_field_with_gradient(pos, 0.2, 8.0, 6, _gen())
    assert torch.allclose(u.cpu(), ref_u, atol=1e-4)


def test_cap_norm():
    u = torch.randn(20, 3, dtype=torch.float64)
    assert cap_norm(u, 0.1).norm(dim=-1).max() <= 0.1 + 1e-12
