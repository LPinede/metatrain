import math

import torch

from metatrain.experimental.ssl_pretraining.high_order.irreps import wigner_d
from metatrain.experimental.ssl_pretraining.high_order.targets import (
    chirality,
    decompose_gradient,
    multipoles,
    residual_multipoles,
)


torch.manual_seed(1)
TET = torch.tensor(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float64
) / math.sqrt(3)


def _o3(improper: bool) -> torch.Tensor:
    q = torch.linalg.qr(torch.randn(1, 3, 3, dtype=torch.float64))[0]
    q = q * torch.linalg.det(q).sign().reshape(-1, 1, 1)
    return -q if improper else q


def _env():
    vec = 1.5 * torch.randn(8, 3, dtype=torch.float64)
    w = torch.rand(8, 2, dtype=torch.float64)
    centers = torch.zeros(8, dtype=torch.long)
    u = 0.05 * torch.randn(8, 3, dtype=torch.float64)
    hidden = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    return vec, w, centers, u, hidden


def test_residual_multipole_equivariant_including_improper():
    vec, w, c, u, hidden = _env()
    ref = residual_multipoles(c, vec, vec + u, hidden, w, 1, 6, 4, 4.0)
    for improper in (False, True):
        g = _o3(improper)[0]
        rot = residual_multipoles(
            c, vec @ g.T, (vec + u) @ g.T, hidden, w, 1, 6, 4, 4.0
        )
        for l in range(7):
            d = wigner_d(g[None], l)[0]
            assert torch.allclose(
                rot[l], torch.einsum("mp,abcp->abcm", d, ref[l]), atol=1e-12
            )


def test_residual_reduces_to_hole_without_displacement():
    vec, w, c, _, hidden = _env()
    res = residual_multipoles(c, vec, vec, hidden, w, 1, 4, 4, 4.0)
    hole = multipoles(c[hidden], vec[hidden], w[hidden], 1, 4, 4, 4.0)
    assert all(torch.allclose(a, b, atol=1e-14) for a, b in zip(res, hole, strict=True))


def test_residual_ignores_hidden_atom_displacement():
    vec, w, c, u, hidden = _env()
    moved = vec.clone()
    moved[0] += 0.3  # observed position of a hidden atom is never used
    a = residual_multipoles(c, vec, vec + u, hidden, w, 1, 4, 4, 4.0)
    b = residual_multipoles(c, vec, moved + u, hidden, w, 1, 4, 4, 4.0)
    assert all(torch.allclose(x, y) for x, y in zip(a, b, strict=True))


def test_chirality_flips_under_mirror_and_vanishes_when_achiral():
    pos = TET * torch.tensor([1.09, 1.94, 1.77, 1.35], dtype=torch.float64)[:, None]
    cw = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    c = torch.zeros(4, dtype=torch.long)
    k = chirality(c, pos, cw, 1, 4.0)
    assert abs(k.item()) > 1e-2
    mirror = torch.tensor([1, 1, -1.0], dtype=torch.float64)
    assert torch.allclose(chirality(c, pos * mirror, cw, 1, 4.0), -k)
    assert torch.allclose(chirality(c, pos @ _o3(False)[0].T, cw, 1, 4.0), k)
    pos2 = TET * torch.tensor([1.09, 1.94, 1.94, 1.77], dtype=torch.float64)[:, None]
    cw2 = torch.tensor(
        [[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float64
    )
    assert abs(chirality(c, pos2, cw2, 1, 4.0).item()) < 1e-12


def test_gradient_decomposition_transformation_laws():
    grad = torch.randn(5, 3, 3, dtype=torch.float64)
    for improper in (False, True):
        g = _o3(improper)
        det = torch.linalg.det(g)
        d0, r0, s0 = decompose_gradient(grad)
        d1, r1, s1 = decompose_gradient(g @ grad @ g.transpose(-1, -2))
        assert torch.allclose(d1, d0, atol=1e-12)
        assert torch.allclose(r1, det * torch.einsum("ab,nb->na", g[0], r0), atol=1e-12)
        assert torch.allclose(
            s1, torch.einsum("mp,np->nm", wigner_d(g, 2)[0], s0), atol=1e-12
        )


def test_shear_is_isometric():
    grad = torch.randn(5, 3, 3, dtype=torch.float64)
    _, _, shear = decompose_gradient(grad)
    sym = 0.5 * (grad + grad.transpose(-1, -2))
    trace = sym.diagonal(dim1=-2, dim2=-1).sum(-1)
    sym = sym - trace[:, None, None] / 3 * torch.eye(3, dtype=torch.float64)
    assert torch.allclose(shear.pow(2).sum(-1), sym.pow(2).sum((-1, -2)), atol=1e-12)
