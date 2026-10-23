import pytest
import torch

from metatrain.experimental.ssl_pretraining.high_order.irreps import (
    back_map,
    irrep_matrix,
    real_clebsch_gordan,
    real_spherical_harmonics,
    wigner_d,
)


torch.manual_seed(0)


def _o3(n: int, improper: bool = False) -> torch.Tensor:
    q = torch.linalg.qr(torch.randn(n, 3, 3, dtype=torch.float64))[0]
    q = q * torch.linalg.det(q).sign().reshape(-1, 1, 1)
    return -q if improper else q


def _eye(n: int) -> torch.Tensor:
    return torch.eye(n, dtype=torch.float64)


@pytest.mark.parametrize("improper", [False, True])
@pytest.mark.parametrize("l", range(9))
def test_harmonics_transform_with_wigner(l, improper):
    rot = _o3(6, improper)
    v = torch.randn(40, 3, dtype=torch.float64)
    y = real_spherical_harmonics(l, v)[l]
    y_rot = real_spherical_harmonics(l, torch.einsum("nab,kb->nka", rot, v))[l]
    d = wigner_d(rot, l)
    assert torch.allclose(y_rot, torch.einsum("nmp,kp->nkm", d, y), atol=1e-10)
    assert torch.allclose(d @ d.transpose(1, 2), _eye(2 * l + 1), atol=1e-10)


def test_homomorphism():
    a, b = _o3(4), _o3(4, improper=True)
    for l in range(6):
        assert torch.allclose(
            wigner_d(a @ b, l), wigner_d(a, l) @ wigner_d(b, l), atol=1e-10
        )


def test_inversion_signs_and_pseudo():
    inv = -torch.eye(3, dtype=torch.float64)[None]
    for l in range(5):
        assert torch.allclose(wigner_d(inv, l)[0], (-1) ** l * _eye(2 * l + 1))
        assert torch.allclose(
            irrep_matrix(inv, l, -1)[0], (-1) ** (l + 1) * _eye(2 * l + 1)
        )


def test_harmonics_orthonormal():
    pts = torch.randn(400_000, 3, dtype=torch.float64)
    y = torch.cat(real_spherical_harmonics(3, pts), dim=-1)
    gram = 4 * torch.pi * (y.T @ y) / len(pts)
    assert torch.allclose(gram, _eye(16), atol=2e-2)


@pytest.mark.parametrize(
    "l1,l2,l", [(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 4), (1, 3, 3)]
)
def test_clebsch_gordan_equivariant_isometry(l1, l2, l):
    c = real_clebsch_gordan(l1, l2, l)
    assert torch.allclose(
        torch.einsum("ijk,ijq->kq", c, c), _eye(2 * l + 1), atol=1e-10
    )
    rot = torch.cat([_o3(3), _o3(3, improper=True)])
    a = torch.randn(2 * l1 + 1, dtype=torch.float64)
    b = torch.randn(2 * l2 + 1, dtype=torch.float64)
    lhs = torch.einsum("ni,nj,ijk->nk", wigner_d(rot, l1) @ a, wigner_d(rot, l2) @ b, c)
    det = torch.linalg.det(rot) ** (l1 + l2 + l)
    rhs = det[:, None] * (wigner_d(rot, l) @ torch.einsum("i,j,ijk->k", a, b, c))
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_back_map_inverts_forward():
    rot = _o3(5, improper=True)
    v = torch.randn(5, 5, 7, dtype=torch.float64)
    d = irrep_matrix(rot, 2, 1)
    assert torch.allclose(
        back_map(torch.einsum("nmp,np...->nm...", d, v), d), v, atol=1e-12
    )


def test_clebsch_gordan_is_cached():
    assert real_clebsch_gordan(1, 1, 2) is real_clebsch_gordan(1, 1, 2)
