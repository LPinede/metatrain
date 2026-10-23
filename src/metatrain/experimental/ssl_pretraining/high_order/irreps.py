"""Irrep toolkit: real spherical harmonics, Wigner matrices, Clebsch-Gordan, back-maps.

All conventions derive from ``real_spherical_harmonics``, so they agree by
construction: ``wigner_d(R, l)`` satisfies Y_l(R r) = D(R) Y_l(r) for proper and
improper R (for an improper R it carries (-1)^l, i.e. it is the proper-parity
representation). Pseudo blocks use ``irrep_matrix(R, l, -1) = det(R) * wigner_d(R, l)``.
"""

import math
from functools import lru_cache
from typing import Dict, List, Tuple

import torch


# Real l=1 harmonics are ordered (y, z, x): ``XYZ_TO_L1[i]`` is the xyz index of
# l=1 component i. The inverse permutation is ``argsort(XYZ_TO_L1)``.
XYZ_TO_L1 = (1, 2, 0)


def _legendre_table(l_max: int, x: torch.Tensor) -> Dict[Tuple[int, int], torch.Tensor]:
    """Associated Legendre P_l^m(x), 0 <= m <= l <= l_max, without Condon-Shortley."""
    p: Dict[Tuple[int, int], torch.Tensor] = {(0, 0): torch.ones_like(x)}
    s = torch.sqrt(torch.clamp(1 - x * x, min=0.0))
    for m in range(1, l_max + 1):
        p[(m, m)] = (2 * m - 1) * s * p[(m - 1, m - 1)]
    for m in range(0, l_max):
        p[(m + 1, m)] = (2 * m + 1) * x * p[(m, m)]
    for m in range(0, l_max + 1):
        for l in range(m + 2, l_max + 1):
            p[(l, m)] = (
                (2 * l - 1) * x * p[(l - 1, m)] - (l + m - 1) * p[(l - 2, m)]
            ) / (l - m)
    return p


def real_spherical_harmonics(l_max: int, vectors: torch.Tensor) -> List[torch.Tensor]:
    """Orthonormal real spherical harmonics of the directions of ``vectors`` (..., 3),
    one tensor (..., 2l+1) per l, ordered m = -l..l."""
    r = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    p = _legendre_table(l_max, z)
    phi = torch.atan2(y, x)
    out = []
    for l in range(l_max + 1):
        comps = []
        for m in range(-l, l + 1):
            am = abs(m)
            norm = math.sqrt(
                (2 * l + 1)
                / (4 * math.pi)
                * math.factorial(l - am)
                / math.factorial(l + am)
            )
            if m == 0:
                comps.append(norm * p[(l, 0)])
            elif m > 0:
                comps.append(math.sqrt(2) * norm * p[(l, am)] * torch.cos(am * phi))
            else:
                comps.append(math.sqrt(2) * norm * p[(l, am)] * torch.sin(am * phi))
        out.append(torch.stack(comps, dim=-1))
    return out


def _fit_points(l: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1234 + l)
    pts = torch.randn(max(4 * (2 * l + 1), 32), 3, generator=gen, dtype=torch.float64)
    return pts.to(device=device, dtype=dtype)


def wigner_d(rotations: torch.Tensor, l: int) -> torch.Tensor:
    """Real Wigner matrices (n, 2l+1, 2l+1), fitted by least squares from
    Y_l(R r_k) = D Y_l(r_k) on fixed points (exact up to round-off)."""
    pts = _fit_points(l, rotations.dtype, rotations.device)
    y = real_spherical_harmonics(l, pts)[l]
    y_rot = real_spherical_harmonics(l, torch.einsum("nab,kb->nka", rotations, pts))[l]
    pinv = torch.linalg.pinv(y)
    return torch.einsum("mk,nkp->npm", pinv, y_rot)


def irrep_matrix(rotations: torch.Tensor, l: int, sigma: int) -> torch.Tensor:
    """Representation matrix of irrep (l, sigma)."""
    d = wigner_d(rotations, l)
    if sigma == -1:
        d = d * torch.linalg.det(rotations).reshape(-1, 1, 1)
    return d


def back_map(values: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Student-frame block values (n, 2l+1, ...) -> teacher frame: D^T v."""
    return torch.einsum("nmp,nm...->np...", matrices, values)


@lru_cache(maxsize=None)
def real_clebsch_gordan(l1: int, l2: int, l: int) -> torch.Tensor:
    """Real coupling tensor C (2l1+1, 2l2+1, 2l+1), an isometry (C^T C = I) solving
    (D_l1 x D_l2) C = C D_l for proper rotations. Under improper rotations the product
    of two proper blocks picks up det(R)^(l1+l2+l): callers book-keep parity.

    The result is cached per ``(l1, l2, l)`` as a shared CPU float64 tensor: callers
    must ``.to()`` it and never modify it in place."""
    gen = torch.Generator().manual_seed(7 + 100 * l1 + 10 * l2 + l)
    rots = torch.linalg.qr(torch.randn(6, 3, 3, generator=gen, dtype=torch.float64))[0]
    rots = rots * torch.linalg.det(rots).sign().reshape(-1, 1, 1)
    d1, d2, d = (wigner_d(rots, k) for k in (l1, l2, l))
    n1, n2, n = 2 * l1 + 1, 2 * l2 + 1, 2 * l + 1
    eye_n = torch.eye(n, dtype=torch.float64)
    eye_12 = torch.eye(n1 * n2, dtype=torch.float64)
    rows = [
        torch.kron(torch.kron(d1[k], d2[k]), eye_n)
        - torch.kron(eye_12, d[k].T.contiguous())
        for k in range(rots.shape[0])
    ]
    _, _, vh = torch.linalg.svd(torch.cat(rows))
    c = vh[-1].reshape(n1 * n2, n)
    c = c / torch.linalg.norm(c) * math.sqrt(n)
    return c.reshape(n1, n2, n)
