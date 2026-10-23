"""Displacement fields: smooth random fields (with exact gradients), windowed l=1
irreducible modes (dilation, rotation, shear), projection onto those modes, caps."""

import math
from typing import Tuple

import torch

from .irreps import XYZ_TO_L1, real_clebsch_gordan


def shear_matrix(shear: torch.Tensor) -> torch.Tensor:
    """(n, 5) real l=2 components -> (n, 3, 3) symmetric traceless (xyz basis)."""
    m_l1 = torch.einsum("ijk,nk->nij", real_clebsch_gordan(1, 1, 2).to(shear), shear)
    inv = torch.argsort(torch.tensor(XYZ_TO_L1))
    return m_l1[:, inv][:, :, inv]


def mode_displacements(
    positions, center, dilation, rotation, shear, window
) -> torch.Tensor:
    """u(r) = W(|r-c|) [(dilation/sqrt3)(r-c) + rotation x (r-c) + S(shear)(r-c)],
    W(d) = exp(-(d/window)^2), which varies within the cutoff so the local rotation is
    identifiable."""
    rel = positions - center
    w = torch.exp(-((rel.norm(dim=-1) / window) ** 2))[:, None]
    s = shear_matrix(shear[None])[0]
    lin = (
        dilation / math.sqrt(3.0) * rel
        + torch.cross(rotation.expand_as(rel), rel, dim=-1)
        + rel @ s.T
    )
    return w * lin


def _waves_to_field(
    positions: torch.Tensor, q: torch.Tensor, a: torch.Tensor, phase: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Field sum_w a_w cos(q_w . r + phase_w) with mean removed, and its gradient."""
    arg = positions @ q.T + phase
    u = torch.cos(arg) @ a
    u = u - u.mean(dim=0)
    grad = -torch.einsum("nw,wa,wb->nab", torch.sin(arg), a, q)
    return u, grad


def smooth_field_with_gradient(positions, amplitude, wavelength, n_waves, generator):
    """Random plane waves with |q| = 2 pi / wavelength; mean displacement removed.
    Returns u (N, 3) and the exact gradient G[i]_ab = du_a/dx_b at each atom.

    The random numbers come from ``generator`` (CPU) and are moved to the device and
    dtype of ``positions`` before any arithmetic."""
    kw = {"device": positions.device, "dtype": positions.dtype}
    q = torch.randn(n_waves, 3, generator=generator, dtype=torch.float64)
    q = (2 * math.pi / wavelength) * q / q.norm(dim=-1, keepdim=True)
    a = torch.randn(n_waves, 3, generator=generator, dtype=torch.float64)
    a = a * amplitude / math.sqrt(n_waves)
    phase = 2 * math.pi * torch.rand(n_waves, generator=generator, dtype=torch.float64)
    return _waves_to_field(positions, q.to(**kw), a.to(**kw), phase.to(**kw))


def periodic_field_with_gradient(positions, cell, amplitude, max_index, generator):
    """Periodic field: wave vectors on the reciprocal lattice, q = 2 pi (n . B), with
    integer n in [-max_index, max_index]^3 \\ {0} (one random n per wave, 6 waves).
    Mean removed (a periodic field cannot contain a global rotation).

    The random numbers come from ``generator`` (CPU) and are moved to the device and
    dtype of ``positions`` before any arithmetic."""
    kw = {"device": positions.device, "dtype": positions.dtype}
    # rows are b_i; the (tiny) inverse is computed on CPU in float64
    recip = 2 * math.pi * torch.linalg.inv(cell.detach().cpu().double()).T
    n = torch.randint(-max_index, max_index + 1, (6, 3), generator=generator).double()
    n[(n == 0).all(dim=1)] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    q = n @ recip
    a = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    a = a * amplitude / math.sqrt(6)
    phase = 2 * math.pi * torch.rand(6, generator=generator, dtype=torch.float64)
    return _waves_to_field(positions, q.to(**kw), a.to(**kw), phase.to(**kw))


def cap_norm(u: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Scale each displacement so that |u_i| <= max_norm (skin rule)."""
    norms = u.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return u * torch.clamp(max_norm / norms, max=1.0)


def project_l1_modes(positions, center, displacement, window):
    """Least-squares (dilation, rotation, shear) amplitudes of ``displacement`` on the
    windowed l=1 modes at ``center``."""
    eye = torch.eye(9, dtype=positions.dtype, device=positions.device)
    basis = [
        mode_displacements(
            positions, center, eye[k, 0], eye[k, 1:4], eye[k, 4:9], window
        ).reshape(-1)
        for k in range(9)
    ]
    coef = torch.linalg.lstsq(
        torch.stack(basis, dim=1), displacement.reshape(-1, 1)
    ).solution[:, 0]
    return coef[0], coef[1:4], coef[4:9]
