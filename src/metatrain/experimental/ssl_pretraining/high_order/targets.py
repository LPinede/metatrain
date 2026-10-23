"""Analytic targets of the high-order track. Every function works in the frame of the
vectors it is given; callers pass teacher-frame vectors."""

import math
from typing import List, Tuple

import torch

from .irreps import XYZ_TO_L1, real_clebsch_gordan, real_spherical_harmonics


def smooth_cutoff(d: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cosine cutoff: 1 at d = 0, 0 for d >= cutoff."""
    return torch.where(
        d < cutoff, 0.5 * (torch.cos(math.pi * d / cutoff) + 1.0), torch.zeros_like(d)
    )


def gaussian_radial(d: torch.Tensor, n_radial: int, cutoff: float) -> torch.Tensor:
    """Gaussian radial basis (n_pairs, n_radial) times the smooth cutoff."""
    centers = torch.linspace(0.0, cutoff, n_radial, dtype=d.dtype, device=d.device)
    width = cutoff / n_radial
    radial = torch.exp(-(((d[:, None] - centers) / width) ** 2))
    return radial * smooth_cutoff(d, cutoff)[:, None]


def multipoles(
    centers, vectors, weights, n_centers, l_max, n_radial, cutoff
) -> List[torch.Tensor]:
    """M^l[c, n, ch, m] = sum over pairs of w[ch] R_n(|r|) Y_lm(r), l = 0..l_max."""
    d = vectors.norm(dim=-1)
    radial = gaussian_radial(d, n_radial, cutoff)
    ylm = real_spherical_harmonics(l_max, vectors)
    out = []
    for l in range(l_max + 1):
        contrib = torch.einsum("pn,pc,pm->pncm", radial, weights, ylm[l])
        acc = torch.zeros(
            n_centers,
            n_radial,
            weights.shape[1],
            2 * l + 1,
            dtype=vectors.dtype,
            device=vectors.device,
        )
        acc.index_add_(0, centers, contrib)
        out.append(acc)
    return out


def residual_multipoles(
    centers,
    clean_vectors,
    observed_vectors,
    hidden,
    weights,
    n_centers,
    l_max,
    n_radial,
    cutoff,
) -> List[torch.Tensor]:
    """R_i = sum_{k hidden} phi(r0_ik) + sum_{k visible} [phi(r0_ik) - phi(r_obs_ik)].
    Equals the clean full-shell multipole minus what the student observes."""
    full = multipoles(
        centers, clean_vectors, weights, n_centers, l_max, n_radial, cutoff
    )
    visible = ~hidden
    seen = multipoles(
        centers[visible],
        observed_vectors[visible],
        weights[visible],
        n_centers,
        l_max,
        n_radial,
        cutoff,
    )
    return [f - s for f, s in zip(full, seen, strict=True)]


def chirality(centers, vectors, channel_weights, n_centers, cutoff) -> torch.Tensor:
    """kappa_i = mu_a . (mu_b x mu_c), channel dipoles mu = sum_j w f(r) r_hat."""
    d = vectors.norm(dim=-1).clamp(min=1e-12)
    unit = vectors / d[:, None]
    f = smooth_cutoff(d, cutoff)
    mu = torch.zeros(n_centers, 3, 3, dtype=vectors.dtype, device=vectors.device)
    mu.index_add_(
        0, centers, (f[:, None] * channel_weights)[:, :, None] * unit[:, None, :]
    )
    return torch.einsum("ix,ix->i", mu[:, 0], torch.cross(mu[:, 1], mu[:, 2], dim=-1))


def decompose_gradient(
    grad_u: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """G_ab = du_a/dx_b -> dilation (n,1) [(0,+)], local rotation (n,3) [(1,-)],
    shear (n,5) [(2,+), real l=2 basis, isometric]."""
    eye = torch.eye(3, dtype=grad_u.dtype, device=grad_u.device)
    tr = grad_u.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
    dilation = tr / math.sqrt(3.0)
    anti = 0.5 * (grad_u - grad_u.transpose(-1, -2))
    rotation = torch.stack([anti[:, 2, 1], anti[:, 0, 2], anti[:, 1, 0]], dim=-1)
    sym = 0.5 * (grad_u + grad_u.transpose(-1, -2)) - tr[..., None] / 3.0 * eye
    perm = torch.tensor(XYZ_TO_L1, device=grad_u.device)
    sym_l1 = sym[:, perm][:, :, perm]
    shear = torch.einsum("nij,ijk->nk", sym_l1, real_clebsch_gordan(1, 1, 2).to(grad_u))
    return dilation, rotation, shear
