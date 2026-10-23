import torch


def haar_o3(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``n`` Haar-distributed O(3) matrices (QR of a Gaussian, with the sign fix)."""
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, dtype=dtype))
    return q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
