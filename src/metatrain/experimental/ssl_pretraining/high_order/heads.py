"""Per-block heads on learned tokens, an invariant one-step context module, and block
normalization."""

from typing import Any, Dict, Tuple, Union

import torch

from ..backbones import HeadInputs
from .targets import gaussian_radial


def _mlp(d_in: int, d_hidden: int, d_out: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, d_hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(d_hidden, d_out),
    )


class BlockHead(torch.nn.Module):
    """One irrep block ``(n, 2l+1, n_out)``.

    Default: an MLP on the pooled feature (node tokens and the cutoff-weighted
    edge-token sum). ``use_edges=True``: Abbott et al.'s Eq. 8, with a nonlinear map per
    edge token before the neighbour sum.

    Accepts only ``HeadInputs`` (learned tokens and scalar cutoff factors), so no raw
    geometric vector can enter a head (the head rule).
    """

    def __init__(
        self,
        dims: Tuple[int, int],
        l: int,
        n_out: int,
        hidden: int,
        use_edges: bool = False,
        channel_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        self.l, self.n_out, self.use_edges = l, n_out, use_edges
        self.dn = max(1, int(round(channel_fraction * dims[0])))
        self.de = max(1, int(round(channel_fraction * dims[1])))
        width = (2 * l + 1) * n_out
        if use_edges:
            self.node_net = _mlp(self.dn, hidden, width)
            self.edge_net = _mlp(self.de, hidden, width)
        else:
            self.node_net = _mlp(self.dn + self.de, hidden, width)

    def forward(self, x: HeadInputs) -> torch.Tensor:
        if not isinstance(x, HeadInputs):
            raise TypeError("BlockHead takes HeadInputs (learned tokens) only")
        node, edge = x.node[:, : self.dn], x.edge[..., : self.de]
        weights = x.cutoff[..., None]
        if self.use_edges:
            out = self.node_net(node) + (self.edge_net(edge) * weights).sum(1)
        else:
            out = self.node_net(torch.cat([node, (edge * weights).sum(1)], dim=-1))
        return out.reshape(-1, 2 * self.l + 1, self.n_out)


class InvariantContext(torch.nn.Module):
    """One message-passing step whose edge inputs are distances only (rotation-
    invariant), used to give masked atoms context for the latent term."""

    def __init__(self, d_in: int, hidden: int, n_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.n_rbf, self.cutoff = n_rbf, cutoff
        self.message = _mlp(2 * d_in + n_rbf, hidden, d_in)
        self.update = _mlp(2 * d_in, hidden, d_in)

    def forward(
        self, features: torch.Tensor, pairs: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        rbf = gaussian_radial(distances, self.n_rbf, self.cutoff).to(features)
        inputs = torch.cat([features[pairs[:, 0]], features[pairs[:, 1]], rbf], dim=-1)
        agg = torch.zeros_like(features).index_add_(
            0, pairs[:, 0], self.message(inputs)
        )
        return self.update(torch.cat([features, agg], dim=-1))


class BlockNormalizer(torch.nn.Module):
    """Loss = mean squared error / EMA of the target variance (floor), per named block.

    The EMA variances (dynamic block names) are module extra state, so they are saved
    and restored with ``state_dict``. They are updated in training mode only; in eval
    mode the stored variance is used (the unstored batch variance for an unseen name).
    """

    def __init__(self, momentum: float = 0.99, floor: float = 1e-6) -> None:
        super().__init__()
        self.momentum, self.floor = momentum, floor
        self.var: Dict[str, float] = {}

    def get_extra_state(self) -> Dict[str, float]:
        return dict(self.var)

    def set_extra_state(self, state: Any) -> None:
        self.var = {str(k): float(v) for k, v in state.items()}

    def forward(
        self, name: str, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            v = float(target.var(unbiased=False)) if target.numel() > 1 else 1.0
            if self.training:
                old = self.var.get(name, v)
                self.var[name] = self.momentum * old + (1 - self.momentum) * v
                var = self.var[name]
            else:
                var = self.var.get(name, v)
        return ((pred - target) ** 2).mean() / max(var, self.floor)

    def scale(self, name: str) -> float:
        return max(self.var.get(name, 1.0), self.floor) ** 0.5


def batch_variance_floor(
    pred: torch.Tensor, gamma: Union[float, torch.Tensor]
) -> torch.Tensor:
    """``relu(gamma - std over the batch)`` per component, averaged; per batch, not per
    atom. ``gamma`` is a level shared by every component, or one level per component
    (shape ``pred.shape[1:]``)."""
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    return torch.relu(gamma - pred.std(dim=0)).mean()
