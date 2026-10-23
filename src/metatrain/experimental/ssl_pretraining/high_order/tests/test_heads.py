import pytest
import torch

from metatrain.experimental.ssl_pretraining.backbones import HeadInputs
from metatrain.experimental.ssl_pretraining.high_order.heads import (
    BlockHead,
    BlockNormalizer,
    InvariantContext,
    batch_variance_floor,
)
from metatrain.experimental.ssl_pretraining.high_order.irreps import (
    back_map,
    irrep_matrix,
)

from . import haar_o3


def _inputs(n=5, k=7, dn=16, de=12):
    cutoff = torch.rand(n, k)
    cutoff[:, -2:] = 0.0  # padded neighbour slots
    return HeadInputs(torch.randn(n, dn), torch.randn(n, k, de), cutoff)


def test_block_head_shape_and_rejects_raw_tensors():
    head = BlockHead((16, 12), 2, 3, 32)
    assert head(_inputs()).shape == (5, 5, 3)
    with pytest.raises(TypeError):
        head(torch.randn(5, 16))  # only HeadInputs (learned tokens) are accepted


@pytest.mark.parametrize("use_edges", [False, True])
def test_padded_edges_contribute_nothing(use_edges):
    head = BlockHead((16, 12), 1, 2, 32, use_edges=use_edges)
    x = _inputs()
    y = HeadInputs(x.node, x.edge.clone(), x.cutoff)
    y.edge[:, -2:] = 100 * torch.randn_like(y.edge[:, -2:])
    assert torch.allclose(head(x), head(y))


def test_edge_head_is_nonlinear_per_edge():
    """Abbott's form is not a function of the pooled edge sum alone: two neighbour
    sets with the same cutoff-weighted sum give different outputs."""
    torch.manual_seed(0)
    head = BlockHead((4, 3), 1, 1, 16, use_edges=True)
    node = torch.zeros(1, 4)
    a = HeadInputs(node, torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0]]]), torch.ones(1, 2))
    b = HeadInputs(node, torch.zeros(1, 2, 3), torch.ones(1, 2))
    assert not torch.allclose(head(a), head(b))
    pooled = BlockHead((4, 3), 1, 1, 16)  # default: pooled feature
    assert torch.allclose(pooled(a), pooled(b))  # sees only the sum


@pytest.mark.parametrize("use_edges", [False, True])
def test_channel_fraction_ignores_other_channels(use_edges):
    head = BlockHead((16, 12), 0, 1, 32, use_edges=use_edges, channel_fraction=0.5)
    x = _inputs()
    y = HeadInputs(x.node.clone(), x.edge.clone(), x.cutoff)
    y.node[:, 8:] += 5.0
    y.edge[..., 6:] += 5.0
    assert torch.allclose(head(x), head(y))


def test_math_invariant_output_pays_full_target_energy():
    """Anchor inequality (math only, no head): an output Phi independent of the student
    rotation pays E||D^T Phi - tau||^2 = ||Phi||^2 + ||tau||^2, since E[D] = 0; so the
    best such output is 0 and pays ||tau||^2."""
    torch.manual_seed(0)
    n = 4000
    tau = torch.randn(5, dtype=torch.float64)
    phi = torch.randn(5, dtype=torch.float64)
    d = irrep_matrix(haar_o3(n), 2, 1)
    assert d.mean(0).abs().max() < 0.05
    phi_n = phi.expand(n, 5)[..., None].contiguous()
    loss = ((back_map(phi_n, d)[..., 0] - tau) ** 2).sum(-1).mean()
    expected = (phi**2).sum() + (tau**2).sum()
    assert torch.allclose(loss, expected, rtol=0.05)
    assert loss > (tau**2).sum()


def _pairs_distances(positions):
    pairs = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]])
    vec = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    return pairs, vec.norm(dim=-1)


def test_invariant_context_depends_on_distances_only():
    torch.manual_seed(0)
    ctx = InvariantContext(8, 16, 8, 4.0)
    feats = torch.randn(4, 8)
    positions = torch.randn(4, 3)
    pairs, dist = _pairs_distances(positions)
    out = ctx(feats, pairs, dist)
    assert out.shape == (4, 8)

    rot = torch.linalg.qr(torch.randn(3, 3))[0]
    pairs_r, dist_r = _pairs_distances(positions @ rot.T)
    assert torch.allclose(out, ctx(feats, pairs_r, dist_r), atol=1e-5)

    moved = positions.clone()
    moved[3] += torch.tensor([0.5, 0.0, 0.0])  # changes the 2-3 distance
    pairs_m, dist_m = _pairs_distances(moved)
    assert not torch.allclose(out, ctx(feats, pairs_m, dist_m), atol=1e-5)


def test_normalizer_scale_free():
    norm = BlockNormalizer(momentum=0.0)
    t = torch.randn(100, 5, 2)
    p = t + 0.1 * torch.randn_like(t)
    a = norm("x", p, t)
    b = BlockNormalizer(momentum=0.0)("x", 10 * p, 10 * t)
    assert torch.allclose(a, b, rtol=1e-5)
    assert abs(norm.scale("x") - float(t.std(unbiased=False))) < 1e-4
    assert BlockNormalizer().scale("unseen") == 1.0


def test_normalizer_state_round_trips_through_state_dict():
    torch.manual_seed(0)
    norm = BlockNormalizer(momentum=0.5)
    norm("a", torch.randn(10, 3), 2 * torch.randn(10, 3))
    norm("b", torch.randn(10, 3), 3 * torch.randn(10, 3))
    state = norm.state_dict()
    fresh = BlockNormalizer(momentum=0.5)
    fresh.load_state_dict(state)
    assert fresh.var == norm.var
    assert fresh.scale("a") == norm.scale("a")
    # a name unseen at load time still works afterwards
    fresh("c", torch.randn(10, 3), torch.randn(10, 3))
    assert "c" in fresh.var


def test_batch_floor_zero_when_spread():
    pred = torch.randn(64, 3, 4)
    assert batch_variance_floor(pred, 0.1) == 0
    assert batch_variance_floor(torch.zeros(64, 3, 4), 0.1) > 0


def test_normalizer_eval_mode_does_not_update():
    norm = BlockNormalizer(momentum=0.5)
    norm("a", torch.randn(10, 3), torch.randn(10, 3))
    before = norm.state_dict()["_extra_state"]
    norm.eval()
    t = 5 * torch.randn(10, 3)
    norm("a", torch.randn(10, 3), t)
    norm("new", torch.randn(10, 3), t)  # unseen name: batch variance, not stored
    assert norm.state_dict()["_extra_state"] == before
    assert "new" not in norm.var
