import copy
import math

import pytest
import torch

from metatrain.experimental.ssl_pretraining.documentation import TermsHypers
from metatrain.experimental.ssl_pretraining.high_order import documentation
from metatrain.experimental.ssl_pretraining.high_order.terms import (
    ActionResponseTerm,
    PositionalSmoothnessTerm,
    RotationalSmoothnessTerm,
    RotationConditionedLatentTerm,
)
from metatrain.experimental.ssl_pretraining.high_order.views import (
    build_high_order_views,
    small_rotations,
)
from metatrain.experimental.ssl_pretraining.terms import TERMS
from metatrain.experimental.ssl_pretraining.trainer import check_hypers
from metatrain.utils.hypers import init_with_defaults

from . import make_step


def _views(batch, seed=0, **over):
    systems, options, hypers = batch
    hypers = copy.deepcopy(hypers)
    hypers["high_order"].update(over)
    return systems, build_high_order_views(
        systems, options, hypers, torch.Generator().manual_seed(seed)
    )


def test_small_rotations_are_rotations_by_the_angle():
    r = small_rotations(16, 0.1, torch.float64, torch.Generator().manual_seed(0))
    eye = torch.eye(3, dtype=torch.float64).expand(16, 3, 3)
    torch.testing.assert_close(r @ r.transpose(1, 2), eye)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(16, dtype=torch.float64))
    angles = torch.arccos((r.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2)
    torch.testing.assert_close(angles, torch.full((16,), 0.1, dtype=torch.float64))


def test_rot_small_is_the_student_view_rotated_by_the_angle(large_batch):
    _, v = _views(large_batch, p_def=0.5, smooth_rotation=0.2)
    for student, rotated in zip(
        v.student, v.extras["student_views"]["rot_small"], strict=True
    ):
        x, y = student.positions.detach(), rotated.positions.detach()
        # y = x R^T for a rotation R by 0.2 rad (fit by least squares)
        r = torch.linalg.lstsq(x, y).solution.T
        angle = torch.arccos((r.trace() - 1) / 2)
        assert abs(float(angle) - 0.2) < 1e-6


def test_disp_small_stays_within_half_the_skin(large_batch):
    systems, v = _views(large_batch, p_def=1.0, smooth_displacement=0.2)
    skin = large_batch[2]["noise"]["skin"]
    ho = v.extras["high_order"]
    offset = 0
    for system, moved, student in zip(
        systems, v.extras["student_views"]["disp_small"], v.student, strict=True
    ):
        n = len(system)
        rotation = ho.rotations[offset]
        clean = system.positions @ rotation.T
        total = (moved.positions.detach() - clean).norm(dim=-1)
        assert float(total.max()) <= skin / 2 + 1e-9
        assert not torch.allclose(moved.positions, student.positions)
        offset += n


def test_smoothness_views_leave_the_other_views_unchanged(large_batch):
    _, a = _views(large_batch, p_def=0.5, second_rotation=True)
    _, b = _views(
        large_batch,
        p_def=0.5,
        second_rotation=True,
        smooth_rotation=0.1,
        smooth_displacement=0.02,
    )
    for sa, sb in zip(a.student + a.teacher, b.student + b.teacher, strict=True):
        assert torch.equal(sa.positions, sb.positions)
    assert set(b.extras["student_views"]) == {"rot2", "rot_small", "disp_small"}


@pytest.mark.parametrize(
    "term_class, option",
    [
        (RotationalSmoothnessTerm, "smooth_rotation"),
        (PositionalSmoothnessTerm, "smooth_displacement"),
    ],
)
def test_smoothness_term(medium_batch, pet_core, small_teacher, term_class, option):
    step = make_step(medium_batch, pet_core, small_teacher, seed=1, **{option: 0.05})
    term = term_class(weight=2.0)
    assert not term.needs_gradients and not term.uses_prediction
    loss, diag = term(step)
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    torch.testing.assert_close(loss, 2.0 * diag[f"{term.name}/loss"])
    loss.backward()
    grads = [p.grad for p in pet_core.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    without = make_step(medium_batch, pet_core, small_teacher, seed=1)
    with pytest.raises(ValueError, match=option):
        term(without)


def test_rotational_smoothness_penalizes_high_orders_more():
    """The math behind the term: under a small rotation about a random axis, the
    squared change of an order-l function is proportional to l(l+1) (checked on the
    real spherical harmonics, order 4 vs order 1)."""
    from metatrain.experimental.ssl_pretraining.high_order.irreps import (
        real_spherical_harmonics,
    )

    torch.manual_seed(0)
    points = torch.randn(2000, 3, dtype=torch.float64)
    r = small_rotations(2000, 0.05, torch.float64, torch.Generator().manual_seed(1))
    rotated = torch.einsum("nab,nb->na", r, points)
    y, yr = real_spherical_harmonics(4, points), real_spherical_harmonics(4, rotated)
    change = {
        l: float((yr[l] - y[l]).pow(2).sum(-1).mean() / y[l].pow(2).sum(-1).mean())
        for l in (1, 4)
    }
    ratio = change[4] / change[1]
    assert abs(ratio - (4 * 5) / (1 * 2)) < 1.0  # l(l+1) scaling


def test_smoothness_terms_are_registered_and_validated(small_batch):
    for name, cls_name in [
        ("rotational_smoothness", "RotationalSmoothnessTermHypers"),
        ("positional_smoothness", "PositionalSmoothnessTermHypers"),
    ]:
        assert TERMS[name].name == name
        assert name in TermsHypers.__annotations__
        option = {"rotational_smoothness": "smooth_rotation"}.get(
            name, "smooth_displacement"
        )
        _, _, hypers = small_batch
        hypers = copy.deepcopy(hypers)
        hypers["views"]["builder"] = "high_order"
        hypers["objective"]["terms"] = {
            "vicreg": {"variance": 1.0, "covariance": 0.04},
            name: copy.deepcopy(init_with_defaults(getattr(documentation, cls_name))),
        }
        with pytest.raises(ValueError, match=option):
            check_hypers(hypers, 4.0)
        hypers["high_order"][option] = 0.05
        check_hypers(hypers, 4.0)


def test_full_vector_compares_the_whole_latent(medium_batch, pet_core, small_teacher):
    step = make_step(medium_batch, pet_core, small_teacher, seed=1, p_def=0.0)
    d = step.teacher_features.shape[1]
    dims = step.extras["student_tokens"].dims
    term = RotationConditionedLatentTerm(dims, d, 2, 16, 4.0, 1.0, full_vector=True)
    target = term.teacher_frame_targets(step)
    assert target.shape[1] == d
    teacher = step.teacher_features
    if "minus" in step.extras["teacher_features"]:
        teacher = 0.5 * (teacher + step.extras["teacher_features"]["minus"])
    torch.testing.assert_close(target, teacher[step.views.latent_targets])
    assert term.predict(step).shape == target.shape
    probed = RotationConditionedLatentTerm(dims, d, 2, 16, 4.0, 1.0)
    assert probed.teacher_frame_targets(step).shape[1] == step.probes.shape[0]


def test_full_vector_response(medium_batch, pet_core, small_teacher):
    step = make_step(medium_batch, pet_core, small_teacher, seed=1)
    d = step.teacher_features.shape[1]
    term = ActionResponseTerm(
        step.extras["student_tokens"].dims, d, 16, 1.0, full_vector=True
    )
    if step.extras["high_order"].centers.numel():
        delta, _ = term.responses(step)
        torch.testing.assert_close(term.teacher_frame_targets(step), delta)
        assert term.predict(step).shape == delta.shape
    assert math.isfinite(float(term(step)[0].detach()))
