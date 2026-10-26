import copy

import pytest
import torch

from metatrain.experimental.ssl_pretraining.documentation import TermsHypers
from metatrain.experimental.ssl_pretraining.high_order import documentation
from metatrain.experimental.ssl_pretraining.high_order.terms import (
    ActionResponseTerm,
    InverseResponseTerm,
)
from metatrain.experimental.ssl_pretraining.high_order.views import (
    build_high_order_views,
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


def test_nudge_views_are_the_student_input_moved_by_the_teacher_modes(large_batch):
    systems, v = _views(large_batch, p_def=0.5, student_nudge=True)
    skin = large_batch[2]["noise"]["skin"]
    ho = v.extras["high_order"]
    plus, minus = (v.extras["student_views"][k] for k in ("nudge_plus", "nudge_minus"))
    offset = 0
    moved_some = False
    for system, student, p, m in zip(systems, v.student, plus, minus, strict=True):
        n = len(system)
        rotation = ho.rotations[offset]
        clean = system.positions @ rotation.T
        for view in (p, m):
            total = (view.positions.detach() - clean).norm(dim=-1)
            assert float(total.max()) <= skin / 2 + 1e-9
        # antithetic around the student input, wherever the cap does not bind
        mid = 0.5 * (p.positions + m.positions).detach()
        uncapped = ((p.positions - m.positions).detach() / 2).norm(dim=-1) < 1e-3
        torch.testing.assert_close(
            mid[uncapped], student.positions.detach()[uncapped], atol=1e-9, rtol=0
        )
        moved_some |= not torch.allclose(p.positions, student.positions)
        offset += n
    assert moved_some


def test_nudge_views_leave_the_other_views_unchanged(large_batch):
    _, a = _views(large_batch, p_def=0.5, smooth_displacement=0.02)
    _, b = _views(large_batch, p_def=0.5, smooth_displacement=0.02, student_nudge=True)
    for sa, sb in zip(a.student + a.teacher, b.student + b.teacher, strict=True):
        assert torch.equal(sa.positions, sb.positions)
    for sa, sb in zip(
        a.extras["student_views"]["disp_small"],
        b.extras["student_views"]["disp_small"],
        strict=True,
    ):
        assert torch.equal(sa.positions, sb.positions)


def test_inverse_response_term(medium_batch, pet_core, small_teacher):
    step = make_step(
        medium_batch, pet_core, small_teacher, seed=1, p_def=1.0, student_nudge=True
    )
    term = InverseResponseTerm(step.extras["student_tokens"].dims, 16, weight=2.0)
    assert not term.needs_gradients and not term.uses_prediction
    ho = step.extras["high_order"]
    assert ho.centers.numel() >= 2
    pred, target = term.decode(step)
    assert pred.shape == target.shape == (ho.centers.numel(), 9)
    # the targets are the student-frame amplitudes over the mode amplitude
    a0 = step.hypers["high_order"]["mode_amplitude"]
    blocks = ActionResponseTerm(
        step.extras["student_tokens"].dims, 8, 8, 1.0
    ).student_amplitudes(step, ho.center_amplitudes)
    torch.testing.assert_close(target, torch.cat(list(blocks.values()), 1) / a0)
    loss, diag = term(step)
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    torch.testing.assert_close(loss, 2.0 * diag[f"{term.name}/loss"])
    loss.backward()
    grads = [p.grad for p in pet_core.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    without = make_step(medium_batch, pet_core, small_teacher, seed=1, p_def=1.0)
    with pytest.raises(ValueError, match="student_nudge"):
        term(without)


def test_inverse_response_without_centres_costs_nothing(
    medium_batch, pet_core, small_teacher
):
    step = make_step(
        medium_batch,
        pet_core,
        small_teacher,
        seed=1,
        p_def=1.0,
        student_nudge=True,
        max_centers=0,
    )
    term = InverseResponseTerm(step.extras["student_tokens"].dims, 16)
    loss, diag = term(step)
    assert float(loss.detach()) == 0.0 and float(diag[f"{term.name}/skipped"]) == 1.0


def test_inverse_response_is_registered_and_validated(small_batch):
    assert TERMS["inverse_response"].name == "inverse_response"
    assert "inverse_response" in TermsHypers.__annotations__
    _, _, hypers = small_batch
    hypers = copy.deepcopy(hypers)
    hypers["views"]["builder"] = "high_order"
    hypers["objective"]["terms"] = {
        "vicreg": {"variance": 1.0, "covariance": 0.04},
        "inverse_response": copy.deepcopy(
            init_with_defaults(documentation.InverseResponseTermHypers)
        ),
    }
    with pytest.raises(ValueError, match="student_nudge"):
        check_hypers(hypers, 4.0)
    hypers["high_order"]["student_nudge"] = True
    check_hypers(hypers, 4.0)
