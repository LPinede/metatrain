import copy
import dataclasses
import math
import types
import warnings

import ase
import ase.io
import numpy as np
import pytest
import torch

from metatrain.experimental.ssl_pretraining.backbones import HeadInputs
from metatrain.experimental.ssl_pretraining.high_order import documentation
from metatrain.experimental.ssl_pretraining.high_order.deformations import (
    mode_displacements,
)
from metatrain.experimental.ssl_pretraining.high_order.irreps import (
    XYZ_TO_L1,
    irrep_matrix,
)
from metatrain.experimental.ssl_pretraining.high_order.terms import (
    ActionResponseTerm,
)
from metatrain.experimental.ssl_pretraining.objective import StepData, grad_jepa_loss
from metatrain.experimental.ssl_pretraining.terms import (
    TERMS,
    TermContext,
    build_terms,
)
from metatrain.experimental.ssl_pretraining.tests import CARBON_PATH, small_pet
from metatrain.experimental.ssl_pretraining.trainer import check_hypers
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.hypers import init_with_defaults

from . import ATOMIC_TYPES, haar_o3, make_batch


NAME = "action_response"
EPS = 0.05


def _term(step, **kwargs):
    kwargs.setdefault("warmup_fraction", 0.0)
    return ActionResponseTerm(
        step.extras["student_tokens"].dims,
        step.teacher_features.shape[1],
        32,
        1.0,
        EPS,
        rank=8,
        **kwargs,
    )


def _with_rotations(step, rotations):
    """``step`` with the student rotations replaced (the rest shared)."""
    ho = copy.copy(step.extras["high_order"])
    ho.rotations = rotations
    return dataclasses.replace(step, extras={**step.extras, "high_order": ho})


def test_fixture_has_centres(step_data):
    assert step_data.extras["high_order"].centers.numel() >= 2


def test_response_term_finite_and_monitors(step_data):
    term = _term(step_data)
    loss, diag = term(step_data)
    assert torch.isfinite(loss) and loss.requires_grad
    for key in [
        "rho_target",
        "rho_l0_dilation",
        "rho_l1_rotation",
        "rho_l2_shear",
        "r2",
        "loss",
    ]:
        assert f"{NAME}/{key}" in diag
    assert all(torch.isfinite(v) for v in diag.values())


def test_target_is_half_the_pair_difference(step_data):
    term = _term(step_data)
    centers = step_data.extras["high_order"].centers
    plus = step_data.teacher_features
    minus = step_data.extras["teacher_features"]["minus"]
    expected = (0.5 * (plus - minus) @ step_data.probes.T)[centers]
    torch.testing.assert_close(term.teacher_frame_targets(step_data), expected)


def test_one_sided_teacher_runs(step_data_one_sided):
    step = step_data_one_sided
    assert "plus" in step.extras["teacher_features"]
    term = _term(step)
    centers = step.extras["high_order"].centers
    plus = step.extras["teacher_features"]["plus"]
    expected = ((plus - step.teacher_features) @ step.probes.T)[centers]
    torch.testing.assert_close(term.teacher_frame_targets(step), expected)
    loss, diag = term(step)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in diag.values())


def test_no_centers_gives_zero(tiny_step_data):
    assert tiny_step_data.extras["high_order"].centers.numel() == 0
    term = _term(tiny_step_data)
    loss, diag = term(tiny_step_data)
    assert loss == 0 and loss.requires_grad
    assert float(diag[f"{NAME}/loss"]) == 0.0
    assert all(torch.isfinite(v) for v in diag.values())


def test_student_frame_amplitudes(step_data):
    term = _term(step_data)
    ho = step_data.extras["high_order"]
    rotations = ho.rotations[ho.centers]
    a = ho.center_amplitudes
    mapped = term.student_amplitudes(step_data, a)
    det = torch.linalg.det(rotations)
    # dilation: a scalar; rotation: an axial vector, det(g) g w; shear: D_2(g) s
    torch.testing.assert_close(mapped["dilation"], a[:, :1])
    axial = det[:, None] * torch.einsum("nab,nb->na", rotations, a[:, 1:4])
    torch.testing.assert_close(mapped["rotation"], axial[:, list(XYZ_TO_L1)])
    shear = torch.einsum("nmp,np->nm", irrep_matrix(rotations, 2, 1), a[:, 4:9])
    torch.testing.assert_close(mapped["shear"], shear)


def test_pseudo_block_uses_determinant(step_data):
    """Under ``g -> -g`` the axial rotation amplitude is unchanged (a proper l=1
    block would flip)."""
    term = _term(step_data)
    assert term.BLOCKS["rotation"] == (1, -1)
    ho = step_data.extras["high_order"]
    a = ho.center_amplitudes
    mirrored = _with_rotations(step_data, -ho.rotations)
    torch.testing.assert_close(
        term.student_amplitudes(mirrored, a)["rotation"],
        term.student_amplitudes(step_data, a)["rotation"],
    )
    rotations = ho.rotations[ho.centers]
    proper = torch.einsum("nmp,np->nm", irrep_matrix(-rotations, 1, 1), a[:, [2, 3, 1]])
    assert not torch.allclose(proper, term.student_amplitudes(mirrored, a)["rotation"])


def test_math_student_frame_amplitudes_move_the_mode_with_the_input():
    """The mode at the student-frame amplitudes, on the rotated positions, is the
    rotated mode: the amplitudes are mapped in the basis of ``mode_displacements``."""
    torch.manual_seed(0)
    rotations = haar_o3(4)
    a = torch.randn(4, 9, dtype=torch.float64)
    term = ActionResponseTerm((4, 4), 6, 8, 1.0, EPS)
    ho = types.SimpleNamespace(rotations=rotations, centers=torch.arange(4))
    step = types.SimpleNamespace(extras={"high_order": ho})
    mapped = term.student_amplitudes(step, a)
    inverse = torch.argsort(torch.tensor(XYZ_TO_L1))
    x = torch.randn(10, 3, dtype=torch.float64)
    zero = torch.zeros(3, dtype=torch.float64)
    for n in range(4):
        r = rotations[n]
        u = mode_displacements(x, zero, a[n, 0], a[n, 1:4], a[n, 4:9], 3.0)
        u_student = mode_displacements(
            x @ r.T,
            zero,
            mapped["dilation"][n, 0],
            mapped["rotation"][n, inverse],
            mapped["shear"][n],
            3.0,
        )
        torch.testing.assert_close(u_student, u @ r.T)


def test_prediction_contracts_the_amplitudes_with_low_rank_heads(step_data):
    term = _term(step_data)
    ho = step_data.extras["high_order"]
    tokens = step_data.extras["student_tokens"].select(ho.centers)
    mapped = term.student_amplitudes(step_data, ho.center_amplitudes)
    vector = 0.0
    for block in term.BLOCKS:
        a = term.heads[block](tokens)  # (n_c, 2l+1, rank)
        phi = torch.einsum("nmk,dk->nmd", a, term.maps[block].weight)
        vector = vector + torch.einsum("nm,nmd->nd", mapped[block].to(phi), phi)
    torch.testing.assert_close(term.predict(step_data), vector @ step_data.probes.T)


def test_shuffle_permutes_the_amplitudes_across_centres(step_data):
    term = _term(step_data, shuffle_amplitudes=True)
    a = step_data.extras["high_order"].center_amplitudes
    generator = step_data.extras["generator"]
    state = generator.get_state()
    shuffled = term.amplitudes(step_data)
    generator.set_state(state)
    perm = torch.randperm(len(a), generator=generator)
    torch.testing.assert_close(shuffled, a[perm])
    # redrawn at every call (the fixture has few centres: some draws are identities)
    assert any(not torch.equal(term.amplitudes(step_data), a) for _ in range(20))
    torch.testing.assert_close(_term(step_data).amplitudes(step_data), a)


def test_l_max_drops_the_higher_blocks(step_data):
    term = _term(step_data, l_max=1)
    assert set(term.heads) == {"dilation", "rotation"}
    loss, diag = term(step_data)
    assert torch.isfinite(loss) and f"{NAME}/rho_l2_shear" not in diag


def test_warm_up_turns_the_term_off(step_data):
    term = _term(step_data, warmup_fraction=0.6).train()
    step = dataclasses.replace(
        step_data, extras={**step_data.extras, "step_fraction": 0.5}
    )
    loss, diag = term(step)
    assert loss == 0 and float(diag[f"{NAME}/loss"]) > 0
    assert float(diag[f"{NAME}/active"]) == 0.0
    step.extras["step_fraction"] = 0.7
    loss, diag = term(step)
    assert loss > 0 and float(diag[f"{NAME}/active"]) == 1.0


def test_loss_is_weighted_normalized_mse_plus_floors(step_data):
    term = _term(step_data, floor_gamma=0.5).eval()
    # zero rotation and shear contributions: their floors are active
    with torch.no_grad():
        for block in ["rotation", "shear"]:
            last = term.heads[block].node_net[-1]
            last.weight.zero_()
            last.bias.zero_()
    term.weight = 3.0
    loss, diag = term(step_data)
    assert float(diag[f"{NAME}/floor"]) > 0
    torch.testing.assert_close(loss, 3.0 * diag[f"{NAME}/loss"])
    no_floor = _term(step_data, floor_gamma=0.0).eval()
    no_floor.load_state_dict(term.state_dict())
    loss_0, _ = no_floor(step_data)
    torch.testing.assert_close(loss / 3.0 - loss_0, 0.5 * diag[f"{NAME}/floor"])


def test_rho_monitors_are_relative_vector_responses(step_data):
    term = _term(step_data).eval()
    _, diag = term(step_data)
    ho = step_data.extras["high_order"]
    plus = step_data.teacher_features[ho.centers]
    minus = step_data.extras["teacher_features"]["minus"][ho.centers]
    reference = (0.5 * (plus + minus)).norm()
    torch.testing.assert_close(
        diag[f"{NAME}/rho_target"], (0.5 * (plus - minus)).norm() / (EPS * reference)
    )
    tokens = step_data.extras["student_tokens"].select(ho.centers)
    mapped = term.student_amplitudes(step_data, ho.center_amplitudes)
    a = torch.einsum(
        "nm,nmk->nk", mapped["shear"].to(plus), term.heads["shear"](tokens)
    )
    shear = a @ term.maps["shear"].weight.T
    torch.testing.assert_close(
        diag[f"{NAME}/rho_l2_shear"], shear.norm() / (EPS * reference)
    )


def test_mode_amplitude_defaults_to_the_views_setting(step_data):
    term = ActionResponseTerm(
        step_data.extras["student_tokens"].dims,
        step_data.teacher_features.shape[1],
        32,
        1.0,
        None,
        warmup_fraction=0.0,
    )
    _, diag = term(step_data)
    _, diag_ref = _term(step_data)(step_data)
    scale = step_data.hypers["high_order"]["mode_amplitude"] / EPS
    torch.testing.assert_close(
        diag[f"{NAME}/rho_target"] * scale, diag_ref[f"{NAME}/rho_target"]
    )


DIM = 6


def _probes(gen, num_probes=2):
    signs = torch.randint(0, 2, (num_probes, DIM), generator=gen) * 2 - 1
    return signs.to(torch.float64) / DIM**0.5


def _synthetic_step(n, seed=0):
    """Invariant student tokens, Haar O(3) rotations, Gaussian amplitudes and a
    teacher whose antithetic response is linear in the student-frame amplitudes:
    ``Delta = sum_blocks (c_b . a_S,b) w_b`` (what an orientation-blind student can
    represent), around a random reference ``h``."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    rotations = haar_o3(n)
    amplitudes = EPS * torch.randn(n, 9, generator=gen, dtype=torch.float64)
    ho = types.SimpleNamespace(
        rotations=rotations,
        centers=torch.arange(n),
        center_amplitudes=amplitudes,
    )
    mapped = ActionResponseTerm((4, 4), DIM, 8, 1.0, EPS).student_amplitudes(
        types.SimpleNamespace(extras={"high_order": ho}), amplitudes
    )
    delta = torch.zeros(n, DIM, dtype=torch.float64)
    for block in ["dilation", "rotation", "shear"]:
        c = torch.randn(mapped[block].shape[1], generator=gen, dtype=torch.float64)
        w = torch.randn(DIM, generator=gen, dtype=torch.float64)
        delta = delta + (mapped[block] @ c)[:, None] * w[None, :] / EPS
    reference = torch.randn(n, DIM, generator=gen, dtype=torch.float64)
    tokens = HeadInputs(
        node=torch.ones(n, 4, dtype=torch.float64),
        edge=torch.zeros(n, 1, 4, dtype=torch.float64),
        cutoff=torch.zeros(n, 1, dtype=torch.float64),
    )
    return StepData(
        views=types.SimpleNamespace(),
        probes=_probes(gen),
        student_features=torch.zeros(n, DIM, dtype=torch.float64),
        teacher_features=reference + delta,
        student_gradients=None,
        teacher_gradients=None,
        prediction={},
        hypers={},
        extras={
            "student_tokens": tokens,
            "teacher_features": {"minus": reference - delta},
            "high_order": ho,
            "step_fraction": 1.0,
            "generator": gen,
        },
    )


def _fit(term, step, n_steps=300):
    """Train ``term`` on ``step`` with new probes (and, with shuffling, a new
    permutation) at every step; the normalized error on fresh probes, with the
    amplitudes the term uses."""
    gen = torch.Generator().manual_seed(1)
    term = term.to(torch.float64).train()
    optimizer = torch.optim.Adam(term.parameters(), lr=1e-2)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss, _ = term(dataclasses.replace(step, probes=_probes(gen)))
        loss.backward()
        optimizer.step()
    term.eval()
    step = dataclasses.replace(step, probes=_probes(gen, 8))
    with torch.no_grad():
        target = term.teacher_frame_targets(step)
        error = (term.predict(step) - target).pow(2).mean()
    return float(error / target.var())


def test_linear_response_is_fitted_with_matched_not_shuffled_amplitudes():
    step = _synthetic_step(1000)

    def term(shuffle):
        torch.manual_seed(0)
        return ActionResponseTerm(
            (4, 4),
            DIM,
            16,
            1.0,
            EPS,
            rank=4,
            shuffle_amplitudes=shuffle,
            floor_gamma=0.0,
            warmup_fraction=0.0,
        )

    assert _fit(term(False), step) < 0.05
    assert _fit(term(True), step) > 0.9


def test_math_shuffled_lstsq_control():
    """If the target is exactly linear in a, the matched regression fits it and
    the shuffled one cannot."""
    torch.manual_seed(0)
    n = 512
    a = torch.randn(n, 9)
    k = torch.randn(9, 4)
    target = a @ k
    fit = torch.linalg.lstsq(a, target).solution
    shuffled = a[torch.randperm(n)]
    fit_s = torch.linalg.lstsq(shuffled, target).solution
    assert ((a @ fit - target) ** 2).mean() < 1e-10
    assert ((shuffled @ fit_s - target) ** 2).mean() > 0.5 * target.var()


def _defaults():
    return copy.deepcopy(init_with_defaults(documentation.ActionResponseTermHypers))


def test_from_hypers_takes_the_feature_dim_from_the_context():
    context = TermContext(
        dims=(24, 20),
        cutoff=3.5,
        atomic_types=[1, 8],
        num_probes=5,
        feature_dim=44,
    )
    hypers = _defaults()
    hypers["head"]["channel_fraction"] = 0.5
    hypers["rank"] = 7
    hypers["shuffle_amplitudes"] = True
    (term,) = build_terms({NAME: hypers}, context)
    assert term.heads["shear"].dn == 12 and term.heads["shear"].n_out == 7
    assert term.maps["shear"].weight.shape == (44, 7)
    assert term.shuffle_amplitudes and term.mode_amplitude is None
    assert not term.needs_gradients and not term.uses_prediction
    assert term.required_targets == {}
    assert TERMS[NAME] is ActionResponseTerm
    context.feature_dim = None
    with pytest.raises(ValueError, match="feature_dim"):
        build_terms({NAME: hypers}, context)


def test_default_hypers_disable_the_term_and_validate_when_enabled():
    options = get_default_hypers("experimental.ssl_pretraining")
    terms = options["training"]["objective"]["terms"]
    assert terms[NAME] is None
    terms[NAME] = _defaults()
    assert terms[NAME]["rank"] == 64
    assert terms[NAME]["l_max"] == 2
    assert terms[NAME]["warmup_fraction"] == 0.05
    assert terms[NAME]["floor_gamma"] == 0.1
    assert terms[NAME]["shuffle_amplitudes"] is False
    check_architecture_options(name="experimental.ssl_pretraining", options=options)


def test_check_hypers_requires_the_high_order_builder(small_batch):
    _, _, hypers = small_batch
    hypers = copy.deepcopy(hypers)
    hypers["objective"]["terms"] = {
        NAME: _defaults(),
        "vicreg": {"variance": 1.0, "covariance": 0.04},
    }
    with pytest.raises(ValueError, match=f"'{NAME}' term needs `views.builder"):
        check_hypers(hypers, 4.0)
    hypers["views"]["builder"] = "high_order"
    check_hypers(hypers, 4.0)


def _jittered_grid():
    rng = np.random.default_rng(0)
    grid = np.stack(
        np.meshgrid(np.arange(3), np.arange(3), np.arange(2), indexing="ij"), -1
    ).reshape(-1, 3)
    positions = 1.5 * grid + 0.1 * rng.standard_normal(grid.shape)
    return ase.Atoms("C" * len(grid), positions=positions)


@pytest.mark.parametrize("teacher_pair", [True, False])
def test_term_contract(teacher_pair):
    """Alone: finite on degenerate batches, no warnings, trains the student only."""
    frames = [
        ase.Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ase.io.read(CARBON_PATH, 0),
        _jittered_grid(),
    ]
    torch.manual_seed(0)
    student = small_pet(ATOMIC_TYPES, torch.float64).train()
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    context = TermContext(
        dims=(
            student.num_readout_layers * student.d_node,
            student.num_readout_layers * student.d_pet,
        ),
        cutoff=4.0,
        atomic_types=ATOMIC_TYPES,
        num_probes=2,
        feature_dim=student.num_readout_layers * (student.d_node + student.d_pet),
    )
    (term,) = build_terms({NAME: _defaults()}, context)
    term = term.to(torch.float64)
    for batch in [frames[:1], frames[:2], frames]:
        systems, options, hypers = make_batch(batch, torch.float64)
        hypers["views"]["builder"] = "high_order"
        hypers["high_order"]["teacher_pair"] = teacher_pair
        hypers["high_order"]["p_def"] = 1.0
        for seed in range(3):
            student.zero_grad()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loss, metrics = grad_jepa_loss(
                    student,
                    teacher,
                    None,
                    [term],
                    systems,
                    options,
                    hypers,
                    torch.Generator().manual_seed(seed),
                    with_rank=True,
                    step_fraction=0.5,
                )
                loss.backward()
            assert torch.isfinite(loss)
            assert all(math.isfinite(v) for v in metrics.values())
            assert all(p.grad is None for p in teacher.parameters())
        # the batch with centres reaches the student
        if len(batch) > 2:
            assert f"{NAME}/rho_target" in metrics
            assert any(p.grad is not None for p in student.parameters())
