import copy
import dataclasses
import types
import warnings

import ase
import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining.backbones import HeadInputs
from metatrain.experimental.ssl_pretraining.high_order import documentation
from metatrain.experimental.ssl_pretraining.high_order.irreps import (
    irrep_matrix,
)
from metatrain.experimental.ssl_pretraining.high_order.terms import (
    RotationConditionedLatentTerm,
)
from metatrain.experimental.ssl_pretraining.objective import StepData, grad_jepa_loss
from metatrain.experimental.ssl_pretraining.terms import (
    TERMS,
    TermContext,
    build_terms,
)
from metatrain.experimental.ssl_pretraining.tests import CARBON_PATH, small_pet
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.hypers import init_with_defaults

from . import ATOMIC_TYPES, haar_o3, make_batch, make_step, randomize_heads


NAME = "rotation_conditioned_latent"
CUTOFF = 4.0


def _term(step, l_max=4, **kwargs):
    kwargs.setdefault("warmup_fraction", 0.0)
    return randomize_heads(
        RotationConditionedLatentTerm(
            step.extras["student_tokens"].dims,
            step.teacher_features.shape[1],
            l_max,
            32,
            CUTOFF,
            1.0,
            **kwargs,
        )
    )


def test_l0_is_plain_invariant_latent(step_data):
    term = _term(step_data, l_max=0)
    assert term.head_keys == [(0, 1)]
    loss, diag = term(step_data)
    assert torch.isfinite(loss) and f"{NAME}/total/rmse" in diag
    # the prediction is the l=0 head on the head inputs, mapped to the latent and
    # contracted with the probes, whatever the rotations
    pred = term.predict(step_data)
    atoms = step_data.views.latent_targets
    inputs = term.head_inputs(step_data).select(atoms)
    latent = term.maps["l0p"](term.heads["l0p"](inputs)[:, 0])
    expected = latent @ step_data.probes.T
    torch.testing.assert_close(pred, expected)
    ho = step_data.extras["high_order"]
    flipped = copy.copy(step_data)
    flipped.extras = {
        **step_data.extras,
        "high_order": copy.copy(ho),
    }
    flipped.extras["high_order"].rotations = haar_o3(len(ho.rotations)).to(ho.rotations)
    torch.testing.assert_close(term.predict(flipped), pred)


def test_target_is_the_probe_contracted_pair_mean(step_data):
    term = _term(step_data)
    plus = step_data.teacher_features
    minus = step_data.extras["teacher_features"]["minus"]
    atoms = step_data.views.latent_targets
    expected = (0.5 * (plus + minus) @ step_data.probes.T)[atoms]
    torch.testing.assert_close(term.teacher_frame_targets(step_data), expected)


def test_one_sided_target_is_the_clean_teacher(medium_batch, pet_core, small_teacher):
    step = make_step(medium_batch, pet_core, small_teacher, seed=1, teacher_pair=False)
    term = _term(step)
    atoms = step.views.latent_targets
    expected = (step.teacher_features @ step.probes.T)[atoms]
    torch.testing.assert_close(term.teacher_frame_targets(step), expected)


def test_masked_atoms_read_the_invariant_context_only(step_data_masked):
    step = step_data_masked
    term = _term(step)
    masked = step.views.masked
    assert masked.any()
    tokens = step.extras["student_tokens"]
    inputs = term.head_inputs(step)
    # visible atoms: the student tokens
    torch.testing.assert_close(inputs.node[~masked], tokens.node[~masked])
    torch.testing.assert_close(inputs.cutoff[~masked], tokens.cutoff[~masked])
    # masked atoms: the context output, and no edge token is read
    assert (inputs.cutoff[masked] == 0).all()
    pairs, distances = term.context_pairs(step)
    assert masked[pairs[:, 0]].all() and not masked[pairs[:, 1]].any()
    ho = step.extras["high_order"]
    # masked structures are undeformed: clean distances are student distances
    assert (distances < CUTOFF).all()
    context = term.context(tokens.node, pairs, distances)
    torch.testing.assert_close(inputs.node[masked], context[masked])
    # every reachable masked atom gets at least one message
    reachable = masked & step.views.latent_targets
    assert reachable.any()
    assert torch.isin(reachable.nonzero()[:, 0], pairs[:, 0]).all()
    assert len(pairs) <= int(masked[ho.pair_index[:, 0]].sum())


@pytest.mark.parametrize("l, sigma", [(0, -1), (1, 1), (1, -1), (2, 1)])
def test_math_orbit_coefficient(l, sigma):
    """The least-squares coefficients of a target on the entries of ``D^(l,sigma)``
    (with the other blocks) are ``d E[D t]`` (Peter-Weyl, the optimum of term C)."""
    torch.manual_seed(0)
    n = 100000
    rots = haar_o3(n)
    blocks = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1)]
    features = {b: irrep_matrix(rots, *b).reshape(n, -1) for b in blocks}
    coefficients = {
        b: torch.randn(features[b].shape[1], dtype=rots.dtype) for b in blocks
    }
    target = sum(features[b] @ coefficients[b] for b in blocks)
    design = torch.cat([features[b] for b in blocks], dim=1)
    solution = torch.linalg.lstsq(design, target[:, None]).solution[:, 0]
    start = sum(features[b].shape[1] for b in blocks[: blocks.index((l, sigma))])
    fitted = solution[start : start + (2 * l + 1) ** 2]
    torch.testing.assert_close(fitted, coefficients[(l, sigma)])
    d = 2 * l + 1
    monte_carlo = d * (features[(l, sigma)] * target[:, None]).mean(0)
    torch.testing.assert_close(monte_carlo, fitted, atol=0.1, rtol=0.0)


DIM = 6


def _synthetic_step(n, content, seed=0):
    """Invariant student tokens (identical for every atom), Haar rotations, and a
    teacher latent ``h = s(g) w`` with an orientation signal ``s`` without l=0
    proper content: ``"l1p"`` is ``(g^T a)_x`` (order 1, proper), ``"l1m"`` is
    ``det(g) (g^T a)_x`` (order 1, pseudo) and ``"l0m"`` is ``det(g)``."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    rotations = haar_o3(n)
    a = torch.randn(3, generator=gen, dtype=torch.float64)
    det = torch.linalg.det(rotations)
    signal = {
        "l1p": torch.einsum("nba,b->na", rotations, a)[:, 0],
        "l1m": det * torch.einsum("nba,b->na", rotations, a)[:, 0],
        "l0m": det,
    }[content]
    w = torch.randn(DIM, generator=gen, dtype=torch.float64)
    teacher = signal[:, None] * w[None, :]
    tokens = HeadInputs(
        node=torch.ones(n, 4, dtype=torch.float64),
        edge=torch.zeros(n, 1, 4, dtype=torch.float64),
        cutoff=torch.zeros(n, 1, dtype=torch.float64),
    )
    ho = types.SimpleNamespace(
        rotations=rotations,
        pair_index=torch.zeros(0, 2, dtype=torch.long),
        clean_vectors=torch.zeros(0, 3, dtype=torch.float64),
    )
    views = types.SimpleNamespace(
        masked=torch.zeros(n, dtype=torch.bool),
        latent_targets=torch.ones(n, dtype=torch.bool),
    )
    return StepData(
        views=views,
        probes=_probes(gen),
        student_features=torch.zeros(n, DIM, dtype=torch.float64),
        teacher_features=teacher,
        student_gradients=None,
        teacher_gradients=None,
        prediction={},
        hypers={},
        extras={
            "student_tokens": tokens,
            "teacher_features": {"minus": teacher},
            "high_order": ho,
            "step_fraction": 1.0,
        },
    )


def _probes(gen, num_probes=2):
    signs = torch.randint(0, 2, (num_probes, DIM), generator=gen) * 2 - 1
    return signs.to(torch.float64) / DIM**0.5


def _fit(term, step, n_steps=300):
    """Train ``term`` on ``step`` with new probes at every step; the normalized
    error on fresh probes."""
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
        error = (term.predict(step) - term.teacher_frame_targets(step)).pow(2).mean()
    return float(error / term.teacher_frame_targets(step).var())


@pytest.mark.parametrize("content", ["l1p", "l1m", "l0m"])
def test_heads_learn_the_orientation_content_with_probes_redrawn(content):
    step = _synthetic_step(2000, content)

    def term(l_max, group):
        torch.manual_seed(0)
        return RotationConditionedLatentTerm(
            (4, 4),
            DIM,
            l_max,
            16,
            CUTOFF,
            1.0,
            rank=4,
            rotation_group=group,
            floor_gamma=0.0,
            warmup_fraction=0.0,
        )

    # the invariant JEPA cannot predict orientation content (C is neutral)
    assert _fit(term(0, "O3"), step) > 0.9
    # order 1 (with the (0, -) block) captures it, although the probes change
    assert _fit(term(1, "O3"), step) < 0.05
    if content != "l1p":
        # without the pseudo heads, the pseudo content stays unexplained
        assert _fit(term(1, "SO3"), step) > 0.9


def test_prediction_contracts_the_rotation_with_the_head_output(step_data):
    term = _term(step_data, l_max=2)
    assert term.active_blocks() == term.head_keys
    atoms = step_data.views.latent_targets
    inputs = term.head_inputs(step_data).select(atoms)
    rotations = step_data.extras["high_order"].rotations[atoms]
    latent = 0.0
    for l, sigma in term.head_keys:
        name = term.key(l, sigma)
        a = term.heads[name](inputs).reshape(-1, 2 * l + 1, 2 * l + 1, term.rank)
        phi = torch.einsum("nmqk,dk->nmqd", a, term.maps[name].weight)
        latent = latent + torch.einsum(
            "nmq,nmqd->nd", irrep_matrix(rotations, l, sigma), phi
        )
    torch.testing.assert_close(term.predict(step_data), latent @ step_data.probes.T)


def test_curriculum_holds_order_zero_during_the_warm_up(step_data):
    term = _term(step_data, warmup_fraction=0.1).train()
    assert term.active_l_max == 2
    step = copy.copy(step_data)
    step.extras = {**step_data.extras, "step_fraction": 0.05}
    _, diag = term(step)
    assert float(diag[f"{NAME}/active_l_max"]) == 0
    assert f"{NAME}/l1/norm_ratio" not in diag
    assert f"{NAME}/l0_pseudo/norm_ratio" not in diag
    assert term.active_blocks() == [(0, 1)]
    # validation (step fraction 1) keeps the order of the training steps
    term.eval()
    _, diag = term(step_data)
    assert float(diag[f"{NAME}/active_l_max"]) == 0
    # no promotion during the warm-up
    for _ in range(3):
        term.maybe_promote({f"{NAME}/l2/norm_ratio": 0.5})
    assert term.active_l_max == 2
    term.train()
    step.extras = {**step_data.extras, "step_fraction": 0.1}
    _, diag = term(step)
    assert float(diag[f"{NAME}/active_l_max"]) == 2
    assert f"{NAME}/l2/norm_ratio" in diag and f"{NAME}/l3/norm_ratio" not in diag
    assert f"{NAME}/l0_pseudo/norm_ratio" in diag


def test_promotion_requires_three_epochs(step_data):
    term = _term(step_data)
    assert term.active_l_max == 2
    for _ in range(2):
        term.maybe_promote({f"{NAME}/l2/norm_ratio": torch.tensor(0.5)})
    assert term.active_l_max == 2
    term.maybe_promote({f"{NAME}/l2/norm_ratio": torch.tensor(0.1)})  # resets
    for _ in range(2):
        term.on_validation_epoch_end({f"{NAME}/l2/norm_ratio": 0.5})
    assert term.active_l_max == 2
    term.on_validation_epoch_end({f"{NAME}/l2/norm_ratio": 0.5})
    assert term.active_l_max == 3
    for _ in range(3):
        term.maybe_promote({f"{NAME}/l2/norm_ratio": 0.5})  # not the top order
    assert term.active_l_max == 3
    for _ in range(6):
        term.maybe_promote({f"{NAME}/l3/norm_ratio": 0.5})
        term.maybe_promote({f"{NAME}/l4/norm_ratio": 0.5})
    assert term.active_l_max == 4  # capped at l_max


def test_curriculum_state_is_saved_with_the_term(step_data):
    term = _term(step_data, warmup_fraction=0.1).train()
    step = copy.copy(step_data)
    step.extras = {**step_data.extras, "step_fraction": 0.5}
    term(step)
    for _ in range(3):
        term.maybe_promote({f"{NAME}/l2/norm_ratio": 0.5})
    term.maybe_promote({f"{NAME}/l3/norm_ratio": 0.5})
    state = term.state_dict()
    fresh = _term(step_data, warmup_fraction=0.1)
    fresh.load_state_dict(state)
    assert fresh.active_l_max == 3
    fresh.eval()
    _, diag = fresh(step_data)
    assert float(diag[f"{NAME}/active_l_max"]) == 3
    fresh.maybe_promote({f"{NAME}/l3/norm_ratio": 0.5})
    fresh.maybe_promote({f"{NAME}/l3/norm_ratio": 0.5})
    assert fresh.active_l_max == 4  # the streak of 1 was restored


def test_low_l_max_starts_at_l_max(step_data):
    assert _term(step_data, l_max=1).active_l_max == 1
    assert _term(step_data, l_max=0).active_l_max == 0


def test_no_pseudo_heads_without_inversions(step_data):
    term = _term(step_data, l_max=2, rotation_group="SO3")
    assert term.head_keys == [(0, 1), (1, 1), (2, 1)]


def test_pseudo_heads_with_inversions(step_data):
    term = _term(step_data, l_max=2)
    assert term.head_keys == [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    # the invariant JEPA has no (0, -) block
    assert _term(step_data, l_max=0).head_keys == [(0, 1)]


def test_order_stays_zero_without_student_rotations(step_data):
    term = _term(step_data, l_max=4, rotation_group="none")
    assert term.head_keys == [(0, 1)] and term.active_l_max == 0
    _, diag = term(step_data)
    assert float(diag[f"{NAME}/active_l_max"]) == 0
    for _ in range(3):
        term.maybe_promote({f"{NAME}/l0/norm_ratio": 0.9})
    assert term.active_l_max == 0


def test_norm_ratio_is_the_spread_ratio(step_data):
    term = _term(step_data, l_max=1).eval()
    _, diag = term(step_data)
    parts = term.contributions(step_data)
    target = term.teacher_frame_targets(step_data)

    def spread(x):
        return (x - x.mean(0)).norm()

    order_1 = parts[(1, 1)] + parts[(1, -1)]
    torch.testing.assert_close(
        diag[f"{NAME}/l1/norm_ratio"], spread(order_1) / spread(target)
    )
    torch.testing.assert_close(
        diag[f"{NAME}/l0/norm_ratio"], spread(parts[(0, 1)]) / spread(target)
    )
    torch.testing.assert_close(
        diag[f"{NAME}/l0_pseudo/norm_ratio"], spread(parts[(0, -1)]) / spread(target)
    )


def test_loss_is_weighted_normalized_mse_plus_floors(step_data):
    term = _term(step_data, floor_gamma=0.5).eval()
    # zero l > 0 contributions: every floor is active
    with torch.no_grad():
        for l, sigma in term.head_keys:
            if l > 0:
                last = term.heads[term.key(l, sigma)].node_net[-1]
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


def test_degenerate_batches_are_finite(tiny_step_data):
    term = _term(tiny_step_data)
    loss, diag = term(tiny_step_data)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in diag.values())


def _defaults():
    return copy.deepcopy(
        init_with_defaults(documentation.RotationConditionedLatentTermHypers)
    )


def test_from_hypers_takes_the_feature_dim_and_rotation_group_from_the_context():
    context = TermContext(
        dims=(24, 20),
        cutoff=3.5,
        atomic_types=[1, 8],
        num_probes=5,
        rotation_group="SO3",
        feature_dim=44,
    )
    hypers = _defaults()
    hypers["head"]["channel_fraction"] = 0.5
    hypers["rank"] = 7
    (term,) = build_terms({NAME: hypers}, context)
    assert term.l_max == 4 and term.active_l_max == 2
    assert all(sigma == 1 for _, sigma in term.head_keys)
    assert term.heads["l1p"].dn == 12 and term.heads["l1p"].n_out == 3 * 7
    assert term.maps["l1p"].weight.shape == (44, 7)
    assert term.context.cutoff == 3.5
    assert not term.needs_gradients and not term.uses_prediction
    assert term.required_targets == {}
    assert TERMS[NAME] is RotationConditionedLatentTerm
    context.feature_dim = None
    with pytest.raises(ValueError, match="feature_dim"):
        build_terms({NAME: hypers}, context)


def test_default_hypers_disable_the_term_and_validate_when_enabled():
    options = get_default_hypers("experimental.ssl_pretraining")
    terms = options["training"]["objective"]["terms"]
    assert terms[NAME] is None
    terms[NAME] = _defaults()
    assert terms[NAME]["l_max"] == 4
    assert terms[NAME]["promote_threshold"] == 0.2
    assert terms[NAME]["warmup_fraction"] == 0.05
    assert terms[NAME]["rank"] == 64
    check_architecture_options(name="experimental.ssl_pretraining", options=options)


def test_term_contract():
    """Alone: finite on degenerate batches, no warnings, trains the student only."""
    frames = [
        ase.Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ase.io.read(CARBON_PATH, 0),
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
        cutoff=CUTOFF,
        atomic_types=ATOMIC_TYPES,
        num_probes=2,
        feature_dim=student.num_readout_layers * (student.d_node + student.d_pet),
    )
    (term,) = build_terms({NAME: _defaults()}, context)
    term = term.to(torch.float64)
    for batch in [frames[:1], frames[:2], frames]:
        systems, options, hypers = make_batch(batch, torch.float64)
        hypers["views"]["builder"] = "high_order"
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
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            assert all(p.grad is None for p in teacher.parameters())
        # the multi-atom batches reach the student
        if len(batch) > 1:
            assert any(p.grad is not None for p in student.parameters())
