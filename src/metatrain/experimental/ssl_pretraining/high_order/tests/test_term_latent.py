import copy
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
    wigner_d,
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

from . import ATOMIC_TYPES, haar_o3, make_batch, make_step


NAME = "rotation_conditioned_latent"
CUTOFF = 4.0


def _term(step, l_max=4, **kwargs):
    kwargs.setdefault("warmup_fraction", 0.0)
    return RotationConditionedLatentTerm(
        step.extras["student_tokens"].dims,
        l_max,
        step.probes.shape[0],
        32,
        CUTOFF,
        1.0,
        **kwargs,
    )


def test_l0_is_plain_invariant_latent(step_data):
    term = _term(step_data, l_max=0)
    assert term.head_keys == [(0, 1)]
    loss, diag = term(step_data)
    assert torch.isfinite(loss) and f"{NAME}/total/rmse" in diag
    # the prediction is the l=0 head on the head inputs, whatever the rotations
    pred = term.predict(step_data)
    atoms = step_data.views.latent_targets
    inputs = term.head_inputs(step_data).select(atoms)
    expected = term.heads["l0p"](inputs)[:, 0]
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


def test_math_orbit_coefficient_l1():
    """Least squares through ``<D(g), Phi>`` with an invariant student feature
    recovers the l=1 content of a target that rotates as a vector."""
    torch.manual_seed(0)
    n = 20000
    rots = haar_o3(n)
    target_dir = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    target = torch.einsum("nba,b->na", rots, target_dir)[:, 0]  # (g^T t)_x
    d = wigner_d(rots, 1).reshape(n, 9)
    coef = torch.linalg.lstsq(d, target[:, None]).solution
    pred = d @ coef
    assert ((pred[:, 0] - target) ** 2).mean() < 1e-10


def _synthetic_step(n, num_probes, sigma, seed=0):
    """Invariant student tokens (identical for every atom), Haar rotations, and a
    teacher feature whose probe contraction is ``c_p det(g)^((1-sigma)/2) (g^T a)_x``:
    l=1 content of parity ``sigma``, no l=0 content."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    rotations = haar_o3(n)
    a = torch.randn(3, generator=gen, dtype=torch.float64)
    signal = torch.einsum("nba,b->na", rotations, a)[:, 0]
    if sigma == -1:
        signal = signal * torch.linalg.det(rotations)
    dim = 6
    w = torch.randn(dim, generator=gen, dtype=torch.float64)
    probes = (torch.randint(0, 2, (num_probes, dim), generator=gen) * 2 - 1).to(w)
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
        probes=probes,
        student_features=torch.zeros(n, dim, dtype=torch.float64),
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


def _fit(term, step, n_steps=300):
    term = term.to(torch.float64).train()
    optimizer = torch.optim.Adam(term.parameters(), lr=1e-2)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss, _ = term(step)
        loss.backward()
        optimizer.step()
    term.eval()
    with torch.no_grad():
        _, diag = term(step)
    return float(diag[f"{NAME}/total/rmse"]) ** 2 / float(
        term.teacher_frame_targets(step).var()
    )


@pytest.mark.parametrize("sigma", [1, -1])
def test_rotation_conditioned_heads_capture_the_l1_content(sigma):
    step = _synthetic_step(2000, 3, sigma)

    def term(l_max, group):
        torch.manual_seed(0)
        return RotationConditionedLatentTerm(
            (4, 4), l_max, 3, 16, CUTOFF, 1.0, rotation_group=group, floor_gamma=0.0
        )

    # the invariant JEPA cannot predict orientation content (C is neutral)
    assert _fit(term(0, "O3"), step) > 0.9
    # order 1 of the right parity captures it
    assert _fit(term(1, "O3"), step) < 0.05
    if sigma == -1:
        # without the pseudo heads, the pseudo content stays unexplained
        assert _fit(term(1, "SO3"), step) > 0.9


def test_prediction_contracts_the_rotation_with_the_head_output(step_data):
    term = _term(step_data, l_max=2)
    atoms = step_data.views.latent_targets
    inputs = term.head_inputs(step_data).select(atoms)
    rotations = step_data.extras["high_order"].rotations[atoms]
    p = step_data.probes.shape[0]
    expected = 0.0
    for l, sigma in term.head_keys:
        phi = term.heads[term.key(l, sigma)](inputs)
        phi = phi.reshape(-1, 2 * l + 1, 2 * l + 1, p)
        expected = expected + torch.einsum(
            "nmq,nmqp->np", irrep_matrix(rotations, l, sigma), phi
        )
    torch.testing.assert_close(term.predict(step_data), expected)


def test_curriculum_holds_order_zero_during_the_warm_up(step_data):
    term = _term(step_data, warmup_fraction=0.1).train()
    assert term.active_l_max == 2
    step = copy.copy(step_data)
    step.extras = {**step_data.extras, "step_fraction": 0.05}
    _, diag = term(step)
    assert float(diag[f"{NAME}/active_l_max"]) == 0
    assert f"{NAME}/l1/norm_ratio" not in diag
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


@pytest.mark.parametrize("group", ["SO3", "none"])
def test_no_pseudo_heads_without_inversions(step_data, group):
    term = _term(step_data, l_max=2, rotation_group=group)
    assert all(sigma == 1 for (_, sigma) in term.head_keys)
    assert [l for l, _ in term.head_keys] == [0, 1, 2]


def test_pseudo_heads_with_inversions(step_data):
    term = _term(step_data, l_max=2)
    assert term.head_keys == [(0, 1), (1, 1), (1, -1), (2, 1), (2, -1)]


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


def test_from_hypers_takes_probes_and_rotation_group_from_the_context():
    context = TermContext(
        dims=(24, 20),
        cutoff=3.5,
        atomic_types=[1, 8],
        num_probes=5,
        rotation_group="SO3",
    )
    hypers = _defaults()
    hypers["head"]["channel_fraction"] = 0.5
    (term,) = build_terms({NAME: hypers}, context)
    assert term.num_probes == 5 and term.l_max == 4 and term.active_l_max == 2
    assert all(sigma == 1 for _, sigma in term.head_keys)
    assert term.heads["l1p"].dn == 12 and term.heads["l1p"].n_out == 3 * 5
    assert term.context.cutoff == 3.5
    assert not term.needs_gradients and not term.uses_prediction
    assert term.required_targets == {}
    assert TERMS[NAME] is RotationConditionedLatentTerm


def test_default_hypers_disable_the_term_and_validate_when_enabled():
    options = get_default_hypers("experimental.ssl_pretraining")
    terms = options["training"]["objective"]["terms"]
    assert terms[NAME] is None
    terms[NAME] = _defaults()
    assert terms[NAME]["l_max"] == 4
    assert terms[NAME]["promote_threshold"] == 0.2
    assert terms[NAME]["warmup_fraction"] == 0.05
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
