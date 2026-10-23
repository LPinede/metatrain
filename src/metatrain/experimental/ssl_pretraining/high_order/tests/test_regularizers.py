import copy
import warnings

import ase
import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining.documentation import TermsHypers
from metatrain.experimental.ssl_pretraining.high_order import documentation
from metatrain.experimental.ssl_pretraining.high_order.irreps import (
    back_map,
    irrep_matrix,
)
from metatrain.experimental.ssl_pretraining.high_order.terms import (
    EquivariantVicregTerm,
    RotationAveragedVicregTerm,
    equivariant_vicreg_block,
    vicreg_on_average,
)
from metatrain.experimental.ssl_pretraining.objective import (
    grad_jepa_loss,
    vicreg_terms,
)
from metatrain.experimental.ssl_pretraining.regularizers import SigregTerm
from metatrain.experimental.ssl_pretraining.terms import (
    TERMS,
    TermContext,
    build_terms,
)
from metatrain.experimental.ssl_pretraining.tests import CARBON_PATH, small_pet
from metatrain.experimental.ssl_pretraining.trainer import check_hypers
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.pydantic import MetatrainValidationError

from . import ATOMIC_TYPES, haar_o3, make_batch


NAME = "experimental.ssl_pretraining"
TWO_ROTATION_TERMS = ["equivariant_vicreg", "vicreg_averaged"]


def _pooled(tokens):
    return torch.cat([tokens.node, (tokens.edge * tokens.cutoff[..., None]).sum(1)], -1)


def test_math_single_view_variance_is_fooled_by_invariant_outputs():
    """Documents why the term needs two rotations: a constant output, back-mapped by
    random rotations, already has a large spread over the batch."""
    torch.manual_seed(0)
    a = torch.randn(1, 5, 1, dtype=torch.float64).expand(4000, 5, 1)
    z = back_map(a, irrep_matrix(haar_o3(4000), 2, 1))
    assert z.std(dim=0).mean() > 0.5 * a[0].norm() / 5**0.5


def test_two_view_loss_separates_equivariant_from_invariant():
    torch.manual_seed(0)
    n = 4000
    d1, d2 = (irrep_matrix(haar_o3(n), 2, 1) for _ in range(2))
    f = torch.randn(n, 5, 4, dtype=torch.float64)  # teacher-frame content per atom
    # an equivariant student outputs D f in its frame, back-mapped to f
    equivariant = (
        back_map(back_map(f, d1.transpose(1, 2)), d1),
        back_map(back_map(f, d2.transpose(1, 2)), d2),
    )
    a = torch.randn(n, 5, 4, dtype=torch.float64)  # same output for both views
    invariant = (back_map(a, d1), back_map(a, d2))
    agreement_eq = equivariant_vicreg_block(*equivariant)[0]
    agreement_inv = equivariant_vicreg_block(*invariant)[0]
    assert agreement_eq < 1e-10
    assert agreement_inv == pytest.approx(2.0, abs=0.1)


def test_variance_hinge_is_scale_free_and_vanishes_for_spread_outputs():
    torch.manual_seed(0)
    z = torch.randn(512, 5, 4, dtype=torch.float64)
    _, hinge, _ = equivariant_vicreg_block(z, z)
    _, hinge_scaled, _ = equivariant_vicreg_block(1e3 * z, 1e3 * z)
    assert float(hinge) == pytest.approx(0.0, abs=1e-3)
    torch.testing.assert_close(hinge, hinge_scaled)
    # one direction only: most components are collapsed
    collapsed = torch.zeros_like(z)
    collapsed[:, 0, 0] = torch.randn(512, dtype=torch.float64)
    assert equivariant_vicreg_block(collapsed, collapsed)[1] > 0.4


def test_block_covariance_is_rotation_invariant_and_sees_correlated_channels():
    torch.manual_seed(0)
    z = torch.randn(256, 5, 4, dtype=torch.float64)
    r = irrep_matrix(haar_o3(1).expand(256, 3, 3), 2, 1)
    rotated = back_map(z, r.transpose(1, 2))  # same rotation for every atom
    torch.testing.assert_close(
        equivariant_vicreg_block(z, z)[2],
        equivariant_vicreg_block(rotated, rotated)[2],
    )
    copies = z.clone()
    copies[..., 1] = copies[..., 0]
    assert (
        equivariant_vicreg_block(copies, copies)[2]
        > 10 * (equivariant_vicreg_block(z, z)[2])
    )


def test_projector_is_linear_so_it_cannot_create_orders():
    term = EquivariantVicregTerm((8, 6), l_max=2, channels=4)
    assert set(term.proj) == {"l1p", "l1m", "l2p", "l2m"}
    for proj in term.proj.values():
        assert isinstance(proj, torch.nn.Linear)
        assert proj.in_features == 14
    assert term.proj["l2p"].out_features == 5 * 4
    proper = EquivariantVicregTerm((8, 6), l_max=3, rotation_group="SO3")
    assert set(proper.proj) == {"l1p", "l2p", "l3p"}


def test_ramp_is_zero_in_warmup_and_one_after():
    term = EquivariantVicregTerm((8, 6), warmup_fraction=0.05)
    assert term.ramp(0.01) == 0.0
    assert term.ramp(0.075) == pytest.approx(0.5)
    assert term.ramp(0.5) == 1.0
    assert EquivariantVicregTerm((8, 6), warmup_fraction=0.0).ramp(0.0) == 1.0


def test_term_runs_on_two_rotations(step_data_two_rotations):
    step = step_data_two_rotations
    term = EquivariantVicregTerm(step.extras["student_tokens"].dims)
    assert not term.needs_gradients and not term.uses_prediction
    step.extras["step_fraction"] = 1.0
    loss, diag = term(step)
    assert torch.isfinite(loss) and loss.requires_grad
    assert "equivariant_vicreg/l2p/agreement" in diag
    assert all(torch.isfinite(v) for v in diag.values())
    # off during the warm-up, still in the graph
    step.extras["step_fraction"] = 0.0
    loss, _ = term(step)
    assert float(loss.detach()) == 0.0 and loss.requires_grad


def test_term_back_maps_each_view_with_its_own_rotation(step_data_two_rotations):
    """With the projected outputs replaced by the student-frame image of a fixed
    teacher-frame block, the two back-mapped views agree exactly."""
    step = step_data_two_rotations
    ho = step.extras["high_order"]
    visible = ~step.views.masked
    term = EquivariantVicregTerm(step.extras["student_tokens"].dims, l_max=1)
    z1, z2 = term.back_mapped(step, 1, -1)
    assert z1.shape == z2.shape == (int(visible.sum()), 3, term.channels)
    # the projector is linear, so z depends on the rotation only through D^T
    p2 = term.proj["l1m"](_pooled(step.extras["student_tokens_extra"]["rot2"]))
    expected = back_map(
        p2[visible].reshape(-1, 3, term.channels),
        irrep_matrix(ho.rotations2[visible], 1, -1),
    )
    torch.testing.assert_close(z2, expected)


def test_two_rotation_terms_need_the_second_rotation(step_data):
    for term in (EquivariantVicregTerm((8, 6)), RotationAveragedVicregTerm()):
        with pytest.raises(ValueError, match="second_rotation"):
            term(step_data)


def test_math_vicreg_averaged_sees_only_the_shared_part():
    torch.manual_seed(0)
    shared, orient = torch.randn(256, 16), torch.randn(256, 16)
    out = vicreg_on_average(shared + orient, shared - orient)  # orientation cancels
    ref = vicreg_terms(shared)
    assert torch.allclose(out[0], ref[0]) and torch.allclose(out[1], ref[1])


def test_vicreg_averaged_reads_the_rows_of_vicreg(step_data_two_rotations):
    step = step_data_two_rotations
    rows = ~step.views.masked
    f1 = step.student_features[rows]
    f2 = _pooled(step.extras["student_tokens_extra"]["rot2"].select(rows))
    variance, covariance = vicreg_terms(0.5 * (f1 + f2))
    term = RotationAveragedVicregTerm(variance=2.0, covariance=0.5)
    assert not term.needs_gradients and not term.uses_prediction
    loss, diag = term(step)
    torch.testing.assert_close(loss, 2.0 * variance + 0.5 * covariance)
    torch.testing.assert_close(diag["vicreg_averaged/variance"], variance.detach())
    torch.testing.assert_close(diag["vicreg_averaged/covariance"], covariance.detach())


def test_sigreg_term_skips_tiny_batches(tiny_step_data):
    loss, _ = SigregTerm(1.0)(tiny_step_data)
    assert torch.isfinite(loss)


def _term_defaults():
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    return copy.deepcopy(
        {
            "equivariant_vicreg": init_with_defaults(
                documentation.EquivariantVicregTermHypers
            ),
            "vicreg_averaged": init_with_defaults(
                documentation.RotationAveragedVicregTermHypers
            ),
        }
    )


def test_terms_are_registered_with_their_hypers():
    for name in TWO_ROTATION_TERMS:
        assert TERMS[name].name == name
        assert name in TermsHypers.__annotations__
    options = get_default_hypers(NAME)
    terms = options["training"]["objective"]["terms"]
    assert all(terms[name] is None for name in TWO_ROTATION_TERMS)
    terms.update(_term_defaults())
    options["training"]["high_order"]["second_rotation"] = True
    check_architecture_options(name=NAME, options=options)
    for key, value in [("l_max", 0), ("channels", 0), ("rotation_group", "none")]:
        bad = copy.deepcopy(options)
        bad["training"]["objective"]["terms"]["equivariant_vicreg"][key] = value
        with pytest.raises(MetatrainValidationError, match=key):
            check_architecture_options(name=NAME, options=bad)


def test_from_hypers_takes_dims_from_the_context():
    context = TermContext(dims=(24, 20), cutoff=3.5, atomic_types=[1, 8], num_probes=2)
    hypers = _term_defaults()
    hypers["equivariant_vicreg"].update(l_max=1, channels=3, warmup_fraction=0.1)
    equivariant, averaged = build_terms(hypers, context)
    assert equivariant.proj["l1p"].in_features == 44
    assert equivariant.proj["l1p"].out_features == 9
    assert equivariant.warmup_fraction == 0.1
    assert averaged.variance == 1.0 and averaged.covariance == 0.04


@pytest.mark.parametrize("name", TWO_ROTATION_TERMS)
def test_check_hypers_requires_the_second_rotation(name, small_batch):
    _, _, hypers = small_batch
    hypers = copy.deepcopy(hypers)
    hypers["objective"]["terms"][name] = _term_defaults()[name]
    hypers["views"]["builder"] = "high_order"
    with pytest.raises(ValueError, match="second_rotation"):
        check_hypers(hypers, 4.0)
    hypers["high_order"]["second_rotation"] = True
    check_hypers(hypers, 4.0)
    hypers["views"]["builder"] = "grad_jepa"
    with pytest.raises(ValueError, match="high_order"):
        check_hypers(hypers, 4.0)


@pytest.mark.parametrize("name", TWO_ROTATION_TERMS)
def test_two_rotation_term_contract(name):
    """Each term alone: finite on degenerate batches, no warnings, trains the
    student (and the projector) only."""
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
        cutoff=4.0,
        atomic_types=ATOMIC_TYPES,
        num_probes=2,
    )
    (term,) = build_terms({name: _term_defaults()[name]}, context)
    if isinstance(term, torch.nn.Module):
        term = term.to(torch.float64)
    for batch in [frames[:1], frames[:2], frames]:
        systems, options, hypers = make_batch(batch, torch.float64)
        hypers["views"]["builder"] = "high_order"
        hypers["high_order"]["second_rotation"] = True
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
                    step_fraction=1.0,
                )
                loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            assert any(p.grad is not None for p in student.parameters())
            assert all(p.grad is None for p in teacher.parameters())
