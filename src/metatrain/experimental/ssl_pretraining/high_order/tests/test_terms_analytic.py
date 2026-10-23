import copy
import dataclasses
import warnings

import ase
import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining.documentation import TermsHypers
from metatrain.experimental.ssl_pretraining.high_order import documentation
from metatrain.experimental.ssl_pretraining.high_order.irreps import irrep_matrix
from metatrain.experimental.ssl_pretraining.high_order.terms import (
    ChiralityTerm,
    DeformationJetTerm,
    ResidualMultipoleTerm,
    jet_targets,
)
from metatrain.experimental.ssl_pretraining.objective import grad_jepa_loss
from metatrain.experimental.ssl_pretraining.terms import (
    TERMS,
    TermContext,
    build_terms,
)
from metatrain.experimental.ssl_pretraining.tests import (
    CARBON_PATH,
    small_pet,
)
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.pydantic import MetatrainValidationError

from . import (
    ATOMIC_TYPES,
    haar_o3,
    make_batch,
    make_step,
    mirrored,
    with_hidden_atom_moved,
)


HIGH_ORDER_TERMS = ["residual_multipole", "deformation_jets", "chirality"]
CUTOFF = 4.0
N_SPECIES = len(ATOMIC_TYPES)


def _dims(step):
    return step.extras["student_tokens"].dims


def _multipole(step, l_max=2, n_radial=4, floor_gamma=0.1):
    return ResidualMultipoleTerm(
        _dims(step), l_max, n_radial, N_SPECIES, CUTOFF, 32, 1.0, floor_gamma
    )


def _jets(step):
    return DeformationJetTerm(_dims(step), 32, 1.0)


def _chirality(step, cutoff=CUTOFF):
    return ChiralityTerm(_dims(step), 32, 1.0, (1, 6, 8), cutoff)


def _all_terms(step):
    return [_multipole(step), _jets(step), _chirality(step)]


def _neighbor_counts(ho, cutoff):
    close = ho.clean_vectors.norm(dim=-1) < cutoff
    return torch.bincount(ho.pair_index[close, 0], minlength=ho.boundary.numel())


def test_fixtures_cover_both_branches(step_data):
    ho = step_data.extras["high_order"]
    assert ho.branch_deformed.any() and ho.branch_masked.any()
    assert ho.boundary.sum() >= 2


def test_terms_return_finite_scalar_and_diagnostics(step_data):
    for term in _all_terms(step_data):
        loss, diag = term(step_data)
        assert loss.dim() == 0 and torch.isfinite(loss)
        assert loss.requires_grad
        assert f"{term.name}/loss" in diag
        assert all(torch.isfinite(v) for v in diag.values())
        assert all(k.startswith(term.name + "/") for k in diag)
    _, diag = _multipole(step_data)(step_data)
    for l in range(3):
        assert f"residual_multipole/l{l}/rmse" in diag
        assert f"residual_multipole/l{l}/norm_ratio" in diag
    _, diag = _jets(step_data)(step_data)
    for block in ("displacement", "dilation", "rotation", "shear"):
        assert f"deformation_jets/{block}/rmse" in diag
    _, diag = _chirality(step_data)(step_data)
    assert "chirality/l0/norm_ratio" in diag


def test_block_shapes_and_sites(step_data):
    ho = step_data.extras["high_order"]
    n_boundary = int(ho.boundary.sum())
    term = _multipole(step_data, l_max=2, n_radial=4)
    preds, targets = term.predict(step_data), term.teacher_frame_targets(step_data)
    assert list(preds) == list(targets) == ["l0", "l1", "l2"]
    for l in range(3):
        shape = (n_boundary, 2 * l + 1, 4 * N_SPECIES)
        assert preds[f"l{l}"].shape == targets[f"l{l}"].shape == shape

    # B: every atom of the deformed structures (all visible)
    n_deformed = int(ho.branch_deformed[ho.structure_index].sum())
    term = _jets(step_data)
    preds, targets = term.predict(step_data), term.teacher_frame_targets(step_data)
    sizes = {"displacement": 3, "dilation": 1, "rotation": 3, "shear": 5}
    assert list(preds) == list(targets) == list(sizes)
    for block, size in sizes.items():
        assert preds[block].shape == targets[block].shape == (n_deformed, size, 1)

    # chirality: visible atoms with at least 3 neighbours within the cutoff
    sites = (ho.boundary | ho.context) & (_neighbor_counts(ho, CUTOFF) >= 3)
    term = _chirality(step_data)
    preds, targets = term.predict(step_data), term.teacher_frame_targets(step_data)
    assert preds["l0"].shape == targets["l0"].shape == (int(sites.sum()), 1, 1)
    assert targets["l0"].abs().max() > 0


def test_residual_targets_are_species_resolved_hidden_multipoles(step_data_masked):
    """Mask-only branch: the target of a boundary atom is the multipole of its
    hidden neighbours, one channel per element (through ``species_index``)."""
    ho = step_data_masked.extras["high_order"]
    n_radial = 4
    target = _multipole(step_data_masked, l_max=0, n_radial=n_radial)
    target = target.teacher_frame_targets(step_data_masked)["l0"]
    target = target.reshape(-1, n_radial, N_SPECIES)
    for row, atom in enumerate(ho.boundary.nonzero().flatten().tolist()):
        pairs = (ho.pair_index[:, 0] == atom) & ho.hidden_source
        hidden = set(ho.species_index[ho.pair_index[pairs, 1]].tolist())
        assert hidden
        for channel in range(N_SPECIES):
            norm = target[row, :, channel].abs().sum()
            assert (norm > 0) if channel in hidden else (norm == 0)


def test_chirality_needs_three_neighbours_within_the_cutoff(step_data):
    ho = step_data.extras["high_order"]
    visible = ho.boundary | ho.context
    for cutoff in (CUTOFF, 2.0):
        sites = visible & (_neighbor_counts(ho, cutoff) >= 3)
        target = _chirality(step_data, cutoff).teacher_frame_targets(step_data)
        assert target["l0"].shape[0] == int(sites.sum())
    # the filter acts at the shorter cutoff
    assert (visible & (_neighbor_counts(ho, 2.0) < 3)).any()


def test_anchor_does_not_see_hidden_positions(medium_batch, pet_core, small_teacher):
    """Moving a masked atom of the clean structure and re-running the student (same
    random stream) changes the target but not the prediction: the student never
    sees the masked atom."""
    step_a = make_step(medium_batch, pet_core, small_teacher, seed=1, p_def=0.0)
    shift = torch.tensor([0.1, -0.05, 0.08])
    moved = with_hidden_atom_moved(medium_batch, step_a, shift)
    step_b = make_step(moved, pet_core, small_teacher, seed=1, p_def=0.0)
    ho_a, ho_b = step_a.extras["high_order"], step_b.extras["high_order"]
    # the mask sampler reads distances: this seed and shift keep the same roles
    assert torch.equal(step_a.views.masked, step_b.views.masked)
    assert torch.equal(ho_a.boundary, ho_b.boundary)

    term = _multipole(step_a)
    pred_a, pred_b = term.predict(step_a), term.predict(step_b)
    target_a = term.teacher_frame_targets(step_a)
    target_b = term.teacher_frame_targets(step_b)
    for block in pred_a:
        torch.testing.assert_close(pred_a[block], pred_b[block])
    assert not torch.allclose(target_a["l1"], target_b["l1"])


def test_parity_is_carried_by_the_back_map_not_the_target(
    medium_batch, pet_core, small_teacher, step_data_deformed
):
    """Teacher-frame targets do not depend on the student rotation; with the same
    head outputs, g -> -g flips the proper l=1 prediction and leaves the pseudo
    (1,-) one unchanged (likewise (0,-) against (0,+))."""
    step = step_data_deformed
    flipped = mirrored(medium_batch, pet_core, small_teacher, seed=0, p_def=1.0)
    ho, ho_flipped = step.extras["high_order"], flipped.extras["high_order"]
    assert torch.equal(ho_flipped.rotations, -ho.rotations)
    term = _jets(step)
    targets, targets_flipped = (
        term.teacher_frame_targets(step),
        term.teacher_frame_targets(flipped),
    )
    for block in targets:
        torch.testing.assert_close(targets[block], targets_flipped[block])

    # same tokens, rotations negated: only the back-map changes
    negated = copy.copy(step)
    negated.extras = {
        **step.extras,
        "high_order": dataclasses.replace(ho, rotations=-ho.rotations),
    }
    pred, pred_neg = term.predict(step), term.predict(negated)
    torch.testing.assert_close(pred_neg["displacement"], -pred["displacement"])
    torch.testing.assert_close(pred_neg["rotation"], pred["rotation"])
    torch.testing.assert_close(pred_neg["dilation"], pred["dilation"])
    torch.testing.assert_close(pred_neg["shear"], pred["shear"])
    assert pred["rotation"].abs().max() > 0

    chirality = _chirality(step)
    kappa, kappa_neg = chirality.predict(step), chirality.predict(negated)
    torch.testing.assert_close(kappa_neg["l0"], -kappa["l0"])
    assert kappa["l0"].abs().max() > 0


def test_jet_targets_transform_as_their_blocks():
    """The (l, sigma) of each block (used by the back-map) is the transformation
    law of its target: tau(h u, h G h^T) = D^(l,sigma)(h) tau(u, G)."""
    torch.manual_seed(0)
    u = torch.randn(6, 3, dtype=torch.float64)
    grad = torch.randn(6, 3, 3, dtype=torch.float64)
    h = haar_o3(1).expand(6, 3, 3)
    h = h * torch.linalg.det(h).sign().reshape(-1, 1, 1) * -1  # improper
    targets = jet_targets(u, grad)
    rotated = jet_targets(
        torch.einsum("nab,nb->na", h, u), h @ grad @ h.transpose(-1, -2)
    )
    for block, (l, sigma) in DeformationJetTerm.BLOCKS.items():
        expected = torch.einsum(
            "nmp,npo->nmo", irrep_matrix(h, l, sigma), targets[block]
        )
        torch.testing.assert_close(rotated[block], expected)


def test_loss_is_weighted_mean_of_normalized_block_losses_plus_floor(step_data):
    """Zero predictions: each block costs mean(t^2) / var(t) (fresh EMA), and every
    l > 0 block pays the full floor 0.1 (A only)."""
    gamma, weight, l_max = 0.5, 2.0, 2
    term = ResidualMultipoleTerm(
        _dims(step_data), l_max, 4, N_SPECIES, CUTOFF, 32, weight, gamma
    )
    for head in term.heads.values():
        torch.nn.init.zeros_(head.node_net[-1].weight)
        torch.nn.init.zeros_(head.node_net[-1].bias)
    targets = term.teacher_frame_targets(step_data)
    blocks = torch.stack(
        [t.pow(2).mean() / t.var(unbiased=False) for t in targets.values()]
    )
    loss, _ = term(step_data)
    expected = weight * (blocks.mean() + gamma * 0.1 * l_max)
    torch.testing.assert_close(loss.detach(), expected.to(loss))


def test_degenerate_batches_are_finite(tiny_step_data):
    for term in _all_terms(tiny_step_data):
        loss, diag = term(tiny_step_data)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(v) for v in diag.values())


def test_terms_are_registered_after_the_grad_jepa_terms():
    assert list(TERMS)[-3:] == HIGH_ORDER_TERMS
    for name in HIGH_ORDER_TERMS:
        assert TERMS[name].name == name
        assert name in TermsHypers.__annotations__


def test_default_hypers_disable_the_terms_and_validate_when_enabled():
    options = get_default_hypers("experimental.ssl_pretraining")
    terms = options["training"]["objective"]["terms"]
    assert all(terms[name] is None for name in HIGH_ORDER_TERMS)
    terms.update(_term_defaults())
    terms["residual_multipole"]["l_max"] = 2
    terms["residual_multipole"]["head"]["use_edges"] = True
    terms["chirality"]["channel_types"] = [1, 6, 7]
    check_architecture_options(name="experimental.ssl_pretraining", options=options)
    terms["chirality"]["channel_types"] = [1, 6, 6]
    with pytest.raises(MetatrainValidationError, match="channel_types"):
        check_architecture_options(name="experimental.ssl_pretraining", options=options)


def _term_defaults():
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    return copy.deepcopy(
        {
            "residual_multipole": init_with_defaults(
                documentation.ResidualMultipoleTermHypers
            ),
            "deformation_jets": init_with_defaults(
                documentation.DeformationJetsTermHypers
            ),
            "chirality": init_with_defaults(documentation.ChiralityTermHypers),
        }
    )


def test_from_hypers_takes_dims_cutoff_and_species_from_the_context():
    context = TermContext(dims=(24, 20), cutoff=3.5, atomic_types=[1, 8], num_probes=2)
    hypers = _term_defaults()
    hypers["residual_multipole"]["head"]["channel_fraction"] = 0.5
    multipole, jets, chirality = build_terms(hypers, context)
    assert multipole.cutoff == chirality.cutoff == 3.5
    assert multipole.n_species == 2
    assert multipole.heads["l0"].dn == 12 and multipole.heads["l0"].de == 10
    assert jets.heads["shear"].dn == 24
    assert chirality.channel_types == (1, 6, 8)
    for term in (multipole, jets, chirality):
        assert not term.needs_gradients and not term.uses_prediction
        assert term.required_targets == {}


@pytest.mark.parametrize("name", HIGH_ORDER_TERMS)
def test_high_order_term_contract(name):
    """Each term alone: finite on degenerate batches, no warnings, trains the
    student only."""
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
    (term,) = build_terms({name: _term_defaults()[name]}, context)
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
                )
                loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            assert any(p.grad is not None for p in student.parameters())
            assert all(p.grad is None for p in teacher.parameters())


@pytest.fixture
def achiral_step_data(pet_core, small_teacher):
    """Two dimers and a planar formic acid: kappa is identically zero."""
    frames = [
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ase.Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]),
        ase.Atoms(
            "HCOOH",
            positions=[
                [-0.95, 0.55, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -1.2, 0.0],
                [1.15, 0.7, 0.0],
                [1.9, 0.1, 0.0],
            ],
        ),
    ]
    return make_step(make_batch(frames, torch.float32), pet_core, small_teacher)


def test_blocks_without_target_variance_are_skipped(achiral_step_data, step_data):
    """A zero-variance block costs nothing, reports ``skipped`` and never touches
    the EMA of its target variance."""
    term = _chirality(achiral_step_data)
    target = term.teacher_frame_targets(achiral_step_data)["l0"]
    assert target.numel() >= 2 and target.abs().max() == 0
    loss, diag = term(achiral_step_data)
    assert loss.requires_grad and float(loss.detach()) == 0.0
    assert float(diag["chirality/l0/skipped"]) == 1.0
    assert term.norm.var == {}

    # a stored EMA is not updated by a zero-variance batch
    _, diag = term(step_data)
    assert float(diag["chirality/l0/skipped"]) == 0.0
    stored = dict(term.norm.var)
    term(achiral_step_data)
    assert term.norm.var == stored


def test_isolated_deformed_atom_is_skipped(pet_core, small_teacher):
    """One deformed atom: every block has a single row, so no batch variance."""
    batch = make_batch([ase.Atoms("H", positions=[[0.0, 0.0, 0.0]])], torch.float32)
    step = make_step(batch, pet_core, small_teacher, p_def=1.0)
    term = _jets(step)
    loss, diag = term(step)
    assert float(loss.detach()) == 0.0
    assert all(float(diag[f"deformation_jets/{b}/skipped"]) == 1.0 for b in term.BLOCKS)
    assert term.norm.var == {}
