import copy
import warnings

import ase
import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining.documentation import TrainerHypers
from metatrain.experimental.ssl_pretraining.objective import (
    FEATURE_OUTPUT,
    build_views,
    grad_jepa_loss,
    random_scalar_gradients,
    relative_gradient_spread,
    vicreg_terms,
)
from metatrain.experimental.ssl_pretraining.predictor import GradJEPAPredictor
from metatrain.experimental.ssl_pretraining.terms import build_terms
from metatrain.experimental.ssl_pretraining.views import (
    build_view,
    get_skin_neighbor_list_transform,
)
from metatrain.utils.hypers import init_with_defaults

from . import CARBON_PATH, QM9_PATH, small_pet, to_systems


def _setup(frames, dtype=torch.float32, species_query=False):
    torch.manual_seed(0)
    student = small_pet([1, 6, 7, 8], dtype).train()
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    options = student.requested_neighbor_lists()[0]
    transform = get_skin_neighbor_list_transform(options, skin=0.5)
    systems, _, _ = transform(to_systems(frames, dtype), {}, {})
    feature_dim = student.num_readout_layers * (student.d_node + student.d_pet)
    predictor = GradJEPAPredictor(
        feature_dim, 32, 8, 8, options.cutoff, [1, 6, 7, 8], species_query
    ).to(dtype)
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    hypers = copy.deepcopy(init_with_defaults(TrainerHypers))
    return student, teacher, predictor, systems, options, hypers


def _terms(hypers):
    return build_terms(hypers["objective"]["terms"])


def test_feature_dim_formula():
    student, _, _, systems, _, _ = _setup(ase.io.read(QM9_PATH, ":2"))
    features = student(systems, FEATURE_OUTPUT)["feature"].block().values
    assert features.shape[1] == student.num_readout_layers * (
        student.d_node + student.d_pet
    )


def test_random_scalar_gradients():
    student, _, _, systems, _, _ = _setup(ase.io.read(QM9_PATH, ":3"))
    positions = [s.positions.detach().requires_grad_(True) for s in systems]
    options = student.requested_neighbor_lists()[0]
    views = [
        build_view(s, options, p, s.cell)
        for s, p in zip(systems, positions, strict=True)
    ]
    features = student(views, FEATURE_OUTPUT)["feature"].block().values
    probes = torch.randn(2, features.shape[1])
    expected = torch.autograd.grad(
        (features @ probes[1]).sum(), positions, retain_graph=True
    )
    gradients = random_scalar_gradients(features, positions, probes, create_graph=False)
    torch.testing.assert_close(gradients[1], torch.cat(expected))
    # translation invariance: per-system gradients sum to zero
    start = 0
    for system in systems:
        end = start + len(system)
        torch.testing.assert_close(
            gradients[:, start:end].sum(dim=1), torch.zeros(2, 3), atol=1e-5, rtol=0
        )
        start = end


def test_loss_backward_reaches_student_and_predictor_only():
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":6")
    )
    hypers["mask"]["probability"] = 1.0
    loss, metrics = grad_jepa_loss(
        student,
        teacher,
        predictor,
        _terms(hypers),
        systems,
        options,
        hypers,
        torch.Generator(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["masked_fraction"] > 0
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is not None for p in student.parameters())
    assert any(p.grad is not None for p in predictor.parameters())


def test_loss_is_deterministic_given_generator():
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":6")
    )
    losses = [
        grad_jepa_loss(
            student,
            teacher,
            predictor,
            _terms(hypers),
            systems,
            options,
            hypers,
            torch.Generator().manual_seed(3),
        )[0]
        for _ in range(2)
    ]
    torch.testing.assert_close(losses[0], losses[1])


def test_views_without_mask_or_noise():
    _, _, _, systems, options, hypers = _setup(ase.io.read(QM9_PATH, ":4"))
    hypers["mask"]["probability"] = 0.0
    hypers["noise"]["probability"] = 0.0
    views = build_views(systems, options, hypers, torch.Generator())
    assert not views.masked.any()
    assert views.latent_targets.all()
    assert torch.all(views.displacements == 0)


def test_swap_masks_the_complement():
    chain = ase.Atoms("C10", positions=[[1.5 * i, 0.0, 0.0] for i in range(10)])
    _, _, _, systems, options, hypers = _setup([chain])
    hypers["mask"].update(
        probability=1.0, generator="k_hop_ego", ego_hops=[1], swap_probability=1.0
    )
    views = build_views(systems, options, hypers, torch.Generator())
    # the 1-hop ego network (2 or 3 atoms) is the context
    assert int(views.masked.sum()) in (7, 8)


@pytest.mark.parametrize("generator", ["nl_ball", "k_hop_ego"])
@pytest.mark.parametrize("species_query", [False, True])
def test_degenerate_batches_are_finite(generator, species_query):
    frames = [
        ase.Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ase.io.read(CARBON_PATH, 0),
    ]
    for batch in [frames[:1], frames]:
        student, teacher, predictor, systems, options, hypers = _setup(
            batch, torch.float64, species_query
        )
        hypers["mask"].update(probability=1.0, generator=generator)
        hypers["noise"]["probability"] = 1.0
        for seed in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                loss, _ = grad_jepa_loss(
                    student,
                    teacher,
                    predictor,
                    _terms(hypers),
                    systems,
                    options,
                    hypers,
                    torch.Generator().manual_seed(seed),
                    with_rank=True,
                )
            assert torch.isfinite(loss)
            loss.backward()


def test_masked_targets_are_reachable():
    """Masked atoms without a context neighbor within the cutoff are not targets."""
    chain = ase.Atoms("C10", positions=[[1.5 * i, 0.0, 0.0] for i in range(10)])
    _, _, _, systems, options, hypers = _setup([chain])
    hypers["mask"].update(
        probability=1.0, generator="k_hop_ego", ego_hops=[1], swap_probability=1.0
    )
    views = build_views(systems, options, hypers, torch.Generator())
    targets = views.latent_targets
    assert torch.all(views.masked[targets])
    assert 0 < int(targets.sum()) < int(views.masked.sum())
    close = views.vectors.norm(dim=-1) < options.cutoff
    pairs = views.pairs[close]
    for atom in torch.nonzero(targets).squeeze(-1).tolist():
        neighbors = pairs[pairs[:, 0] == atom, 1]
        assert torch.any(~views.masked[neighbors])


def test_displacements_are_centered():
    _, _, _, systems, options, hypers = _setup(ase.io.read(QM9_PATH, ":4"))
    hypers["noise"]["probability"] = 1.0
    views = build_views(systems, options, hypers, torch.Generator())
    start = 0
    for system in systems:
        end = start + len(system)
        delta = views.displacements[start:end]
        assert torch.any(delta != 0)
        torch.testing.assert_close(delta.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
        start = end


def test_vicreg_terms_are_scale_free_and_detect_collapse():
    torch.manual_seed(0)
    features = torch.randn(256, 16, dtype=torch.float64)
    variance, covariance = vicreg_terms(features)
    variance_scaled, covariance_scaled = vicreg_terms(100.0 * features + 3.0)
    # scale-free up to the epsilon of the layer norm
    torch.testing.assert_close(variance, variance_scaled, rtol=1e-3, atol=0)
    torch.testing.assert_close(covariance, covariance_scaled, rtol=1e-3, atol=0)
    # rank-1 features (plus a common offset): strongly correlated dimensions
    direction = torch.randn(16, dtype=torch.float64)
    collapsed = torch.randn(256, 1, dtype=torch.float64) * direction + 5.0
    assert vicreg_terms(collapsed)[1] > 10 * covariance


def test_relative_gradient_spread_is_scale_free():
    torch.manual_seed(0)
    features = torch.randn(20, 8, dtype=torch.float64)
    gradients = torch.randn(2, 20, 3, dtype=torch.float64)
    probes = torch.randn(2, 8, dtype=torch.float64)
    select = torch.ones(20, dtype=torch.bool)
    spread = relative_gradient_spread(features, gradients, probes, select)
    scaled = relative_gradient_spread(10 * features, 10 * gradients, probes, select)
    torch.testing.assert_close(spread, scaled)


def test_all_masked_by_default_has_no_unmasked_terms():
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":6")
    )
    _, metrics = grad_jepa_loss(
        student,
        teacher,
        predictor,
        _terms(hypers),
        systems,
        options,
        hypers,
        torch.Generator(),
    )
    assert metrics["latent_unmasked"] == 0.0
    assert metrics["gradient_unmasked"] == 0.0
    assert metrics["latent_masked"] > 0.0
