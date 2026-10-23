"""The shared-pipeline hooks: view builders, extras, predictor gating, step inputs."""

import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining import objective
from metatrain.experimental.ssl_pretraining.backbones import HeadInputs
from metatrain.experimental.ssl_pretraining.objective import (
    VIEW_BUILDERS,
    Views,
    build_step_data,
    build_views,
    encode_extra_teacher_views,
    grad_jepa_loss,
    grad_jepa_term_losses,
    register_view_builder,
)
from metatrain.experimental.ssl_pretraining.terms import TERMS, build_terms
from metatrain.experimental.ssl_pretraining.trainer import check_hypers
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.pydantic import MetatrainValidationError

from . import QM9_PATH
from .test_objective import _setup


VICREG_ONLY = {"vicreg": {"variance": 1.0, "covariance": 0.04}}


def test_registry_contains_default():
    assert VIEW_BUILDERS["grad_jepa"] is build_views


def test_register_view_builder(monkeypatch):
    monkeypatch.setattr(objective, "VIEW_BUILDERS", dict(VIEW_BUILDERS))

    def other(systems, options, hypers, generator, position_gradients=True):
        return build_views(systems, options, hypers, generator, position_gradients)

    register_view_builder("other", other)
    register_view_builder("other", other)  # idempotent for the same callable
    assert objective.VIEW_BUILDERS["other"] is other
    with pytest.raises(ValueError, match="already registered"):
        register_view_builder("other", build_views)


def test_views_extras_default_empty(small_batch):
    systems, options, hypers = small_batch
    out = build_views(systems, options, hypers, torch.Generator().manual_seed(0))
    assert out.extras == {}


def test_views_grad_jepa_fields_are_optional(small_batch):
    systems, _, _ = small_batch
    n_atoms = sum(len(s) for s in systems)
    views = Views(
        student=systems,
        teacher=systems,
        rotations=torch.eye(3).expand(n_atoms, 3, 3),
        masked=torch.zeros(n_atoms, dtype=torch.bool),
    )
    for name in [
        "displacements",
        "in_masked_structure",
        "latent_targets",
        "gradient_targets",
        "pairs",
        "vectors",
    ]:
        assert getattr(views, name) is None
    assert views.extras == {}


def test_extra_teacher_views_are_encoded_without_grad(small_batch, small_teacher):
    systems, _, _ = small_batch
    extras = {"teacher_views": {"plus": systems, "minus": systems}}
    feats = encode_extra_teacher_views(small_teacher, extras)
    assert set(feats) == {"plus", "minus"}
    assert not feats["plus"].requires_grad
    assert torch.allclose(feats["plus"], feats["minus"])
    assert encode_extra_teacher_views(small_teacher, {}) == {}


def test_unknown_builder_rejected():
    hypers = get_default_hypers("experimental.ssl_pretraining")["training"]
    assert hypers["views"] == {"builder": "grad_jepa"}
    hypers["views"]["builder"] = "nope"
    with pytest.raises(ValueError, match="unknown view builder 'nope'"):
        check_hypers(hypers, cutoff=4.0)


def test_views_hypers_reject_extra_keys():
    options = get_default_hypers("experimental.ssl_pretraining")
    options["training"]["views"]["bulider"] = "grad_jepa"
    with pytest.raises(MetatrainValidationError, match="bulider"):
        check_architecture_options(name="experimental.ssl_pretraining", options=options)


def test_step_inputs_reach_extras():
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":3"), torch.float64
    )
    generator = torch.Generator().manual_seed(0)
    step = build_step_data(
        student,
        teacher,
        predictor,
        systems,
        options,
        hypers,
        generator,
        needs_gradients=False,
        step_fraction=0.25,
    )
    assert step.extras["step_fraction"] == 0.25
    assert step.extras["generator"] is generator
    assert step.extras["teacher_features"] == {}
    # the Grad-JEPA builder keeps the plain feature call
    assert "student_tokens" not in step.extras


def test_other_builder_gets_tokens_and_extra_teacher_features(monkeypatch):
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":3"), torch.float64
    )
    monkeypatch.setattr(objective, "VIEW_BUILDERS", dict(VIEW_BUILDERS))
    seen = {}

    def tagged(systems, options, hypers, generator, position_gradients=True):
        seen["position_gradients"] = position_gradients
        views = build_views(systems, options, hypers, generator, position_gradients)
        views.extras = {"tag": 1, "teacher_views": {"same": views.teacher}}
        return views

    register_view_builder("tagged", tagged)
    hypers["views"]["builder"] = "tagged"
    step = build_step_data(
        student,
        teacher,
        predictor,
        systems,
        options,
        hypers,
        torch.Generator().manual_seed(0),
        needs_gradients=False,
    )
    assert seen["position_gradients"] is False
    assert step.extras["tag"] == 1
    assert step.extras["step_fraction"] == 0.0
    torch.testing.assert_close(
        step.extras["teacher_features"]["same"], step.teacher_features
    )
    tokens = step.extras["student_tokens"]
    assert isinstance(tokens, HeadInputs)
    assert tokens.node.requires_grad
    assert tokens.node.shape[0] == step.student_features.shape[0]


@pytest.mark.parametrize(
    "key", ["teacher_features", "step_fraction", "generator", "student_tokens"]
)
def test_reserved_extras_keys_are_rejected(monkeypatch, key):
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":2"), torch.float64
    )
    monkeypatch.setattr(objective, "VIEW_BUILDERS", dict(VIEW_BUILDERS))

    def clashing(systems, options, hypers, generator, position_gradients=True):
        views = build_views(systems, options, hypers, generator, position_gradients)
        views.extras = {key: None}
        return views

    register_view_builder("clashing", clashing)
    hypers["views"]["builder"] = "clashing"
    with pytest.raises(ValueError, match=f"reserved extras key.*{key}"):
        build_step_data(
            student,
            teacher,
            predictor,
            systems,
            options,
            hypers,
            torch.Generator().manual_seed(0),
            needs_gradients=False,
        )


def test_uses_prediction_flags():
    flags = {name: TERMS[name].uses_prediction for name in TERMS}
    assert flags == {
        "latent": True,
        "gradient": True,
        "vicreg": False,
        "gradient_spread": False,
    }


class _Forbidden(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("the predictor ran without a term using it")


class _NoFlag:
    """A term without ``uses_prediction`` (treated as not using the predictor)."""

    name = "no_flag"
    needs_gradients = False
    required_targets: dict = {}

    def __call__(self, step):
        assert step.prediction == {}
        return step.student_features.pow(2).mean(), {}


def test_predictor_not_run_when_no_term_uses_it():
    student, teacher, _, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":3"), torch.float64
    )
    for active in [build_terms(VICREG_ONLY), [_NoFlag()]]:
        for predictor in [_Forbidden(), None]:
            loss, _ = grad_jepa_loss(
                student,
                teacher,
                predictor,
                active,
                systems,
                options,
                hypers,
                torch.Generator().manual_seed(0),
            )
            assert torch.isfinite(loss)


def test_predictor_required_when_a_term_uses_it():
    student, teacher, _, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":2"), torch.float64
    )
    with pytest.raises(ValueError, match="predictor"):
        grad_jepa_loss(
            student,
            teacher,
            None,
            build_terms({"latent": {"weight": 1.0}}),
            systems,
            options,
            hypers,
            torch.Generator().manual_seed(0),
        )


def test_term_losses_sum_to_the_loss():
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":3"), torch.float64
    )
    active = build_terms(hypers["objective"]["terms"])
    args = (student, teacher, predictor, active, systems, options, hypers)
    loss, metrics = grad_jepa_loss(*args, torch.Generator().manual_seed(1))
    losses, term_metrics = grad_jepa_term_losses(
        *args, torch.Generator().manual_seed(1)
    )
    assert list(losses) == [term.name for term in active]
    assert all(value.requires_grad for value in losses.values())
    torch.testing.assert_close(sum(losses.values()), loss, rtol=0, atol=0)
    assert term_metrics == metrics
