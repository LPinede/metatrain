import copy
import warnings

import ase
import ase.io
import pytest
import torch
from omegaconf import OmegaConf

from metatrain.experimental.grad_jepa import objective, terms
from metatrain.experimental.grad_jepa.documentation import TermsHypers
from metatrain.experimental.grad_jepa.objective import build_step_data, grad_jepa_loss
from metatrain.experimental.grad_jepa.terms import TERMS, build_terms, register_term
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.pydantic import MetatrainValidationError

from . import CARBON_PATH, QM9_PATH
from .test_objective import _setup


LATENT_ONLY = {
    "latent": {"weight": 1.0},
    "vicreg": {"variance": 1.0, "covariance": 0.04},
}
GRADIENT_METRICS = {
    "gradient_masked",
    "gradient_unmasked",
    "gradient_cosine",
    "teacher_gradient_rms",
    "gradient_spread",
}

# Loss and metrics of the pre-refactor objective (``grad_jepa_loss`` at commit
# 6b0aaff9, with the flat ``loss_weights``), captured before the refactor for the
# setup of ``_golden_inputs``. To regenerate them, check out that commit and run
# the same setup with ``gradient_std_floor`` in place of
# ``objective.terms.gradient_spread.floor``.
GOLDEN = {
    "default": (
        {"seed": 3},
        3.392390381009938,
        {
            "latent_masked": 1.0404535411024338,
            "latent_unmasked": 0.0,
            "gradient_masked": 0.9993935688749016,
            "gradient_unmasked": 0.0,
            "variance": 0.6820084057307527,
            "covariance": 16.763371632546235,
            "gradient_spread": 0.1754128130422482,
            "gradient_cosine": 0.07457421650144588,
            "teacher_gradient_rms": 0.20774287610585945,
            "masked_fraction": 0.5652173913043478,
            "rankme": 3.1255672920934674,
        },
    ),
    # unmasked structures and teacher noise: every latent/gradient term is active
    "partial_mask": (
        {"seed": 5, "mask_probability": 0.5, "noise_probability": 1.0},
        3.2517560006921005,
        {
            "latent_masked": 1.027148648891092,
            "latent_unmasked": 1.0742791658129291,
            "gradient_masked": 1.0002843513424795,
            "gradient_unmasked": 1.1190844593865101,
            "variance": 0.4308809096068309,
            "covariance": 6.127529613795955,
            "gradient_spread": 0.1361535159642078,
            "gradient_cosine": -0.09329825655464562,
            "teacher_gradient_rms": 0.238829860626727,
            "masked_fraction": 0.30434782608695654,
            "rankme": 7.238897714270256,
        },
    ),
    # a high floor: the gradient-spread hinge is active
    "active_floor": (
        {"seed": 7, "mask_probability": 0.5, "noise_probability": 1.0, "floor": 1.0},
        3.5191544662402996,
        {
            "latent_masked": 1.0489488092487527,
            "latent_unmasked": 1.0573916825308467,
            "gradient_masked": 1.1398983312754127,
            "gradient_unmasked": 1.2521690022243668,
            "variance": 0.40820511871936904,
            "covariance": 6.378964277142133,
            "gradient_spread": 0.1044653527772322,
            "gradient_cosine": -0.05343811813247703,
            "teacher_gradient_rms": 0.2271456098252474,
            "masked_fraction": 0.17391304347826086,
            "rankme": 7.734858296149519,
        },
    ),
}


def _golden_inputs(config):
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":6"), torch.float64
    )
    if "mask_probability" in config:
        hypers["mask"]["probability"] = config["mask_probability"]
        hypers["noise"]["probability"] = config["noise_probability"]
    if "floor" in config:
        hypers["objective"]["terms"]["gradient_spread"]["floor"] = config["floor"]
    return student, teacher, predictor, systems, options, hypers


def _only(name):
    """Terms hypers with only ``name`` enabled, at its default hypers."""
    defaults = copy.deepcopy(get_default_hypers("experimental.grad_jepa"))
    return {name: defaults["training"]["objective"]["terms"][name]}


def _architecture_options(terms_hypers):
    options = copy.deepcopy(get_default_hypers("experimental.grad_jepa"))
    options["training"]["objective"]["terms"] = terms_hypers
    return options


def test_registry_matches_the_hypers():
    assert list(TERMS) == ["latent", "gradient", "vicreg", "gradient_spread"]
    # every registered term has its hypers entry (and nothing else has one)
    assert set(TermsHypers.__annotations__) == set(TERMS)
    for name, cls in TERMS.items():
        assert cls.name == name


def test_needs_gradients():
    flags = {t.name: t.needs_gradients for t in build_terms(_only_all())}
    assert flags == {
        "latent": False,
        "gradient": True,
        "vicreg": False,
        "gradient_spread": True,
    }


def _only_all():
    defaults = copy.deepcopy(get_default_hypers("experimental.grad_jepa"))
    return defaults["training"]["objective"]["terms"]


def test_build_terms_absent_or_null_is_disabled():
    names = [t.name for t in build_terms(LATENT_ONLY)]
    assert names == ["latent", "vicreg"]
    with_nulls = {**_only_all(), "gradient": None, "gradient_spread": None}
    assert [t.name for t in build_terms(with_nulls)] == ["latent", "vicreg"]


def test_build_terms_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="unknown loss term 'span'"):
        build_terms({**LATENT_ONLY, "span": {"weight": 1.0}})
    with pytest.raises(ValueError, match="at least one"):
        build_terms({})
    with pytest.raises(ValueError, match="at least one"):
        build_terms({"latent": None})


def test_register_term(monkeypatch):
    monkeypatch.setattr(terms, "TERMS", dict(TERMS))

    class Constant:
        name = "constant"
        needs_gradients = False

        def __init__(self, value: float) -> None:
            self.value = value
            self.required_targets: dict = {}

        def __call__(self, step):
            loss = step.student_features.sum() * 0.0 + self.value
            return loss, {"constant": loss}

    assert register_term(Constant) is Constant
    (term,) = build_terms({"constant": {"value": 2.0}})
    assert term.value == 2.0
    with pytest.raises(ValueError, match="already registered"):
        register_term(Constant)


@pytest.mark.parametrize("name", list(GOLDEN))
def test_behaviour_preservation(name):
    """The default terms reproduce the pre-refactor objective exactly."""
    config, expected_loss, expected_metrics = GOLDEN[name]
    student, teacher, predictor, systems, options, hypers = _golden_inputs(config)
    loss, metrics = grad_jepa_loss(
        student,
        teacher,
        predictor,
        build_terms(hypers["objective"]["terms"]),
        systems,
        options,
        hypers,
        torch.Generator().manual_seed(config["seed"]),
        with_rank=True,
    )
    assert set(metrics) == set(expected_metrics)
    torch.testing.assert_close(
        float(loss.detach()), expected_loss, rtol=1e-10, atol=1e-12
    )
    for key, value in expected_metrics.items():
        torch.testing.assert_close(metrics[key], value, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", list(TERMS))
def test_term_contract(name):
    """Each term alone: finite on degenerate batches, trains the student only."""
    frames = [
        ase.Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        ase.io.read(CARBON_PATH, 0),
    ]
    for batch in [frames[:1], frames[:2], frames]:
        student, teacher, predictor, systems, options, hypers = _setup(
            batch, torch.float64
        )
        hypers["mask"]["probability"] = 1.0
        hypers["noise"]["probability"] = 1.0
        hypers["objective"]["terms"] = _only(name)
        active = build_terms(hypers["objective"]["terms"])
        for seed in range(3):
            student.zero_grad()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loss, metrics = grad_jepa_loss(
                    student,
                    teacher,
                    predictor,
                    active,
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


def test_latent_only_creates_no_position_gradients(monkeypatch):
    student, teacher, predictor, systems, options, hypers = _setup(
        ase.io.read(QM9_PATH, ":6"), torch.float64
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("probe gradients computed in latent-only mode")

    monkeypatch.setattr(objective, "random_scalar_gradients", forbidden)
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
    assert step.student_gradients is None
    assert step.teacher_gradients is None
    assert set(step.prediction) == {"latent"}
    positions = [s.positions for s in step.views.student + step.views.teacher]
    assert not any(p.requires_grad for p in positions)
    assert not step.teacher_features.requires_grad

    loss, metrics = grad_jepa_loss(
        student,
        teacher,
        predictor,
        build_terms(LATENT_ONLY),
        systems,
        options,
        hypers,
        torch.Generator().manual_seed(0),
    )
    loss.backward()
    assert not GRADIENT_METRICS & set(metrics)
    assert any(p.grad is not None for p in student.parameters())


def test_latent_only_equals_full_objective_without_gradient_terms():
    config = GOLDEN["partial_mask"][0]
    student, teacher, predictor, systems, options, hypers = _golden_inputs(config)
    full_terms = build_terms(hypers["objective"]["terms"])

    full = build_step_data(
        student,
        teacher,
        predictor,
        systems,
        options,
        hypers,
        torch.Generator().manual_seed(config["seed"]),
        needs_gradients=True,
    )
    expected = sum(
        term(full)[0] for term in full_terms if term.name in ("latent", "vicreg")
    )
    _, full_metrics = grad_jepa_loss(
        student,
        teacher,
        predictor,
        full_terms,
        systems,
        options,
        hypers,
        torch.Generator().manual_seed(config["seed"]),
        with_rank=True,
    )

    loss, metrics = grad_jepa_loss(
        student,
        teacher,
        predictor,
        build_terms(LATENT_ONLY),
        systems,
        options,
        hypers,
        torch.Generator().manual_seed(config["seed"]),
        with_rank=True,
    )
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-14)
    assert set(metrics) == set(full_metrics) - GRADIENT_METRICS
    for key, value in metrics.items():
        torch.testing.assert_close(value, full_metrics[key], rtol=1e-12, atol=1e-14)


def test_yaml_accepts_the_latent_only_baseline():
    check_architecture_options(
        name="experimental.grad_jepa", options=_architecture_options(LATENT_ONLY)
    )


@pytest.mark.parametrize(
    "terms_hypers, match",
    [
        ({**LATENT_ONLY, "span": {"weight": 1.0}}, "span"),
        ({}, "at least one"),
        ({"latent": None, "vicreg": None}, "at least one"),
        ({"latent": {"weight": 1.0, "wieght": 2.0}}, "wieght"),
        ({"gradient_spread": {"weight": 0.1, "flor": 0.01}}, "flor"),
    ],
)
def test_yaml_rejects_invalid_terms(terms_hypers, match):
    with pytest.raises(MetatrainValidationError, match=match):
        check_architecture_options(
            name="experimental.grad_jepa",
            options=_architecture_options(terms_hypers),
        )


def test_null_disables_a_default_term_after_merging():
    """``mtt train`` merges the defaults: absent terms come back, null removes."""
    defaults = get_default_hypers("experimental.grad_jepa")
    user = {"training": {"objective": {"terms": {"gradient": None}}}}
    merged = OmegaConf.to_container(OmegaConf.merge(defaults, user))
    check_architecture_options(name="experimental.grad_jepa", options=merged)
    names = [t.name for t in build_terms(merged["training"]["objective"]["terms"])]
    assert names == ["latent", "vicreg", "gradient_spread"]
