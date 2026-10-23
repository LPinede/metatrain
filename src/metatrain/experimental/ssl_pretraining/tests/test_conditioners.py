import copy
import warnings

import ase
import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining import conditioners
from metatrain.experimental.ssl_pretraining.conditioners import (
    CONDITIONERS,
    build_conditioners,
    register_conditioner,
)
from metatrain.experimental.ssl_pretraining.documentation import ConditionersHypers
from metatrain.experimental.ssl_pretraining.objective import grad_jepa_loss
from metatrain.experimental.ssl_pretraining.predictor import GradJEPAPredictor
from metatrain.experimental.ssl_pretraining.terms import build_terms
from metatrain.experimental.ssl_pretraining.views import random_rotations
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.pydantic import MetatrainValidationError

from . import CARBON_PATH
from .conftest import N
from .test_objective import _setup
from .test_predictor import _predictor


class Dummy:
    """Test-only conditioner: reads an extra per-atom vector field ``step.field``
    (``(n_atoms, 3)``, teacher frame) and builds invariants and vectors from it."""

    name = "dummy"
    scalar_dim = 2
    vector_dim = 2

    def __init__(self) -> None:
        self.required_targets: dict = {}

    def edge_inputs(self, step, centers, neighbors, directions):
        field = step["field"]
        scalars = torch.stack(
            [
                field[neighbors].norm(dim=-1),
                (directions * field[centers]).sum(dim=-1),
            ],
            dim=-1,
        )
        vectors = torch.stack([field[neighbors], field[centers] + directions], dim=1)
        return scalars, vectors


class StepReader:
    """Test-only conditioner reading real ``StepData`` fields: the teacher-frame
    displacements (a polar vector) and the student features."""

    name = "step_reader"
    scalar_dim = 1
    vector_dim = 1

    def __init__(self) -> None:
        self.required_targets: dict = {}

    def edge_inputs(self, step, centers, neighbors, directions):
        displacements = step.views.displacements
        scalars = step.student_features[neighbors].mean(dim=-1, keepdim=True)
        return scalars, displacements[neighbors].unsqueeze(1)


@pytest.fixture
def registered(monkeypatch):
    """The registry with the test-only conditioners, for the test only."""
    monkeypatch.setattr(conditioners, "CONDITIONERS", dict(CONDITIONERS))
    register_conditioner(Dummy)
    register_conditioner(StepReader)
    return conditioners.CONDITIONERS


def _dummy_step():
    return {"field": torch.randn(N, 3, dtype=torch.float64)}


def _rotate_dummy_step(step, rotation):
    return {"field": step["field"] @ rotation.T}


# Per conditioner name: a step for the predictor-level tests, and how a rotation
# acts on its frame-dependent fields. A newly registered conditioner needs an entry.
PREDICTOR_CASES = {"dummy": (_dummy_step, _rotate_dummy_step)}
CONTRACT_NAMES = list(PREDICTOR_CASES)  # the test-only ones, then the registered
CONTRACT_NAMES += [name for name in CONDITIONERS if name not in PREDICTOR_CASES]


def _conditioned(inputs, name="dummy", num_layers=1):
    assert name in PREDICTOR_CASES, f"add a predictor case for conditioner '{name}'"
    torch.manual_seed(2)
    make_step, _ = PREDICTOR_CASES[name]
    conditioner = conditioners.CONDITIONERS[name]()
    return (
        _predictor(num_layers=num_layers, conditioners=[conditioner]),
        {**inputs, "step": make_step()},
    )


def test_registry_is_empty_in_v1():
    assert CONDITIONERS == {}
    # every registered conditioner has its hypers entry (and nothing else has one)
    assert set(ConditionersHypers.__annotations__) == set(CONDITIONERS)
    assert build_conditioners({}) == []


def test_register_and_build(registered):
    assert registered["dummy"] is Dummy
    assert "dummy" not in CONDITIONERS  # the shipped registry is untouched
    (conditioner,) = build_conditioners({"dummy": {}})
    assert isinstance(conditioner, Dummy)
    assert build_conditioners({"dummy": None}) == []
    with pytest.raises(ValueError, match="already registered"):
        register_conditioner(Dummy)
    with pytest.raises(ValueError, match="unknown conditioner 'other'"):
        build_conditioners({"other": {}})


@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("improper", [False, True])
@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_conditioned_predictor_is_covariant(
    registered, name, inputs, improper, num_layers
):
    predictor, conditioned = _conditioned(inputs, name, num_layers=num_layers)
    rotation = random_rotations(
        1, "SO3", torch.float64, torch.Generator().manual_seed(0)
    )[0]
    if improper:
        rotation = -rotation
        assert torch.det(rotation) < 0
    rotated = dict(conditioned)
    for key in ["gradients", "displacements", "vectors"]:
        rotated[key] = conditioned[key] @ rotation.T
    rotated["step"] = PREDICTOR_CASES[name][1](conditioned["step"], rotation)
    out, out_rotated = predictor(**conditioned), predictor(**rotated)
    torch.testing.assert_close(out["gradient"] @ rotation.T, out_rotated["gradient"])
    torch.testing.assert_close(out["latent"], out_rotated["latent"])


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_conditioned_predictor_is_probe_linear(registered, name, inputs, num_layers):
    predictor, conditioned = _conditioned(inputs, name, num_layers=num_layers)
    scaled = dict(conditioned)
    scaled["probes"] = 3.0 * conditioned["probes"]
    scaled["gradients"] = 3.0 * conditioned["gradients"]
    torch.testing.assert_close(
        3.0 * predictor(**conditioned)["gradient"], predictor(**scaled)["gradient"]
    )


def test_conditioner_channels_reach_both_heads(registered, inputs):
    predictor, conditioned = _conditioned(inputs, "dummy")
    changed = dict(conditioned)
    changed["step"] = {"field": 2.0 * conditioned["step"]["field"]}
    out, out_changed = predictor(**conditioned), predictor(**changed)
    assert not torch.allclose(out["latent"], out_changed["latent"])
    assert not torch.allclose(out["gradient"], out_changed["gradient"])


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_conditioned_predictor_ignores_masked_features(registered, name, inputs):
    predictor, conditioned = _conditioned(inputs, name, num_layers=2)
    changed = dict(conditioned)
    changed["features"] = conditioned["features"].clone()
    changed["features"][conditioned["masked"]] += 10.0
    out, out_changed = predictor(**conditioned), predictor(**changed)
    torch.testing.assert_close(out["latent"], out_changed["latent"])
    torch.testing.assert_close(out["gradient"], out_changed["gradient"])


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_conditioned_predictor_on_degenerate_batches(
    registered, name, inputs, num_layers
):
    """No pairs (isolated atoms), and no masked atoms: finite, no warnings."""
    predictor, conditioned = _conditioned(inputs, name, num_layers=num_layers)
    empty = dict(conditioned)
    empty["pairs"] = conditioned["pairs"][:0]
    empty["vectors"] = conditioned["vectors"][:0]
    unmasked = dict(conditioned)
    unmasked["masked"] = torch.zeros_like(conditioned["masked"])
    for batch in (empty, unmasked):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = predictor(**batch)
            (out["latent"].sum() + out["gradient"].sum()).backward()
        assert torch.isfinite(out["latent"]).all()
        assert torch.isfinite(out["gradient"]).all()


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_latent_only_with_conditioner(registered, name, inputs):
    predictor, conditioned = _conditioned(inputs, name)
    conditioned["gradients"] = None
    assert set(predictor(**conditioned)) == {"latent"}


@pytest.mark.parametrize("num_layers", [1, 2])
def test_objective_contract_with_a_conditioner(registered, num_layers):
    """Through ``grad_jepa_loss`` on degenerate batches, with a conditioner reading
    real ``StepData`` fields: finite, no warnings, student trained, teacher not."""
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
        (conditioner,) = build_conditioners({"step_reader": {}})
        torch.manual_seed(1)
        predictor = GradJEPAPredictor(
            predictor.query_context.in_features,
            16,
            4,
            8,
            options.cutoff,
            [1, 6, 7, 8],
            False,
            num_layers=num_layers,
            conditioners=[conditioner],
        ).double()
        terms = build_terms(hypers["objective"]["terms"])
        for seed in range(3):
            student.zero_grad()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loss, metrics = grad_jepa_loss(
                    student,
                    teacher,
                    predictor,
                    terms,
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


def _options(predictor_hypers):
    options = copy.deepcopy(get_default_hypers("experimental.ssl_pretraining"))
    options["training"]["predictor"].update(predictor_hypers)
    return options


def test_yaml_defaults_and_depth():
    options = _options({"num_layers": 3, "conditioners": {}})
    check_architecture_options(name="experimental.ssl_pretraining", options=options)
    default = get_default_hypers("experimental.ssl_pretraining")["training"][
        "predictor"
    ]
    assert default["num_layers"] == 1
    assert default["conditioners"] == {}


@pytest.mark.parametrize(
    "predictor_hypers, match",
    [
        ({"conditioners": {"magic": {}}}, "magic"),
        ({"num_layers": 0}, "num_layers"),
        ({"num_layer": 2}, "num_layer"),
    ],
)
def test_yaml_rejects_invalid_predictor_hypers(predictor_hypers, match):
    with pytest.raises(MetatrainValidationError, match=match):
        check_architecture_options(
            name="experimental.ssl_pretraining", options=_options(predictor_hypers)
        )
