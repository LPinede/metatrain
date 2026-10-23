import copy
import warnings

import pytest
import torch

from metatrain.experimental.grad_jepa import conditioners
from metatrain.experimental.grad_jepa.conditioners import (
    CONDITIONERS,
    build_conditioners,
    register_conditioner,
)
from metatrain.experimental.grad_jepa.documentation import ConditionersHypers
from metatrain.experimental.grad_jepa.views import random_rotations
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.pydantic import MetatrainValidationError

from .conftest import N
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


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(conditioners, "CONDITIONERS", dict(CONDITIONERS))
    register_conditioner(Dummy)
    return conditioners.CONDITIONERS


def _all_conditioners():
    return ["dummy"] + list(CONDITIONERS)


def _conditioned(inputs, name="dummy", num_layers=1):
    torch.manual_seed(2)
    field = torch.randn(N, 3, dtype=torch.float64)
    conditioner = Dummy()
    assert conditioner.name == name
    return (
        _predictor(num_layers=num_layers, conditioners=[conditioner]),
        {**inputs, "step": {"field": field}},
    )


def test_registry_is_empty_in_v1():
    assert CONDITIONERS == {}
    # every registered conditioner has its hypers entry (and nothing else has one)
    assert set(ConditionersHypers.__annotations__) == set(CONDITIONERS)
    assert build_conditioners({}) == []


def test_register_and_build(registered):
    assert registered["dummy"] is Dummy
    (conditioner,) = build_conditioners({"dummy": {}})
    assert isinstance(conditioner, Dummy)
    assert build_conditioners({"dummy": None}) == []
    with pytest.raises(ValueError, match="already registered"):
        register_conditioner(Dummy)
    with pytest.raises(ValueError, match="unknown conditioner 'other'"):
        build_conditioners({"other": {}})


@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("improper", [False, True])
def test_conditioned_predictor_is_covariant(inputs, improper, num_layers):
    predictor, conditioned = _conditioned(inputs, num_layers=num_layers)
    rotation = random_rotations(
        1, "SO3", torch.float64, torch.Generator().manual_seed(0)
    )[0]
    if improper:
        rotation = -rotation
        assert torch.det(rotation) < 0
    rotated = dict(conditioned)
    for key in ["gradients", "displacements", "vectors"]:
        rotated[key] = conditioned[key] @ rotation.T
    rotated["step"] = {"field": conditioned["step"]["field"] @ rotation.T}
    out, out_rotated = predictor(**conditioned), predictor(**rotated)
    torch.testing.assert_close(out["gradient"] @ rotation.T, out_rotated["gradient"])
    torch.testing.assert_close(out["latent"], out_rotated["latent"])


@pytest.mark.parametrize("num_layers", [1, 2])
def test_conditioned_predictor_is_probe_linear(inputs, num_layers):
    predictor, conditioned = _conditioned(inputs, num_layers=num_layers)
    scaled = dict(conditioned)
    scaled["probes"] = 3.0 * conditioned["probes"]
    scaled["gradients"] = 3.0 * conditioned["gradients"]
    torch.testing.assert_close(
        3.0 * predictor(**conditioned)["gradient"], predictor(**scaled)["gradient"]
    )


def test_conditioner_channels_reach_both_heads(inputs):
    predictor, conditioned = _conditioned(inputs)
    changed = dict(conditioned)
    changed["step"] = {"field": 2.0 * conditioned["step"]["field"]}
    out, out_changed = predictor(**conditioned), predictor(**changed)
    assert not torch.allclose(out["latent"], out_changed["latent"])
    assert not torch.allclose(out["gradient"], out_changed["gradient"])


def test_conditioned_predictor_ignores_masked_features(inputs):
    predictor, conditioned = _conditioned(inputs, num_layers=2)
    changed = dict(conditioned)
    changed["features"] = conditioned["features"].clone()
    changed["features"][conditioned["masked"]] += 10.0
    out, out_changed = predictor(**conditioned), predictor(**changed)
    torch.testing.assert_close(out["latent"], out_changed["latent"])
    torch.testing.assert_close(out["gradient"], out_changed["gradient"])


@pytest.mark.parametrize("num_layers", [1, 2])
def test_conditioned_predictor_on_degenerate_batches(inputs, num_layers):
    """No pairs (isolated atoms), and no masked atoms: finite, no warnings."""
    predictor, conditioned = _conditioned(inputs, num_layers=num_layers)
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


def test_latent_only_with_conditioner(inputs):
    predictor, conditioned = _conditioned(inputs)
    conditioned["gradients"] = None
    assert set(predictor(**conditioned)) == {"latent"}


def _options(predictor_hypers):
    options = copy.deepcopy(get_default_hypers("experimental.grad_jepa"))
    options["training"]["predictor"].update(predictor_hypers)
    return options


def test_yaml_defaults_and_depth():
    options = _options({"num_layers": 3, "conditioners": {}})
    check_architecture_options(name="experimental.grad_jepa", options=options)
    default = get_default_hypers("experimental.grad_jepa")["training"]["predictor"]
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
            name="experimental.grad_jepa", options=_options(predictor_hypers)
        )
