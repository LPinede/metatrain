import copy
import shutil
import subprocess
import sys
import warnings

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from metatrain.experimental.ssl_pretraining.documentation import (
    TermsHypers,
    TrainerHypers,
)
from metatrain.experimental.ssl_pretraining.high_order.presets import (
    PENDING_PRESETS,
    PRESETS,
    analytic_terms,
    apply_preset,
)
from metatrain.experimental.ssl_pretraining.objective import grad_jepa_loss
from metatrain.experimental.ssl_pretraining.terms import TermContext, build_terms
from metatrain.experimental.ssl_pretraining.tests import QM9_PATH
from metatrain.experimental.ssl_pretraining.trainer import check_hypers
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.hypers import init_with_defaults


NAME = "experimental.ssl_pretraining"
IMPLEMENTED = sorted(PRESETS)
ANALYTIC = {"residual_multipole", "deformation_jets", "chirality"}
EXPECTED_TERMS = {
    "high_order_analytic_only": ANALYTIC | {"vicreg"},
    "high_order_edge_heads": ANALYTIC | {"vicreg"},
    "high_order_dedicated_channels": ANALYTIC | {"vicreg"},
    "high_order_sigreg": ANALYTIC | {"sigreg"},
    "high_order_vicreg_averaged": ANALYTIC | {"vicreg_averaged"},
    "high_order_equivariant_vicreg": ANALYTIC | {"vicreg", "equivariant_vicreg"},
    "invariant_jepa": {"rotation_conditioned_latent", "vicreg"},
}
# the presets whose terms read the antithetic teacher pair
TEACHER_PAIR = {"invariant_jepa"}


def _defaults() -> dict:
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    return copy.deepcopy(init_with_defaults(TrainerHypers))


def _hypers(preset: str) -> dict:
    return apply_preset(preset, _defaults())


def _active(hypers: dict) -> set:
    return {k for k, v in hypers["objective"]["terms"].items() if v is not None}


def _context(core) -> TermContext:
    n = core.num_readout_layers
    return TermContext(
        dims=(n * core.d_node, n * core.d_pet),
        cutoff=4.0,
        atomic_types=[1, 6, 7, 8],
        num_probes=2,
        feature_dim=n * (core.d_node + core.d_pet),
    )


def test_implemented_presets():
    assert set(IMPLEMENTED) == set(EXPECTED_TERMS)
    assert not set(PRESETS) & set(PENDING_PRESETS)


@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_preset_sets_the_expected_terms(preset):
    hypers = _hypers(preset)
    assert _active(hypers) == EXPECTED_TERMS[preset]
    assert hypers["views"]["builder"] == "high_order"
    assert hypers["high_order"]["teacher_pair"] is (preset in TEACHER_PAIR)


@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_preset_writes_every_term_explicitly(preset):
    terms = PRESETS[preset]["objective"]["terms"]
    assert set(terms) == set(TermsHypers.__annotations__)
    assert all(terms[name] is None for name in set(terms) - EXPECTED_TERMS[preset])


@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_preset_validates(preset):
    hypers = _hypers(preset)
    options = get_default_hypers(NAME)
    options["training"] = hypers
    check_architecture_options(name=NAME, options=options)
    check_hypers(hypers, cutoff=4.0)


@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_preset_survives_the_merge_with_the_defaults(preset):
    """``mtt train`` merges the options with the defaults: no default term may come
    back, so an options file written from a partial base gives the preset."""
    defaults = get_default_hypers(NAME)
    user = {"training": apply_preset(preset, {"num_epochs": 1})}
    merged = OmegaConf.to_container(OmegaConf.merge(defaults, user))
    assert _active(merged["training"]) == EXPECTED_TERMS[preset]
    assert merged["training"]["num_epochs"] == 1
    check_architecture_options(name=NAME, options=merged)
    check_hypers(merged["training"], cutoff=4.0)


def test_apply_preset_does_not_modify_its_inputs():
    base = _defaults()
    before = copy.deepcopy(base)
    preset = copy.deepcopy(PRESETS["high_order_sigreg"])
    hypers = apply_preset("high_order_sigreg", base)
    assert base == before
    assert PRESETS["high_order_sigreg"] == preset
    hypers["objective"]["terms"]["sigreg"]["weight"] = 7.0
    assert PRESETS["high_order_sigreg"]["objective"]["terms"]["sigreg"]["weight"] != 7.0


def test_apply_preset_keeps_the_other_base_keys():
    base = _defaults()
    base["high_order"]["p_def"] = 0.25
    base["num_probes"] = 5
    hypers = apply_preset("high_order_analytic_only", base)
    assert hypers["high_order"]["p_def"] == 0.25
    assert hypers["num_probes"] == 5


@pytest.mark.parametrize("preset", PENDING_PRESETS)
def test_presets_with_teacher_terms_are_pending(preset):
    with pytest.raises(NotImplementedError, match="H9"):
        apply_preset(preset, _defaults())


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="unknown preset 'nope'"):
        apply_preset("nope", _defaults())


def test_family_weights_set_the_term_weights():
    terms = analytic_terms(analytic_weight=2.0, chirality_weight=0.5)
    assert terms["residual_multipole"]["weight"] == 2.0
    assert terms["deformation_jets"]["weight"] == 2.0
    assert terms["chirality"]["weight"] == 1.0
    default = analytic_terms()
    assert default["chirality"]["weight"] == pytest.approx(0.1)


def test_head_options_reach_the_heads(pet_core):
    context = _context(pet_core)
    dn = context.dims[0]

    def built(preset):
        terms = build_terms(_hypers(preset)["objective"]["terms"], context)
        return {t.name: t for t in terms}

    default = built("high_order_analytic_only")
    edge = built("high_order_edge_heads")
    dedicated = built("high_order_dedicated_channels")
    for name, heads in [
        ("residual_multipole", lambda t: list(t.heads.values())),
        ("deformation_jets", lambda t: list(t.heads.values())),
        ("chirality", lambda t: [t.head]),
    ]:
        assert all(not h.use_edges and h.dn == dn for h in heads(default[name]))
        assert all(h.use_edges and h.dn == dn for h in heads(edge[name]))
        assert all(h.dn == dn // 2 for h in heads(dedicated[name]))


def test_invariant_jepa_is_the_order_zero_latent_on_masked_views(pet_core):
    hypers = _hypers("invariant_jepa")
    assert hypers["high_order"]["p_def"] == 0.0
    assert hypers["high_order"]["second_rotation"] is False
    latent = hypers["objective"]["terms"]["rotation_conditioned_latent"]
    assert latent["l_max"] == 0
    (term,) = [
        t
        for t in build_terms(hypers["objective"]["terms"], _context(pet_core))
        if t.name == "rotation_conditioned_latent"
    ]
    assert term.head_keys == [(0, 1)] and term.active_l_max == 0


@pytest.mark.parametrize(
    "preset", ["high_order_vicreg_averaged", "high_order_equivariant_vicreg"]
)
def test_two_rotation_presets_encode_the_second_rotation(preset):
    hypers = _hypers(preset)
    assert hypers["high_order"]["second_rotation"] is True
    equivariant = hypers["objective"]["terms"]["equivariant_vicreg"]
    if equivariant is not None:
        assert equivariant["rotation_group"] == hypers["rotation_group"] == "O3"


@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_preset_trains_one_step(preset, small_batch, pet_core, small_teacher):
    systems, options, base = small_batch
    hypers = apply_preset(preset, base)
    hypers["atomic_types"] = base["atomic_types"]
    check_hypers(hypers, options.cutoff)
    terms = build_terms(hypers["objective"]["terms"], _context(pet_core))
    assert {t.name for t in terms} == EXPECTED_TERMS[preset]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss, metrics = grad_jepa_loss(
            pet_core,
            small_teacher,
            None,
            terms,
            systems,
            options,
            hypers,
            torch.Generator().manual_seed(0),
            step_fraction=0.5,
        )
        loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(p.grad is not None for p in pet_core.parameters())


# validation rules of ``check_hypers``


def _analytic() -> dict:
    return _hypers("high_order_analytic_only")


def test_two_anti_collapse_terms_rejected():
    hypers = _analytic()
    hypers["objective"]["terms"]["sigreg"] = _hypers("high_order_sigreg")["objective"][
        "terms"
    ]["sigreg"]
    with pytest.raises(ValueError, match="exactly one anti-collapse term"):
        check_hypers(hypers, cutoff=4.0)


def test_no_anti_collapse_term_needs_the_pilot_flag():
    hypers = _analytic()
    hypers["objective"]["terms"]["vicreg"] = None
    with pytest.raises(ValueError, match="allow_no_anti_collapse"):
        check_hypers(hypers, cutoff=4.0)
    hypers["allow_no_anti_collapse"] = True
    check_hypers(hypers, cutoff=4.0)


def test_grad_jepa_latent_only_without_anti_collapse_needs_the_flag():
    hypers = _defaults()
    terms = hypers["objective"]["terms"]
    terms.update(gradient=None, gradient_spread=None, vicreg=None)
    with pytest.raises(ValueError, match="allow_no_anti_collapse"):
        check_hypers(hypers, cutoff=4.0)
    hypers["allow_no_anti_collapse"] = True
    check_hypers(hypers, cutoff=4.0)


@pytest.mark.parametrize(
    "preset", ["high_order_vicreg_averaged", "high_order_equivariant_vicreg"]
)
def test_two_rotation_terms_need_the_second_rotation(preset):
    hypers = _hypers(preset)
    hypers["high_order"]["second_rotation"] = False
    with pytest.raises(ValueError, match="second_rotation"):
        check_hypers(hypers, cutoff=4.0)


@pytest.mark.parametrize("name", sorted(ANALYTIC))
def test_high_order_terms_need_the_high_order_builder(name):
    hypers = _analytic()
    hypers["views"]["builder"] = "grad_jepa"
    for other in ANALYTIC - {name}:
        hypers["objective"]["terms"][other] = None
    with pytest.raises(ValueError, match=f"'{name}' term needs `views.builder"):
        check_hypers(hypers, cutoff=4.0)


@pytest.mark.parametrize("name", ["latent", "gradient", "gradient_spread"])
def test_grad_jepa_terms_rejected_with_the_high_order_builder(name):
    hypers = _analytic()
    hypers["objective"]["terms"][name] = _defaults()["objective"]["terms"][name]
    with pytest.raises(ValueError, match=f"'{name}' term needs the Grad-JEPA views"):
        check_hypers(hypers, cutoff=4.0)


def test_equivariant_vicreg_rotation_group_must_match_the_trainer():
    hypers = _hypers("high_order_equivariant_vicreg")
    hypers["rotation_group"] = "SO3"
    with pytest.raises(ValueError, match="rotation_group"):
        check_hypers(hypers, cutoff=4.0)
    hypers["objective"]["terms"]["equivariant_vicreg"]["rotation_group"] = "SO3"
    check_hypers(hypers, cutoff=4.0)


@pytest.mark.parametrize(
    "preset, term",
    [
        ("high_order_equivariant_vicreg", "equivariant_vicreg"),
        ("high_order_vicreg_averaged", "vicreg_averaged"),
    ],
)
def test_rotation_terms_need_student_rotations(preset, term):
    hypers = _hypers(preset)
    hypers["rotation_group"] = "none"
    with pytest.raises(ValueError, match=f"'{term}' term needs random student"):
        check_hypers(hypers, cutoff=4.0)


@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_preset_does_not_write_the_rotation_group(preset):
    assert "rotation_group" not in PRESETS[preset]
    hypers = apply_preset(preset, {"rotation_group": "SO3"})
    assert hypers["rotation_group"] == "SO3"


def test_mtt_train_with_the_analytic_preset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy(QM9_PATH, "qm9.xyz")
    training = apply_preset(
        "high_order_analytic_only",
        {"num_epochs": 1, "batch_size": 8, "num_workers": 0},
    )
    options = {
        "seed": 0,
        "architecture": {
            "name": NAME,
            "model": {
                "d_pet": 16,
                "d_head": 16,
                "d_node": 16,
                "d_feedforward": 16,
                "num_heads": 2,
                "num_attention_layers": 1,
                "num_gnn_layers": 2,
                "cutoff": 4.0,
            },
            "training": training,
        },
        "training_set": {
            "systems": {"read_from": "qm9.xyz", "length_unit": "angstrom"},
            "targets": {},
        },
        "validation_set": 0.1,
        "test_set": 0.0,
    }
    OmegaConf.save(options, "options.yaml")
    result = subprocess.run(
        [sys.executable, "-m", "metatrain", "train", "options.yaml", "-o", "model.pt"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    assert (
        "Active loss terms: vicreg, residual_multipole, deformation_jets, chirality"
        in log
    ), log
    checkpoint = torch.load("model.ckpt", weights_only=False)
    assert checkpoint["core"]["architecture_name"] == "pet"
    # the heads of the analytic terms are saved with the run
    saved = checkpoint["grad_jepa"]["terms_state_dict"]
    assert any(key.startswith("residual_multipole.") for key in saved)
    assert (tmp_path / "model.pt").exists()


def _dump(tmp_path, preset, base=None):
    command = [
        sys.executable,
        "-W",
        "error::RuntimeWarning",
        "-m",
        "metatrain.experimental.ssl_pretraining.high_order.presets",
        "dump",
        preset,
        "-o",
        str(tmp_path / "out.yaml"),
    ]
    if base is not None:
        (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
        command += ["--base", str(tmp_path / "base.yaml")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return yaml.safe_load((tmp_path / "out.yaml").read_text())


@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_dump_validates_and_survives_the_merge(preset, tmp_path):
    dumped = _dump(tmp_path, preset, base={"num_epochs": 1})
    assert dumped == apply_preset(preset, {"num_epochs": 1})
    terms = dumped["objective"]["terms"]
    assert set(terms) == set(TermsHypers.__annotations__)
    assert {k for k, v in terms.items() if v is not None} == EXPECTED_TERMS[preset]
    # the dumped YAML is a valid ``architecture.training`` section; ``mtt train``
    # merges it over the defaults, which must not bring back disabled terms
    defaults = get_default_hypers(NAME)
    merged = OmegaConf.to_container(OmegaConf.merge(defaults, {"training": dumped}))
    assert _active(merged["training"]) == EXPECTED_TERMS[preset]
    check_architecture_options(name=NAME, options=merged)
    check_hypers(merged["training"], cutoff=4.0)


def test_dump_without_base_and_with_a_pending_preset(tmp_path):
    assert _dump(tmp_path, "high_order_sigreg") == apply_preset("high_order_sigreg", {})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "metatrain.experimental.ssl_pretraining.high_order.presets",
            "dump",
            "high_order",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "teacher terms" in result.stderr
