import ast
import inspect

import pytest

from metatrain.experimental.ssl_pretraining.high_order import documentation
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.pydantic import MetatrainValidationError


NAME = "experimental.ssl_pretraining"

DEFAULTS = {
    "p_def": 0.5,
    "field_amplitude": 0.15,
    "wavelength": 8.0,
    "n_waves": 6,
    "max_wave_index": 2,
    "iid_sigma": 0.03,
    "mode_amplitude": 0.05,
    "mode_window": 3.0,
    "max_centers": 4,
    "teacher_pair": True,
    "l_max_multipole": 4,
    "l_max_latent": 4,
    "n_radial": 6,
    "analytic_weight": 1.0,
    "teacher_weight": 1.0,
    "warmup_fraction": 0.05,
    "floor_gamma": 0.1,
    "chirality_weight": 0.1,
    "chirality_types": [1, 6, 8],
    "head_inputs": "node",
    "anchor_channel_fraction": 1.0,
    "anti_collapse": "vicreg",
    "second_rotation": False,
    "equivariant_vicreg_weight": 0.1,
}


def test_defaults_reach_the_trainer_hypers():
    assert get_default_hypers(NAME)["training"]["high_order"] == DEFAULTS


def test_documentation_module_is_torch_free():
    tree = ast.parse(inspect.getsource(documentation))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


def _check(**over):
    options = get_default_hypers(NAME)
    options["training"]["high_order"].update(over)
    check_architecture_options(name=NAME, options=options)


def test_valid_overrides_pass():
    _check(p_def=0.0, teacher_pair=False, head_inputs="node_edge", iid_sigma=0.0)
    _check(p_def=1.0, anti_collapse="sigreg", max_centers=1)


def test_unknown_key_rejected():
    with pytest.raises(MetatrainValidationError, match="p_deff"):
        _check(p_deff=0.5)


@pytest.mark.parametrize(
    "over",
    [
        {"p_def": 1.5},
        {"p_def": -0.1},
        {"field_amplitude": 0.0},
        {"wavelength": -1.0},
        {"n_waves": 0},
        {"max_wave_index": 0},
        {"iid_sigma": -0.01},
        {"mode_amplitude": 0.0},
        {"mode_window": 0.0},
        {"max_centers": 0},
        {"l_max_multipole": -1},
        {"l_max_latent": -1},
        {"n_radial": 0},
        {"analytic_weight": -1.0},
        {"teacher_weight": -1.0},
        {"warmup_fraction": 1.5},
        {"floor_gamma": -0.1},
        {"chirality_weight": -0.1},
        {"chirality_types": []},
        {"head_inputs": "edge"},
        {"anchor_channel_fraction": 0.0},
        {"anchor_channel_fraction": 1.5},
        {"anti_collapse": "barlow"},
        {"equivariant_vicreg_weight": -0.1},
    ],
)
def test_out_of_range_rejected(over):
    with pytest.raises(MetatrainValidationError, match=next(iter(over))):
        _check(**over)
