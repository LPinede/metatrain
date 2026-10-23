import copy

import pytest
import torch

from metatrain.experimental.ssl_pretraining.backbones import BACKBONES, HeadInputs
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.pydantic import MetatrainValidationError

from . import small_model_hypers


PET = BACKBONES["pet"]


def test_feature_dim_formula(pet_core):
    expected = pet_core.num_readout_layers * (
        pet_core.hypers["d_node"] + pet_core.hypers["d_pet"]
    )
    assert PET.feature_dim(pet_core) == expected


def test_features_have_feature_dim(pet_core, small_batch):
    systems, _, _ = small_batch
    features = PET.features(pet_core, systems)
    n_atoms = sum(len(s) for s in systems)
    assert features.shape == (n_atoms, PET.feature_dim(pet_core))


def test_neighbor_list_options(pet_core):
    options = PET.neighbor_list_options(pet_core)
    assert options == pet_core.requested_neighbor_lists()[0]


def test_tokens_reproduce_feature_output(pet_core, small_batch):
    """Node tokens plus the cutoff-weighted edge-token sum equal PET's "feature"
    output."""
    systems, _, _ = small_batch
    feats, tok = PET.forward_with_tokens(pet_core, systems)
    assert isinstance(tok, HeadInputs)
    n_atoms = sum(len(s) for s in systems)
    assert tok.node.shape[0] == tok.edge.shape[0] == tok.cutoff.shape[0] == n_atoms
    assert tok.edge.shape[1] == tok.cutoff.shape[1]
    assert sum(tok.dims) == PET.feature_dim(pet_core)
    pooled = (tok.edge * tok.cutoff[..., None]).sum(1)
    assert torch.allclose(torch.cat([tok.node, pooled], dim=1), feats, atol=1e-5)
    assert torch.allclose(feats, PET.features(pet_core, systems), atol=1e-6)


def test_tokens_carry_gradients(pet_core, small_batch):
    systems, _, _ = small_batch
    _, tok = PET.forward_with_tokens(pet_core, systems)
    tok.edge.sum().backward()
    assert any(p.grad is not None for p in pet_core.parameters())


def test_token_capture_restores_the_backend(pet_core, small_batch):
    systems, _, _ = small_batch
    PET.forward_with_tokens(pet_core, systems)
    assert "calculate_features" not in vars(pet_core.backend)
    copy.deepcopy(pet_core)  # no captured closure left behind


def test_head_inputs_select_bool_and_index():
    tok = HeadInputs(
        node=torch.arange(4.0)[:, None].expand(4, 3),
        edge=torch.zeros(4, 5, 2),
        cutoff=torch.ones(4, 5),
    )
    assert tok.dims == (3, 2)
    mask = torch.tensor([True, False, True, False])
    by_mask = tok.select(mask)
    by_index = tok.select(torch.tensor([0, 2]))
    for selected in [by_mask, by_index]:
        assert selected.node[:, 0].tolist() == [0.0, 2.0]
        assert selected.edge.shape == (2, 5, 2)
        assert selected.cutoff.shape == (2, 5)


def test_validate_rejects_long_range_and_adaptive_cutoff():
    hypers = small_model_hypers()
    PET.validate(hypers)
    long_range = copy.deepcopy(hypers)
    long_range["long_range"]["enable"] = True
    with pytest.raises(ValueError, match="long-range"):
        PET.validate(long_range)
    adaptive = copy.deepcopy(hypers)
    adaptive["num_neighbors_adaptive"] = 16
    with pytest.raises(ValueError, match="adaptive cutoff"):
        PET.validate(adaptive)


def test_backbone_hyper():
    options = copy.deepcopy(get_default_hypers("experimental.ssl_pretraining"))
    assert options["model"]["backbone"] == "pet"
    check_architecture_options(name="experimental.ssl_pretraining", options=options)
    options["model"]["backbone"] = "space"
    with pytest.raises(MetatrainValidationError):
        check_architecture_options(name="experimental.ssl_pretraining", options=options)


def test_setup_builds_a_stock_pet(small_batch):
    """The ``backbone`` key selects the adapter and is not passed on to PET, so that
    the checkpoint stays a stock PET checkpoint."""
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6, 7, 8], targets={})
    model = PET.setup({**small_model_hypers(), "backbone": "pet"}, info)
    assert "backbone" not in model.core.hypers
    assert model.core.__class__.__name__ == "PET"
