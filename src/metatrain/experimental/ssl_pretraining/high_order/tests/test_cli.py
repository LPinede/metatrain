import shutil

import torch

from metatrain.experimental.ssl_pretraining.high_order.presets import apply_preset
from metatrain.experimental.ssl_pretraining.tests import (
    ETHANOL_PATH,
    QM9_PATH,
    small_pet,
)
from metatrain.experimental.ssl_pretraining.tests.test_cli import MODEL, _mtt_train


SSL_MARKERS = ("grad_jepa", "high_order", "predictor", "terms", "teacher")


def test_high_order_pretrain_then_finetune_as_pet(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy(QM9_PATH, "qm9.xyz")
    shutil.copy(ETHANOL_PATH, "ethanol.xyz")

    # there is no ``preset`` hyper: the preset is merged into the YAML here
    training = apply_preset(
        "high_order", {"num_epochs": 1, "batch_size": 8, "num_workers": 0}
    )
    _mtt_train(
        {
            "seed": 0,
            "architecture": {
                "name": "experimental.ssl_pretraining",
                "model": MODEL,
                "training": training,
            },
            "training_set": {
                "systems": {"read_from": "qm9.xyz", "length_unit": "angstrom"},
                "targets": {},
            },
            "validation_set": 0.1,
            "test_set": 0.0,
        },
        "pretrained",
    )
    pretrained = torch.load("pretrained.ckpt", weights_only=False)
    assert pretrained["core"]["architecture_name"] == "pet"
    terms = pretrained["grad_jepa"]["terms_state_dict"]
    for term in ("residual_multipole", "deformation_jets", "action_response"):
        assert any(key.startswith(f"{term}.") for key in terms), term

    _mtt_train(
        {
            "seed": 0,
            "architecture": {
                "name": "pet",
                "model": MODEL,
                "training": {
                    "num_epochs": 1,
                    "batch_size": 8,
                    "num_workers": 0,
                    "learning_rate": 1e-12,
                    "finetune": {"read_from": "pretrained.ckpt", "method": "full"},
                },
            },
            "training_set": {
                "systems": {"read_from": "ethanol.xyz", "length_unit": "angstrom"},
                "targets": {
                    "energy": {
                        "key": "energy",
                        "unit": "eV",
                        "forces": {"key": "forces"},
                    }
                },
            },
            "validation_set": 0.1,
            "test_set": 0.0,
        },
        "finetuned",
    )
    assert (tmp_path / "finetuned.pt").exists()
    finetuned = torch.load("finetuned.ckpt", weights_only=False)
    assert finetuned["core"]["architecture_name"] == "pet"

    # the exported PET is stock PET: no SSL keys anywhere
    assert "grad_jepa" not in finetuned
    for key in finetuned["core"]["best_model_state_dict"]:
        assert not any(marker in key for marker in SSL_MARKERS), key

    # provenance: with a negligible learning rate the backbone stays at the
    # pretrained weights and is far from a fresh initialisation
    pre = pretrained["core"]["best_model_state_dict"]
    fine = finetuned["core"]["best_model_state_dict"]
    fresh = small_pet([1, 6, 8]).state_dict()
    keys = [
        k
        for k, v in pre.items()
        if "gnn_layers" in k
        and v.is_floating_point()
        and v.numel() > 100
        and k in fine
        and k in fresh
        and fresh[k].shape == v.shape
    ]
    assert keys
    for k in keys:
        assert torch.allclose(fine[k], pre[k], atol=1e-6), k
    assert any(not torch.allclose(fine[k], fresh[k], atol=1e-3) for k in keys)
