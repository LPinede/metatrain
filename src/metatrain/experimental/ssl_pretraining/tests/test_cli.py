import shutil
import subprocess
import sys

import torch
from omegaconf import OmegaConf

from . import ETHANOL_PATH, QM9_PATH, small_pet


MODEL = {
    "d_pet": 16,
    "d_head": 16,
    "d_node": 16,
    "d_feedforward": 16,
    "num_heads": 2,
    "num_attention_layers": 1,
    "num_gnn_layers": 2,
    "cutoff": 4.0,
}


def _mtt_train(options: dict, name: str) -> None:
    OmegaConf.save(options, f"{name}.yaml")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "metatrain",
            "train",
            f"{name}.yaml",
            "-o",
            f"{name}.pt",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)


def test_pretrain_then_finetune(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy(QM9_PATH, "qm9.xyz")
    shutil.copy(ETHANOL_PATH, "ethanol.xyz")

    _mtt_train(
        {
            "seed": 0,
            "architecture": {
                "name": "experimental.ssl_pretraining",
                "model": MODEL,
                "training": {
                    "num_epochs": 1,
                    "batch_size": 8,
                    "num_workers": 0,
                    "predictor": {"hidden_dim": 16, "probe_dim": 4, "num_rbf": 8},
                },
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
    checkpoint = torch.load("pretrained.ckpt", weights_only=False)
    assert checkpoint["core"]["architecture_name"] == "pet"
    assert "grad_jepa" in checkpoint
    assert (tmp_path / "pretrained.pt").exists()

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
    assert (tmp_path / "finetuned.ckpt").exists()

    # Provenance: with a negligible learning rate, the fine-tuned backbone must stay
    # at the pretrained weights and be far from a fresh initialisation.
    pre = torch.load("pretrained.ckpt", weights_only=False)["core"][
        "best_model_state_dict"
    ]
    fine = torch.load("finetuned.ckpt", weights_only=False)["core"][
        "best_model_state_dict"
    ]
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
