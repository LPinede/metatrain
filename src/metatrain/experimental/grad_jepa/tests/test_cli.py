import shutil
import subprocess
import sys

import torch
from omegaconf import OmegaConf

from . import ETHANOL_PATH, QM9_PATH


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
    subprocess.check_call(
        [sys.executable, "-m", "metatrain", "train", f"{name}.yaml", "-o", f"{name}.pt"]
    )


def test_pretrain_then_finetune(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy(QM9_PATH, "qm9.xyz")
    shutil.copy(ETHANOL_PATH, "ethanol.xyz")

    _mtt_train(
        {
            "seed": 0,
            "architecture": {
                "name": "experimental.grad_jepa",
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
