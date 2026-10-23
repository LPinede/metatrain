import copy
from pathlib import Path
from typing import List

import ase
import torch
from metatomic.torch import System, systems_to_torch

from metatrain.pet import PET
from metatrain.pet import Trainer as PETTrainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo


RESOURCES = Path(__file__).parents[5] / "tests" / "resources"
QM9_PATH = str(RESOURCES / "qm9_reduced_100.xyz")
CARBON_PATH = str(RESOURCES / "carbon_reduced_100.xyz")
ETHANOL_PATH = str(RESOURCES / "ethanol_reduced_100.xyz")


def small_model_hypers() -> dict:
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    hypers.update(
        d_pet=16,
        d_head=16,
        d_node=16,
        d_feedforward=16,
        num_heads=2,
        num_attention_layers=1,
        num_gnn_layers=2,
        cutoff=4.0,
    )
    return hypers


def small_pet(atomic_types: List[int], dtype: torch.dtype = torch.float32) -> PET:
    info = DatasetInfo(length_unit="angstrom", atomic_types=atomic_types, targets={})
    trainer = PETTrainer(get_default_hypers("pet")["training"])
    return trainer.setup(small_model_hypers(), info).core.to(dtype)


def to_systems(frames: List[ase.Atoms], dtype: torch.dtype) -> List[System]:
    return [system.to(dtype) for system in systems_to_torch(frames)]
