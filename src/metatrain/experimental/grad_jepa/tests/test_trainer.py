import copy

import ase.io
import pytest
import torch

from metatrain.experimental.grad_jepa.documentation import TrainerHypers
from metatrain.experimental.grad_jepa.trainer import (
    Trainer,
    check_hypers,
    get_collate_fns,
)
from metatrain.pet import Trainer as PETTrainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.io import model_from_checkpoint

from . import QM9_PATH, small_model_hypers, to_systems


ATOMIC_TYPES = [1, 6, 7, 8]
INFO = DatasetInfo(length_unit="angstrom", atomic_types=ATOMIC_TYPES, targets={})


def _trainer_hypers(**overrides) -> dict:
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    hypers = copy.deepcopy(init_with_defaults(TrainerHypers))
    hypers.update(
        batch_size=8,
        num_epochs=2,
        num_workers=0,
        learning_rate=1e-3,
        checkpoint_interval=1,
    )
    hypers["predictor"].update(hidden_dim=16, probe_dim=4, num_rbf=8)
    hypers["mask"]["ball_radius_range"] = [1.0, 2.5]
    hypers.update(overrides)
    return hypers


def _dataset() -> Dataset:
    frames = ase.io.read(QM9_PATH, ":24")
    return Dataset.from_dict({"system": to_systems(frames, torch.float64)})


def test_setup_rejects_targets():
    energy = get_energy_target_info("energy", {"unit": "eV", "quantity": "energy"})
    info = DatasetInfo(
        length_unit="angstrom", atomic_types=ATOMIC_TYPES, targets={"energy": energy}
    )
    with pytest.raises(ValueError, match="label-free"):
        Trainer(_trainer_hypers()).setup(small_model_hypers(), info)


def test_setup_rejects_adaptive_cutoff():
    model_hypers = small_model_hypers()
    model_hypers["num_neighbors_adaptive"] = 16
    with pytest.raises(ValueError, match="adaptive cutoff"):
        Trainer(_trainer_hypers()).setup(model_hypers, INFO)


def test_setup_rejects_long_range():
    model_hypers = small_model_hypers()
    model_hypers["long_range"]["enable"] = True
    with pytest.raises(ValueError, match="long-range"):
        Trainer(_trainer_hypers()).setup(model_hypers, INFO)


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("mask", "ball_radius_range"), [1.0, 5.0], "cutoff"),
        (("mask", "topology_cutoff"), 5.0, "cutoff"),
        (("mask", "fraction_range"), [0.5, 0.2], "low <= high"),
        (("mask", "ego_hops"), [], "ego_hops"),
        (("num_probes",), 0, "num_probes"),
        (("ema_tau_range",), [1.0, 0.9], "low <= high"),
    ],
)
def test_check_hypers(path, value, match):
    hypers = copy.deepcopy(init_with_defaults(TrainerHypers))
    check_hypers(hypers, cutoff=4.5)
    section = hypers
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        check_hypers(hypers, cutoff=4.5)


def test_train_save_resume_and_finetune(tmp_path):
    torch.manual_seed(0)
    dataset = _dataset()
    trainer = Trainer(_trainer_hypers(checkpoint_interval=10))
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(
        model, torch.float32, [torch.device("cpu")], [dataset], [dataset], str(tmp_path)
    )
    path = tmp_path / "ssl.ckpt"
    trainer.save_checkpoint(model, path)

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["core"]["architecture_name"] == "pet"
    assert checkpoint["epoch"] == 1
    assert set(checkpoint["grad_jepa"]) == {
        "step",
        "teacher_state_dict",
        "predictor_state_dict",
    }
    # epoch 0 and the last epoch are always saved
    assert (tmp_path / "model_0.ckpt").exists()
    assert (tmp_path / "model_1.ckpt").exists()
    # by default, the weights handed to fine-tuning are the final EMA teacher
    assert checkpoint["best_epoch"] == 1
    for name, value in checkpoint["grad_jepa"]["teacher_state_dict"].items():
        if not isinstance(value, torch.Tensor):
            continue  # ``_extra_state`` entries
        torch.testing.assert_close(
            checkpoint["core"]["best_model_state_dict"][name], value
        )

    # resume one more epoch with ``restart_from``
    resumed = Trainer(_trainer_hypers(num_epochs=3, restart_from=str(path)))
    resumed_model = resumed.setup(small_model_hypers(), INFO)
    resumed.train(
        resumed_model,
        torch.float32,
        [torch.device("cpu")],
        [dataset],
        [dataset],
        str(tmp_path),
    )
    assert resumed.epoch == 2
    assert resumed.step == 3 * checkpoint["grad_jepa"]["step"] // 2

    # the checkpoint fine-tunes as a plain PET with a new energy target
    pet = model_from_checkpoint(copy.deepcopy(checkpoint), context="finetune")
    energy = get_energy_target_info(
        "energy", {"unit": "eV", "quantity": "energy"}, add_position_gradients=True
    )
    info = DatasetInfo(
        length_unit="angstrom", atomic_types=ATOMIC_TYPES, targets={"energy": energy}
    )
    pet = PETTrainer(get_default_hypers("pet")["training"]).restart(
        pet, info, model_hypers=small_model_hypers()
    )
    assert "energy" in pet.core.supported_outputs()
    best = {
        k.removeprefix("core."): v
        for k, v in trainer.best_model_state_dict.items()
        if k.startswith("core.")
    }
    state = pet.core.state_dict()
    for name, value in best.items():
        if name != "finetune_config" and isinstance(value, torch.Tensor):
            torch.testing.assert_close(state[name], value)


def test_best_rankme_student_selection(tmp_path):
    torch.manual_seed(0)
    dataset = _dataset()
    trainer = Trainer(_trainer_hypers(export_weights="best_rankme_student"))
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(
        model, torch.float32, [torch.device("cpu")], [dataset], [dataset], str(tmp_path)
    )
    # the selection metric is minus the validation RankMe
    assert trainer.best_metric < -1.0
    assert trainer.best_epoch in (0, 1)


@pytest.mark.parametrize("augment", [True, False])
def test_collate_global_augmentation(augment):
    from metatrain.pet import PET

    options = PET(small_model_hypers(), INFO).requested_neighbor_lists()[0]
    collate_train, collate_val = get_collate_fns(
        options, _trainer_hypers(global_augmentation=augment)
    )
    systems = to_systems(ase.io.read(QM9_PATH, ":2"), torch.float64)
    torch.manual_seed(0)
    train, _, _ = collate_train.callables[0](systems, {}, {})
    val, _, _ = collate_val.callables[0](systems, {}, {})
    assert len(collate_val.callables) == 1
    assert len(collate_train.callables) == (2 if augment else 1)
    # validation is never augmented: same positions, skin neighbor list added
    torch.testing.assert_close(val[0].positions, systems[0].positions)
    assert options in val[0].known_neighbor_lists()
    if augment:
        assert not torch.allclose(train[0].positions, systems[0].positions)
        # a global O(3) transformation preserves interatomic distances
        torch.testing.assert_close(
            torch.cdist(train[0].positions, train[0].positions),
            torch.cdist(systems[0].positions, systems[0].positions),
        )


def test_train_latent_only(tmp_path):
    torch.manual_seed(0)
    dataset = _dataset()
    hypers = _trainer_hypers(num_epochs=1)
    hypers["objective"]["terms"] = {
        "latent": {"weight": 1.0},
        "vicreg": {"variance": 1.0, "covariance": 0.04},
    }
    trainer = Trainer(hypers)
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(
        model, torch.float32, [torch.device("cpu")], [dataset], [dataset], str(tmp_path)
    )
    assert trainer.epoch == 0
    assert (tmp_path / "model_0.ckpt").exists()
