import copy

import ase.io
import pytest
import torch

from metatrain.experimental.ssl_pretraining import objective
from metatrain.experimental.ssl_pretraining import terms as terms_module
from metatrain.experimental.ssl_pretraining import trainer as trainer_module
from metatrain.experimental.ssl_pretraining.documentation import TrainerHypers
from metatrain.experimental.ssl_pretraining.trainer import (
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


def _assert_same_tensors(state, expected) -> None:
    assert set(state) == set(expected)
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):  # skip ``_extra_state`` entries
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)


def _spy_first_step(monkeypatch) -> dict:
    """Record the student, teacher, predictor and term states at the first step."""
    first: dict = {}
    loss_function = trainer_module.grad_jepa_loss

    def spy(student, teacher, predictor, terms, *args, **kwargs):
        if not first:
            modules = {
                "student": student,
                "teacher": teacher,
                "predictor": predictor,
            }
            for term in terms:
                if isinstance(term, torch.nn.Module):
                    modules[f"term:{term.name}"] = term
            for name, module in modules.items():
                first[name] = copy.deepcopy(module.state_dict())
        return loss_function(student, teacher, predictor, terms, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "grad_jepa_loss", spy)
    return first


def test_train_save_resume_and_finetune(tmp_path, monkeypatch):
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
    assert checkpoint["trainer_ckpt_version"] == 2
    assert set(checkpoint["grad_jepa"]) == {
        "step",
        "teacher_state_dict",
        "predictor_state_dict",
        "terms_state_dict",
    }
    assert checkpoint["grad_jepa"]["terms_state_dict"] == {}
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
    first = _spy_first_step(monkeypatch)
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
    # the first resumed step sees the saved student, teacher and predictor
    _assert_same_tensors(first["student"], model.core.state_dict())
    _assert_same_tensors(
        first["teacher"], checkpoint["grad_jepa"]["teacher_state_dict"]
    )
    _assert_same_tensors(
        first["predictor"], checkpoint["grad_jepa"]["predictor_state_dict"]
    )

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


def test_train_latent_only(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("probe gradients computed in latent-only mode")

    monkeypatch.setattr(objective, "random_scalar_gradients", forbidden)
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


def _labels_devices(module: torch.nn.Module) -> set:
    """Devices of the metatensor buffers (e.g. PET's ``single_label``) of a module."""
    devices = set()
    for submodule in module.modules():
        for name in getattr(submodule, "_mts_buffer_names", []):
            value = getattr(submodule, name)
            if hasattr(value, "device"):
                devices.add(torch.device(value.device).type)
    return devices


def _train_then_restart(tmp_path, device: torch.device) -> None:
    """A fresh 1-epoch run, then a ``restart_from`` run for one more epoch."""
    torch.manual_seed(0)
    dataset = _dataset()
    trainer = Trainer(_trainer_hypers(num_epochs=1))
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(model, torch.float32, [device], [dataset], [dataset], str(tmp_path))
    path = tmp_path / "ssl.ckpt"
    trainer.save_checkpoint(model, path)
    resumed = Trainer(_trainer_hypers(num_epochs=2, restart_from=str(path)))
    resumed_model = resumed.setup(small_model_hypers(), INFO)
    resumed.train(
        resumed_model, torch.float32, [device], [dataset], [dataset], str(tmp_path)
    )


def test_modules_are_moved_after_copies_and_loads(tmp_path, monkeypatch):
    """CPU check of the device bug fixed for GPUs.

    ``copy.deepcopy`` and ``load_state_dict`` leave metatensor Labels buffers on the
    CPU, whatever the device of the module. A device move (``_apply``) must therefore
    come after the teacher is copied and after every state load, for both the
    student and the teacher. On a CPU-only machine the devices cannot differ, so the
    order of these events is recorded and checked instead.
    """
    from metatrain.pet import PET

    moved: dict = {}  # id(PET) -> True if moved since its last copy or load
    original_deepcopy = copy.deepcopy
    original_apply = PET._apply
    original_load = PET._load_from_state_dict

    def deepcopy(value, *args, **kwargs):
        result = original_deepcopy(value, *args, **kwargs)
        if isinstance(result, PET):
            moved[id(result)] = False
        return result

    def apply(self, *args, **kwargs):
        result = original_apply(self, *args, **kwargs)
        moved[id(self)] = True
        return result

    def load(self, *args, **kwargs):
        moved[id(self)] = False
        return original_load(self, *args, **kwargs)

    seen = []
    loss_function = trainer_module.grad_jepa_loss

    def spy(student, teacher, *args, **kwargs):
        seen.append((moved.get(id(student), False), moved.get(id(teacher), False)))
        return loss_function(student, teacher, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", deepcopy)
    monkeypatch.setattr(PET, "_apply", apply)
    monkeypatch.setattr(PET, "_load_from_state_dict", load)
    monkeypatch.setattr(trainer_module, "grad_jepa_loss", spy)
    _train_then_restart(tmp_path, torch.device("cpu"))
    assert len(seen) > 0
    assert all(student and teacher for student, teacher in seen)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_train_and_restart_on_mps(tmp_path, monkeypatch):
    """Every metatensor buffer of the student and teacher is on the device."""
    seen = []
    loss_function = trainer_module.grad_jepa_loss

    def spy(student, teacher, *args, **kwargs):
        seen.append(_labels_devices(student) | _labels_devices(teacher))
        return loss_function(student, teacher, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "grad_jepa_loss", spy)
    _train_then_restart(tmp_path, torch.device("mps"))
    assert len(seen) > 0
    assert all(devices == {"mps"} for devices in seen)


class _ScaleTerm(torch.nn.Module):
    """A test-only term with a parameter and a buffer counting training calls."""

    name = "scale"
    needs_gradients = False

    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight
        self.required_targets: dict = {}
        # float64, so that the test can check the trainer's dtype move
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
        self.register_buffer("training_calls", torch.zeros((), dtype=torch.float64))

    def forward(self, step):
        if self.training:
            self.training_calls += 1
        loss = self.weight * (self.scale * step.student_features).pow(2).mean()
        return loss, {"scale": self.scale.detach()}


def _with_scale_term(monkeypatch, **overrides) -> dict:
    monkeypatch.setitem(terms_module.TERMS, "scale", _ScaleTerm)
    hypers = _trainer_hypers(**overrides)
    hypers["objective"]["terms"]["scale"] = {"weight": 1.0}
    return hypers


def test_parametric_term_is_trained_saved_and_restored(tmp_path, monkeypatch):
    torch.manual_seed(0)
    dataset = _dataset()
    trainer = Trainer(_with_scale_term(monkeypatch, num_epochs=2))
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(
        model, torch.float32, [torch.device("cpu")], [dataset], [dataset], str(tmp_path)
    )
    path = tmp_path / "ssl.ckpt"
    trainer.save_checkpoint(model, path)
    state = torch.load(path, weights_only=False)["grad_jepa"]["terms_state_dict"]
    assert set(state) == {"scale.scale", "scale.training_calls"}
    # moved to the training dtype, and trained by the optimizer
    assert state["scale.scale"].dtype == torch.float32
    assert state["scale.scale"] != 1.0
    # in training mode for the training steps of both epochs, not for validation
    assert state["scale.training_calls"] == trainer.step

    first = _spy_first_step(monkeypatch)
    resumed = Trainer(
        _with_scale_term(monkeypatch, num_epochs=3, restart_from=str(path))
    )
    resumed_model = resumed.setup(small_model_hypers(), INFO)
    resumed.train(
        resumed_model,
        torch.float32,
        [torch.device("cpu")],
        [dataset],
        [dataset],
        str(tmp_path),
    )
    _assert_same_tensors(
        {f"scale.{name}": value for name, value in first["term:scale"].items()}, state
    )
    assert resumed.terms_state_dict["scale.training_calls"] == resumed.step


def _v1_checkpoint(tmp_path) -> dict:
    """A checkpoint as saved by trainer version 1 (no term state)."""
    torch.manual_seed(0)
    dataset = _dataset()
    trainer = Trainer(_trainer_hypers(num_epochs=1))
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(
        model, torch.float32, [torch.device("cpu")], [dataset], [dataset], str(tmp_path)
    )
    trainer.save_checkpoint(model, tmp_path / "ssl.ckpt")
    checkpoint = torch.load(tmp_path / "ssl.ckpt", weights_only=False)
    checkpoint["trainer_ckpt_version"] = 1
    checkpoint["grad_jepa"].pop("terms_state_dict", None)
    return checkpoint


def test_upgrade_checkpoint_v1_to_v2(tmp_path):
    checkpoint = Trainer.upgrade_checkpoint(_v1_checkpoint(tmp_path))
    assert checkpoint["trainer_ckpt_version"] == 2
    assert checkpoint["grad_jepa"]["terms_state_dict"] == {}


def test_restart_from_upgrades_a_v1_checkpoint(tmp_path):
    path = tmp_path / "v1.ckpt"
    torch.save(_v1_checkpoint(tmp_path), path)
    resumed = Trainer(_trainer_hypers(num_epochs=2, restart_from=str(path)))
    resumed_model = resumed.setup(small_model_hypers(), INFO)
    resumed.train(
        resumed_model,
        torch.float32,
        [torch.device("cpu")],
        [_dataset()],
        [_dataset()],
        str(tmp_path),
    )
    assert resumed.epoch == 1


@pytest.mark.parametrize("version", [0, 99])
def test_restart_from_rejects_unknown_versions(tmp_path, version):
    checkpoint = _v1_checkpoint(tmp_path)
    checkpoint["trainer_ckpt_version"] = version
    path = tmp_path / "unknown.ckpt"
    torch.save(checkpoint, path)
    trainer = Trainer(_trainer_hypers(restart_from=str(path)))
    model = trainer.setup(small_model_hypers(), INFO)
    with pytest.raises((RuntimeError, ValueError), match=f"version {version}"):
        trainer.train(
            model,
            torch.float32,
            [torch.device("cpu")],
            [_dataset()],
            [_dataset()],
            str(tmp_path),
        )


class _Recorder:
    """A test-only term built with the build context; records the step fractions."""

    name = "recorder"
    needs_gradients = False
    uses_prediction = False
    required_targets: dict = {}
    built: list = []

    def __init__(self, context) -> None:
        self.context = context
        self.fractions: list = []
        _Recorder.built.append(self)

    @classmethod
    def from_hypers(cls, hypers, context):
        return cls(context, **hypers)

    def __call__(self, step):
        self.fractions.append(step.extras["step_fraction"])
        return step.student_features.sum() * 0.0, {}


def test_trainer_passes_the_build_context_and_step_fraction(tmp_path, monkeypatch):
    monkeypatch.setitem(terms_module.TERMS, "recorder", _Recorder)
    monkeypatch.setattr(_Recorder, "built", [])
    hypers = _trainer_hypers(num_epochs=1, num_probes=3)
    hypers["objective"]["terms"]["recorder"] = {}
    trainer = Trainer(hypers)
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(
        model,
        torch.float32,
        [torch.device("cpu")],
        [_dataset()],
        [_dataset()],
        str(tmp_path),
    )
    (term,) = _Recorder.built
    core = model.core
    n = core.num_readout_layers
    assert term.context.dims == (n * core.d_node, n * core.d_pet)
    assert term.context.cutoff == core.requested_neighbor_lists()[0].cutoff
    assert term.context.atomic_types == ATOMIC_TYPES
    assert term.context.num_probes == 3
    n_train = trainer.step
    assert term.fractions[:n_train] == [i / n_train for i in range(n_train)]
    assert len(term.fractions) > n_train
    assert all(f == 1.0 for f in term.fractions[n_train:])


class _EpochHook(_Recorder):
    """A test-only term with the validation-epoch hook; records its arguments."""

    name = "epoch_hook"

    def __init__(self, context) -> None:
        super().__init__(context)
        self.epochs: list = []

    def __call__(self, step):
        loss, _ = super().__call__(step)
        return loss, {"epoch_hook/value": loss.new_tensor(2.0)}

    def on_validation_epoch_end(self, diagnostics) -> None:
        self.epochs.append(dict(diagnostics))


def test_trainer_calls_the_validation_epoch_hook(tmp_path, monkeypatch):
    monkeypatch.setitem(terms_module.TERMS, "epoch_hook", _EpochHook)
    monkeypatch.setattr(_Recorder, "built", [])
    hypers = _trainer_hypers(num_epochs=2)
    hypers["objective"]["terms"]["epoch_hook"] = {}
    trainer = Trainer(hypers)
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(
        model,
        torch.float32,
        [torch.device("cpu")],
        [_dataset()],
        [_dataset()],
        str(tmp_path),
    )
    (term,) = _Recorder.built
    # once per epoch, with the epoch-averaged validation diagnostics
    assert len(term.epochs) == 2
    assert all(epoch["epoch_hook/value"] == 2.0 for epoch in term.epochs)
    assert all("loss" in epoch for epoch in term.epochs)
    assert term.context.rotation_group == "O3"
    core = model.core
    n = core.num_readout_layers
    assert term.context.feature_dim == n * (core.d_node + core.d_pet)


def test_training_without_prediction_terms_builds_no_predictor(tmp_path):
    hypers = _trainer_hypers(num_epochs=1)
    hypers["objective"]["terms"] = {"vicreg": {"variance": 1.0, "covariance": 0.04}}
    trainer = Trainer(hypers)
    model = trainer.setup(small_model_hypers(), INFO)
    trainer.train(
        model,
        torch.float32,
        [torch.device("cpu")],
        [_dataset()],
        [_dataset()],
        str(tmp_path),
    )
    assert trainer.predictor_state_dict is None
    trainer.save_checkpoint(model, tmp_path / "ssl.ckpt")
    resumed = Trainer(
        {**hypers, "num_epochs": 2, "restart_from": str(tmp_path / "ssl.ckpt")}
    )
    resumed_model = resumed.setup(small_model_hypers(), INFO)
    resumed.train(
        resumed_model,
        torch.float32,
        [torch.device("cpu")],
        [_dataset()],
        [_dataset()],
        str(tmp_path),
    )
    assert resumed.epoch == 1


def _core_parameters(model) -> dict:
    return {k: v.detach().clone() for k, v in model.core.named_parameters()}


def test_freeze_backbone_trains_only_the_heads(tmp_path, monkeypatch):
    torch.manual_seed(0)
    dataset = _dataset()
    trainer = Trainer(_with_scale_term(monkeypatch, num_epochs=2, freeze_backbone=True))
    model = trainer.setup(small_model_hypers(), INFO)
    initial = _core_parameters(model)
    trainer.train(
        model, torch.float32, [torch.device("cpu")], [dataset], [dataset], str(tmp_path)
    )
    for name, value in model.core.named_parameters():
        assert not value.requires_grad
        torch.testing.assert_close(value, initial[name])
    # the term and the predictor are still trained
    assert trainer.terms_state_dict["scale.scale"] != 1.0
    assert trainer.predictor_state_dict is not None
    # the teacher is not updated and stays equal to the student
    teacher = trainer.teacher_state_dict
    for name, value in model.core.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(teacher[name], value)
    # the exported weights are the frozen random backbone
    for name, value in initial.items():
        torch.testing.assert_close(trainer.best_model_state_dict[f"core.{name}"], value)

    # checkpoint and restart
    path = tmp_path / "frozen.ckpt"
    trainer.save_checkpoint(model, path)
    resumed = Trainer(
        _with_scale_term(
            monkeypatch, num_epochs=3, freeze_backbone=True, restart_from=str(path)
        )
    )
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
    for name, value in resumed_model.core.named_parameters():
        torch.testing.assert_close(value, initial[name])
    assert (
        resumed.terms_state_dict["scale.scale"]
        != trainer.terms_state_dict["scale.scale"]
    )


def test_freeze_backbone_needs_something_to_train(tmp_path):
    hypers = _trainer_hypers(freeze_backbone=True)
    vicreg = hypers["objective"]["terms"]["vicreg"]
    hypers["objective"]["terms"] = {name: None for name in hypers["objective"]["terms"]}
    hypers["objective"]["terms"]["vicreg"] = vicreg
    trainer = Trainer(hypers)
    model = trainer.setup(small_model_hypers(), INFO)
    dataset = _dataset()
    with pytest.raises(ValueError, match="freeze_backbone"):
        trainer.train(
            model,
            torch.float32,
            [torch.device("cpu")],
            [dataset],
            [dataset],
            str(tmp_path),
        )
