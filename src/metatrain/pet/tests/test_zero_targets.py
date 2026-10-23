from metatomic.torch import load_atomistic_model

from metatrain.pet import Trainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo


def test_export_without_targets(tmp_path):
    """A PET without targets (e.g. self-supervised pretraining) can be exported."""
    hypers = get_default_hypers("pet")
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets={})
    model = Trainer(hypers["training"]).setup(hypers["model"], info)

    path = str(tmp_path / "model.pt")
    model.export().save(path)
    assert "feature" in load_atomistic_model(path).capabilities().outputs
