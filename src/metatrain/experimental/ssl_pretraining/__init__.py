from metatrain.pet import PET

from . import high_order  # noqa: F401 (registers the ``high_order`` view builder)
from .trainer import Trainer


__model__ = PET
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("Lucas Pinède", "@LPinede"),
]

__maintainers__ = [
    ("Lucas Pinède", "@LPinede"),
]
