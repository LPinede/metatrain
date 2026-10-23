from metatrain.pet import PET

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
