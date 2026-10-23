"""Upgrades of the ssl_pretraining trainer checkpoints, one version at a time.

The trainer state is stored under the checkpoint key ``"grad_jepa"`` (the
architecture's name before it became ``experimental.ssl_pretraining``). The key is
kept as is, so that checkpoints written before the rename are read unchanged.
"""


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """Update a v1 checkpoint to v2: add the (empty) state of the loss terms.

    :param checkpoint: The checkpoint to update, in place.
    """
    checkpoint["grad_jepa"]["terms_state_dict"] = {}
