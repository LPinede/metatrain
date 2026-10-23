"""Upgrades of the grad_jepa trainer checkpoints, one version at a time."""


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """Update a v1 checkpoint to v2: add the (empty) state of the loss terms.

    :param checkpoint: The checkpoint to update, in place.
    """
    checkpoint["grad_jepa"]["terms_state_dict"] = {}
