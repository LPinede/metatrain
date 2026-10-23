"""Backbone adapters (ADR 0001).

The SSL wrapper reaches its backbone only through an adapter: set-up, the checks
on the model hypers, the per-atom feature dimension, the neighbor list options,
the per-atom ``"feature"`` output and, for heads that read them, the node and edge
tokens. Only PET is supported (``backbone: pet``).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from metatomic.torch import ModelOutput, NeighborListOptions, System

from metatrain.pet import Trainer as PETTrainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.wrapper import MetatrainModel


FEATURE_OUTPUT = {"feature": ModelOutput(sample_kind="atom")}


@dataclass
class HeadInputs:
    """Per-atom backbone tokens that SSL heads may read."""

    node: torch.Tensor  # (n_atoms, Dn)
    edge: torch.Tensor  # (n_atoms, max_neighbors, De)
    cutoff: torch.Tensor  # (n_atoms, max_neighbors), 0 on padded neighbors

    def select(self, index: torch.Tensor) -> "HeadInputs":
        """The tokens of a subset of atoms.

        :param index: A boolean mask or an integer index over the atoms.
        :return: The selected tokens.
        """
        return HeadInputs(self.node[index], self.edge[index], self.cutoff[index])

    @property
    def dims(self) -> Tuple[int, int]:
        """``(Dn, De)``, the node and edge token dimensions."""
        return self.node.shape[1], self.edge.shape[2]


class Backbone(Protocol):
    """What the SSL wrapper needs from a backbone."""

    name: str

    def setup(
        self, model_hypers: Dict[str, Any], dataset_info: DatasetInfo
    ) -> MetatrainModel:
        """Build a fresh model; ``model_hypers["backbone"]`` is not passed on.

        :param model_hypers: The model hypers of the SSL wrapper.
        :param dataset_info: The dataset info.
        :return: The model, whose core is a stock backbone.
        """

    def feature_dim(self, core: torch.nn.Module) -> int:
        """Dimension of the per-atom ``"feature"`` output.

        :param core: The backbone core.
        :return: The feature dimension.
        """

    def token_dims(self, core: torch.nn.Module) -> Tuple[int, int]:
        """The node and edge token dimensions (:attr:`HeadInputs.dims`).

        :param core: The backbone core.
        :return: ``(Dn, De)``.
        """

    def neighbor_list_options(self, core: torch.nn.Module) -> NeighborListOptions:
        """The neighbor list the backbone reads.

        :param core: The backbone core.
        :return: The neighbor list options.
        """

    def features(self, core: torch.nn.Module, systems: List[System]) -> torch.Tensor:
        """The per-atom ``"feature"`` output.

        :param core: The backbone core.
        :param systems: The systems, with the backbone's neighbor list.
        :return: The features, ``(n_atoms, feature_dim)``.
        """

    def forward_with_tokens(
        self, core: torch.nn.Module, systems: List[System]
    ) -> Tuple[torch.Tensor, Optional[HeadInputs]]:
        """The per-atom features and the tokens they are built from.

        :param core: The backbone core.
        :param systems: The systems, with the backbone's neighbor list.
        :return: The features (as :meth:`features`) and the tokens, or ``None``
            if the backbone does not expose them.
        """

    def validate(self, model_hypers: Dict[str, Any]) -> None:
        """Reject the model hypers that the SSL wrapper does not support.

        :param model_hypers: The model hypers of the SSL wrapper.
        """


class _CaptureTokens:
    """Intercepts ``core.backend.calculate_features`` for one forward pass and
    keeps the node and edge tokens it returns (PET itself is not modified).

    :param core: The PET core.
    """

    def __init__(self, core: torch.nn.Module) -> None:
        self.backend = core.backend
        self.node: Optional[torch.Tensor] = None
        self.edge: Optional[torch.Tensor] = None
        self.cutoff: Optional[torch.Tensor] = None

    def __enter__(self) -> "_CaptureTokens":
        original = self.backend.calculate_features

        def wrapped(
            batch_data: Dict[str, torch.Tensor], **kwargs: Any
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
            nodes, edges = original(batch_data, **kwargs)
            self.node = torch.cat(nodes, dim=1)
            self.edge = torch.cat(edges, dim=2)
            self.cutoff = batch_data["cutoff_factors"]
            return nodes, edges

        # an instance attribute shadows the method until ``__exit__`` removes it
        setattr(self.backend, "calculate_features", wrapped)  # noqa: B010
        return self

    def __exit__(self, *exc: Any) -> None:
        delattr(self.backend, "calculate_features")


class PETBackbone:
    """The PET adapter."""

    name = "pet"

    def setup(
        self, model_hypers: Dict[str, Any], dataset_info: DatasetInfo
    ) -> MetatrainModel:
        pet_hypers = {k: v for k, v in model_hypers.items() if k != "backbone"}
        pet_trainer = PETTrainer(get_default_hypers("pet")["training"])
        return pet_trainer.setup(pet_hypers, dataset_info)  # type: ignore[arg-type]

    def feature_dim(self, core: torch.nn.Module) -> int:
        return int(core.num_readout_layers * (core.d_node + core.d_pet))

    def token_dims(self, core: torch.nn.Module) -> Tuple[int, int]:
        n = int(core.num_readout_layers)
        return n * int(core.d_node), n * int(core.d_pet)

    def neighbor_list_options(self, core: torch.nn.Module) -> NeighborListOptions:
        return core.requested_neighbor_lists()[0]

    def features(self, core: torch.nn.Module, systems: List[System]) -> torch.Tensor:
        return core(systems, FEATURE_OUTPUT)["feature"].block().values

    def forward_with_tokens(
        self, core: torch.nn.Module, systems: List[System]
    ) -> Tuple[torch.Tensor, Optional[HeadInputs]]:
        with _CaptureTokens(core) as capture:
            features = self.features(core, systems)
        assert capture.node is not None
        assert capture.edge is not None
        assert capture.cutoff is not None
        return features, HeadInputs(capture.node, capture.edge, capture.cutoff)

    def validate(self, model_hypers: Dict[str, Any]) -> None:
        if model_hypers["long_range"]["enable"]:
            # the captured node tokens precede the long-range addition
            raise ValueError(
                "ssl_pretraining does not support PET's long-range module yet"
            )
        if model_hypers["num_neighbors_adaptive"] is not None:
            # masking removes neighbors, which would change the adaptive cutoffs
            raise ValueError(
                "ssl_pretraining does not support PET's adaptive cutoff yet"
            )


BACKBONES: Dict[str, Backbone] = {"pet": PETBackbone()}
"""The backbone adapters, by the ``backbone`` model hyperparameter."""
