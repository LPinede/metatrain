import copy
from typing import List
from unittest import mock

import ase
import torch
from metatomic.torch import NeighborListOptions

from metatrain.experimental.ssl_pretraining.documentation import TrainerHypers
from metatrain.experimental.ssl_pretraining.high_order import views as ho_views
from metatrain.experimental.ssl_pretraining.objective import StepData, build_step_data
from metatrain.experimental.ssl_pretraining.tests import to_systems
from metatrain.experimental.ssl_pretraining.views import (
    build_view,
    get_skin_neighbor_list_transform,
)
from metatrain.utils.hypers import init_with_defaults


ATOMIC_TYPES = [1, 6, 7, 8]
# the neighbor list options of the small PET of ``ssl_pretraining/tests``
OPTIONS = NeighborListOptions(cutoff=4.0, full_list=True, strict=False)


def haar_o3(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``n`` Haar-distributed O(3) matrices (QR of a Gaussian, with the sign fix)."""
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, dtype=dtype))
    return q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]


def make_batch(frames: List[ase.Atoms], dtype: torch.dtype):
    """``(systems, options, hypers)``: the frames with the skin neighbor list, the
    neighbor list options and the default trainer hypers (with the runtime
    ``atomic_types`` key that the trainer adds)."""
    # deep copy: ``init_with_defaults`` shares the mutable defaults
    hypers = copy.deepcopy(init_with_defaults(TrainerHypers))
    hypers["atomic_types"] = list(ATOMIC_TYPES)
    transform = get_skin_neighbor_list_transform(OPTIONS, hypers["noise"]["skin"])
    systems, _, _ = transform(to_systems(frames, dtype), {}, {})
    return systems, OPTIONS, hypers


def make_step(batch, student, teacher, seed: int = 0, **over) -> StepData:
    """The step data of ``batch`` with the high-order view builder (no probe
    gradients, no predictor), with ``over`` applied to the ``high_order`` hypers."""
    systems, options, hypers = batch
    hypers = copy.deepcopy(hypers)
    hypers["views"]["builder"] = "high_order"
    hypers["high_order"].update(over)
    return build_step_data(
        student,
        teacher,
        None,
        systems,
        options,
        hypers,
        torch.Generator().manual_seed(seed),
        needs_gradients=False,
    )


def with_hidden_atom_moved(batch, step: StepData, shift: torch.Tensor):
    """``batch`` with the first masked atom of ``step`` moved by ``shift`` in the
    clean structure (the neighbor list samples are kept)."""
    systems, options, hypers = batch
    atom = int(step.views.masked.nonzero()[0])
    structure = int(step.extras["high_order"].structure_index[atom])
    start = sum(len(s) for s in systems[:structure])
    system = systems[structure]
    positions = system.positions.detach().clone()
    positions[atom - start] += shift.to(positions)
    moved = list(systems)
    moved[structure] = build_view(system, options, positions, system.cell)
    return moved, options, hypers


def mirrored(batch, student, teacher, seed: int = 0, **over) -> StepData:
    """:func:`make_step` with every student rotation ``g`` replaced by ``-g`` (the
    random stream is unchanged)."""
    original = ho_views.random_rotations

    def negated(*args, **kwargs):
        return -original(*args, **kwargs)

    with mock.patch.object(ho_views, "random_rotations", negated):
        return make_step(batch, student, teacher, seed, **over)


def randomize_heads(module: torch.nn.Module) -> torch.nn.Module:
    """Give the last layer of every ``BlockHead`` in ``module`` PyTorch's default
    random init (heads start at zero), for tests of the structure of non-zero
    predictions. Returns ``module``."""
    from metatrain.experimental.ssl_pretraining.high_order.heads import BlockHead

    for head in module.modules():
        if isinstance(head, BlockHead):
            for net in [head.node_net] + ([head.edge_net] if head.use_edges else []):
                net[-1].reset_parameters()
    return module
