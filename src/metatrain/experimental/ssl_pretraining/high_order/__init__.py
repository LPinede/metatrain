"""The high-order SSL method: irreps, analytic targets, deformations, views and loss
terms.

Importing this package registers the ``high_order`` view builder and the loss terms
``residual_multipole``, ``deformation_jets``, ``chirality``,
``equivariant_vicreg`` and ``vicreg_averaged``.
"""

from ..objective import register_view_builder
from ..terms import register_term
from .terms import (
    ChiralityTerm,
    DeformationJetTerm,
    EquivariantVicregTerm,
    ResidualMultipoleTerm,
    RotationAveragedVicregTerm,
)
from .views import build_high_order_views


register_view_builder("high_order", build_high_order_views)
for _term in (
    ResidualMultipoleTerm,
    DeformationJetTerm,
    ChiralityTerm,
    EquivariantVicregTerm,
    RotationAveragedVicregTerm,
):
    register_term(_term)
