"""The high-order SSL method: irreps, analytic targets, deformations and views.

Importing this package registers the ``high_order`` view builder.
"""

from ..objective import register_view_builder
from .views import build_high_order_views


register_view_builder("high_order", build_high_order_views)
