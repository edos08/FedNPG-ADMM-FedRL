"""
Second-order optimization primitives shared by every algorithm here.

These live in `common` rather than inside `fednpg_admm` because the
`baselines` package needs the identical conjugate-gradient solver and
Fisher-vector product to implement standard-average FedNPG (paper eq. 12),
and sibling packages deliberately never import each other -- a duplicated
copy would be two things to keep in sync, and any drift between them would
silently invalidate the comparison between the two methods.
"""

from .cg import conjugate_gradient
from .line_search import line_search_step
from .fisher import kl_divergence_loss, make_fisher_vector_product

__all__ = ["conjugate_gradient", "line_search_step", "kl_divergence_loss", "make_fisher_vector_product"]
