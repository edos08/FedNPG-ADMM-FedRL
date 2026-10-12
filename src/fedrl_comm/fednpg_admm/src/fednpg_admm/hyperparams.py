"""
Re-export of the paper's Table 4 values, which now live in
`common.hyperparams` so the `baselines` package can read the identical
numbers without importing this one (`baselines` and `fednpg_admm` never
import each other). Kept as a module so existing
`from .hyperparams import DEFAULT_HYPERPARAMS` imports keep working.
"""

from common.hyperparams import DEFAULT_HYPERPARAMS, HyperParams

__all__ = ["DEFAULT_HYPERPARAMS", "HyperParams"]
