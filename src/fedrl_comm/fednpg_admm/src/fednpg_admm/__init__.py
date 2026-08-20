"""
fednpg_admm — Algorithm 1 from Lan et al. 2023 (arXiv:2310.19807).

CLI: fedrl-train-admm (see train.py). Also directly importable for use
outside the CLI (e.g. from a notebook, or from quantization's sweep).
"""

from .agent import AgentState, agent_update
from .cg import conjugate_gradient
from .fisher import kl_divergence_loss, make_fisher_vector_product
from .hyperparams import DEFAULT_HYPERPARAMS, HyperParams
from .server import server_update
from .train import train

__version__ = "0.1.0"

__all__ = [
    "AgentState",
    "agent_update",
    "conjugate_gradient",
    "kl_divergence_loss",
    "make_fisher_vector_product",
    "DEFAULT_HYPERPARAMS",
    "HyperParams",
    "server_update",
    "train",
]
