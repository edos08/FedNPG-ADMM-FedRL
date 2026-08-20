"""
Per-agent ADMM update — Algorithm 1, lines 5-9.

theta is held fixed throughout this function (loaded into `policy` via
common.networks.set_flat_params before calling this, by train.py) — the
superscript k on H_i^k and g_i^k in the paper indexes the ITERATION's data,
not a change in theta mid-iteration. theta only changes at the outer
server update (line 13), once per iteration, after every agent has already
finished lines 5-9.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from common.networks.policy import GaussianMLPPolicy
from common.networks.utils import get_flat_grad
from common.rollout.buffer import RolloutBatch

from .cg import conjugate_gradient
from .fisher import make_fisher_vector_product


@dataclass
class AgentState:
    """
    Persistent per-agent ADMM state, carried across iterations. y_i is
    THIS agent's own y_i^{k-1} from its last local solve; lam_i is its
    dual variable. Both start at zero, per the paper's initialization
    ("Initial ... {y_i^0 = y^0}, {lambda_i}" -- y^0 is conventionally zero,
    and the paper doesn't specify a nonzero lambda_i init).
    """

    y_i: torch.Tensor
    lam_i: torch.Tensor

    @classmethod
    def initial(cls, dim: int, device: "torch.device | str" = "cpu") -> "AgentState":
        return cls(y_i=torch.zeros(dim, device=device), lam_i=torch.zeros(dim, device=device))


def agent_update(
    policy: GaussianMLPPolicy,
    batch: RolloutBatch,
    state: AgentState,
    y_prev: torch.Tensor,
    rho: float,
    cg_iters: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, AgentState]:
    """
    Run Algorithm 1, lines 5-9, for one agent, one iteration.

    `y_prev`: the SERVER's broadcast y^{k-1} from the end of the previous
    round (Algorithm 1 line 3) -- distinct from state.y_i, which is THIS
    agent's own previous local solve. Both appear in the paper's equations
    and are generally different vectors; that gap between them is exactly
    what the dual variable lam_i is tracking.

    Returns (y_i^k, g_i^k, updated AgentState). y_i^k and g_i^k are what
    get sent to the server -- through a common.comms.Channel, which this
    function does NOT touch; that's train.py's job, since whether/how
    those tensors get transformed in transit (quantization) is an
    orchestration concern, not an algorithm concern.
    """
    # line 6: dual variable update
    lam_i = state.lam_i + rho * (state.y_i - y_prev)

    # line 7: local policy gradient, score-function estimator.
    # advantages are DETACHED -- they come from the frozen critic/GAE
    # computation, not a function of the actor's parameters, and must not
    # contribute their own gradient here.
    log_probs = policy.log_prob(batch.obs, batch.actions)
    surrogate = (log_probs * batch.advantages.detach()).mean()
    g_i = get_flat_grad(surrogate, policy, retain_graph=True)

    # line 8: local ADMM solve, y_i^k = (H_i + rho*I)^-1 (g_i - lam_i + rho*y_prev)
    fvp = make_fisher_vector_product(policy, batch.obs)

    def Avp(v: torch.Tensor) -> torch.Tensor:
        return fvp(v) + rho * v

    b = g_i - lam_i + rho * y_prev
    y_i = conjugate_gradient(Avp, b, num_iters=cg_iters)

    new_state = AgentState(y_i=y_i, lam_i=lam_i)
    return y_i, g_i, new_state