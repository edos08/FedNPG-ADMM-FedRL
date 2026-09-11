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
from typing import Callable

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


def compute_local_gradient(
    policy: GaussianMLPPolicy, batch: RolloutBatch, rho: float
) -> tuple[torch.Tensor, "Callable[[torch.Tensor], torch.Tensor]"]:
    """
    Algorithm 1 line 7 plus the (H_i + rho*I) operator line 8 solves
    against, as one reusable pair.

    Split out of agent_update so BOTH can be computed once per iteration
    and reused across several ADMM rounds (train.py's --admm-rounds): g_i
    and H_i depend only on (theta, this agent's batch), and neither
    changes while the outer theta is held fixed -- only y^{k-1} and
    lambda_i move between rounds. Recomputing them per round would be
    pure waste, and rebuilding the Fisher graph repeatedly is the
    expensive half.
    """
    # advantages are DETACHED -- they come from the frozen critic/GAE
    # computation, not a function of the actor's parameters, and must not
    # contribute their own gradient here.
    log_probs = policy.log_prob(batch.obs, batch.actions)
    surrogate = (log_probs * batch.advantages.detach()).mean()
    g_i = get_flat_grad(surrogate, policy, retain_graph=True)

    fvp = make_fisher_vector_product(policy, batch.obs)

    def Avp(v: torch.Tensor) -> torch.Tensor:
        return fvp(v) + rho * v

    return g_i, Avp


def local_admm_solve(
    Avp: "Callable[[torch.Tensor], torch.Tensor]",
    g_i: torch.Tensor,
    state: AgentState,
    y_prev: torch.Tensor,
    rho: float,
    cg_iters: int = 10,
) -> tuple[torch.Tensor, AgentState]:
    """
    Algorithm 1 lines 6 and 8 -- the only two lines that change between
    ADMM rounds for a fixed theta and batch:

        lambda_i <- lambda_i + rho * (y_i - y^{k-1})            (line 6)
        y_i      <- (H_i + rho*I)^-1 (g_i - lambda_i + rho*y^{k-1})  (line 8)

    Calling this repeatedly with `y_prev` refreshed to the latest
    mean_i(y_i) is standard consensus ADMM, and is what --admm-rounds > 1
    does. Calling it exactly once per iteration is Algorithm 1 as
    literally written.
    """
    lam_i = state.lam_i + rho * (state.y_i - y_prev)
    b = g_i - lam_i + rho * y_prev
    y_i = conjugate_gradient(Avp, b, num_iters=cg_iters)
    return y_i, AgentState(y_i=y_i, lam_i=lam_i)


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

    One round only -- see compute_local_gradient / local_admm_solve,
    which this is now a thin composition of, for the multi-round path.
    """
    g_i, Avp = compute_local_gradient(policy, batch, rho)  # line 7
    y_i, new_state = local_admm_solve(Avp, g_i, state, y_prev, rho, cg_iters)  # lines 6, 8
    return y_i, g_i, new_state