"""
Standard-average FedNPG -- eq. (12) in the paper. The O(d^2) baseline that
FedNPG-ADMM exists to replace.

Paper eq. (12), verbatim in structure:

    theta <- theta + sqrt( 2*N*delta / (sum_i g_i)^T (sum_i H_i)^-1 (sum_i g_i) )
                     * (sum_i H_i)^-1 (sum_i g_i)

Each agent uplinks its full Fisher H_i (d x d) and its gradient g_i (d), so
the per-agent uplink is O(d^2) per iteration. FedNPG-ADMM transmits two
d-vectors instead, which is the entire point of the comparison.

WHY THIS SOLVES HESSIAN-FREE, AND WHY THAT IS STILL HONEST
----------------------------------------------------------
An earlier note in this file said the baseline must actually materialize
and invert H, so as to genuinely pay the O(d^2) cost. That is right about
the ACCOUNTING and impossible in fact: a dense H is 0.095 GB on Swimmer,
1.36 GB on Hopper (11 GB across 8 agents), and **2.1 PB on Humanoid**
(16.9 PB across 8 agents) -- four orders of magnitude beyond any machine.
The baseline would simply not exist for the paper's largest environment,
which is the environment where its O(d^2) cost matters most.

What the paper's claim is about is UPLINK BYTES, not server FLOPs: Fig. 3
measures "the number of transmitted parameters with double precision in
each agent". So this module:

  * solves the same linear system matrix-free, by conjugate gradient on the
    POOLED Fisher-vector product, which is mathematically the same
    direction that materializing sum_i H_i and inverting it would give; and
  * charges the channel the full d^2 elements per agent per iteration
    (train.py, via Channel.record_unmaterialized_send), so the reported
    communication overhead is exactly what eq. (12) would really cost.

The result is a baseline that is faithful on the axis being compared and
merely feasible on the axis that is not. Do NOT "fix" this by dropping the
d^2 metering -- that would make FedNPG-ADMM look free by comparison and the
whole experiment would measure nothing.

sum_i H_i v == N * fvp_pooled(v): each H_i is a mean over its own agent's
batch, and the pooled Fisher-vector product is a mean over all N batches
concatenated, so scaling by N recovers the sum.
"""

from __future__ import annotations

import torch

from common.networks.policy import GaussianMLPPolicy

from common.optim import conjugate_gradient, make_fisher_vector_product


def standard_average_step(
    policy: GaussianMLPPolicy,
    pooled_obs: torch.Tensor,
    g_sum: torch.Tensor,
    num_agents: int,
    delta: float,
    damping: float,
    cg_iters: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eq. (12)'s direction and step size. Returns (direction, step_size)
    WITHOUT applying them, so train.py can run the same TRPO-style line search
    the FedNPG-ADMM path uses -- otherwise the two arms would differ in their
    safeguards as well as in their aggregation, and the comparison would be
    confounded.

    `damping` is added to the pooled system exactly as rho damps the per-agent
    solves in Algorithm 1 line 8, so the two arms differ ONLY in how the linear
    system is solved. More CG iterations than the ADMM path's 10 because this
    solve is the reference the approximation is measured against.

    Raises ValueError when (sum g_i)^T d <= 0, matching server.py's contract:
    the step size is undefined rather than silently NaN. With no consensus
    error in this arm, that points at the gradient or critic, not aggregation.
    """
    fvp = make_fisher_vector_product(policy, pooled_obs)

    def Avp(v: torch.Tensor) -> torch.Tensor:
        return num_agents * fvp(v) + damping * v

    direction = conjugate_gradient(Avp, g_sum, num_iters=cg_iters).detach()

    denom = torch.dot(g_sum, direction)
    if denom.item() <= 0:
        raise ValueError(
            f"(sum g_i)^T (sum H_i)^-1 (sum g_i) = {denom.item():.6g} is non-positive; "
            "eq. (12)'s step size is undefined. This arm has no consensus error, so "
            "suspect the gradient or a badly-fit critic rather than the aggregation."
        )
    step_size = torch.sqrt(2 * num_agents * delta / denom)
    return direction, step_size
