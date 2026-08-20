"""
Server-side aggregation — Algorithm 1, lines 11-13.
"""

from __future__ import annotations

import torch


def server_update(
    theta: torch.Tensor,
    y_list: list[torch.Tensor],
    g_list: list[torch.Tensor],
    delta: float,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate N agents' (y_i^k, g_i^k) into the new global y^k and theta^k.

        y^k     <- mean_i(y_i^k)                                   (line 12)
        theta^k <- theta^{k-1} + eta * sqrt(2*N*delta / (sum_i g_i^k)^T y^k) * y^k  (line 13)

    Returns (theta_new, y_new) -- y_new is broadcast to every agent as
    next round's y^{k-1} (line 3); theta_new gets loaded into every
    agent's policy via set_flat_params before the next rollout.

    `eta` is NOT from Table 4 -- see fednpg_admm.hyperparams module
    docstring for why. Defaults to 1.0 (take the ADMM-solved step
    unscaled), the first thing to tune if training is unstable.

    Raises if (sum g_i)^T y <= 0, rather than silently producing NaN from
    sqrt() of a negative number. Algorithm 1 has no line search and no
    fallback for this case (see the earlier TRPO-vs-NPG discussion) -- it
    can genuinely happen, most often early in training with a poorly fit
    critic, and corrupting theta with NaN silently is a far worse failure
    mode than stopping here with an explicit reason.
    """
    y = torch.stack(y_list).mean(dim=0)
    g_sum = torch.stack(g_list).sum(dim=0)
    N = len(y_list)

    denom = torch.dot(g_sum, y)
    if denom.item() <= 0:
        raise ValueError(
            f"(sum g_i)^T y = {denom.item():.6g} is non-positive -- the trust-region "
            "step size sqrt(2*N*delta/denom) is undefined. The aggregated ADMM "
            "direction isn't an ascent direction this round. Algorithm 1 as written "
            "has no fallback for this (no line search). Most common early in "
            "training with a badly-fit critic producing noisy advantages -- check "
            "critic_loss in the results before assuming the ADMM/CG code is wrong."
        )

    step_size = eta * torch.sqrt(2 * N * delta / denom)
    theta_new = theta + step_size * y
    return theta_new, y
