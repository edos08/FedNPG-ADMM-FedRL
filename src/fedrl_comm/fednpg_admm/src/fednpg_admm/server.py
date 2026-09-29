"""
Server-side aggregation — Algorithm 1, lines 11-13 — plus an optional
TRPO-style backtracking line search around line 13's closed-form step.

The line search is NOT in Algorithm 1's pseudocode, which applies the
closed-form step unconditionally. It exists because the paper's own Setup
section (Simulations, page 7) states: "We follow the practical settings in
TRPO with line search [38]" (ref. [38] is Schulman et al.'s TRPO paper) --
i.e. the paper's OWN reported Fig. 2/Table 2 results were produced with a
practical safeguard the pseudocode doesn't mention. Without it, a single
noisy round's step (bad H_i estimate, unlucky advantage estimates) gets
applied to theta in full and compounds every iteration, with nothing to
reject or shrink it -- Algorithm 1's closed-form step size only controls
the LINEAR approximation of KL moved, not the true, exact KL, so it can
silently overshoot the intended trust region.
"""

from __future__ import annotations

import torch
from torch.distributions import Independent, Normal, kl_divergence

from common.networks.policy import GaussianMLPPolicy
from common.networks.utils import set_flat_params


def _compute_candidate(
    y_list: list[torch.Tensor],
    g_list: list[torch.Tensor],
    delta: float,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shared by server_update and compute_candidate_step: aggregate into
    y^k (line 12) and the closed-form step size (line 13), without
    applying it to theta yet. Raises ValueError under the same condition
    server_update documents (see its docstring).
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
    return y, step_size


def server_update(
    theta: torch.Tensor,
    y_list: list[torch.Tensor],
    g_list: list[torch.Tensor],
    delta: float,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate N agents' (y_i^k, g_i^k) into the new global y^k and theta^k,
    applying Algorithm 1 line 13's closed-form step UNCONDITIONALLY -- no
    line search. This is the paper-literal path (used when line search is
    disabled); see compute_candidate_step + line_search_step for the
    practical path this project uses by default.

        y^k     <- mean_i(y_i^k)                                   (line 12)
        theta^k <- theta^{k-1} + eta * sqrt(2*N*delta / (sum_i g_i^k)^T y^k) * y^k  (line 13)

    Returns (theta_new, y_new) -- y_new is broadcast to every agent as
    next round's y^{k-1} (line 3); theta_new gets loaded into every
    agent's policy via set_flat_params before the next rollout.

    `eta` is NOT from Table 4 -- see fednpg_admm.hyperparams module
    docstring for why. Defaults to 1.0 (take the ADMM-solved step
    unscaled), the first thing to tune if training is unstable.

    Raises if (sum g_i)^T y <= 0, rather than silently producing NaN from
    sqrt() of a negative number -- see _compute_candidate.
    """
    y, step_size = _compute_candidate(y_list, g_list, delta, eta)
    theta_new = theta + step_size * y
    return theta_new, y


def compute_candidate_step(
    y_list: list[torch.Tensor],
    g_list: list[torch.Tensor],
    delta: float,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Public entry point for a caller that wants (y_global, step_size)
    BEFORE committing to theta_new -- e.g. line_search_step below, which
    needs the closed-form step as its backtracking starting point. Same
    ValueError semantics as server_update (see _compute_candidate).
    """
    return _compute_candidate(y_list, g_list, delta, eta)


# Re-exported so `from .server import line_search_step` keeps working; the
# implementation moved to common.optim so `baselines` can use the same one.
from common.optim import line_search_step  # noqa: E402,F401
