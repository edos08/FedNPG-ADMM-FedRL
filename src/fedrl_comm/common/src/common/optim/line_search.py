"""
TRPO-style backtracking line search, shared by every algorithm here.

Lives in `common` because both FedNPG-ADMM (fednpg_admm/server.py) and the
standard-average FedNPG baseline (baselines/) must apply the IDENTICAL
safeguard -- if the two arms differed in their line search as well as in
their aggregation, the paper's Figure 3 comparison would be confounded by
the safeguard rather than measuring the algorithms.
"""

from __future__ import annotations

import torch
from torch.distributions import Independent, Normal, kl_divergence

from common.networks.policy import GaussianMLPPolicy
from common.networks.utils import set_flat_params


@torch.no_grad()
def line_search_step(
    policy: GaussianMLPPolicy,
    theta: torch.Tensor,
    y: torch.Tensor,
    step_size: torch.Tensor,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    delta: float,
    max_backtracks: int = 10,
    backtrack_coeff: float = 0.8,
) -> tuple[torch.Tensor, int, float]:
    """
    TRPO-style backtracking line search around Algorithm 1 line 13's
    closed-form step (see module docstring for why this exists).

    Starts from `step_size` (line 13's value, optimal only under a LINEAR
    approximation of the KL constraint) and shrinks by `backtrack_coeff`
    up to `max_backtracks` times, accepting the first candidate theta for
    which BOTH:
      - the ACTUAL KL divergence moved (not the linear approximation) is
        within `delta`, and
      - TRPO's importance-sampled surrogate objective
        `mean(advantage * pi_new(a|s)/pi_old(a|s))` actually improves
        versus theta (where at theta the ratio is 1, so the baseline is
        simply `advantages.mean()`),
    hold, evaluated on `obs`/`actions`/`advantages`. These are expected to
    be POOLED across every agent's local batch -- the paper doesn't
    specify how a federated line search should evaluate KL/improvement;
    pooling mirrors the same choice already made for critic fitting in
    train.py, evaluating the shared policy's candidate update against
    every agent's collected data at once rather than any single agent's.

    The acceptance test uses the RATIO surrogate, matching sb3-contrib's
    TRPO (`(advantages * ratio).mean()`) and Schulman's original, rather
    than the score-function form `mean(log_prob * advantage)` this used to
    compare. That earlier form is the right thing for the GRADIENT --
    Algorithm 1 line 7 defines g_i exactly that way, and the two coincide
    there anyway since the ratio is 1 at theta -- but it is NOT a valid
    measure of surrogate improvement at a DISPLACED theta, so using it to
    accept/reject a step was comparing the wrong quantity.

    Returns (accepted_theta, num_backtracks_used, achieved_kl). If no
    candidate in the sequence satisfies both conditions, returns theta
    UNCHANGED with num_backtracks_used == max_backtracks and achieved_kl
    as NaN -- the same "skip this iteration's theta update" outcome as
    server_update's ValueError path, just for a different failure mode (a
    well-defined but empirically-bad step, rather than an undefined one).

    Mutates `policy`'s live parameters as a side effect of evaluating
    each candidate (via set_flat_params) -- always left holding whichever
    theta this function returns by the time it returns, so callers don't
    need to re-sync afterward themselves.
    """
    # `policy` still holds theta on entry (train.py broadcasts it at the
    # top of the iteration and agent_update only reads it), so this IS
    # log pi_theta_old(a|s) -- the denominator of the importance ratio.
    old_log_prob = policy.log_prob(obs, actions)
    old_mean = policy.mean(obs)
    old_std = policy.std()
    old_dist = Independent(Normal(old_mean, old_std), 1)
    old_surrogate = advantages.mean()  # ratio == 1 at theta, by construction

    for i in range(max_backtracks):
        candidate_theta = theta + step_size * y
        set_flat_params(policy, candidate_theta)

        new_dist = policy.distribution(obs)
        kl = kl_divergence(old_dist, new_dist).mean()
        ratio = torch.exp(policy.log_prob(obs, actions) - old_log_prob)
        new_surrogate = (advantages * ratio).mean()

        if kl.item() <= delta and new_surrogate.item() > old_surrogate.item():
            return candidate_theta, i, kl.item()

        step_size = step_size * backtrack_coeff

    # No candidate satisfied both conditions this round -- revert theta
    # and the policy's live params to match, matching the existing
    # --on-invalid-step=skip convention: leave theta unchanged rather
    # than applying a step that overshoots the trust region or makes the
    # surrogate objective worse.
    set_flat_params(policy, theta)
    return theta, max_backtracks, float("nan")
