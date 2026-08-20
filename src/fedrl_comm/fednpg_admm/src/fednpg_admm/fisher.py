"""
Fisher-vector products via KL-divergence double backward.

H_i in Algorithm 1 is the Fisher information matrix, not the Hessian of
any performance objective. The standard identity this relies on: the
Hessian of KL(pi_old || pi_theta), evaluated at the point theta = theta_old
(i.e. where the two distributions coincide), equals the Fisher information
matrix at that point. So: build a KL-divergence loss between a DETACHED
snapshot of the current policy and the current (differentiable) policy at
the same input, then take its Hessian-vector product via double backward
(same trick demonstrated in common.networks.utils.get_flat_grad's
docstring, applied here to a KL loss instead of a generic loss).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.distributions import Independent, Normal, kl_divergence

from common.networks.policy import GaussianMLPPolicy
from common.networks.utils import get_flat_grad


def kl_divergence_loss(policy: GaussianMLPPolicy, obs: torch.Tensor) -> torch.Tensor:
    """
    Mean KL(pi_old || pi_theta) over a batch of observations, where pi_old
    is a DETACHED snapshot of the policy's current output (no gradient
    tracking) and pi_theta is the same policy's live, differentiable
    output at the same theta. Value is exactly 0 at this point (both sides
    are numerically identical) -- that's expected and correct. What
    matters is the SECOND derivative w.r.t. theta, which is the Fisher
    matrix; the first derivative is also 0, which is exactly the condition
    under which "Hessian of KL" and "Fisher information" coincide.
    """
    with torch.no_grad():
        old_mean = policy.mean(obs)
        old_std = torch.exp(policy.log_std)
    old_dist = Independent(Normal(old_mean, old_std), 1)
    new_dist = policy.distribution(obs)
    return kl_divergence(old_dist, new_dist).mean()


def make_fisher_vector_product(
    policy: GaussianMLPPolicy, obs: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns fvp(v) = F @ v, where F is the Fisher information matrix at
    the policy's CURRENT parameters, computed on the given observation
    batch, without ever materializing F itself.

    The returned closure can be called repeatedly (once per CG iteration)
    -- retain_graph=True on every call, both the one that builds grad_kl
    here and the one inside fvp(), because CG calls fvp() many times
    against the SAME underlying graph, not just once.
    """
    kl = kl_divergence_loss(policy, obs)
    grad_kl = get_flat_grad(kl, policy, retain_graph=True, create_graph=True)

    def fvp(v: torch.Tensor) -> torch.Tensor:
        gv = (grad_kl * v).sum()
        return get_flat_grad(gv, policy, retain_graph=True)

    return fvp
