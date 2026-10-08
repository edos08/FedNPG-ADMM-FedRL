"""
Federated PPO -- the first-order baseline in the paper's Figure 3
("first-order FedPPO algorithms. The number of federated agents N is fixed
to 8. PPO clipping parameter is set as 0.2").

WHAT THE PAPER LEAVES UNSPECIFIED
---------------------------------
It names FedPPO and gives the clip parameter, and nothing else: not how
often agents communicate, not whether the server averages gradients or
parameters, not the local epoch count. Two readings are defensible:

  (a) FedSGD-style -- every agent computes ONE clipped-surrogate gradient
      on its batch, the server averages those and takes one step. Exactly
      parallels FedNPG's structure (one aggregation per iteration), but it
      throws away the thing that makes PPO work: the multiple epochs of
      minibatch reuse its clipping exists to make safe. A PPO restricted to
      one gradient step per rollout is a straw man, and beating it would
      say nothing.

  (b) FedAvg-style -- every agent runs several local PPO epochs on its own
      batch starting from the shared theta, then the server averages the
      resulting parameters. This is what "FedPPO" means in the federated
      learning literature, and it gives PPO its normal strength.

This implements **(b)**, and the choice is recorded in run metadata as
`fedppo_variant`. Uplink is one d-vector (the parameter delta) per agent
per iteration, i.e. O(d) -- the same order as FedNPG-ADMM, which is what
makes the Figure 3 comparison about first- vs second-order rather than
about bytes.

CONTROLLED COMPARISON
---------------------
Everything except the actor update is identical to the other two arms:
same environments, same observation/reward normalization, same GAE, same
POOLED critic, same logged columns. Only the policy update rule differs --
clipped surrogate with local Adam epochs here, natural-gradient trust
region there. Anything else would confound Figure 3.

A fresh Adam is built per agent per iteration rather than carried across
rounds. Carrying it would leave the moment estimates describing a
parameter trajectory that jumped at every averaging step; resetting is the
standard FedAvg treatment of local optimizer state and avoids that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from common.networks import get_flat_params, set_flat_params
from common.networks.policy import GaussianMLPPolicy


@dataclass
class PPOStats:
    policy_loss: float
    clip_fraction: float
    approx_kl: float


def local_ppo_update(
    policy: GaussianMLPPolicy,
    theta: torch.Tensor,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    lr: float,
    clip: float = 0.2,
    epochs: int = 10,
    batch_size: int = 64,
) -> tuple[torch.Tensor, PPOStats]:
    """
    Run `epochs` passes of minibatch PPO on ONE agent's batch, starting
    from the shared `theta`, and return (theta_i, stats) without touching
    the caller's copy of theta.

    `policy` is mutated as working storage and left holding theta_i; the
    caller re-broadcasts the averaged theta at the top of the next
    iteration, exactly as the FedNPG-ADMM path does.

    The ratio is taken against log-probs computed at the SHARED theta and
    held fixed for all epochs -- that is what makes the clip meaningful:
    it bounds how far this agent's local policy may drift from the
    broadcast one, which is precisely the quantity federated averaging
    then has to reconcile.
    """
    set_flat_params(policy, theta)
    with torch.no_grad():
        old_log_prob = policy.log_prob(obs, actions).detach()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = obs.shape[0]
    clipped_total, seen, kl_sum, loss_last = 0.0, 0, 0.0, 0.0

    for _ in range(epochs):
        perm = torch.randperm(n, device=obs.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            log_prob = policy.log_prob(obs[idx], actions[idx])
            ratio = torch.exp(log_prob - old_log_prob[idx])
            adv = advantages[idx]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            loss = -torch.min(unclipped, clipped).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                clipped_total += ((ratio - 1.0).abs() > clip).float().sum().item()
                kl_sum += (old_log_prob[idx] - log_prob.detach()).sum().item()
                seen += idx.numel()
            loss_last = loss.item()

    stats = PPOStats(
        policy_loss=loss_last,
        clip_fraction=clipped_total / max(seen, 1),
        approx_kl=kl_sum / max(seen, 1),
    )
    return get_flat_params(policy), stats
