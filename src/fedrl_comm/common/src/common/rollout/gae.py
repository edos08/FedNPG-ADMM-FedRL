"""
Generalized Advantage Estimation (Schulman et al. 2016) + critic fitting.

GAE isn't part of the FedNPG-ADMM paper itself — it's inherited from the
paper's own stated basis ("generalized advantage estimation (0.95) [39]" in
the Setup section), so this transcribes the standard GAE recursion, not
anything specific to ADMM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import RolloutBatch


def compute_gae(batch: RolloutBatch, gamma: float = 0.99, lam: float = 0.95) -> RolloutBatch:
    """
    Fill in batch.advantages and batch.returns (in place, and returned for
    chaining) via the standard backward recursion, correctly handling
    MULTIPLE episodes within one batch:

        delta_t = r_t + gamma * next_values_t - values_t
        A_t     = delta_t                                    if episode_boundary_t
                = delta_t + gamma * lam * A_{t+1}             otherwise

    No `last_value` argument, on purpose — earlier versions of this
    function required the caller to correctly supply "0.0 if the batch
    ended on a true termination, value_fn(next_obs) if truncated instead",
    which handled only the batch's FINAL step and silently did nothing for
    episode boundaries occurring mid-batch (a real bug: for any env
    without a genuine terminal state, e.g. Swimmer, EVERY boundary is a
    truncation, and there was no mechanism at all to stop GAE from
    bootstrapping across a mid-batch reset into an unrelated episode).
    batch.next_values now carries the correct bootstrap value for every
    single step, resolved by collect_rollout at collection time when it
    still has the real post-step observation — this function no longer
    needs, or accepts, information from outside the batch.

    Does NOT normalize advantages. That's a per-algorithm choice (common in
    PPO/TRPO implementations, absent from this paper's Algorithm 1) left to
    whoever consumes this batch — not silently applied here.
    """
    T = len(batch)
    advantages = torch.zeros(T, device=batch.rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        delta = batch.rewards[t] + gamma * batch.next_values[t] - batch.values[t]
        if bool(batch.episode_boundary[t]):
            gae = delta  # cut the recursion: t+1 belongs to a different episode
        else:
            gae = delta + gamma * lam * gae
        advantages[t] = gae

    batch.advantages = advantages
    batch.returns = advantages + batch.values
    return batch


def compute_episode_returns(batch: RolloutBatch) -> list[float]:
    """
    Sum of rewards for each COMPLETED episode within the batch. An episode
    ending via true termination OR truncation both count as "completed"
    for this purpose (matching how reward curves are conventionally
    reported, e.g. in stable-baselines3 — the paper's own stated basis —
    even though GAE bootstrapping must still treat the two differently).
    An episode still running when the batch ends is NOT included; its
    eventual total is unknown. Can return an empty list if no episode
    completed within this batch (a real possibility when an episode is
    long relative to the timestep budget T) — callers must handle that,
    not assume at least one value comes back.
    """
    returns: list[float] = []
    running = 0.0
    for t in range(len(batch)):
        running += batch.rewards[t].item()
        if bool(batch.episode_boundary[t]):
            returns.append(running)
            running = 0.0
    return returns


def fit_value_function(
    value_fn: nn.Module,
    obs: torch.Tensor,
    returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
) -> list[float]:
    """
    Regress the critic toward `returns` via plain MSE + whatever optimizer
    is passed in (Adam, per the paper's Setup section — see the "Adam is
    for the critic, not the actor" note from earlier discussion; the
    actor's update is always the NPG trust-region step, never this).

    Takes obs/returns tensors directly rather than a RolloutBatch: a caller
    pooling multiple agents' data into one shared critic update (which
    fednpg_admm's train.py does — the paper doesn't specify how the critic
    is trained federatedly at all) only HAS pooled obs/returns, not a
    well-formed RolloutBatch with matching actions/rewards/etc.

    Returns per-epoch loss so a caller can sanity-check the critic is
    actually learning rather than failing silently.
    """
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = value_fn(obs)
        loss = F.mse_loss(pred, returns)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses