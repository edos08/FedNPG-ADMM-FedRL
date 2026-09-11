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
    whoever consumes this batch — not silently applied here. See
    `normalize_advantages` below for an explicit, opt-in helper.
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


def normalize_advantages(batch: RolloutBatch, eps: float = 1e-8) -> RolloutBatch:
    """
    Rescale batch.advantages in place to mean 0, std 1 (and return the batch
    for chaining) -- NOT part of Algorithm 1, same status as the module
    docstring's other NPG/TRPO-standard additions. Separate from
    compute_gae itself so the "does not normalize" guarantee documented
    there stays true; call this explicitly, per agent's own local batch, if
    normalization is wanted (mirrors how e.g. tianshou's NPG/TRPO normalize
    per collected batch by default). `eps` guards against a degenerate
    all-equal-advantage batch producing a divide-by-zero -> NaN.
    """
    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + eps)
    return batch


def compute_episode_returns(batch: RolloutBatch, carry_over: float = 0.0) -> tuple[list[float], float]:
    """
    Sum of rewards for each COMPLETED episode within the batch. An episode
    ending via true termination OR truncation both count as "completed"
    for this purpose (matching how reward curves are conventionally
    reported, e.g. in stable-baselines3 — the paper's own stated basis —
    even though GAE bootstrapping must still treat the two differently).

    `carry_over`: reward this same agent's episode-in-progress had
    already accumulated BEFORE this batch started -- i.e. the second
    return value from this function's own PREVIOUS call, threaded through
    exactly like collect_rollout's `obs` resume point. This is not
    optional bookkeeping: whenever `timesteps` doesn't evenly divide the
    episode length (true for every env/T combination this project uses --
    e.g. Swimmer's 1000-step episodes against T=2048), most iterations'
    FIRST captured episode boundary belongs to an episode that actually
    started one or more iterations ago. Without carry_over, its reported
    "return" silently truncates to whatever fraction of it happened to
    fall inside just this batch -- not the episode's real, full total --
    with the missing fraction's size depending entirely on the iteration's
    phase against that mismatch. Left unfixed, this manufactures a
    perfectly periodic (period = timesteps / gcd(timesteps, episode_length)
    iterations) rise-then-decay pattern in the reported metric that looks
    exactly like training instability but never touches training at all:
    compute_gae uses batch.rewards/next_values/episode_boundary directly
    and is completely unaffected by this; only this reporting path was
    wrong.

    Sums `batch.raw_rewards`, NOT `batch.rewards` -- if reward
    normalization is active (common.envs.normalize.NormalizingVectorEnv),
    `rewards` holds the rescaled values GAE/the critic actually train on,
    which is exactly what should NOT be reported here: this function's
    whole purpose is producing a number comparable to the paper's Table 2,
    and an arbitrarily-rescaled reward defeats that.

    Returns (list of completed episodes' full returns this call,
    updated carry_over for whatever episode is still running when this
    batch ends -- 0.0 if the batch happened to end exactly on a boundary).
    An empty returns list is a real possibility (episode long relative to
    T) — callers must handle that, not assume at least one value comes
    back.
    """
    returns: list[float] = []
    running = carry_over
    for t in range(len(batch)):
        running += batch.raw_rewards[t].item()
        if bool(batch.episode_boundary[t]):
            returns.append(running)
            running = 0.0
    return returns, running


def fit_value_function(
    value_fn: nn.Module,
    obs: torch.Tensor,
    returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    batch_size: int = 64,
) -> list[float]:
    """
    Regress the critic toward `returns` via minibatch SGD -- `epochs`
    passes over a freshly-shuffled permutation of the batch each pass
    (matching stable-baselines3/tianshou's convention: the paper's own
    stated basis is stable-baselines3, and its critic fitting is
    minibatched too), not full-batch gradient descent repeated `epochs`
    times against the IDENTICAL data every step (what this function did
    before).

    That earlier full-batch version let the critic take up to
    `critic_epochs` deterministic steps toward exactly the same finite
    sample every iteration, with no stochasticity to limit how tightly it
    could fit (rather than generalize to) that round's particular noisy
    returns. Diagnosed as the likely cause of two symptoms once
    critic_epochs was pushed well above the original default of 5:
    Swimmer's reward oscillating indefinitely even after critic_loss had
    collapsed to near-zero (a classic overfitting signature, not evidence
    of a good fit), and Hopper's critic_loss staying large and noisy for
    the entire run instead of settling. Shuffled minibatches reintroduce
    the gradient noise that acts as an implicit regularizer against this.

    Takes obs/returns tensors directly rather than a RolloutBatch: a caller
    pooling multiple agents' data into one shared critic update (which
    fednpg_admm's train.py does — the paper doesn't specify how the critic
    is trained federatedly at all) only HAS pooled obs/returns, not a
    well-formed RolloutBatch with matching actions/rewards/etc.

    Returns per-minibatch-step loss (not per-epoch) so a caller can still
    sanity-check the critic is learning; the last entry (last minibatch of
    the last epoch) is what gets logged per training iteration.
    """
    n = obs.shape[0]
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=obs.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = value_fn(obs[idx])
            loss = F.mse_loss(pred, returns[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """
    1 - Var[returns - values] / Var[returns] -- the standard PPO/TRPO
    critic-quality diagnostic (OpenAI Baselines, stable-baselines3), not
    logged anywhere in this pipeline until now. Unlike raw MSE loss, this
    is scale-invariant across environments with very different reward
    magnitudes (Swimmer's tiny per-step rewards vs. Hopper's much larger
    ones make their critic_loss numbers not directly comparable; this is).
    1.0 means the critic perfectly predicts returns, 0.0 means it's no
    better than always predicting the mean return, and it CAN go negative
    -- worse than the mean -- as an explicit red flag a shrinking critic_loss
    alone would never surface.

    `values` should be the critic's PRE-update predictions (i.e.
    batch.values, captured at collection time before this iteration's
    fit) against this iteration's `returns` -- the standard convention,
    answering "how good was the value function that GAE actually used,"
    not "how well did the critic memorize what it was just fit to."
    """
    var_returns = returns.var()
    if var_returns.item() == 0.0:
        return float("nan")
    return (1.0 - (returns - values).var() / var_returns).item()