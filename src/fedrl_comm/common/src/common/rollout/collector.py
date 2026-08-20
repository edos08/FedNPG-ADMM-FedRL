"""
Trajectory collection: step a policy through an env for a fixed number of
timesteps, recording everything GAE needs.

Collects a FIXED number of steps per call (Table 4's per-env "Timesteps
(T)"), not fixed-length episodes — standard on-policy convention, matching
the paper's stated basis in stable-baselines3. An episode ending mid-batch
doesn't truncate the batch early: the env auto-resets and collection
continues until exactly num_steps transitions are recorded.

CORRECTNESS NOTE (found via testing, not designed in from the start): the
bootstrap value for GAE at each step must be resolved HERE, not inferred
later from buffer adjacency. If episode boundaries are ignored, "the value
at buffer index t+1" silently gets treated as "the value of the true
successor state of index t" — which is simply false the moment a reset
happens mid-batch, since index t+1 then belongs to an unrelated new
episode. This function computes the correct bootstrap value for every
step, including the batch's final one, so gae.compute_gae never has to
guess.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from common.networks.policy import GaussianMLPPolicy
from common.networks.value import ValueMLP

from .buffer import RolloutBatch


@torch.no_grad()
def collect_rollout(
    env,
    policy: GaussianMLPPolicy,
    value_fn: ValueMLP,
    num_steps: int,
    obs: np.ndarray | None = None,
) -> tuple[RolloutBatch, np.ndarray]:
    """
    Step `env` for exactly `num_steps` timesteps under `policy`.

    Runs under torch.no_grad(): log_probs/values recorded here are a frozen
    snapshot at collection time (theta_old in the paper's notation) — they
    get RE-computed with gradients enabled later, at training time, against
    whatever theta is current then. Collection and gradient computation are
    deliberately two separate passes, not one.

    Device is inferred from `policy`'s own parameters, not passed
    separately — the module already carries this information, and it
    avoids yet another parameter to keep in sync across every call site.
    MuJoCo's env.step() itself is always CPU/numpy regardless of this;
    only the policy/critic forward passes move (see common.device's
    module docstring for why that isn't automatically a speed win here).

    `obs`: resume point, so consecutive calls don't discard the env's
    ongoing state between training iterations. Pass None only on the very
    first call for a given env. Returns the final observation too, so the
    caller can thread it into the next call.
    """
    device = next(policy.parameters()).device

    if obs is None:
        obs, _ = env.reset()

    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
    term_buf, boundary_buf, next_val_buf = [], [], []

    for _ in range(num_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_t, log_prob_t = policy.sample(obs_t)
        value_t = value_fn(obs_t)

        # RAW action (may exceed env bounds — see GaussianMLPPolicy.sample
        # docstring). Stored as-is; only the copy handed to env.step() is
        # clipped, so log_prob and the stored action always refer to the
        # same value. .cpu() before .numpy() is required whenever `device`
        # isn't CPU — torch refuses to convert a CUDA/MPS tensor directly.
        action_np = action_t.squeeze(0).cpu().numpy()
        clipped_action = action_np.clip(env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, _ = env.step(clipped_action)

        obs_buf.append(obs)
        act_buf.append(action_np)
        logp_buf.append(log_prob_t.item())
        rew_buf.append(reward)
        val_buf.append(value_t.item())
        term_buf.append(terminated)
        boundary_buf.append(terminated or truncated)

        if terminated:
            # True end: no future reward exists, bootstrap is exactly 0 —
            # not "unknown", not "borrow the next episode's start value".
            next_val_buf.append(0.0)
        elif truncated:
            # Time-limit cutoff, not a real end: the episode conceptually
            # continues, so bootstrap from the ACTUAL next state — computed
            # right now, before it's overwritten by the reset below.
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_val_buf.append(value_fn(next_obs_t).item())
        else:
            # Normal continuation: the true successor state IS the next
            # buffer entry, whose value we're about to compute anyway on
            # the next loop iteration. Placeholder now (NaN), backfilled
            # below without any extra critic forward passes.
            next_val_buf.append(float("nan"))

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # Backfill normal-continuation placeholders from the already-computed
    # value of the following buffer entry.
    for t in range(num_steps - 1):
        if math.isnan(next_val_buf[t]):
            next_val_buf[t] = val_buf[t + 1]
    # The batch's final step, if it wasn't itself a boundary, has no
    # following buffer entry to borrow from — resolve it with one extra
    # critic call on the true next observation (the same `obs` this
    # function returns to the caller for the next call anyway).
    if math.isnan(next_val_buf[-1]):
        last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        next_val_buf[-1] = value_fn(last_obs_t).item()

    batch = RolloutBatch(
        obs=torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=device),
        log_probs=torch.as_tensor(logp_buf, dtype=torch.float32, device=device),
        rewards=torch.as_tensor(rew_buf, dtype=torch.float32, device=device),
        values=torch.as_tensor(val_buf, dtype=torch.float32, device=device),
        terminated=torch.as_tensor(term_buf, dtype=torch.bool, device=device),
        episode_boundary=torch.as_tensor(boundary_buf, dtype=torch.bool, device=device),
        next_values=torch.as_tensor(next_val_buf, dtype=torch.float32, device=device),
        advantages=torch.zeros(num_steps, device=device),
        returns=torch.zeros(num_steps, device=device),
    )
    return batch, obs


def collect_rollout_for_agents(
    envs: list,
    policy: GaussianMLPPolicy,
    value_fn: ValueMLP,
    num_steps: int,
    obs_list: "list[np.ndarray | None] | None" = None,
) -> tuple[list[RolloutBatch], list[np.ndarray]]:
    """
    Collect one RolloutBatch per agent, all under the SAME (just-broadcast,
    shared) policy — this is the federated data-collection step: N agents,
    N independent envs, one global theta.
    """
    if obs_list is None:
        obs_list = [None] * len(envs)
    batches, next_obs_list = [], []
    for env, obs in zip(envs, obs_list):
        batch, next_obs = collect_rollout(env, policy, value_fn, num_steps, obs=obs)
        batches.append(batch)
        next_obs_list.append(next_obs)
    return batches, next_obs_list