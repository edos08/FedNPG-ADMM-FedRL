"""
Container for one agent's collected rollout.

Every field is a torch.Tensor with a leading time dimension of length T
(the paper's per-iteration timestep count — Table 4's "Timesteps (T)").
advantages/returns start as zeros and get filled in by gae.compute_gae();
everything else is filled in at collection time by collector.collect_rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor  # (T, obs_dim)
    actions: torch.Tensor  # (T, act_dim) — RAW actions, pre-clip (see collector.py)
    log_probs: torch.Tensor  # (T,) — log pi_theta_old(a_t|s_t), frozen at collection time
    rewards: torch.Tensor  # (T,) — what GAE/the critic actually train on: reward-normalized
    # (see common.envs.normalize.NormalizingVectorEnv) if that's active, otherwise identical
    # to raw_rewards.
    raw_rewards: torch.Tensor  # (T,) — the TRUE, unnormalized env reward, regardless of
    # whether reward normalization is active. Reporting/comparison-to-the-paper's-Table-2
    # code (compute_episode_returns, train.py's logged mean_reward) must use THIS, never
    # `rewards` -- otherwise a run's numbers become an arbitrary rescaling that's no longer
    # comparable to anything, defeating the whole point of reward normalization (which
    # exists to help TRAINING, not to change what "reward" means to a human reading results).
    values: torch.Tensor  # (T,) — V(s_t) at collection time
    terminated: torch.Tensor  # (T,) bool — TRUE env termination, not truncation
    episode_boundary: torch.Tensor  # (T,) bool — terminated OR truncated: a reset
    # happened right after this step. GAE must not carry its recursive
    # accumulator across this boundary, regardless of which kind it is.
    next_values: torch.Tensor  # (T,) — the CORRECT V(s_{t+1}) to bootstrap from at
    # each step: 0.0 if terminated[t]; V(true next_obs) if truncated[t] (computed
    # before the reset overwrote it) or if it's simply the batch's final step;
    # otherwise identical to values[t+1] for a normal, non-boundary continuation.
    advantages: torch.Tensor  # (T,) — filled in by compute_gae; zeros until then
    returns: torch.Tensor  # (T,) — advantages + values; the critic's regression target

    def __len__(self) -> int:
        return self.obs.shape[0]
