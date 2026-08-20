"""
Trajectory collection + GAE — turns a policy and an env into the
(obs, action, log_prob, advantage, return) batches every algorithm
package trains on.
"""

from .buffer import RolloutBatch
from .collector import collect_rollout, collect_rollout_for_agents
from .gae import compute_episode_returns, compute_gae, fit_value_function

__all__ = [
    "RolloutBatch",
    "collect_rollout",
    "collect_rollout_for_agents",
    "compute_episode_returns",
    "compute_gae",
    "fit_value_function",
]
