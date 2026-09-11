"""
Value network (critic) for GAE-based advantage estimation.

NOT specified anywhere in the paper's tables — Table 4 only covers the
policy MLP's architecture. Every TRPO/PPO implementation needs a critic to
compute advantages, including the paper's own stated basis
(stable-baselines3), but the paper never describes one. Defaulting the
critic's hidden sizes to match the policy's is a common convention, not a
requirement — there's no rule that actor and critic share architecture.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .utils import orthogonal_init_


class ValueMLP(nn.Module):
    """
    Maps a state to a scalar value estimate V(s). No output activation —
    unlike the policy's action mean, V(s) is not bounded to any range.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

        if ortho_init:
            # sqrt(2) hidden / 1.0 on the value head, biases zeroed --
            # stable-baselines3's ActorCriticPolicy default for the critic
            # (its value head uses gain 1, unlike the policy's 0.01 action
            # head). See common.networks.utils.orthogonal_init_.
            orthogonal_init_(self.net, hidden_gain=2.0**0.5, head_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

    @classmethod
    def from_env(cls, env, hidden_sizes: Sequence[int] = (64, 64), **kwargs) -> "ValueMLP":
        """Build a critic sized to match a given env — same rationale as
        GaussianMLPPolicy.from_env(): read obs_dim off the env itself."""
        obs_dim = env.observation_space.shape[0]
        return cls(obs_dim, hidden_sizes=hidden_sizes, **kwargs)
