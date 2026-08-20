"""
Gaussian MLP policy for continuous control (Swimmer/Hopper/Humanoid).

Architecture follows the paper's Table 4: ReLU hidden activations, Tanh
output activation. Action bounds are NOT uniform across these three envs —
Swimmer-v4 and Hopper-v4 use [-1, 1], but Humanoid-v4 uses [-0.4, 0.4].
The action_low/action_high rescaling below is load-bearing, not defensive —
always build via GaussianMLPPolicy.from_env() rather than hardcoding [-1, 1] 
yourself.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


class GaussianMLPPolicy(nn.Module):
    """
    Diagonal-Gaussian policy: mean is an MLP(state), std is a single
    state-independent learned vector (standard choice in NPG/TRPO
    implementations — state-dependent std is a design choice this project
    isn't making by default). action = mean + std * noise, mean squashed to
    the env's action range via the output activation.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] = nn.Tanh,
        log_std_init: float = -0.5,
        action_low: torch.Tensor | float = -1.0,
        action_high: torch.Tensor | float = 1.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, act_dim), output_activation()]
        self.mean_net = nn.Sequential(*layers)

        # State-independent log std, one per action dim. -0.5 is a common
        # TRPO-style default (std ~ 0.6) — it is NOT specified in the
        # paper's Table 4 (which only lists network/training hyperparams,
        # not exploration noise init), so treat this as our own choice to
        # tune, not a transcription of the paper.
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("_action_center", (high + low) / 2.0)
        self.register_buffer("_action_scale", (high - low) / 2.0)

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        squashed = self.mean_net(obs)
        return self._action_center + self._action_scale * squashed

    def distribution(self, obs: torch.Tensor) -> Independent:
        mean = self.mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample + its log-prob, for use inside training.

        NOTE: the mean is squashed to the action range via the output
        activation, but Gaussian noise added on top of it is NOT bounded —
        a sampled action can legitimately fall outside env.action_space's
        bounds. This is expected (it's how the paper's own TRPO /
        stable-baselines3 basis behaves too): use the RAW action returned
        here for log_prob / training math, and only clip to
        env.action_space.low/high at the point of calling env.step(). Don't
        clip before computing log_prob — that would be scoring a different
        action than the one actually used for training.
        """
        dist = self.distribution(obs)
        action = dist.rsample()
        return action, dist.log_prob(action)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.distribution(obs).log_prob(action)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean action, no sampling — for evaluation, not training rollouts."""
        with torch.no_grad():
            return self.mean(obs)

    @classmethod
    def from_env(
        cls, env, hidden_sizes: Sequence[int] = (64, 64), **kwargs
    ) -> "GaussianMLPPolicy":
        """
        Build a policy sized and bounded to match a given Gymnasium env —
        the recommended constructor path, since it reads obs/action
        dimensions and action bounds from the env itself instead of trusting
        a hardcoded number that might drift out of sync with reality.
        """
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        return cls(
            obs_dim,
            act_dim,
            hidden_sizes=hidden_sizes,
            action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
            action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
            **kwargs,
        )
