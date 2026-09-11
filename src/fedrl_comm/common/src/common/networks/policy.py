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

from .utils import orthogonal_init_


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
        log_std_min: float = -2.0,
        log_std_max: float = 0.5,
        action_low: torch.Tensor | float = -1.0,
        action_high: torch.Tensor | float = 1.0,
        ortho_init: bool = True,
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

        if ortho_init:
            # Orthogonal initialization, matching stable-baselines3's
            # ActorCriticPolicy (ortho_init=True by default there): gain
            # sqrt(2) on hidden layers, 0.01 on the action head, biases
            # zeroed. NOT specified by the paper -- Table 4 covers layer
            # sizes and activations only -- but the Setup section cites
            # stable-baselines3's practical settings, and this is one of
            # them. The 0.01 action-head gain is the load-bearing part: it
            # makes the initial mean action near-zero, whereas PyTorch's
            # default Kaiming-uniform init (what this used to do) pushes
            # Kaiming-scale outputs straight into the Tanh below, which can
            # start training already near saturation -- worse the larger
            # the net gets (Hopper's 128x128, Humanoid's 512x512x512).
            orthogonal_init_(self.mean_net, hidden_gain=2.0**0.5, head_gain=0.01)

        # State-independent log std, one per action dim. -0.5 is a common
        # TRPO-style default (std ~ 0.6) — it is NOT specified in the
        # paper's Table 4 (which only lists network/training hyperparams,
        # not exploration noise init), so treat this as our own choice to
        # tune, not a transcription of the paper.
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        # log_std is part of theta, so Algorithm 1's NPG/ADMM step moves it
        # exactly like any mean-network weight -- and Algorithm 1 has no
        # entropy term or bounds to counteract that. Empirically (see
        # docs/npg_admm_theory_and_implementation.md, swimmer diagnostics)
        # this is unstable in BOTH directions, not just collapse-to-zero:
        # with num_agents=8, log_std shrinks monotonically (ADMM's average
        # over 8 agents' y_i smooths out any single noisy round); with
        # num_agents=1 (no averaging to cancel a bad round), log_std for one
        # action dim instead RAN AWAY UPWARD around iteration ~110-160
        # (std 0.6 -> 7+), coinciding exactly with reward collapsing into
        # noise and never recovering -- once one action's noise gets large
        # enough to be dominated by env-bound clipping, that dimension's
        # advantage signal becomes uninformative, so the next round's log_std
        # gradient is just as likely to push further up as back down: a
        # one-sided runaway with nothing to stop it. log_std_min/log_std_max
        # floor and ceiling the EFFECTIVE std used for sampling/log_prob (via
        # `std()`/`distribution()`); the raw parameter can still be pushed
        # past either bound, but the clamp's zero gradient there stops the
        # ADMM step from having any further effect once it does. Neither
        # bound is in the paper -- this project's fix for a failure mode
        # Algorithm 1 doesn't guard against in either direction.
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("_action_center", (high + low) / 2.0)
        self.register_buffer("_action_scale", (high - low) / 2.0)

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        squashed = self.mean_net(obs)
        return self._action_center + self._action_scale * squashed

    def std(self) -> torch.Tensor:
        """Effective per-dim action std: exp(log_std), clamped to [log_std_min, log_std_max]."""
        return torch.exp(self.log_std.clamp(min=self.log_std_min, max=self.log_std_max))

    def distribution(self, obs: torch.Tensor) -> Independent:
        mean = self.mean(obs)
        std = self.std().expand_as(mean)
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
