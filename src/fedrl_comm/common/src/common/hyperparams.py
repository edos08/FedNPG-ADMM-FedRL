"""
Per-env default hyperparameters, transcribed from the paper's Table 4.

Two things in here are NOT plain transcription, flagged rather than
presented as fact:

1. `eta` (the actor's step multiplier, Algorithm 1 line 13) is not in
   Table 4 at all. Theorem 4.3 derives it analytically from
   Lipschitz/smoothness constants (mu_F, G, L_J) specific to the policy
   and environment, which the paper never reports numerically for
   Swimmer/Hopper/Humanoid. Defaulted to 1.0 here -- "take the ADMM-solved
   step unscaled" -- which is a choice, not a value from the paper. If
   training diverges or crawls, this is the first knob to turn.

2. Table 4's "Learning rate" column (3e-4 / 3e-4 / 1e-5) is used here as
   the CRITIC's Adam learning rate, not the actor's step size. Reasoning:
   3e-4 and 1e-5 are characteristic Adam learning rates (3e-4 in particular
   is a famous default in deep RL), not natural-gradient step multipliers
   -- and the paper's own Theorem 4.3 defines the actor's step
   analytically rather than as a tunable constant, which is a strong
   signal Table 4's "Learning rate" refers to something else: the critic.
   This is an INTERPRETATION of an ambiguous table, not a certainty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HyperParams:
    rho: float  # ADMM penalty (Table 4: "Penalty")
    delta: float  # trust-region radius (Table 4: "Radius")
    gamma: float  # discount factor (Table 4: "Discount")
    gae_lambda: float  # GAE lambda (Setup section text: "generalized advantage estimation (0.95)")
    timesteps: int  # T, per-agent per-iteration (Table 4: "Timesteps")
    iterations: int  # K (Table 4: "Iterations")
    critic_lr: float  # Table 4: "Learning rate" -- see module docstring point 2
    eta: float = 1.0  # NOT from Table 4 -- see module docstring point 1


DEFAULT_HYPERPARAMS: dict[str, HyperParams] = {
    "swimmer": HyperParams(
        rho=0.1, delta=0.01, gamma=0.99, gae_lambda=0.95,
        timesteps=2048, iterations=1000, critic_lr=3e-4,
    ),
    "hopper": HyperParams(
        rho=0.1, delta=0.01, gamma=0.99, gae_lambda=0.95,
        timesteps=1024, iterations=2000, critic_lr=3e-4,
    ),
    "humanoid": HyperParams(
        rho=0.01, delta=0.01, gamma=0.99, gae_lambda=0.95,
        timesteps=512, iterations=3000, critic_lr=1e-5,
    ),
}
