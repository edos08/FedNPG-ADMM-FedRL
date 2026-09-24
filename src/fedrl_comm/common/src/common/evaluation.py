"""
Deterministic policy evaluation and checkpointing.

WHY this exists: until now every number this project reported was the
STOCHASTIC TRAINING return -- the return of the behaviour policy, which
injects Gaussian exploration noise (std ~0.16-0.37 in practice) into every
action, measured on the very rollouts used to compute the gradient. The
reference implementations this project benchmarks itself against report
something different: the return of the DETERMINISTIC policy (the mean
action, no noise) on separate evaluation episodes. On Hopper the gap is
not cosmetic -- action noise is precisely what makes the hopper fall, and
falling ends the episode -- so comparing our stochastic number against
their deterministic one understates this implementation systematically.

Both are logged from here on: `ep_rew_mean` (stochastic, on-policy, free)
stays the training signal, and `eval_ep_rew_mean` (deterministic, separate
env) is the number comparable to published benchmarks.

Nothing here is in the paper; it is measurement apparatus, not algorithm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class EvalResult:
    mean_return: float
    std_return: float
    mean_length: float
    num_episodes: int

    @property
    def as_metrics(self) -> dict[str, float]:
        return {
            "eval_ep_rew_mean": self.mean_return,
            "eval_ep_rew_std": self.std_return,
            "eval_ep_len_mean": self.mean_length,
        }


@torch.no_grad()
def evaluate_policy(
    policy: Any,
    env: Any,
    num_episodes: int = 10,
    obs_rms: Any = None,
    device: "torch.device | str" = "cpu",
    deterministic: bool = True,
) -> EvalResult:
    """
    Roll out `num_episodes` complete episodes on `env` and return their RAW
    (un-normalized) returns.

    `env` must be a single, unwrapped-for-reward env -- pass a plain
    make_env() instance, NOT the training NormalizingVectorEnv: reward
    normalization would make the number incomparable to any published
    benchmark, and sharing the training env would corrupt its running
    state mid-iteration.

    `obs_rms` is the TRAINING observation normalizer, applied here but
    never updated -- the policy was trained on that input distribution, so
    evaluating it on raw observations would measure the wrong function.
    Passing None means the policy was trained without obs normalization.

    `deterministic=True` uses the distribution's mean action (no
    exploration noise), which is what reference implementations report.
    """
    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            x = obs_rms.normalize(np.asarray(obs, dtype=np.float64)) if obs_rms is not None else obs
            obs_t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device).unsqueeze(0)
            if deterministic:
                action = policy.mean(obs_t)
            else:
                action, _ = policy.sample(obs_t)
            act = action.squeeze(0).cpu().numpy()
            # The Gaussian mean is already inside the action bounds when the
            # policy squashes it, but clip anyway: --output-activation
            # identity does not squash, and an out-of-range action would be
            # silently clipped by MuJoCo rather than by us.
            act = np.clip(act, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(act)
            total += float(reward)
            steps += 1
            done = bool(terminated or truncated)
        returns.append(total)
        lengths.append(steps)

    return EvalResult(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_length=float(np.mean(lengths)),
        num_episodes=num_episodes,
    )


def save_checkpoint(
    path: "str | Path",
    policy: Any,
    value_fn: Any = None,
    obs_rms: Any = None,
    extra: "dict | None" = None,
) -> None:
    """
    Save everything needed to reload and evaluate a trained policy later.

    The observation normalizer is part of the model here, not an
    environment detail: a policy trained on normalized observations is
    meaningless without the exact mean/var it was trained against, and
    those live only in the training process's memory. Omitting them was
    why no earlier run could be re-evaluated after the fact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"policy": policy.state_dict()}
    if value_fn is not None:
        payload["value_fn"] = value_fn.state_dict()
    if obs_rms is not None:
        # getattr with defaults: the two projects' RunningMeanStd classes
        # differ (npg_trpo takes `clip` as a normalize() argument rather
        # than storing it, and has no `epsilon` attribute), and a
        # checkpoint writer has no business caring which one it was handed.
        payload["obs_rms"] = {
            "mean": np.asarray(obs_rms.mean),
            "var": np.asarray(obs_rms.var),
            "count": obs_rms.count,
            "clip": getattr(obs_rms, "clip", 10.0),
            "epsilon": getattr(obs_rms, "epsilon", 1e-8),
        }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: "str | Path", policy: Any, value_fn: Any = None) -> dict:
    """Restore a checkpoint into `policy`/`value_fn`; returns the raw payload
    so the caller can rebuild the obs normalizer from payload["obs_rms"]."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"])
    if value_fn is not None and "value_fn" in payload:
        value_fn.load_state_dict(payload["value_fn"])
    return payload
