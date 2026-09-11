"""
Environment registry + factory.

Maps the paper's env names (Table 3) to Gymnasium/MuJoCo IDs, and records
the observation/action dimensions and per-env MLP sizes (Table 4) needed to
build a matching policy network via GaussianMLPPolicy.from_env().
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from .normalize import NormalizingVectorEnv, RunningMeanStd


@dataclass(frozen=True)
class EnvSpec:
    """Static description of one of the paper's benchmark envs."""

    name: str
    gym_id: str
    state_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...]


# Table 3 (agent / action dim / state dim) + Table 4 (MLP sizes) from the
# paper. state_dim/action_dim here are what the paper reports, not
# independently re-derived — they match Gymnasium's default observation
# space for these three envs (Swimmer-v4 excludes x/y position by default,
# Hopper-v4 excludes x position, Humanoid-v4's default 376-dim obs includes
# body inertias/velocities), so no exclude_current_positions_from_observation
# overrides are needed. If you ever pass extra gym_kwargs that change the
# observation space, state_dim below will silently stop matching reality —
# build the policy from env.observation_space (see GaussianMLPPolicy.from_env)
# rather than trusting this number blindly.
ENV_SPECS: dict[str, EnvSpec] = {
    "swimmer": EnvSpec("swimmer", "Swimmer-v4", state_dim=8, action_dim=2, hidden_sizes=(64, 64)),
    "hopper": EnvSpec("hopper", "Hopper-v4", state_dim=11, action_dim=3, hidden_sizes=(128, 128)),
    "humanoid": EnvSpec(
        "humanoid", "Humanoid-v4", state_dim=376, action_dim=17, hidden_sizes=(512, 512, 512)
    ),
}


def get_env_spec(name: str) -> EnvSpec:
    """Look up a benchmark env by the paper's short name (case-insensitive).

    Raises KeyError if `name` isn't recognized — deliberately no silent
    fallback, since a typo here should fail loud, not quietly hand you the
    wrong hidden_sizes three files later.
    """
    key = name.lower()
    if key not in ENV_SPECS:
        raise KeyError(
            f"Unknown env '{name}'. Known envs: {sorted(ENV_SPECS)}. "
            "For an arbitrary Gymnasium env not in this table, call "
            "make_env(gym_id) directly instead of get_env_spec()."
        )
    return ENV_SPECS[key]


def make_env(name_or_gym_id: str, seed: int | None = None, **gym_kwargs) -> gym.Env:
    """
    Create a single environment instance.

    Accepts either one of the paper's short names ("swimmer", "hopper",
    "humanoid") or a raw Gymnasium id (e.g. "HalfCheetah-v4") — an
    unrecognized name is passed straight to gym.make(), so this isn't
    locked to only the three envs the paper happens to use.
    """
    key = name_or_gym_id.lower()
    gym_id = ENV_SPECS[key].gym_id if key in ENV_SPECS else name_or_gym_id

    env = gym.make(gym_id, **gym_kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_envs_for_agents(
    name_or_gym_id: str,
    num_agents: int,
    base_seed: int = 0,
    **gym_kwargs,
) -> list[gym.Env]:
    """
    Create `num_agents` independent environment instances for a federated
    run — one per agent, each seeded distinctly (base_seed + i) so
    trajectories differ across agents even though they'll share the same
    policy, with no state shared between the instances.
    """
    return [make_env(name_or_gym_id, seed=base_seed + i, **gym_kwargs) for i in range(num_agents)]


def make_vector_env_for_agents(
    name_or_gym_id: str,
    num_agents: int,
    base_seed: int = 0,
    **gym_kwargs,
) -> AsyncVectorEnv:
    """
    Same per-agent envs as make_envs_for_agents, but stepped in `num_agents`
    separate OS subprocesses via Gymnasium's AsyncVectorEnv instead of one
    Python process looping over them one at a time -- lets every agent's
    MuJoCo physics step run on its own CPU core in parallel, and lets
    common.rollout.collect_rollout_for_agents batch the policy/critic
    forward pass to size `num_agents` per timestep instead of one agent at
    a time.

    autoreset_mode=SAME_STEP is required, not just convenient. It's what
    makes a sub-env's step() return the POST-reset observation immediately
    on a boundary (matching make_env's single-env behavior: no step is
    ever wasted on a dummy reset transition), while still exposing the
    TRUE pre-reset observation via info["final_obs"] for GAE's
    bootstrap-value calculation. The NEXT_STEP default instead returns the
    real terminal observation on the boundary step and defers the actual
    reset to a following all-zero-reward dummy step -- using that here
    would silently shrink every agent's per-iteration transition count
    below num_steps.

    Each factory seeds its own env at construction time exactly like
    make_env does (base_seed + i) -- functools.partial over the
    module-level make_env, not a closure/lambda, so factories stay
    picklable regardless of the multiprocessing start method in use.
    """
    env_fns = [
        functools.partial(make_env, name_or_gym_id, seed=base_seed + i, **gym_kwargs)
        for i in range(num_agents)
    ]
    return AsyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)