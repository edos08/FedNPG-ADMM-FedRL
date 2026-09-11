"""
Running observation and reward normalization for a vector env.

NOT part of Algorithm 1 -- the paper's pseudocode never normalizes anything.
Flagged the same way as `eta`/`critic_lr`/log_std bounds elsewhere in this
codebase: a practical necessity nearly every real NPG/TRPO/PPO reproduction
applies (e.g. OpenAI Baselines'/stable-baselines3's VecNormalize, tianshou's
VectorEnvNormObs), absent from the paper's literal pseudocode.

Observation normalization: raw MuJoCo observations have wildly different
per-dimension scales, which starves both the policy and critic MLPs of a
well-conditioned input -- added because Swimmer training was plateauing far
below the paper's Table 2 numbers even with the log_std floor/ceiling fix
and eta sweeps, neither of which touches this.

Reward normalization: stable-baselines3's VecNormalize -- the paper's own
cited practical basis -- also normalizes reward by default, dividing by a
running estimate of the discounted return's standard deviation (NOT
centering the mean, so the sign/relative ordering of rewards is preserved).
Added because Hopper's return distribution is far more heavy-tailed than
Swimmer's (a policy that mostly hops for the full episode but sometimes
falls over early produces returns ranging from the hundreds to the
thousands), which makes the critic's raw-reward-scale regression target
much harder to fit consistently than Swimmer's narrower distribution ever
was -- plausibly contributing to both plain-TRPO's weak/slow direction and
ADMM's instability on Hopper specifically.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RunningMeanStd:
    """
    Streaming mean/variance via Chan et al.'s parallel-variance algorithm
    (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm),
    the same numerically-stable form used by OpenAI Baselines' VecNormalize
    and tianshou's RunningMeanStd. `mean`/`var` start as scalars and are
    promoted to per-dimension arrays (or stay scalar, for reward
    normalization's single running variance) by numpy broadcasting on the
    first `update()` call -- no shape needs to be passed in up front.
    """

    def __init__(self, clip: float | None = 10.0, epsilon: float = 1e-8) -> None:
        self.mean: "float | np.ndarray" = 0.0
        self.var: "float | np.ndarray" = 1.0
        self.count = epsilon
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return out


class NormalizingVectorEnv:
    """
    Wraps a vector env (e.g. from make_vector_env_for_agents), normalizing
    observations and/or rewards via running statistics updated online as
    rollouts come in -- composable, not baked into make_vector_env_for_agents
    itself, so callers that want the paper-literal (unnormalized) path can
    just not wrap.

    OBSERVATION correctness note: with autoreset_mode=SAME_STEP (see
    make_vector_env_for_agents), a boundary env's step() returns the
    POST-reset observation, with the true PRE-reset observation carried
    separately in info["final_obs"] -- collect_rollout_for_agents reads
    that key directly to compute GAE's bootstrap value for truncated
    episodes. That path bypasses this wrapper's own `step()` return value,
    so final_obs must be normalized here too (using the SAME running
    statistics) or the critic sees an inconsistent, unnormalized input
    distribution for every truncation-bootstrap call -- a bug that would be
    easy to miss since it only affects one specific, less-common code path.

    REWARD normalization, mirroring stable-baselines3's VecNormalize
    exactly: maintain a per-env running discounted-return accumulator
    (`self.returns = self.returns * gamma + reward`), feed THAT into a
    running variance estimate (not the raw reward directly -- the point is
    to scale by how large returns typically get, not by single-step reward
    magnitude), and divide reward by that running std (never subtracting
    the mean, so a reward's sign and relative ordering are preserved).
    `self.returns` resets to 0 for any env that hit a boundary this step.

    The RAW (unnormalized) reward is always additionally stashed into
    `infos["raw_reward"]`, regardless of whether normalize_reward is on --
    collect_rollout_for_agents reads it into RolloutBatch.raw_rewards,
    which reporting code (compute_episode_returns, train.py's logged
    mean_reward) must use instead of the normalized `rewards` field, or a
    run's logged numbers become an arbitrary rescaling no longer
    comparable to the paper's Table 2.
    """

    def __init__(
        self,
        venv: Any,
        gamma: float = 0.99,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        reward_clip: float | None = 10.0,
        update_stats: bool = True,
    ) -> None:
        self.venv = venv
        self.gamma = gamma
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.update_stats = update_stats
        self.obs_rms = RunningMeanStd()
        self.ret_rms = RunningMeanStd(clip=None)  # reward itself is clipped, not the stat
        self.reward_clip = reward_clip
        self.returns = np.zeros(venv.num_envs, dtype=np.float64)

    def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, Any]:
        obs, info = self.venv.reset(*args, **kwargs)
        if self.update_stats and self.normalize_obs:
            self.obs_rms.update(obs)
        self.returns[:] = 0.0
        norm_obs = self.obs_rms.normalize(obs) if self.normalize_obs else obs
        return norm_obs, info

    def step(self, action: Any) -> tuple[np.ndarray, Any, Any, Any, Any]:
        obs, reward, terminated, truncated, infos = self.venv.step(action)
        raw_reward = np.array(reward, copy=True)

        if self.normalize_obs:
            if self.update_stats:
                # BOTH observations a boundary env produced this step need to
                # feed the running stats -- the true pre-reset final obs
                # (final_obs, below) AND the post-reset obs SAME_STEP already
                # substituted into `obs` -- or the stats silently never see
                # episode-final states at all (every Swimmer episode ends this
                # way, since it only ever truncates, never terminates). Update
                # with final_obs FIRST so the "true final" observation is
                # folded in before the post-reset one, mirroring the order a
                # non-vectorized single-env collector naturally produces them.
                if "final_obs" in infos:
                    final_valid = [o for o in infos["final_obs"] if o is not None]
                    if final_valid:
                        self.obs_rms.update(np.stack(final_valid))
                self.obs_rms.update(obs)
            norm_obs = self.obs_rms.normalize(obs)
        else:
            norm_obs = obs

        infos = dict(infos)
        if self.normalize_obs and "final_obs" in infos:
            infos["final_obs"] = np.array(
                [
                    self.obs_rms.normalize(np.asarray(o)) if o is not None else o
                    for o in infos["final_obs"]
                ],
                dtype=object,
            )

        if self.normalize_reward:
            self.returns = self.returns * self.gamma + reward
            if self.update_stats:
                self.ret_rms.update(self.returns)
            norm_reward = reward / np.sqrt(self.ret_rms.var + self.ret_rms.epsilon)
            if self.reward_clip is not None:
                norm_reward = np.clip(norm_reward, -self.reward_clip, self.reward_clip)
            done = terminated | truncated
            self.returns[done] = 0.0
        else:
            norm_reward = reward

        infos["raw_reward"] = raw_reward
        return norm_obs, norm_reward, terminated, truncated, infos

    def close(self) -> None:
        self.venv.close()

    def __getattr__(self, name: str) -> Any:
        # Forwards anything this wrapper doesn't define itself (num_envs,
        # single_action_space, single_observation_space, ...) straight to
        # the wrapped env -- only reached when normal attribute lookup on
        # this object fails, so it never shadows reset/step/close/venv above.
        return getattr(self.venv, name)
