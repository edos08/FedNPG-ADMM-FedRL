"""
Trailing-episode-window reward aggregation, for reading results back.

WHY this exists: `mean_episode_return` is averaged over only the episodes
that happened to COMPLETE inside one iteration's fixed timestep budget
(Table 4's T). On Hopper that count collapses as the policy improves --
episodes get longer, so with T=1024 it falls from ~16 episodes/iteration
early to exactly 1 by iteration ~300 at num_agents=1. A one-episode
average of a bimodal return distribution (Hopper either survives to the
1000-step time limit, ~3400, or falls early, ~900) swings wildly, which
reads as training instability but is purely a sample-size-per-logged-point
artifact: measured on this project's own runs, the coefficient of
variation drops 3-18x when the same data is aggregated over a trailing
100-episode window instead, and the underlying curve is smooth and
monotone.

Both references this project follows report it the trailing way:
  - stable-baselines3's headline `rollout/ep_rew_mean` is a mean over the
    last `stats_window_size=100` EPISODES (a deque spanning however many
    iterations that takes), not over the current rollout's episodes.
  - the paper's own Figure 2/3 curves are averaged over 10 seeds with 95%
    confidence bands (Setup section, page 7), so a single-seed
    per-iteration curve was never going to look like theirs.

New runs log `ep_rew_mean` directly (see each project's train.py). This
module reconstructs the identical quantity for runs logged BEFORE that
column existed, which is exact rather than approximate: for a given
iteration, `mean_episode_return * num_completed_episodes` is the SUM of
those episodes' returns, so summing numerator and denominator across a
trailing span of iterations recovers the true trailing-episode mean.
"""

from __future__ import annotations

from collections import deque

import pandas as pd


def trailing_episode_mean(
    metrics: pd.DataFrame,
    window_episodes: int = 100,
    return_col: str = "mean_episode_return",
    count_col: str = "num_completed_episodes",
) -> "pd.Series | None":
    """
    Trailing mean over the last `window_episodes` completed episodes, one
    value per row of `metrics` (aligned to its index).

    Returns None if the required columns aren't present, so callers can
    fall back to plotting the raw per-iteration series rather than
    erroring on a run logged by some other producer.

    The window is in EPISODES, not iterations -- matching
    stable-baselines3's `stats_window_size` -- so early iterations (many
    short episodes) span fewer iterations than late ones (few long
    episodes). That is the point: it holds the number of samples behind
    each plotted value roughly constant, which is exactly what the raw
    per-iteration metric fails to do.
    """
    if return_col not in metrics.columns or count_col not in metrics.columns:
        return None

    values: list[float] = []
    # (sum of returns, episode count) for each contributing iteration
    buf: deque[tuple[float, float]] = deque()
    running_sum = 0.0
    running_count = 0.0

    for mean_ret, n_eps in zip(metrics[return_col], metrics[count_col]):
        n = float(n_eps) if pd.notna(n_eps) else 0.0
        if n > 0 and pd.notna(mean_ret):
            buf.append((float(mean_ret) * n, n))
            running_sum += float(mean_ret) * n
            running_count += n
        # Drop whole iterations off the tail while doing so still leaves
        # at least one -- keeping one guarantees a defined value even when
        # a single iteration holds more episodes than the whole window.
        while running_count > window_episodes and len(buf) > 1:
            old_sum, old_count = buf.popleft()
            running_sum -= old_sum
            running_count -= old_count
        values.append(running_sum / running_count if running_count > 0 else float("nan"))

    return pd.Series(values, index=metrics.index, name=f"ep_rew_mean_{window_episodes}")
