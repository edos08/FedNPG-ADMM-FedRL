"""
Plotly figure builders. Each function takes {run_id: RunResult} (already
filtered to whichever runs the user selected) and returns a go.Figure.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html

from common.results import RunResult, trailing_episode_mean

_TEMPLATE = "plotly_white"

# Episodes (not iterations) behind each smoothed point -- stable-baselines3's
# stats_window_size default. See common.results.smoothing.
_REWARD_WINDOW_EPISODES = 100


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_layout(template=_TEMPLATE, xaxis_visible=False, yaxis_visible=False, height=360)
    return fig


def build_reward_figure(runs: dict[str, RunResult]) -> go.Figure:
    """
    Plots mean_episode_return, NOT mean_reward -- the latter is a
    per-TIMESTEP average and isn't comparable to the paper's Fig. 2 (whose
    y-axis is per-episode totals). mean_episode_return can be NaN on
    iterations where no agent completed a full episode within the
    timestep budget; Plotly draws a gap there rather than a fake zero,
    which is the honest behavior -- don't fill it in.

    No shaded variance band here (unlike the old per-step version): that
    would need a per-agent breakdown of episode returns, which isn't
    logged (only the pooled mean across all agents' completed episodes
    is) -- faking a spread from data that isn't there would be worse than
    just not drawing one.

    Draws TWO traces per run: the raw per-iteration value as a faint thin
    line, and a trailing 100-EPISODE mean as the bold one. The bold line
    is the number worth reading -- `mean_episode_return` averages only the
    episodes that finished inside one iteration's timestep budget, a count
    that collapses to 1-2 on Hopper once episodes get long, so its swings
    are a sample-size artifact rather than training instability (see
    common.results.smoothing). The raw trace stays visible on purpose:
    smoothing here is the correct aggregation, not a way to hide anything.

    Prefers the `ep_rew_mean` column when a run logged it directly, and
    otherwise reconstructs the identical quantity from
    mean_episode_return x num_completed_episodes -- so runs started before
    that column existed still get a smoothed curve.
    """
    fig = go.Figure()
    any_data = False
    for run_id, run in runs.items():
        if run.metrics.empty or "mean_episode_return" not in run.metrics.columns:
            continue
        any_data = True
        # Raw first so the smoothed line draws on top of it, and pin both
        # traces of a run to one color via a shared legend group.
        fig.add_trace(
            go.Scatter(
                x=run.metrics["iteration"], y=run.metrics["mean_episode_return"],
                mode="lines", name=f"{run_id} (per-iteration)", connectgaps=False,
                legendgroup=run_id, opacity=0.35, line=dict(width=1),
                hovertemplate="iter %{x}<br>per-iteration %{y:.1f}<extra></extra>",
            )
        )
        if "ep_rew_mean" in run.metrics.columns:
            smoothed = run.metrics["ep_rew_mean"]
        else:
            smoothed = trailing_episode_mean(run.metrics, window_episodes=_REWARD_WINDOW_EPISODES)
        if smoothed is not None:
            fig.add_trace(
                go.Scatter(
                    x=run.metrics["iteration"], y=smoothed,
                    mode="lines", name=f"{run_id} (last {_REWARD_WINDOW_EPISODES} episodes)",
                    connectgaps=False, legendgroup=run_id, line=dict(width=2.5),
                    hovertemplate=f"iter %{{x}}<br>trailing-{_REWARD_WINDOW_EPISODES}ep %{{y:.1f}}<extra></extra>",
                )
            )
    if not any_data:
        return _empty_figure("No reward data yet -- select a run, or wait for the first completed episode.")
    fig.update_layout(
        template=_TEMPLATE, height=360, margin=dict(l=50, r=20, t=30, b=40),
        xaxis_title="iteration",
        yaxis_title=f"episode return (bold = trailing {_REWARD_WINDOW_EPISODES} episodes)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_comm_overhead_figure(runs: dict[str, RunResult]) -> go.Figure:
    fig = go.Figure()
    any_data = False
    for run_id, run in runs.items():
        if run.metrics.empty or "cumulative_comm_bytes" not in run.metrics.columns:
            continue
        any_data = True
        fig.add_trace(
            go.Scatter(
                x=run.metrics["iteration"], y=run.metrics["cumulative_comm_bytes"],
                mode="lines", name=run_id,
            )
        )
    if not any_data:
        return _empty_figure("No communication data yet.")
    fig.update_layout(
        template=_TEMPLATE, height=360, margin=dict(l=60, r=20, t=30, b=40),
        xaxis_title="iteration", yaxis_title="cumulative bytes (per agent, log scale)",
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_critic_loss_figure(runs: dict[str, RunResult]) -> go.Figure:
    fig = go.Figure()
    any_data = False
    for run_id, run in runs.items():
        if run.metrics.empty or "critic_loss" not in run.metrics.columns:
            continue
        any_data = True
        fig.add_trace(
            go.Scatter(x=run.metrics["iteration"], y=run.metrics["critic_loss"], mode="lines", name=run_id)
        )
    if not any_data:
        return _empty_figure("No critic loss data yet.")
    fig.update_layout(
        template=_TEMPLATE, height=360, margin=dict(l=50, r=20, t=30, b=40),
        xaxis_title="iteration", yaxis_title="critic MSE loss", yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _kpi_card(label: str, value: str, icon: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.I(className=f"fa-solid {icon} me-2 text-muted"),
                            html.Span(label, className="text-muted small"),
                        ]
                    ),
                    html.H4(value, className="mb-0 mt-1"),
                ]
            ),
            className="shadow-sm",
        ),
        md=3,
        sm=6,
    )


def build_kpi_cards(runs: dict[str, RunResult]) -> list[dbc.Col]:
    """KPI row for one selection of runs (one dashboard section) -- kept
    per-section rather than global, same reason each section gets its own
    graphs: a "best reward" or "total comm" pooled across e.g. both
    Swimmer AND Hopper runs at once would mix two incomparable scales into
    one meaningless number."""
    completed = [r for r in runs.values() if not r.metrics.empty]
    total_iterations = sum(int(r.metrics["iteration"].max()) for r in completed) if completed else 0

    # Best SUSTAINED return, i.e. the peak of the trailing-episode-window
    # curve -- not the peak of the raw per-iteration column, which on
    # Hopper is often a single lucky episode (that column averages only
    # whatever finished inside one iteration's budget, sometimes exactly
    # one episode). Reporting that max would flatter every run by showing
    # its luckiest episode rather than its actual level.
    def _best(run: RunResult):
        if "ep_rew_mean" in run.metrics.columns:
            series = run.metrics["ep_rew_mean"]
        else:
            series = trailing_episode_mean(run.metrics, window_episodes=_REWARD_WINDOW_EPISODES)
        if series is None:
            series = run.metrics.get("mean_episode_return")
        return series.max() if series is not None and not series.empty else None

    best_candidates = [b for b in (_best(r) for r in completed) if b is not None and b == b]
    best_reward = max(best_candidates) if best_candidates else None
    if best_reward is not None and best_reward != best_reward:  # NaN check
        best_reward = None
    total_bytes = (
        sum(r.metrics["cumulative_comm_bytes"].max() for r in completed if "cumulative_comm_bytes" in r.metrics.columns)
        if completed
        else 0
    )

    return [
        _kpi_card("Runs selected", str(len(runs)), "fa-layer-group"),
        _kpi_card("Iterations logged (max)", f"{total_iterations:,}", "fa-forward"),
        _kpi_card(f"Best return (last {_REWARD_WINDOW_EPISODES} eps)", f"{best_reward:,.1f}" if best_reward is not None else "--", "fa-arrow-trend-up"),
        _kpi_card("Total comm (all selected)", f"{total_bytes:.3e} B", "fa-tower-broadcast"),
    ]
