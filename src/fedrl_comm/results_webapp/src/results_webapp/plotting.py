"""
Plotly figure builders. Each function takes {run_id: RunResult} (already
filtered to whichever runs the user selected) and returns a go.Figure.
"""

from __future__ import annotations

import plotly.graph_objects as go

from common.results import RunResult

_TEMPLATE = "plotly_white"


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
    """
    fig = go.Figure()
    any_data = False
    for run_id, run in runs.items():
        if run.metrics.empty or "mean_episode_return" not in run.metrics.columns:
            continue
        any_data = True
        fig.add_trace(
            go.Scatter(
                x=run.metrics["iteration"], y=run.metrics["mean_episode_return"],
                mode="lines+markers", name=run_id, connectgaps=False,
                marker=dict(size=4),
            )
        )
    if not any_data:
        return _empty_figure("No reward data yet -- select a run, or wait for the first completed episode.")
    fig.update_layout(
        template=_TEMPLATE, height=360, margin=dict(l=50, r=20, t=30, b=40),
        xaxis_title="iteration", yaxis_title="mean episode return",
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
