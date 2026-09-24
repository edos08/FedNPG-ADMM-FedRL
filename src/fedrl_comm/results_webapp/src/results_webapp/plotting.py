"""
Plotly figure builders. Each function takes {run_id: RunResult} (already
filtered to whichever runs the user selected) and returns a go.Figure.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html

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
    ONE trace per run: `mean_episode_return` vs iteration, raw.

    This is deliberately the paper's own metric and nothing else. Section 5
    defines its reward as "the average trajectory rewards across the batch
    collected at each iteration", plotted against iterations, so this chart
    is directly comparable to Figure 2 and Table 2.

    Earlier versions layered a trailing-100-episode curve, a deterministic
    evaluation curve and a seed-averaged mean over this. All three made the
    line look calmer without making the agent any better, which is the
    opposite of useful while the swing itself is the open problem: on
    Hopper at num_agents=1 roughly 9% of episodes end in an early fall and
    each plotted point is ~1.2 episodes, so the swing between ~1000 and
    ~3700 is real behaviour that needs fixing, not smoothing. Those columns
    are still logged in metrics.csv for anyone who wants them; they just
    don't belong on this chart.

    mean_episode_return is NaN on iterations where no episode finished
    inside the timestep budget; Plotly leaves a gap rather than drawing a
    fake zero (connectgaps=False), which is the honest rendering.
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
                mode="lines", name=run_id, connectgaps=False,
                hovertemplate="iter %{x}<br>reward %{y:.0f}<extra></extra>",
            )
        )
    if not any_data:
        return _empty_figure("No reward data yet -- select a run, or wait for the first completed episode.")
    fig.update_layout(
        template=_TEMPLATE, height=360, margin=dict(l=50, r=20, t=30, b=40),
        xaxis_title="iteration", yaxis_title="reward (batch avg per iteration)",
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


_FINAL_WINDOW_ITERS = 100


def build_kpi_cards(runs: dict[str, RunResult]) -> list[dbc.Col]:
    """KPI row for one selection of runs (one dashboard section) -- kept
    per-section rather than global, same reason each section gets its own
    graphs: a "best reward" pooled across e.g. both Swimmer AND Hopper runs
    at once would mix two incomparable scales into one meaningless number."""
    completed = [r for r in runs.values() if not r.metrics.empty]
    total_iterations = sum(int(r.metrics["iteration"].max()) for r in completed) if completed else 0

    def _best(run: RunResult):
        """FINAL reward: the mean of `mean_episode_return` over the last
        _FINAL_WINDOW_ITERS iterations -- the same quantity the paper's
        Table 2 reports as "final rewards", on the same column the chart
        plots.

        Deliberately not the MAX of that column: at ~1.2 episodes per
        iteration its maximum is whichever single episode got luckiest,
        which flatters every run. Averaging a window of the run's own tail
        is both honest and directly comparable to Table 2.
        """
        s = run.metrics.get("mean_episode_return")
        if s is None:
            return None
        s = s.dropna().tail(_FINAL_WINDOW_ITERS)
        return s.mean() if not s.empty else None

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
        _kpi_card(f"Final reward (last {_FINAL_WINDOW_ITERS} iters)", f"{best_reward:,.1f}" if best_reward is not None else "--", "fa-arrow-trend-up"),
        _kpi_card("Total comm (all selected)", f"{total_bytes:.3e} B", "fa-tower-broadcast"),
    ]
