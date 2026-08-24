"""
CLI entry point: fedrl-results-webapp.

A live Dash dashboard: polls the results directory on an interval (default
every 3s) and re-renders, so an in-progress training run's latest logged
iteration shows up without restarting the dashboard.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from .data_loader import load_all_runs, run_options
from .plotting import build_comm_overhead_figure, build_critic_loss_figure, build_reward_figure

REFRESH_INTERVAL_MS = 3000


def build_app(results_path: str) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY, "https://use.fontawesome.com/releases/v6.5.1/css/all.css"],
        title="FedNPG-ADMM results",
    )

    app.layout = dbc.Container(
        fluid=True,
        className="py-4",
        children=[
            dcc.Interval(id="poll-interval", interval=REFRESH_INTERVAL_MS, n_intervals=0),
            dbc.Row(
                [
                    dbc.Col(html.H2("FedNPG-ADMM results", className="mb-0"), width="auto"),
                    dbc.Col(
                        html.Div(id="status-text", className="text-muted small pt-2"),
                        width="auto",
                        className="ms-auto",
                    ),
                ],
                align="center",
                className="mb-3",
            ),
            dbc.Row(id="kpi-row", className="mb-4 g-3"),
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        id="run-selector",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Select runs to compare (defaults to all)",
                    ),
                    width=12,
                ),
                className="mb-4",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(dcc.Graph(id="reward-graph", config={"displaylogo": False}), label="Reward"),
                    dbc.Tab(
                        dcc.Graph(id="comm-graph", config={"displaylogo": False}),
                        label="Communication overhead",
                    ),
                    dbc.Tab(dcc.Graph(id="critic-graph", config={"displaylogo": False}), label="Critic loss"),
                ]
            ),
        ],
    )

    @app.callback(
        Output("run-selector", "options"),
        Output("run-selector", "value"),
        Input("poll-interval", "n_intervals"),
        State("run-selector", "value"),
    )
    def refresh_dropdown(n_intervals, current_value):
        runs = load_all_runs(results_path)
        options = run_options(runs)
        if n_intervals == 0 or not current_value:
            # First load, or nothing selected yet: default to every run found,
            # so the dashboard shows something before the user touches anything.
            return options, [opt["value"] for opt in options]
        return options, dash.no_update

    @app.callback(
        Output("reward-graph", "figure"),
        Output("comm-graph", "figure"),
        Output("critic-graph", "figure"),
        Output("status-text", "children"),
        Output("kpi-row", "children"),
        Input("poll-interval", "n_intervals"),
        Input("run-selector", "value"),
    )
    def refresh_graphs(n_intervals, selected_run_ids):
        all_runs = load_all_runs(results_path)
        selected = selected_run_ids or list(all_runs.keys())
        filtered = {rid: r for rid, r in all_runs.items() if rid in selected}

        status = f"{len(all_runs)} run(s) in {results_path} -- last refreshed {datetime.now():%H:%M:%S}"

        completed = [r for r in filtered.values() if not r.metrics.empty]
        total_iterations = sum(int(r.metrics["iteration"].max()) for r in completed) if completed else 0
        best_reward = (
            max(
                (r.metrics["mean_episode_return"].max() for r in completed if "mean_episode_return" in r.metrics.columns),
                default=None,
            )
            if completed
            else None
        )
        if best_reward is not None and best_reward != best_reward:  # NaN check
            best_reward = None
        total_bytes = (
            sum(r.metrics["cumulative_comm_bytes"].max() for r in completed if "cumulative_comm_bytes" in r.metrics.columns)
            if completed
            else 0
        )

        def kpi_card(label: str, value: str, icon: str) -> dbc.Col:
            return dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div([html.I(className=f"fa-solid {icon} me-2 text-muted"), html.Span(label, className="text-muted small")]),
                            html.H4(value, className="mb-0 mt-1"),
                        ]
                    ),
                    className="shadow-sm",
                ),
                md=3,
                sm=6,
            )

        kpis = [
            kpi_card("Runs selected", str(len(filtered)), "fa-layer-group"),
            kpi_card("Iterations logged (max)", f"{total_iterations:,}", "fa-forward"),
            kpi_card("Best episode return", f"{best_reward:.4f}" if best_reward is not None else "--", "fa-arrow-trend-up"),
            kpi_card("Total comm (all selected)", f"{total_bytes:.3e} B", "fa-tower-broadcast"),
        ]

        return (
            build_reward_figure(filtered),
            build_comm_overhead_figure(filtered),
            build_critic_loss_figure(filtered),
            status,
            kpis,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Live dashboard for FedNPG-ADMM training results")
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    # Werkzeug's dev server logs every request at INFO -- with the dashboard
    # polling every REFRESH_INTERVAL_MS, that's a line spammed every 3s per
    # open tab. Only startup/error messages are worth seeing here.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    app = build_app(args.results_path)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
