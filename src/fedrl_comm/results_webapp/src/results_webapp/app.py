"""
CLI entry point: fedrl-results-webapp.

A live Dash dashboard: polls the results directory on an interval (default
every 3s) and re-renders, so an in-progress training run's latest logged
iteration shows up without restarting the dashboard.

Dashboard is organized into independent SECTIONS (an "+ Add section"
button creates more), each with its own run-selector dropdown, KPI row,
and Reward/Communication/Critic-loss graphs -- e.g. one section following
Swimmer runs and another following Hopper runs, whose reward scales differ
by orders of magnitude and would otherwise squash each other flat on a
single shared graph. Sections are pure client-visible UI state (a Dash
dcc.Store), not persisted anywhere -- reloading the page resets to one
empty section.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, dcc, html

from .data_loader import load_all_runs, run_options
from .plotting import build_comm_overhead_figure, build_critic_loss_figure, build_kpi_cards, build_reward_figure

REFRESH_INTERVAL_MS = 3000


def _make_section(section_id: int) -> dbc.Card:
    """One self-contained dashboard section: its own run selector, KPIs,
    and graphs, all keyed by `section_id` via Dash's pattern-matching ids
    so a single callback can drive an arbitrary number of these at once
    (see the ALL-based callbacks below). The outer Card itself also
    carries a pattern-matching id (section-card/index) purely so
    manage_sections can find-and-remove one specific section later without
    touching any sibling section's live state -- see that callback's
    docstring for why it never regenerates the whole list."""
    return dbc.Card(
        id={"type": "section-card", "index": section_id},
        children=dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id={"type": "section-run-selector", "index": section_id},
                                options=[],
                                value=[],
                                multi=True,
                                placeholder="Select runs for this section (e.g. all swimmer-* runs)",
                            ),
                            width=True,
                        ),
                        dbc.Col(
                            dbc.Button(
                                html.I(className="fa-solid fa-xmark"),
                                id={"type": "remove-section-btn", "index": section_id},
                                color="link",
                                className="text-muted",
                                title="Remove this section",
                            ),
                            width="auto",
                        ),
                    ],
                    align="center",
                    className="mb-3 g-2",
                ),
                dbc.Row(id={"type": "section-kpi-row", "index": section_id}, className="mb-3 g-3"),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            dcc.Graph(id={"type": "section-reward-graph", "index": section_id}, config={"displaylogo": False}),
                            label="Reward",
                        ),
                        dbc.Tab(
                            dcc.Graph(id={"type": "section-comm-graph", "index": section_id}, config={"displaylogo": False}),
                            label="Communication overhead",
                        ),
                        dbc.Tab(
                            dcc.Graph(id={"type": "section-critic-graph", "index": section_id}, config={"displaylogo": False}),
                            label="Critic loss",
                        ),
                    ]
                ),
            ]
        ),
        className="shadow-sm mb-4",
    )


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
            # `section-ids` is the source of truth for which sections exist
            # and in what order; `next-section-id` hands out ever-increasing
            # ids so a removed section's id is never reused within a page
            # session (reusing one could confuse an in-flight pattern-
            # matching callback keyed by that id).
            dcc.Store(id="section-ids", data=[0]),
            dcc.Store(id="next-section-id", data=1),
            dbc.Row(
                [
                    dbc.Col(html.H2("FedNPG-ADMM results", className="mb-0"), width="auto"),
                    dbc.Col(
                        html.Div(id="status-text", className="text-muted small pt-2"),
                        width="auto",
                        className="ms-auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="fa-solid fa-plus me-2"), "Add section"],
                            id="add-section-btn",
                            color="primary",
                            size="sm",
                            outline=True,
                        ),
                        width="auto",
                    ),
                ],
                align="center",
                className="mb-4",
            ),
            html.Div(id="sections-container", children=[_make_section(0)]),
        ],
    )

    @app.callback(
        Output("sections-container", "children"),
        Output("section-ids", "data"),
        Output("next-section-id", "data"),
        Input("add-section-btn", "n_clicks"),
        Input({"type": "remove-section-btn", "index": ALL}, "n_clicks"),
        State("sections-container", "children"),
        State("section-ids", "data"),
        State("next-section-id", "data"),
        prevent_initial_call=True,
    )
    def manage_sections(_add_clicks, _remove_clicks, existing_children, section_ids, next_id):
        """
        Appends to / filters `existing_children` (the ACTUAL current
        children, from State) rather than rebuilding every section fresh
        from `section_ids` via _make_section -- rebuilding would replace
        every sibling section's live component tree too, silently
        resetting whatever runs the user had already selected in them
        back to empty every time a section is merely added or removed
        elsewhere on the page.
        """
        triggered = ctx.triggered_id
        if triggered == "add-section-btn":
            new_children = existing_children + [_make_section(next_id)]
            return new_children, [*section_ids, next_id], next_id + 1

        if isinstance(triggered, dict) and triggered.get("type") == "remove-section-btn":
            # A remove button's OWN creation also fires this callback (a
            # new component just joined the ALL-matched group) with
            # n_clicks=None -- only act on an actual click, not that
            # spurious appearance.
            if not ctx.triggered[0]["value"]:
                raise dash.exceptions.PreventUpdate
            removed_index = triggered["index"]
            new_children = [
                child
                for child in existing_children
                if child.get("props", {}).get("id", {}).get("index") != removed_index
            ]
            new_ids = [sid for sid in section_ids if sid != removed_index]
            return new_children, new_ids, next_id

        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output({"type": "section-run-selector", "index": ALL}, "options"),
        Input("poll-interval", "n_intervals"),
        State({"type": "section-run-selector", "index": ALL}, "id"),
    )
    def refresh_dropdown_options(_n_intervals, ids):
        # Every section shares the same universe of runs to choose from;
        # each section's OWN selection (`value`) is left untouched here --
        # only the list of what's selectABLE refreshes, so a run started
        # after page load becomes pickable without wiping any section's
        # existing choices.
        options = run_options(load_all_runs(results_path))
        return [options for _ in ids]

    @app.callback(
        Output({"type": "section-reward-graph", "index": ALL}, "figure"),
        Output({"type": "section-comm-graph", "index": ALL}, "figure"),
        Output({"type": "section-critic-graph", "index": ALL}, "figure"),
        Output({"type": "section-kpi-row", "index": ALL}, "children"),
        Output("status-text", "children"),
        Input("poll-interval", "n_intervals"),
        Input({"type": "section-run-selector", "index": ALL}, "value"),
    )
    def refresh_sections(_n_intervals, all_selected):
        all_runs = load_all_runs(results_path)
        status = f"{len(all_runs)} run(s) in {results_path} -- last refreshed {datetime.now():%H:%M:%S}"

        reward_figs, comm_figs, critic_figs, kpi_rows = [], [], [], []
        for selected in all_selected:
            filtered = {rid: r for rid, r in all_runs.items() if rid in (selected or [])}
            reward_figs.append(build_reward_figure(filtered))
            comm_figs.append(build_comm_overhead_figure(filtered))
            critic_figs.append(build_critic_loss_figure(filtered))
            kpi_rows.append(build_kpi_cards(filtered))

        return reward_figs, comm_figs, critic_figs, kpi_rows, status

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
