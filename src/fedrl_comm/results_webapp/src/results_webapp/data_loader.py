"""
Loads runs from disk for the dashboard. Thin wrapper around
common.results.load_runs -- exists as its own module (rather than calling
load_runs directly from app.py) so the polling/refresh behavior lives in
one obvious place, not scattered across callbacks.
"""

from __future__ import annotations

from common.results import RunResult, load_runs


def load_all_runs(results_path: str) -> dict[str, RunResult]:
    """
    Re-scans `results_path` from scratch every call -- no caching. This is
    deliberate: results directories are small (a handful of runs, each a
    small CSV), and re-reading from disk on every dashboard refresh means
    an in-progress training run's latest logged iteration is always
    reflected, with no cache-invalidation logic to get wrong.
    """
    return load_runs(results_path)


def run_options(runs: dict[str, RunResult]) -> list[dict]:
    """Dropdown options for a dcc.Dropdown, sorted by most-recently-started first."""
    sorted_runs = sorted(
        runs.values(), key=lambda r: r.metadata.get("started_at", ""), reverse=True
    )
    return [{"label": r.run_id, "value": r.run_id} for r in sorted_runs]
