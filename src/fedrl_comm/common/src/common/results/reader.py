"""
Results reader — the other half of writer.py's contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class RunResult:
    run_id: str
    metadata: dict
    metrics: pd.DataFrame  # empty DataFrame if metrics.csv doesn't exist yet (run just started)


def load_run(run_dir: "str | Path") -> "RunResult | None":
    """
    Load a single run directory. Returns None (not an exception) if the
    directory doesn't look like a run at all (no metadata.json) — this is
    the expected case while scanning a results root that might contain
    unrelated files, not an error condition.
    """
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        metrics = pd.read_csv(metrics_path)
    else:
        # Run exists (metadata written) but hasn't logged an iteration yet —
        # a real state for a just-started live run, not a bug.
        metrics = pd.DataFrame()

    return RunResult(run_id=metadata.get("run_id", run_dir.name), metadata=metadata, metrics=metrics)


def load_runs(results_root: "str | Path") -> dict[str, RunResult]:
    """
    Scan `results_root` for run directories (one level deep) and load every
    one that has a metadata.json. Silently skips anything else in that
    directory (stray files, non-run subdirectories) rather than raising —
    a results directory accumulates clutter over a project's lifetime, and
    that's not this function's problem to police.
    """
    results_root = Path(results_root)
    if not results_root.exists():
        return {}

    runs: dict[str, RunResult] = {}
    for entry in sorted(results_root.iterdir()):
        if not entry.is_dir():
            continue
        run = load_run(entry)
        if run is not None:
            runs[run.run_id] = run
    return runs
