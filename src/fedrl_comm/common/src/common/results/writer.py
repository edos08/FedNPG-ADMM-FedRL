"""
Results writer.

Directory layout, one directory per run:

    <results_root>/<run_id>/
        metadata.json   # static: env, algorithm, num_agents, hyperparams, timestamps
        metrics.csv     # one row per iteration, APPENDED incrementally during
                         # training so a live dashboard can show an in-progress
                         # run without waiting for it to finish

run_id is caller-supplied — this module doesn't invent naming schemes, it
just reads/writes whatever directory it's given.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


class ResultsWriter:
    def __init__(self, results_root: "str | Path", run_id: str, metadata: dict) -> None:
        self.run_dir = Path(results_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.run_dir / "metadata.json"
        self.metrics_path = self.run_dir / "metrics.csv"

        full_metadata = {
            **metadata,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.metadata_path.write_text(json.dumps(full_metadata, indent=2))
        # A fresh ResultsWriter means a fresh run: drop any metrics.csv left
        # over from a prior run under this same run_id, or log_iteration's
        # append-mode writes would pile new rows under stale ones (or raise
        # a ragged-columns error if the two runs logged different metrics).
        self.metrics_path.unlink(missing_ok=True)
        self._fieldnames: "list[str] | None" = None

    def log_iteration(self, iteration: int, **metrics) -> None:
        """
        Append one row. The FIRST call fixes the CSV's columns from
        whatever keys are passed; every subsequent call must pass exactly
        the same keys, or this raises — a silently ragged CSV (rows with
        different columns) is worse than a loud crash, since it corrupts
        every downstream reader in a way that's easy to miss.
        """
        row = {"iteration": iteration, **metrics}
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        elif set(row.keys()) != set(self._fieldnames):
            raise ValueError(
                f"log_iteration called with keys {sorted(row.keys())}, but the "
                f"first call fixed the columns as {sorted(self._fieldnames)}. "
                "Every iteration must log the same set of metrics."
            )

        write_header = not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def finalize(self, **extra_metadata) -> None:
        """Merge extra fields (e.g. completed_at, final reward) into metadata.json."""
        metadata = json.loads(self.metadata_path.read_text())
        metadata.update(extra_metadata)
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata_path.write_text(json.dumps(metadata, indent=2))
