# results_webapp

> 🚧 Not implemented.

Live Dash dashboard for watching training runs happen instead of tailing a
log file and squinting at numbers — same idea as the `results_webapp`
package in the old BSS repo, pointed at whatever `common.results` schema
ends up being.

## Planned contents

```
src/results_webapp/
├── app.py          # CLI entry point: fedrl-results-webapp, Dash app setup
├── data_loader.py  # reads whatever common.results writes, with polling/
│                   # auto-refresh for in-progress runs
└── plotting.py     # reward curves, communication-overhead curves,
                     # quantization bit-width comparisons — the live
                     # equivalents of the report's final figures
```

## Installation

```bash
cd results_webapp
pip install -e .
```

## Usage (once implemented)

```bash
fedrl-results-webapp --results-path results/ --port 8050
```

Then open http://localhost:8050.
