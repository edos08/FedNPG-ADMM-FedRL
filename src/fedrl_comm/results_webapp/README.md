# results_webapp

Live Dash dashboard for training runs. It reads the run directories that
`common.results` writes (`metadata.json` + `metrics.csv`, appended every
iteration) and refreshes every 3 seconds, so runs in progress update without a
restart.

## Contents

```
src/results_webapp/
├── app.py          # Dash app and CLI: fedrl-results-webapp
├── data_loader.py  # loads every run under the results directory
└── plotting.py     # reward, communication-overhead and critic-loss figures
```

## Usage

```bash
fedrl-results-webapp --results-path results --port 8050
```

Then open http://localhost:8050. `--host 0.0.0.0` serves it on the network.

## Installation

Installed together with the other packages from the repository root; see the
main [README](../../../README.md#installation).
