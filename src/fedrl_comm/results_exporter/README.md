# results_exporter

> 🚧 Not implemented.

Turns whatever's sitting in `results/` into the static tables and figures
that actually go in the PDF you hand to Subhra — separate from
`results_webapp`, which is for watching things live while they run, not for
producing the final camera-ready output. Different job, different package;
don't merge them just because they both read the same data.

## Planned contents

```
src/results_exporter/
├── export.py   # CLI entry point: fedrl-export-results
├── tables.py   # LaTeX table generation (final rewards ± std per
│               # algorithm/env, à la the paper's Table 1/2)
└── figures.py  # matplotlib figures, report-quality (not the interactive
                # plotly ones from results_webapp) — reward curves,
                # communication overhead, quantization sweep results
```

## Installation

```bash
cd results_exporter
pip install -e .
```

## Usage (once implemented)

```bash
fedrl-export-results --results-path results/ --output-path report/assets/
```
