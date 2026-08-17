# quantization

> 🚧 Not implemented. This is task **(iv)** — the one part of this project
> that isn't in the paper, so there's no Figure to reproduce and check your
> work against. Budget real time for this one.

The paper's own limitations section admits it: no communication-efficiency
trick beyond ADMM itself was explored. This package is that trick.

## Planned contents

```
src/quantization/
├── quantizers.py  # quantization schemes applied to whatever tensor a
│                  # `common.comms.Channel` sends (y_i, g_i for ADMM;
│                  # H_i, g_i for the standard baseline). Start with
│                  # something simple and standard — e.g. uniform stochastic
│                  # quantization a la QSGD (the paper cites Q-GADMM [9],
│                  # which does exactly this for ADMM — worth reading before
│                  # implementing) — before reaching for anything fancier.
└── sweep.py       # CLI entry point: fedrl-quantize-sweep
                    # runs the same training loop at multiple bit-widths
                    # (e.g. 1, 2, 4, 8, 32-bit / no quantization) and
                    # records: final reward, convergence curve, and actual
                    # communication bytes at each resolution.
```

## Design constraint

This package should NOT need to modify `fednpg_admm` or `baselines` code to
work. It wraps `common.comms.Channel` — quantize on `.send()`, dequantize
(or just accept the noise) on `.receive()`. If you find yourself editing
`agent.py` in `fednpg_admm` to make quantization work, the abstraction in
`common.comms` wasn't built right — fix that instead of special-casing here.

## What "study the performance" should produce for the report

At minimum, for each algorithm (FedNPG-ADMM, and ideally the standard
baseline too, for a fair comparison):
- reward-vs-iterations curves at each bit-width, overlaid
- final reward vs. bit-width (single plot, x-axis = bits, y-axis = reward)
- communication bytes vs. bit-width (to show what you're actually buying
  with the accuracy loss)

## Installation

```bash
cd quantization
pip install -e .
```

## Usage (once implemented)

```bash
fedrl-quantize-sweep --algo fednpg-admm --env swimmer --bits 1,2,4,8,32
```
