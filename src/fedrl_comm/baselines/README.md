# baselines

> 🚧 Not implemented. This is the comparison side of task **(iii)**.

Implements the algorithms FedNPG-ADMM needs to be compared against, so the
report's claims ("ADMM keeps the convergence rate, drops the communication
cost") are backed by something Claude — sorry, *we* — actually ran, not just
copied from the paper's own Figure 2/3.

## Planned contents

```
src/baselines/
├── fednpg_standard.py  # standard-average FedNPG, eq. (12) in the paper:
│                        # server collects the full H_i (d x d) and g_i from
│                        # every agent, averages, does the exact NPG update.
│                        # O(d^2) communication — this is the thing
│                        # FedNPG-ADMM is supposed to beat, so it needs to
│                        # exist as working code, not just a number pasted
│                        # from Table 1 of the paper.
├── fedppo.py            # first-order baseline (federated PPO), used in the
│                         # paper's Figure 3 comparison. Lower priority than
│                         # fednpg_standard.py — the O(d^2) vs O(d) story is
│                         # the one that actually matters for this project.
└── train.py             # CLI entry point: fedrl-train-baseline --algo {fednpg,fedppo}
```

## Why this is separate from `fednpg_admm`

Both packages share environment/network/rollout code (that's `common`).
Keeping the *competing algorithms* in their own package means the ADMM
implementation isn't tangled up with the thing it's being benchmarked
against, and it makes clear in the repo history which commits are
"reproducing the paper's baseline" vs "implementing the paper's actual
contribution."

## Installation

```bash
cd baselines
pip install -e .
```

## Usage (once implemented)

```bash
fedrl-train-baseline --algo fednpg-standard --env swimmer --num-agents 8
fedrl-train-baseline --algo fedppo --env swimmer --num-agents 8
```
