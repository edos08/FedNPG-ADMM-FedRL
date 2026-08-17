# common

> 🚧 Not implemented. This README describes the *plan*, not current behavior.

Shared library used by every other package in this project. No CLI — this
is imported, not run. If you find yourself copy-pasting an env wrapper or a
policy network between `fednpg_admm` and `baselines`, it belongs here
instead.

## Planned contents

```
src/common/
├── envs/       # thin wrappers around gymnasium.make() for Swimmer-v4 /
│               # Hopper-v4 / Humanoid-v4 (paper's Table 3), plus whatever
│               # small-state-space env we pick for task (iii)
├── networks/   # policy MLP: architecture + activation per paper's Table 4
│               # (64x64 ReLU/Tanh for Swimmer, up to 512x512x512 for Humanoid)
├── rollout/    # trajectory collection + advantage estimation (GAE, γ=0.99
│               # per Table 4) — shared by every algorithm variant
├── comms/      # THE important one: an abstraction for "agent sends X to
│               # server" that every algorithm routes through. This is
│               # where `quantization` plugs in later without needing to
│               # touch fednpg_admm or baselines code at all.
└── results/    # results schema (what gets written to disk after a run)
                # + read helpers, shared by results_webapp and
                # results_exporter so they don't diverge on format
```

## Why this exists as its own package

`fednpg_admm` and `baselines` both need: an environment, a policy network,
a rollout collector, and a way to send vectors/matrices from N agents to a
server. The only thing that actually differs between them is the *update
rule*. Without this package, that infra gets duplicated (or worse, drifts
apart) between the two algorithm packages, and you end up debugging two
slightly-different rollout collectors instead of one.

## Installation

```bash
cd common
pip install -e .
```

Or from repo root: `pip install -e src/fedrl_comm/common`.
