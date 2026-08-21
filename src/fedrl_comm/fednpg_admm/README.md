# fednpg_admm

Implementation of **Algorithm 1** from the paper (page 6): FedNPG where the
global NPG direction is estimated via ADMM instead of averaging Hessians —
task **(ii)** from the professor's email. Implemented and runnable.

## Contents

```
src/fednpg_admm/
├── agent.py       # per-agent ADMM update (Algo 1, lines 5-9): dual variable
│                  # update, local policy gradient g_i, local ADMM solve for
│                  # y_i via conjugate gradient against a Fisher-vector product
│                  # (H_i is never materialized as a d x d matrix)
├── server.py      # server-side aggregation (Algo 1, lines 11-13): averages
│                  # y_i into y, takes the trust-region step on theta
├── cg.py          # conjugate gradient solver used by agent.py
├── fisher.py      # Fisher-vector product (no explicit Hessian construction)
├── hyperparams.py # per-env defaults transcribed from the paper's Table 4
└── train.py       # CLI entry point: fedrl-train-admm
                    # orchestrates: common.envs -> common.rollout -> agent.py
                    # -> common.comms (send y_i, g_i) -> server.py -> broadcast theta
```

## Installation

This package depends on `fedrl-common`, which is **not published anywhere** —
it only resolves when installed from the monorepo root with every sibling
package's local path passed to the *same* `pip install` invocation, so pip's
resolver satisfies `fedrl-common` (and friends) locally instead of trying to
fetch them from PyPI:

```bash
cd /path/to/FedNPG-ADMM-FedRL   # repo root, not this directory
pip install -e . \
  -e src/fedrl_comm/common \
  -e src/fedrl_comm/baselines \
  -e src/fedrl_comm/fednpg_admm \
  -e src/fedrl_comm/quantization \
  -e src/fedrl_comm/results_webapp \
  -e src/fedrl_comm/results_exporter
```

`cd src/fedrl_comm/fednpg_admm && pip install -e .` on its own will **not**
work, for the same reason — it needs every other `-e <path>` in the same
command.

**GPU note:** `torch` is pinned to `torch==2.4.1` (see this package's
`pyproject.toml`) rather than left open-ended. Newer `torch` wheels drop
compiled kernels for older GPU architectures (Pascal/Volta, compute
capability < 7.5 — e.g. TITAN Xp/TITAN V) and training will fail with `CUDA
error: no kernel image is available for execution on the device` on those
GPUs otherwise. If you deliberately bump this pin, re-check
`python -c "import torch; print(torch.cuda.get_arch_list())"` against
`nvidia-smi --query-gpu=compute_cap --format=csv` for every GPU you plan to
train on first.

Verify the install:

```bash
fedrl-train-admm --help
```

## Usage

```bash
fedrl-train-admm --env swimmer --num-agents 8 --seed 0
```

`--env` (`swimmer` | `hopper` | `humanoid`) selects hyperparameter defaults
from the paper's Table 4 (`hyperparams.py`). Any of these can be overridden
from the CLI — omitted flags fall back to that env's default:

```
--iterations, --timesteps, --rho, --delta, --gamma, --gae-lambda,
--critic-lr, --eta
```

Other flags:

```
--num-agents N        number of simulated agents (default 8)
--seed N               (default 0)
--cg-iters N            conjugate-gradient iterations per agent update (default 10)
--critic-epochs N       critic-fitting epochs per outer iteration (default 5)
--results-path PATH     where run directories are written (default "results")
--run-id ID             override the auto-generated run id
--device {cpu,mps,cuda:N}
--on-invalid-step {skip,raise}
--show-warnings         don't suppress Gymnasium/etc. deprecation warnings
```

`--device`: validated against what's actually usable on this machine before
training starts (see `common.device.resolve_device`) — a `ValueError` here
almost always means a torch/GPU architecture mismatch (see the GPU note
above), not a bug in this package.

`--on-invalid-step` (default `skip`): Algorithm 1 as written has no line
search, so the trust-region step size on `theta` can be undefined for a
given iteration (see `server.py`'s docstring). `skip` logs a warning and
keeps training with `theta` unchanged that iteration; `raise` aborts
immediately. If you see frequent skips, check `critic_loss` in the logged
metrics before suspecting the ADMM/CG code — a badly-fit critic producing
noisy advantages is the most common cause.

Each run writes `metadata.json` + `metrics.csv` under
`<results-path>/<run-id>/`. Watch a run live while it trains with:

```bash
fedrl-results-webapp --results-path results
```
