# fednpg_admm

FedNPG-ADMM, Algorithm 1 of Lan et al. (NeurIPS 2023): federated natural policy
gradient in which the global natural-gradient direction is estimated by one
round of consensus ADMM per iteration, so each agent uplinks two
`d`-dimensional vectors (`y_i`, `g_i`) instead of a `d x d` Fisher matrix.

## Contents

```
src/fednpg_admm/
├── agent.py       # agent side (Algorithm 1, lines 5-9): dual update, local
│                  # score-function gradient g_i, and the local solve
│                  # y_i = (H_i + rho I)^-1 (g_i - lambda_i + rho y) by
│                  # conjugate gradient; H_i is never formed
├── server.py      # server side (lines 11-13): average the y_i and take the
│                  # trust-region step on theta
├── hyperparams.py # Table 4 defaults (re-exported from common.hyperparams)
└── train.py       # training loop and CLI: fedrl-train-admm
```

Agent and server code know nothing about channels, quantization or the CLI;
`train.py` wires them together. Its `train()` accepts an optional
`channel_factory`, which is how the `quantization` package swaps in
quantizing channels.

## Usage

```bash
fedrl-train-admm --env swimmer --num-agents 8 --seed 0
fedrl-train-admm --help
```

`--env` (`swimmer`, `hopper`, `humanoid`) selects the defaults of the paper's
Table 4. Each of them can be overridden: `--iterations`, `--timesteps`, `--rho`,
`--delta`, `--gamma`, `--gae-lambda`, `--critic-lr`, `--eta`.

The practical settings around the algorithm are on by default and can be
switched off individually: `--no-obs-normalization`,
`--no-reward-normalization`, `--no-advantage-normalization`, `--no-line-search`,
`--no-ortho-init`, and the log-std clamp through `--log-std-min/--log-std-max`.
Hopper's published results switch off observation, reward and advantage
normalization and the line search, and leave the log-std unclamped; see
`final_sweep.sh`.

Other useful flags: `--num-agents`, `--seed`, `--device` (`cpu` is usually the
fastest choice here), `--eval-interval` (deterministic evaluation every k
iterations), `--save-checkpoint`, `--results-path`, `--run-id`.

`--on-invalid-step` (default `skip`): Algorithm 1's step size is undefined when
`(sum_i g_i)^T y <= 0`. `skip` leaves `theta` unchanged for that iteration and
keeps training; `raise` stops.

Each run writes `metadata.json` and `metrics.csv` to
`<results-path>/<run-id>/`. For the per-iteration metric, prefer `ep_rew_mean`
(trailing mean over the last 100 episodes) to `mean_episode_return`, which
averages only the episodes that finished inside one iteration.

## Installation

Installed together with the other packages from the repository root; see the
main [README](../../../README.md#installation).
