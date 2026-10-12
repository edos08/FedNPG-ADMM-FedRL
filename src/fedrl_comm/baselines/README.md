# baselines

The two methods FedNPG-ADMM is compared against in the paper's Figure 3, built
on the same environments, networks, rollout, critic, normalization and
line search (all from `common`), so the comparison differs only in how the
server forms the policy update.

## Contents

```
src/baselines/
├── fednpg_standard.py  # standard-average FedNPG, eq. (12) of the paper:
│                       # the exact global direction (sum_i H_i)^-1 sum_i g_i
├── fedppo.py           # federated PPO, FedAvg style: local clipped-PPO epochs,
│                       # agents send parameter deltas, the server averages
└── train.py            # CLI: fedrl-train-baseline
```

**Standard-average FedNPG** computes its direction matrix-free, by conjugate
gradient on the pooled Fisher, because materializing `sum_i H_i` is not
feasible at Humanoid scale (about 2 PB in double precision). The byte meter is
nevertheless charged the full `d^2 + d` values per agent per round, which is what
the method would transmit, so the communication comparison is exact.

**FedPPO** uses the clipping parameter the paper gives (0.2). The paper does not
specify the aggregation or the number of local epochs; this implementation runs
several local PPO epochs per round and averages the agents' parameter deltas.

## Usage

```bash
fedrl-train-baseline --algo fednpg-standard --env swimmer --num-agents 8 --seed 0
fedrl-train-baseline --algo fedppo          --env swimmer --num-agents 8 --seed 0
fedrl-train-baseline --help
```

The flags for the practical settings match `fedrl-train-admm`
(`--no-obs-normalization`, `--no-line-search`, `--log-std-min`, ...), so a
baseline can run in exactly the configuration of the FedNPG-ADMM runs it is
compared with. `run_sweep.sh base-swimmer|base-humanoid` and
`final_sweep.sh hopper-baselines` run the published comparisons.

## Installation

Installed together with the other packages from the repository root; see the
main [README](../../../README.md#installation).
