# common

Shared library used by every other package. It has no CLI. `fednpg_admm`,
`baselines` and `quantization` differ only in how the server turns the agents'
messages into a policy update, so everything else lives here once.

## Contents

```
src/common/
├── envs/          # Gymnasium/MuJoCo env factory (Swimmer-v4, Hopper-v4,
│                  # Humanoid-v4), per-agent vector env, and running
│                  # observation/reward normalization (normalize.py)
├── networks/      # Gaussian MLP policy and value network (Table 4 of the
│                  # paper: hidden sizes, ReLU hidden layers, Tanh mean head)
├── rollout/       # parallel rollout collection across agents, batch
│                  # container, GAE and critic fitting
├── optim/         # conjugate gradient, Fisher-vector products, TRPO-style
│                  # backtracking line search
├── comms/         # Channel and ByteMeter: every agent->server transmission
│                  # goes through a Channel, which applies an optional
│                  # Transform (quantization plugs in here) and counts bytes
├── results/       # run directory schema: metadata.json + metrics.csv,
│                  # written incrementally during training
├── hyperparams.py # Table 4 values per environment
├── evaluation.py  # deterministic evaluation and checkpoints
└── device.py      # device selection and validation
```

## Byte accounting

`ByteMeter` counts `(number of elements, bits per element)` rather than reading
the tensor's in-memory size. Unquantized sends are counted at 64 bits per value,
the convention of the paper, and a quantizing `Transform` reports its own bit
width, so the communication numbers of every algorithm come from the same code.
`Channel.send()` returns the possibly degraded tensor, and callers must use that
return value, so the server aggregates exactly what was transmitted.

## Installation

Installed together with the other packages from the repository root; see the
main [README](../../../README.md#installation).
