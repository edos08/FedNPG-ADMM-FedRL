# fednpg_admm

> 🚧 Not implemented. This is task **(ii)** from the professor's email.

Implementation of **Algorithm 1** from the paper (page 6): FedNPG where the
global NPG direction is estimated via ADMM instead of averaging Hessians.

## Planned contents

```
src/fednpg_admm/
├── agent.py    # per-agent ADMM update (Algo 1, lines 5-9):
│               #   - dual variable update: λ_i ← λ_i + ρ(y_i^{k-1} - y^{k-1})
│               #   - local gradient: g_i^k from collected trajectories
│               #   - local ADMM solve: y_i^k = (H_i^k + ρI)^-1 (g_i^k - λ_i + ρy^{k-1})
│               #     (use conjugate gradient here, not literal matrix inversion —
│               #     inverting H_i defeats the entire point of the paper)
├── server.py   # server-side aggregation (Algo 1, lines 11-13):
│               #   - y^k ← average of y_i^k across agents
│               #   - θ^k ← θ^{k-1} + η * sqrt(2Nδ / (Σg_i^k)ᵀy^k) * y^k
└── train.py    # CLI entry point: fedrl-train-admm
                 # orchestrates: common.envs -> common.rollout -> agent.py loop
                 # -> common.comms (send y_i, g_i) -> server.py -> broadcast θ
```

## Notes for implementation

- `agent.py` and `server.py` should NOT construct or invert any `d x d`
  matrix directly — the whole point of ADMM here is you never materialize
  `H_i`. Use Hessian-vector products (`torch.autograd.functional.hvp` or a
  manual double-backward) plus conjugate gradient for the linear solve in
  line 8.
- Every tensor agents send to the server (`y_i`, `g_i`) should go through
  `common.comms.Channel`, not be passed directly — that's what lets
  `quantization` intercept it later, and what lets us count communicated
  bytes for the report's O(d) vs O(d²) comparison.
- Hyperparameters per env (ρ, δ, γ, timesteps, iterations, learning rate)
  are in the paper's Table 4 — don't reinvent them, just transcribe.

## Installation

```bash
cd fednpg_admm
pip install -e .
```

## Usage (once implemented)

```bash
fedrl-train-admm --env swimmer --num-agents 8 --iterations 1000
```
