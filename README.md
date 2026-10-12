# FedNPG-ADMM: communication-efficient federated natural policy gradient

A PyTorch reimplementation and extension of

> G. Lan, H. Wang, J. Anderson, C. Brinton, V. Aggarwal, *"Improved
> Communication Efficiency in Federated Natural Policy Gradient via
> ADMM-based Gradient Updates"*, NeurIPS 2023.
> [arXiv:2310.19807](https://arxiv.org/abs/2310.19807)

Federated natural policy gradient (FedNPG) normally needs every agent to send
its local Fisher matrix, `d x d` values, to the server each round. FedNPG-ADMM
writes the global natural-gradient direction as a consensus problem and solves
it with one round of ADMM per iteration, so each agent sends two `d`-dimensional
vectors instead, at the same convergence rate.

This repository contains:

- **FedNPG-ADMM** (Algorithm 1 of the paper), written from scratch.
- **The paper's two baselines**: standard-average FedNPG, metered at its full
  `O(d^2)` uplink, and federated PPO.
- **Quantization of the agent-to-server uplink**, which the paper does not
  study: stochastic uniform and reduced-precision float schemes, with sweeps
  over bit width.
- **A live dashboard** for monitoring runs.

The full write-up, with derivations, results and figures, is
[report.pdf](report.pdf).

Authors: Edoardo Scarpel, Daniele Zebele.

## Results

Final reward on MuJoCo, mean ± standard deviation over five seeds (each seed
summarized by its mean over the last 100 iterations, as in the paper's
Table 2). Paper values are from its Table 2, over ten seeds.

| Environment | Source | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|---|
| Swimmer-v4 | ours | 123.9 ± 2.3 | 126.0 ± 1.3 | 127.0 ± 0.8 | 128.4 ± 0.8 |
| | paper | 109.4 ± 5.9 | 123.6 ± 10.3 | 127.2 ± 2.2 | 128.5 ± 1.9 |
| Hopper-v4 | ours | 1941 ± 494 | 1636 ± 282 | 2536 ± 116 | 2656 ± 286 |
| | paper | 1473 ± 383 | 2384 ± 371 | 2507 ± 230 | 2719 ± 173 |
| Humanoid-v4 | ours | 993 ± 187 | 3391 ± 796 | 5343 ± 121 | 6329 ± 334 |

- **Scaling with the number of agents.** More agents learn faster on all three
  environments. On Swimmer and Humanoid the final reward also increases
  monotonically with N; on Hopper, four and eight agents match the paper and
  finish well above one and two.
- **Baselines at N=8.** FedNPG-ADMM matches standard-average FedNPG on Hopper
  (2656 ± 286 against 2494 ± 677) at 9219 times less uplink per round, and
  exceeds it on Humanoid (6329 ± 334 against 4699 ± 298). FedPPO sends half as
  much per round as FedNPG-ADMM and finishes below it on Hopper and Humanoid.
- **Quantization.** At 8 bits, stochastic quantization keeps both the final
  reward (within the seed spread) and the faster learning with more agents on
  all three environments, for a further 8x less uplink than the paper's 64-bit
  accounting. At 6 bits Humanoid fails at every agent count; Swimmer and Hopper
  hold their final reward.

Hopper runs with a reduced configuration: no reward, advantage or observation
normalization, no line search and no log-std clamp. Swimmer and Humanoid use the
standard TRPO and stable-baselines3 practical settings. See
[docs/hopper_agent_scaling_findings.md](docs/hopper_agent_scaling_findings.md).

## Installation

Python 3.11 or later. Each component is its own package under
`src/fedrl_comm/`, and they must be installed together, in one command:

```bash
git clone https://github.com/edos08/FedNPG-ADMM-FedRL
cd FedNPG-ADMM-FedRL
pip install -e . \
  -e src/fedrl_comm/common \
  -e src/fedrl_comm/baselines \
  -e src/fedrl_comm/fednpg_admm \
  -e src/fedrl_comm/quantization \
  -e src/fedrl_comm/results_webapp
```

The sub-packages are not published on PyPI. Passing every local path in the
same `pip install` lets pip resolve them against each other; `pip install -e .`
alone does not install them. The packages pin `torch==2.11.0`; install the
build that matches your CUDA version first if you plan to use a GPU (for these
small policies the CPU is usually faster).

## Usage

```bash
# FedNPG-ADMM
fedrl-train-admm --env swimmer --num-agents 8 --seed 0

# baselines
fedrl-train-baseline --algo fednpg-standard --env swimmer --num-agents 8 --seed 0
fedrl-train-baseline --algo fedppo          --env swimmer --num-agents 8 --seed 0

# quantization
fedrl-quantize-sweep --env swimmer --scheme stochastic --bits 4,6,8,32
python scripts/run_grid.py --env humanoid --num-seeds 5 --agents 2,4,8 --bits 6,8

# live dashboard
fedrl-results-webapp --results-path results --port 8050
```

`--env` is `swimmer`, `hopper` or `humanoid` and loads that environment's
hyperparameters from the paper's Table 4; every one of them can be overridden
on the command line (`--help` lists them). Runs are written to
`results/<run-id>/` as `metadata.json` and `metrics.csv`.

### Reproducing the results

```bash
./run_sweep.sh swimmer            # FedNPG-ADMM, 5 seeds x N = 1, 2, 4, 8
./run_sweep.sh humanoid
./run_sweep.sh base-swimmer       # baselines at N=8
./run_sweep.sh base-humanoid

./final_sweep.sh hopper           # Hopper, in its reduced configuration
./final_sweep.sh hopper-baselines
./final_sweep.sh hopper-quant

./sweep_status.py --sweep final-hopper   # progress, liveness and ETA
```

`final_sweep.sh` opens a tmux session with one pane per seed and live progress
bars. Both runners skip completed runs, so they can be restarted after an
interruption. A Humanoid run at N=8 takes several hours on CPU.

## Repository layout

```
.
├── README.md
├── report.pdf              # full write-up
├── pyproject.toml          # root package, depends on the five below
├── run_sweep.sh            # Swimmer and Humanoid sweeps
├── final_sweep.sh          # Hopper sweeps
├── sweep_status.py         # progress and ETA for running sweeps
├── scripts/run_grid.py     # seeds x agents x bit widths
├── docs/                   # theory notes and experiment records
└── src/fedrl_comm/
    ├── common/             # envs, networks, rollout, optimizers, comms, results
    ├── fednpg_admm/        # Algorithm 1: fedrl-train-admm
    ├── baselines/          # standard-average FedNPG, FedPPO: fedrl-train-baseline
    ├── quantization/       # uplink quantization: fedrl-quantize-sweep
    └── results_webapp/     # live dashboard: fedrl-results-webapp
```

Each package has its own README. Every agent-to-server transmission goes
through `common.comms.Channel`, which counts bytes the same way for all
algorithms and is where the quantizers plug in.

## Documentation

- [docs/npg_admm_theory_and_implementation.md](docs/npg_admm_theory_and_implementation.md):
  the theory behind NPG and FedNPG-ADMM, mapped onto the code.
- [docs/final_configuration_and_findings.md](docs/final_configuration_and_findings.md):
  the configuration of every environment and the measurements behind it.
- [docs/hopper_agent_scaling_findings.md](docs/hopper_agent_scaling_findings.md):
  why Hopper uses a different configuration.
- [docs/quantization_report.md](docs/quantization_report.md): how the
  quantization works, with results (in Italian).

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{lan2023fednpgadmm,
  title     = {Improved Communication Efficiency in Federated Natural Policy
               Gradient via ADMM-based Gradient Updates},
  author    = {Lan, Guangchen and Wang, Han and Anderson, James and
               Brinton, Christopher and Aggarwal, Vaneet},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  eprint    = {2310.19807}
}
```

## License

MIT; see [LICENSE](LICENSE).
