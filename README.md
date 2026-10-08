# FedNPG-ADMM — Communication-Efficient Federated Natural Policy Gradient

> **Status:** `fednpg_admm` (task ii) and `baselines` (task iii) are complete,
> five seeds per configuration. **Swimmer-v4 reproduces Table 2 at every agent
> count** (123.9/126.0/127.0/128.4 against the paper's 109.4/123.6/127.2/128.5),
> and **Humanoid-v4 scales 993 -> 6329** from N=1 to N=8, both under the
> original configuration. **Hopper-v4 uses its own configuration**: under the
> original one, reward and advantage normalization made larger federations
> collapse late in training and inverted the agent-count trend. With those
> (and observation normalization, line search and the log-std clamp) turned
> off, N=4/N=8 reach 2536/2656 against the paper's 2507/2719 and learn faster
> than N=1/N=2. The per-environment settings, the isolation experiment and
> the refuted hypotheses are in
> [docs/hopper_agent_scaling_findings.md](docs/hopper_agent_scaling_findings.md);
> the full configuration is in
> [docs/final_configuration_and_findings.md](docs/final_configuration_and_findings.md).
> Hopper runs are launched with `./final_sweep.sh hopper` (and
> `hopper-baselines`); `./sweep_status.py` tracks progress. `quantization`
> (task iv) has a stochastic-scheme sweep; `results_exporter` is still a stub.

Course project reproducing and extending:

> G. Lan, H. Wang, J. Anderson, C. Brinton, V. Aggarwal, *"Improved
> Communication Efficiency in Federated Natural Policy Gradient via
> ADMM-based Gradient Updates"*, NeurIPS 2023. [arXiv:2310.19807](https://arxiv.org/abs/2310.19807)

Authors: Edoardo Scarpel, Daniele Zebele.

---

## 1. What the paper actually does

Federated NPG (second-order, better convergence than PG/PPO) requires every
agent to send its local Hessian $H_i ∈ R^{d×d}$ to the server every round —
$O(d²)$ communication. **FedNPG-ADMM** reformulates the global NPG direction
as a distributed quadratic problem solved via ADMM, so agents only ever
transmit two $d$-dimensional vectors ($y_i$, $g_i$) — $O(d)$ communication —
with the *same* convergence rate (Theorem 4.3) and *no* extra sample
complexity. Empirically validated on Swimmer-v4 / Hopper-v4 / Humanoid-v4
(MuJoCo).

## 2. What the professor actually wants

Straight from the email, turned into a checklist:

- [ ] **(i)** Report section explaining the paper's main idea.
- [x] **(ii)** Pseudocode for FedNPG-ADMM + a working implementation
      (Algorithm 1 in the paper — agent-side ADMM update + server-side
      aggregation). → `fednpg_admm` — **done**; see
      [docs/final_configuration_and_findings.md](docs/final_configuration_and_findings.md)
      for the final hyperparameters, the reproduction tables, and the
      hypotheses tested and refuted along the way.
- [ ] **(iii)** *If time allows:* one small state-space instance and one
      large state-space instance, benchmarked against other FedRL methods —
      at minimum standard-average FedNPG ($O(d²)$, eq. 12 in the paper),
      ideally FedPPO too. → `baselines`
- [ ] **(iv)** Quantize the agent→server transmissions and study performance
      vs. quantization resolution. **Not in the original paper** — this is
      the actual novel contribution of the project. → `quantization`

Report deadline: 12 October 2026.

## 3. Repo structure

```
FedNPG-ADMM-FedRL/
├── README.md                      # this file
├── pyproject.toml                 # root meta-package (installs everything)
├── LICENSE
└── src/
    └── fedrl_comm/
        ├── common/                # shared infra — envs, nets, rollout, comms, logging
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/common/
        │       ├── envs/          # Gymnasium/MuJoCo env wrappers (Swimmer/Hopper/Humanoid)
        │       ├── networks/      # policy MLP (per Table 4 of the paper)
        │       ├── rollout/       # trajectory collection + GAE
        │       ├── comms/         # agent<->server communication channel abstraction
        │       │                  # (this is where quantization plugs in)
        │       └── results/       # results schema + read/write helpers
        │
        ├── fednpg_admm/           # (ii): Algorithm 1 from the paper
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/fednpg_admm/
        │       ├── agent.py       # agent-side ADMM update (Algo 1, lines 5-9)
        │       ├── server.py      # server-side aggregation (Algo 1, lines 11-13)
        │       └── train.py       # CLI: fedrl-train-admm
        │
        ├── baselines/             # (iii): the competing FedRL algorithms
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/baselines/
        │       ├── fednpg_standard.py  # standard-average FedNPG, eq. 12 (O(d²))
        │       ├── fedppo.py           # first-order baseline
        │       └── train.py            # CLI: fedrl-train-baseline
        │
        ├── quantization/          # (iv): the novel part — not in the paper
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/quantization/
        │       ├── quantizers.py  # quantization schemes, pluggable into common/comms
        │       └── sweep.py       # CLI: fedrl-quantize-sweep
        │
        ├── results_webapp/        # live dashboard while experiments run
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/results_webapp/
        │       ├── app.py         # CLI: fedrl-results-webapp
        │       ├── data_loader.py
        │       └── plotting.py
        │
        └── results_exporter/      # static tables/figures for the actual report
            ├── README.md
            ├── pyproject.toml
            └── src/results_exporter/
                ├── export.py      # CLI: fedrl-export-results
                ├── tables.py       # LaTeX table generation
                └── figures.py      # camera-ready plots
```

## 4. Setup

```bash
git clone https://github.com/edos08/FedNPG-ADMM-FedRL
cd FedNPG-ADMM-FedRL
# Activate your Python venv of choice, then:
pip install -e . \
  -e src/fedrl_comm/common \
  -e src/fedrl_comm/baselines \
  -e src/fedrl_comm/fednpg_admm \
  -e src/fedrl_comm/quantization \
  -e src/fedrl_comm/results_webapp \
  -e src/fedrl_comm/results_exporter
```

The root `fedrl-comm` package (see root `pyproject.toml`) depends on all six
sub-packages by name, but those names aren't published anywhere — they only
resolve because every sub-package's local path is *also* passed with `-e` in
this same `pip install` invocation, so pip's resolver satisfies each
dependency from the local editable candidate instead of trying (and failing)
to fetch it from PyPI. **`pip install -e .` alone will NOT work** — it needs
every `-e <path>` above in the same command.

Verify install:

```bash
fedrl-train-admm --help
fedrl-train-baseline --help
fedrl-quantize-sweep --help
fedrl-results-webapp --help
fedrl-export-results --help
```

(All of these will currently just tell you they're not implemented yet —
that's expected at this stage. The point is that the packaging plumbing
works *now*, before anyone's written a line of algorithm code, so nobody
loses a day later debugging `pyproject.toml` instead of debugging ADMM.)

## 5. Package details

| Package | Depends on | CLI | Maps to task |
|---|---|---|---|
| `common` | — | — (library only) | shared infra |
| `fednpg_admm` | `common` | `fedrl-train-admm` | (ii) |
| `baselines` | `common` | `fedrl-train-baseline` | (iii) |
| `quantization` | `common` | `fedrl-quantize-sweep` | (iv) |
| `results_webapp` | `common` | `fedrl-results-webapp` | live monitoring |
| `results_exporter` | `common` | `fedrl-export-results` | report figures/tables |

## 6. Suggested rough milestones (edit or delete freely — this is a guess, not a commitment either of us made to Subhra)

- Weeks 1–2: `common` (envs + policy net + rollout) working end to end with
  a *single* non-federated NPG update, just to validate the RL plumbing.
- Weeks 3–4: `fednpg_admm` (Algorithm 1) + `baselines` (eq. 12 standard
  average) running on the small state-space env, results matching the
  paper's Fig. 2 qualitatively.
- Week 5: large state-space env, `quantization` sweep.
- Week 6: `results_exporter`, report writing, buffer for whatever broke.

## 7. Citation

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

## 8. License

MIT — see `LICENSE`. Course project; adjust if your department wants
something else.
