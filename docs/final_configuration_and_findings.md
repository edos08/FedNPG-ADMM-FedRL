# Final configuration and reproduction findings (single-agent Hopper)

Written 2026-09-24, closing the N=1 investigation. Everything here is measured on
this repo's own runs unless a citation says otherwise. Intended as raw material
for the report's experimental section.

## 1. Headline result

| metric | our FedNPG-ADMM, N=1 | reference |
|---|---|---|
| paper metric (batch avg reward / iteration, final 300 iters) | **3205** | Table 2, Hopper N=1: **1473 ± 383** |
| deterministic evaluation (mean action, 10 episodes) | **3749 ± 4**, episode length **1000/1000** | rl-zoo TRPO 3618 ± 357, sb3-contrib TRPO 3372 ± 215, tianshou TRPO 2046 ± 1038 (all Hopper, 1M steps) |
| plain TRPO, same code base, deterministic | **1688 ± 282**, episode length 411 | — |

Two things follow. **The reproduction succeeds**: single-agent FedNPG-ADMM scores
2.2x the paper's own N=1 figure and is competitive with the best published TRPO
implementations, at 2.05M environment steps (1024 timesteps x 2000 iterations).
And **ADMM is not a handicap at N=1**: it beats the plain-TRPO baseline built from
the same components by more than 2x, reaching the 1000-step time limit where plain
TRPO falls at ~410 steps.

## 2. The two reward metrics, and why they differ

- **`mean_episode_return`** — the paper's metric. Section 5: *"Rewards: the average
  trajectory rewards across the batch collected at each iteration"*, plotted against
  iterations. This is what the dashboard plots and what compares to Table 2.
- **`eval_ep_rew_mean`** — deterministic evaluation: the mean action, no exploration
  noise, on a separate environment, raw (un-normalized) rewards. This is what
  rl-zoo / sb3-contrib / tianshou report, and it runs ~1.17-1.20x higher.

Quoting one against the other is a category error, and it hid the true performance
of this implementation for weeks: every number reported before deterministic
evaluation existed was the stochastic one.

## 3. Why the single-seed curve swings between ~1000 and ~3700

This was the last open question at N=1. It is **not a bug, and not fixable by tuning.**

**The return distribution is bimodal.** Over the last 500 iterations, 57.6% of logged
episodes land in 3600-4000 (they reach the 1000-step limit) and 9% fall below 2000.
The mean, 3190, sits in a gap the policy rarely occupies. With **~1.22 episodes
completed per iteration** at T=1024, each plotted point is essentially a single draw
from those two modes -- hence the apparent thrashing.

**The same policy with noise off never falls**: 3749 ± 4 over 10 episodes. So the
falls are caused by exploration noise, not by a defective policy.

**But the fall rate cannot be tuned away.** Four separate interventions were run:

| intervention | mean action std | falls | swing |
|---|---|---|---|
| baseline (`log_std_min=-2`) | 0.171 | 8-9% | 21% |
| release floor (`--log-std-min -20`) | 0.179 | 12% | 24% |
| release floor *and* ceiling | 0.292 (vs 0.301 clamped) | — | — |
| anneal ceiling to std 0.082 | **0.085** | **12%** | **24%** |

The anneal is the decisive one. A fixed-policy sweep had predicted **0% falls at
std 0.085**; training *down to* that std produced **12%**. The reason: the fall rate
is an **equilibrium property, not a function of the noise level**. A policy trained
at low noise builds a correspondingly smaller robustness margin -- TRPO pushes for a
faster, more marginal gait until falls return to ~10%. Deterministic performance got
*worse* under the anneal (3513 ± 284, length 934, vs 3749 ± 4, length 1000).

Releasing the floor also revealed *why* bound-tweaking cannot work: the three action
dimensions **diverge**. With the floor removed, one dimension collapsed to std 0.056
while another drifted up to 0.311, leaving the mean unchanged at 0.179. TRPO has no
entropy term and the trust region bounds the KL of the *joint* distribution, so a
dimension whose actions barely move the advantage receives almost no gradient on its
`log_std` and random-walks.

**The correct interpretation:** TRPO maximizes *expected* return. On Hopper the
optimum is a gait that runs fast, reaches the time limit ~90% of the time, and falls
~10% of the time; a never-falling policy would have to hop conservatively and score
lower on average. The swing is the optimum, not a defect.

**Why the paper's N=1 curve looks smoother:** at 1473 their policy ends episodes at
~410 steps essentially every time, so its returns are unimodal with nothing to swing
between. Our curve swings *because* our policy is 2.5x better and reaches the ceiling.
On this metric one cannot have both the smooth curve and the strong policy.

## 4. Hypotheses tested and refuted

Recording these so they are not re-run. Each was refuted by measurement, not argument.

1. **ADMM consensus quality.** At N=8 the agents' `y_i` are near-orthogonal
   (pairwise cosine ~0.05) and their mean retains only ~0.57 alignment with the true
   pooled natural gradient. Forcing consensus with 5 ADMM rounds per iteration raised
   cosine to ~0.76 but scored *worse* (1426 vs 1827); an exact pooled solve, with zero
   consensus error, landed within noise of one round. Consensus quality is not the
   bottleneck.
2. **Trust-region step-size overshoot.** Algorithm 1 line 13's closed-form step is
   exact only at ADMM convergence. Measured overshoot factor was 0.68-1.04 and the
   achieved KL 0.0054-0.0107 against delta = 0.01 -- i.e. slightly *conservative*, at
   both N=1 and N=8.
3. **Architecture (Table 4's ReLU hidden + Tanh-squashed mean).** Suspected because no
   reference implementation squashes the Gaussian mean. Tested all four combinations:
   Table 4 as written won (3712 deterministic), a linear mean alone was worst (2088,
   63% of evaluations collapsed). Natural gradient largely absorbs the
   reparameterization, so the 50x tanh gradient attenuation does not bite.
4. **Exploration noise magnitude.** Section 3 above.

Also fixed along the way, and load-bearing: a metrics-reporting bug that truncated
episode returns spanning iteration boundaries (the original "zig-zag"), missing
torch/numpy/random seeding, and observation statistics that excluded terminal states.

## 5. Final configuration

Table 4 values are transcribed from the paper. Everything else is a practical setting;
the paper's Setup section says it follows *"the practical settings in TRPO with line
search and in stable-baselines3 with generalized advantage estimation (0.95) and the
Adam optimizer"*, which is the basis for each choice below.

### From Table 4 (per environment)

| | Swimmer-v4 | Hopper-v4 | Humanoid-v4 |
|---|---|---|---|
| rho | 0.1 | 0.1 | 0.01 |
| delta | 0.01 | 0.01 | 0.01 |
| gamma | 0.99 | 0.99 | 0.99 |
| timesteps T | 2048 | 1024 | 512 |
| iterations K | 1000 | 2000 | 3000 |
| critic lr | 3e-4 | 3e-4 | 1e-5 |
| hidden sizes | 64 x 64 | 128 x 128 | 512 x 512 x 512 |
| hidden activation | ReLU | ReLU | ReLU |
| policy mean output | Tanh | Tanh | Tanh |

`gae_lambda = 0.95` comes from the Setup text, not the table.

### Not in the paper (this project's choices, all defaults in the code)

| setting | value | basis |
|---|---|---|
| observation normalization | on, running mean/std, clip 10 | SB3 `VecNormalize` |
| reward normalization | on, scale-only by running std of discounted return, clip 10 | SB3 `VecNormalize` |
| advantage normalization | on, per-agent | SB3 |
| orthogonal init | sqrt(2) hidden, 0.01 policy head, 1.0 value head | SB3 `ortho_init=True` |
| `log_std_init` | -0.5 (std ~0.61) | tianshou / Spinning Up |
| `log_std` clamp | [-2.0, 0.5] | ours; see section 3 -- keeping it is the *best* measured setting |
| critic fitting | 10 epochs, minibatch 64, Adam | SB3 PPO defaults |
| CG iterations / damping | 10 / rho | classic TRPO; damping = rho follows Algorithm 1 line 8 |
| line search | on, 10 backtracks, shrink 0.8, ratio surrogate + true KL | paper's Setup cites TRPO line search |
| `eta` | 1.0 | not specified anywhere; 1.0 = take the ADMM step unscaled |
| `--on-invalid-step` | skip | Algorithm 1 has no fallback when (sum g_i)^T y <= 0 |

### Commands

Every Table 4 default is applied automatically per `--env`; the flags below are only
the measurement apparatus.

```bash
# FedNPG-ADMM -- the configuration to report
fedrl-train-admm --env hopper --num-agents 1 --seed 0 --device cuda:0 \
  --eval-interval 25 --eval-episodes 10 --save-checkpoint --run-id <id>

# plain TRPO baseline (sibling project /mnt/edoardo/NPG), same settings
npg-train --env hopper --seed 0 --device cuda:0 \
  --eval-interval 25 --eval-episodes 10 --save-checkpoint --run-id <id>
```

Swap `--env swimmer|hopper|humanoid` and `--num-agents 1|2|4|8`; nothing else changes.
For figures matching the paper, run seeds 0-9 and average with a 95% CI across seeds
(their Section 5 protocol). Note this smooths the *plotted* curve by ~sqrt(10); it does
not change the underlying behaviour described in section 3.

## 6. Still open

- **N = 8 underperforms N = 1** (~2000 vs 3749 deterministic), inverting the paper's
  central claim that more agents converge faster and higher. Consensus quality and
  step size are both ruled out (section 4). The one implemented-but-untested lead is
  `--advantage-normalization-scope pooled`: advantages are currently normalized
  *within each agent's own batch* before `g_i` is formed, so `sum_i g_i` weights every
  agent equally regardless of how informative its rollout was. This is provably inert
  at N=1 (verified bit-identical) and active only at N>1 -- the right shape for the
  bug being hunted.
- `results_exporter` is still a `NotImplementedError` stub (README task iv).
- The `baselines` package (standard-average FedNPG, FedPPO) is unimplemented, so the
  paper's Figure 3 comparison cannot be reproduced yet.
