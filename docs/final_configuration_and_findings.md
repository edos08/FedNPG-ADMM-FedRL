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
  rl-zoo and sb3-contrib report, and it runs ~1.17-1.20x higher. Note that OpenAI
  Spinning Up reports its ON-POLICY algorithms (VPG/TRPO/PPO) *stochastically*
  instead, so each of our two numbers has its own correct comparator: stochastic
  ~3200 against Spinning Up's TRPO ~1200, deterministic 3749 against rl-zoo's 3618.

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

## 3b. Why the curve is noisy: early termination, confirmed across all three environments

The full sweep settles this. The swing is governed entirely by whether an
environment can end an episode early, and Swimmer is the control that proves it.

| env | terminates early? | episode length | swing N=1 | swing N=8 | `healthy_reward` |
|---|---|---|---|---|---|
| Swimmer-v4 | **never** | **999 +- 0** | **2%** | **0%** | none |
| Hopper-v4 | yes | 368-779 | 26% | 8% | 1.0 / step |
| Humanoid-v4 | yes | 195-874 | 32% | 15% | 5.0 / step |

Swimmer-v4 hardcodes both flags false in `gymnasium/envs/mujoco/swimmer_v4.py`
(`return observation, reward, False, False, info`) and the docs state plainly:
*"The Swimmer never terminates."* Every episode runs the full 1000 steps, so the
curve is flat to within 1-2% -- with the same algorithm, the same code path, and
the std floor fully pinned at 0.1353 from iteration 150 onward.

On the terminating environments, return is approximately **affine in survival
time** (~3.75 reward/step on Hopper), so the per-batch mean is a draw from a
two-component mixture: a tight cluster at the time limit plus a broad spread
below it. With 1-2 episodes finishing per iteration, a 1000-to-3700 swing is the
expected behaviour of the estimator, not instability of the policy. The swing
falls with N exactly as sampling theory predicts (Hopper 26/20/13/8% as episodes
per point go 1.3/5.4/10.5/21.7), which is the paper's own observation that both
algorithms "have lower variance ... when more agents are engaged".

**No published MuJoCo learning curve plots this metric raw.** OpenAI Spinning Up
benchmarks *exactly* our metric -- *"the average trajectory return across the
batch collected at each epoch"* -- and states *"Learning curves are smoothed by
averaging over a window of 11 epochs"*, over 10 seeds
(https://spinningup.openai.com/en/latest/spinningup/bench.html). rl-baselines3-zoo
uses a rolling window of 100 episodes (`rl_zoo3/plots/plot_train.py`,
`--episode-window` default 100); openrlbenchmark/CleanRL uses `rolling: int = 100`;
Weights & Biases applies time-weighted EMA by default. If a figure is published
from this metric, it should be smoothed and the smoothing stated in the caption,
as the Open RL Benchmark paper asks (arXiv:2402.03046): *"smoothing sacrifices
data variability information. Authors should provide clear explanations of this
post-treatment to prevent misinterpretation."*

**Truncation handling is not the cause** and was independently re-checked against
Pardo et al., *Time Limits in Reinforcement Learning* (arXiv:1712.00378), SB3's
`on_policy_algorithm.py`, and tianshou's `value_mask`/`compute_episodic_return`:
this repo already bootstraps 0 on `terminated`, `V(final_obs)` on `truncated`,
and cuts the GAE trace on either. Same semantics as both references.

**Two observations here appear to be unpublished.** No source was found that (a)
quantifies the failure rate or characterises the return distribution of a trained
stochastic MuJoCo policy as bimodal, or (b) documents per-dimension `log_std`
divergence in TRPO/PPO without an entropy bonus. Both should be presented as our
own measurements rather than cited.

## 3c. Multi-seed results, and why Hopper alone fails to scale

Final rewards, paper metric (mean of each seed's last 100 iterations), then
mean +- standard deviation **across seeds** -- the same quantity Table 2 reports.

| | N=1 | N=2 | N=4 | N=8 | seeds |
|---|---|---|---|---|---|
| **Swimmer** ours | 94.7 +- 44.0 | **126.1 +- 1.0** | **127.1 +- 0.8** | **128.3 +- 0.7** | 8-9 |
| Swimmer paper | 109.4 +- 5.9 | 123.6 +- 10.3 | 127.2 +- 2.2 | 128.5 +- 1.9 | 10 |
| **Hopper** ours | **2165 +- 705** | 1987 +- 396 | 1719 +- 234 | 1611 +- 145 | 8-9 |
| Hopper paper | 1473 +- 383 | 2384 +- 371 | 2507 +- 230 | 2719 +- 173 | 10 |
| **Humanoid** ours | 945 +- 178 | 3456 +- 901 | 5348 +- 168 | **6212 +- 397** | 9-10 |

**Swimmer reproduces the paper at N>=2 and beats its seed variance** (+-0.7 vs
+-1.9 at N=8). At N=1 our 94.7 +- 44.0 is bimodal across seeds -- six of nine
reach ~123 while three fall into Swimmer's known local optimum
(45, 15, 54). Two or more agents eliminate that failure entirely.

**Humanoid scales 6.6x** from N=1 to N=8, monotonically.

**Hopper is the single reproduction failure**: monotonically *decreasing* in N
where the paper increases. Our N=1 exceeds the paper's (2165 vs 1473); our N=8
reaches 1611 against their 2719.

The per-seed values expose one mechanism behind all three:

```
swimmer N=1: 121 125 125 126  45 122  15 119  54   <- trimodal, 3 seeds stuck
hopper  N=1: 2895 1562 3244 2392 2391 1226 2493 1271 2013   <- wide, sometimes escapes
hopper  N=8: 1664 1585 1367 1805 1603 1460 1641 1764 1424   <- tight, always the same optimum
```

More agents reduce gradient noise, so training converges more *consistently* --
seed std falls from 44.0 to 0.7 on Swimmer and 705 to 145 on Hopper. **Whether
that helps depends on which basin the reduced exploration settles into.** On
Swimmer and Humanoid it reliably avoids the bad optimum; on Hopper it reliably
lands in it, because Hopper's `healthy_reward` of 1.0 is small next to its
forward reward, so a faster-but-falls gait is locally optimal (reward/step rises
3.76 -> 4.49 with N while episode length falls 779 -> 368). Humanoid's
`healthy_reward` of 5.0 aligns speed with survival, and there both rise together.

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
