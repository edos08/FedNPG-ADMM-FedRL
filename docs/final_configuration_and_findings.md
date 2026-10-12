# Final configuration and reproduction findings (single-agent Hopper)

Written 2026-09-24, closing the N=1 investigation. Everything here is measured on
this repo's own runs unless a citation says otherwise.

> **Status, 2026-10-09.** Sections 1-4 describe the *original* configuration and
> remain accurate as a record of it. Section 3c's multi-seed tables and its
> "Hopper fails to scale" conclusion were superseded: the cause was found
> (reward and advantage normalization) and Hopper now runs with its own
> configuration. Section 3c below now gives the final five-seed numbers; the
> investigation itself is in
> [hopper_agent_scaling_findings.md](hopper_agent_scaling_findings.md).
> Section 5's configuration table is per environment, and section 6 lists what is
> actually still open.

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

## 3c. Multi-seed results (final, five seeds)

Metric: mean of each seed's last 100 iterations of `mean_episode_return`,
then mean +- sample sd over seeds 0-4 -- the quantity Table 2 of the paper
reports. Swimmer and Humanoid use the original configuration; Hopper uses the
Hopper configuration (section 5). Runs: `{swimmer,humanoid}-N{n}-s{0..4}` and
`final-hopper-N{n}-s{0..4}`.

| | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| **Swimmer** ours | 123.9 +- 2.3 | 126.0 +- 1.3 | 127.0 +- 0.8 | 128.4 +- 0.8 |
| Swimmer paper | 109.4 +- 5.9 | 123.6 +- 10.3 | 127.2 +- 2.2 | 128.5 +- 1.9 |
| **Hopper** ours | 1941 +- 494 | 1636 +- 282 | 2536 +- 116 | 2656 +- 286 |
| Hopper paper | 1473 +- 383 | 2384 +- 371 | 2507 +- 230 | 2719 +- 173 |
| **Humanoid** ours | 993 +- 187 | 3391 +- 796 | 5343 +- 121 | 6329 +- 334 |

- **Swimmer** is inside the paper's spread at N=2/4/8 and above it at N=1, with
  smaller seed variance at every agent count. No N=1 seed falls into Swimmer's
  local optimum (per-seed 120.9-126.0). More agents learn faster at every
  checkpoint (training return at iteration 50: 32/50/65/102 for N=1/2/4/8).
- **Humanoid** is absent from the paper's Table 2; it scales 6.4x from N=1 to
  N=8, monotonically, and faster with more agents throughout.
- **Hopper** (Hopper configuration) matches the paper within one sd at N=4 and
  N=8 and learns faster with more agents early on (smoothed training return at
  iteration 100: 598/638/986/1092). It is not strictly monotone: N=2 finishes
  below N=1, inside the seed spread. Five seeds resolve two groups, {1, 2} and
  {4, 8}.

Under the original configuration Hopper read 2497 +- 635 / 2140 +- 386 /
1658 +- 273 / 1605 +- 159 (same metric; those runs were later deleted) -- the
inverted trend this section used to report as an unexplained failure.

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

### Not in the paper (this project's choices) -- differs between environments

**Hopper uses a different configuration from Swimmer and Humanoid** (decided
2026-10-07). Why each setting differs, with the measurements, is in
`docs/hopper_agent_scaling_findings.md` sections 3-5. In short: reward and
advantage normalization each cause Hopper's late collapse, which grows with the
number of agents; observation normalization stops it reaching the stable gait.
Swimmer and Humanoid already reproduce the paper's agent-count ordering with the
original configuration, so they keep it.

| setting | Swimmer, Humanoid | Hopper | basis |
|---|---|---|---|
| observation normalization | on, running mean/std, clip 10 | **off** | SB3 `VecNormalize` |
| reward normalization | on, scale-only by running std of discounted return, clip 10 | **off** | SB3 `VecNormalize` |
| advantage normalization | on, per-agent | **off** | SB3 |
| line search | on, 10 backtracks, shrink 0.8, ratio surrogate + true KL | **off** | paper's Setup cites TRPO line search |
| `log_std` clamp | [-2.0, 0.5] | **off** (`--log-std-min -20 --log-std-max 20`) | ours |
| orthogonal init | sqrt(2) hidden, 0.01 policy head, 1.0 value head | same | SB3 `ortho_init=True` |
| `log_std_init` | -0.5 (std ~0.61) | same | tianshou / Spinning Up |
| critic fitting | 10 epochs, minibatch 64, Adam | same | SB3 PPO defaults |
| CG iterations / damping | 10 / rho | same | classic TRPO; damping = rho follows Algorithm 1 line 8 |
| `eta` | 1.0 | same | not specified anywhere; 1.0 = take the ADMM step unscaled |
| `--on-invalid-step` | skip | same | Algorithm 1 has no fallback when (sum g_i)^T y <= 0 |

Final runs: Swimmer and Humanoid `{env}-N{1,2,4,8}-s{0..4}` (the code defaults,
launched by `run_sweep.sh`); Hopper `final-hopper-N{1,2,4,8}-s{0..4}` (launched by
`final_sweep.sh`).

### Commands

Every Table 4 default is applied automatically per `--env`; the flags below are only
the measurement apparatus.

```bash
# FedNPG-ADMM, Swimmer and Humanoid -- code defaults
fedrl-train-admm --env swimmer --num-agents 1 --seed 0 --device cpu \
  --eval-interval 25 --eval-episodes 10 --save-checkpoint --run-id <id>

# FedNPG-ADMM, Hopper -- five settings off (see table above)
fedrl-train-admm --env hopper --num-agents 1 --seed 0 --device cpu \
  --no-line-search --no-obs-normalization --no-reward-normalization \
  --no-advantage-normalization --log-std-min -20 --log-std-max 20 \
  --eval-interval 25 --eval-episodes 10 --save-checkpoint --run-id <id>
```

(The plain-TRPO numbers in section 1 come from a separate single-agent TRPO
implementation that is not part of this repository.)

Swap `--env swimmer|humanoid` (first command) and `--num-agents 1|2|4|8`; Hopper always uses the second command. All runs on CPU: it measured faster than the GPU for every environment, Humanoid included (re-checked 2026-10-07).
For figures matching the paper, run seeds 0-9 and average with a 95% CI across seeds
(their Section 5 protocol). Note this smooths the *plotted* curve by ~sqrt(10); it does
not change the underlying behaviour described in section 3.

## 6. Still open (2026-10-09)

- **Hopper quantization under the Hopper configuration** is running
  (`./final_sweep.sh hopper-quant`, 30 runs). The current Hopper quantization
  results use the original configuration.
- **Humanoid: standard-average FedNPG finishes below FedNPG-ADMM** at N=8
  (4699 +- 298 against 6329 +- 334), although it computes the exact direction
  ADMM approximates. Cause not established.
- **The Hopper mechanism is an account, not a tested result**: the fixed-KL step
  plus signal normalization explanation fits the isolation table but was not
  tested in isolation (a step that shrinks as the gradient vanishes would).
- **The isolation that chose Hopper's configuration is one seed per setting.**
- The reduced-precision **float** quantization scheme is implemented but never run.

Resolved since the first version of this list: N=8 underperforming N=1 on
Hopper (cause found, see hopper_agent_scaling_findings.md), pooled advantage
normalization (tested, worse), and the baselines (standard-average FedNPG and
FedPPO, both implemented and run).
