# Why Hopper got worse with more agents, and the configuration that fixes it

Investigation record, 2026-09-24 to 2026-10-08.
All numbers are deterministic evaluation return (10 episodes, every 25 iterations)
unless stated otherwise. Supersedes the Hopper analysis in
`final_configuration_and_findings.md` §3c, whose explanation ("Hopper's healthy
reward makes a fast-but-falls gait locally optimal") describes a symptom of the
cause below rather than the cause.

## 1. The problem

Under the original configuration, Hopper's final return fell as the number of
agents grew, the reverse of the paper's Table 2. Swimmer and Humanoid scaled
correctly.

| Hopper-v4, final training return (5 seeds) | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| ours, original configuration | 2362 ± 551 | 2018 ± 377 | 1621 ± 257 | 1587 ± 145 |
| paper (FedNPG-ADMM, Table 2) | 1473 ± 383 | 2384 ± 371 | 2507 ± 230 | 2719 ± 173 |

(Our row is the trailing-episode metric `ep_rew_mean`, averaged over each seed's
last 500 iterations, with population sd. The final results use Table 2's own metric,
`mean_episode_return` over the last 100 iterations with sample sd, under which
the same runs read 2497 ± 635 / 2140 ± 386 / 1658 ± 273 / 1605 ± 159 (section
9). Same runs, same conclusion; quote the Table 2-metric numbers.)

## 2. What was actually happening: learn, then collapse

Deterministic evaluation showed that the paper's effect was reproduced early in
training and destroyed later.

| iteration | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| 100 | 817 ± 298 | 1491 ± 1096 | 2293 ± 1188 | 2325 ± 968 |
| 2000 | 3062 ± 870 | 2891 ± 785 | 1934 ± 316 | 1658 ± 248 |

At iteration 100 return increases with N, as in the paper. By iteration 2000 the
order is inverted. On seed 0, N=8 reached **3854 at iteration 575 with
1000-step episodes**, the best Hopper policy in the project and above the paper's
best Hopper number, and then decayed **51%** to 1870, with episodes shortening to
~410 steps. N=1 held ~3640 flat from iteration ~500 to 2000.

Two consequences:

- The reported metric (mean of the last 100 iterations of *stochastic* training
  return over a 2000-iteration run) measured the collapsed policy, not the learned one.
- The paper's Hopper curves are likely still climbing at K=2000: its N=1 result
  (1473) is well below a standard TRPO baseline on Hopper, so it probably never
  reaches the regime where the collapse appears.

## 3. Cause: two of the "practical settings" we added (a third also hurts)

Our implementation added six settings on top of Algorithm 1, following the paper's
statement that it uses "the practical settings in TRPO with line search, and in
stable-baselines3": TRPO line search, observation normalization, reward
normalization, advantage normalization, orthogonal initialization, and a log-std
clamp ([-2, 0.5]). With **all six removed** ("bare"), the Hopper N=8 collapse
disappeared. An isolation experiment then restored exactly one setting at a time
(Hopper, N=8, seed 0, 2000 iterations):

| configuration | final eval | episode length | peak → final |
|---|---|---|---|
| bare + orthogonal init | 3678 | 1000 | −0% |
| bare + line search | 3530 | 971 | −3% |
| bare + log-std clamp | 3503 | 1000 | −1% |
| bare | 3458 | 989 | −3% |
| bare + advantage normalization | 2513 | 685 | **−31%** (peak 3650) |
| bare + observation normalization | 2000 | 451 | −12% (never above 2268) |
| bare + reward normalization | 1833 | 470 | **−48%** (peak 3550) |
| all six (original configuration) | 1870 | 412 | **−51%** (peak 3854) |

- **Reward normalization** and **advantage normalization** each reproduce the
  late collapse on their own, with the same signature as the original
  configuration: a good peak followed by decay into short, falling episodes.
- **Observation normalization** hurts Hopper differently: the policy never reaches
  the stable gait.
- **Line search, orthogonal init and the log-std clamp** are harmless. Orthogonal
  init ended highest, but on one seed it is within noise of the other two.

Under the bare configuration the agent-count ordering matches the paper
(seed 0, final eval):

| | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| bare | 624 | 2621 | 2170 | 3458 |
| original configuration | 3632 | 3544 | 1426 | 2096 |
| paper | 1473 | 2384 | 2507 | 2719 |

A plausible mechanism, consistent with the measurements but not proven
separately: Algorithm 1 normalizes every step to move a fixed KL distance δ,
however small the gradient has become. Measured on the original configuration at
N=8, ‖y‖ fell from 0.34 to 0.086 over training while the closed-form step size
rose from 0.80 to 0.93 and KL held at 0.0081. So a converged policy keeps being
pushed a full trust region every iteration. Reward and advantage normalization
both rescale the learning signal to a fixed magnitude every iteration, which hides
the fact that the real signal has become small. More agents give a less noisy
direction, so these pushes accumulate consistently instead of cancelling out, and
the collapse grows with N.

## 4. Swimmer and Humanoid keep the original configuration

The original configuration already reproduces the paper's claim on Swimmer and
Humanoid: at every checkpoint, more agents learn faster (training return, mean
of 5 seeds).

| Swimmer | it 25 | 50 | 75 | 100 | 200 | final |
|---|---|---|---|---|---|---|
| N=1 | 24 | 32 | 45 | 57 | 89 | 124 |
| N=2 | 30 | 50 | 74 | 96 | 117 | 126 |
| N=4 | 39 | 65 | 101 | 116 | 123 | 127 |
| N=8 | 47 | 102 | 120 | 122 | 126 | 128 |

| Humanoid | it 500 | 750 | 1000 | 1500 | 3000 |
|---|---|---|---|---|---|
| N=1 | 435 | 477 | 502 | 558 | 981 |
| N=2 | 459 | 513 | 602 | 830 | 3318 |
| N=4 | 543 | 664 | 922 | 3135 | 5256 |
| N=8 | 746 | 1824 | 3836 | 5039 | 6477 |

The Hopper configuration of §5 was also tried on both. With all six settings
off ("bare"), Humanoid failed outright (676 against 6430 at iteration ~1500) and
Swimmer N=1 reached only 68. Bare plus observation normalization reached
higher *final* values on Swimmer (138–145 at N=4/8), but it lost the
ordering: N=8 rose no faster than N=4 and plateaued lower. Since the
agent-count ordering is the paper's claim, Swimmer and Humanoid stay on the
original configuration. (Those test runs were deleted on 2026-10-07; the
numbers here are the record.)

## 5. Final configuration

Table 4 values are unchanged in every environment. Hopper's fix removes
**three** settings, not two: reward and advantage normalization (each causes
the late collapse) and observation normalization (alone it never lets the
policy reach the stable gait, 2000 at iteration 2000). Line search and the
log-std clamp are also off but measured harmless; they are off only so the
configuration is exactly the one measured.

| setting | Hopper | Swimmer, Humanoid | in the paper? |
|---|---|---|---|
| reward normalization | off | on | not in Algorithm 1 |
| advantage normalization | off | on | not in Algorithm 1 |
| observation normalization | off | on | not in Algorithm 1 |
| line search | off | on | paper's Setup says it uses TRPO line search |
| log-std clamp [-2, 0.5] | off | on | not in the paper |
| orthogonal initialization | on | on | not specified |

Swimmer and Humanoid use the configuration the original 5-seed runs were made
with (`{swimmer,humanoid}-N{1,2,4,8}-s{0..4}`). Hopper uses the configuration
measured as `test-hopper-iso-orthoinit` (3678, no decay), run as
`final-hopper-N{1,2,4,8}-s{0..4}` by `final_sweep.sh`. Hopper turns off reward and
advantage normalization because each one causes its late collapse (§3). It
turns off observation normalization because it stopped Hopper from reaching the
stable gait. Line search and the clamp were harmless on Hopper; they are off
there to keep the configuration exactly as measured.

### Baselines (Figure 3) follow the same split

The comparison is only fair if all three algorithms share the same practical
settings for a given environment. Under the original settings both Hopper
baselines showed the same late decay as FedNPG-ADMM (5 seeds, N=8, training
return peak to end): standard-average FedNPG −10 to −19%, FedPPO −22 to −49%.
So on Hopper both baselines are re-run with Hopper's configuration
(`final-hopper-{fednpg-standard,fedppo}-N8-s{0..4}`, `./final_sweep.sh
hopper-baselines`). Swimmer and Humanoid baselines keep their existing runs,
which used the original configuration, the same as their FedNPG-ADMM runs.
Line search does not apply to FedPPO; the remaining switches act on the shared
environment wrapper (observation, reward normalization) and on the advantages
each algorithm consumes.

## 6. Hypotheses tested and refuted

Each was refuted by measurement. None needs re-running.

| hypothesis | evidence against |
|---|---|
| ADMM consensus quality | 5 ADMM rounds raised consensus cosine 0.05 → 0.76 and scored worse (1426 vs 1827); an exact pooled solve was within noise of one round |
| step-size overshoot | achieved KL 0.0054–0.0107 against δ=0.01 at every N; KL per step 0.008 at every N |
| architecture (Table 4 Tanh head) | Table 4 as written beat all three alternatives |
| ADMM dual variable λ blowing up | ‖λ‖/‖g‖ fell from 0.12 to 0.04 over training; bounded |
| Tanh saturation | N=2 saturates as much as N=8 (32% vs 33% pre-activations beyond ±2) and does not collapse |
| log-std clamp binding | bare Hopper with no clamp settles at std 0.142, essentially the clamp's 0.135; restoring the clamp alone is harmless (§3) |
| critic over-fitting (Adam steps ∝ N) | holding critic steps at the N=1 level reduced the collapse but did not stop it (−33% vs −51%) |
| pooled vs per-agent advantage normalization | pooled scored worse at every checkpoint |
| entropy bonus | matched N=1's action-std trajectory to 3 digits and still scored 1648 |
| fixed federation-wide sample budget (T/N per agent) | runs at N·T=2048 plateaued at 1300–1500 (partial runs, 40–79% complete) |
| stricter TRPO line-search acceptance (ratio 0.1) | worse: 1606 |
| smaller step multiplier η = 0.5 | worse: 1519 |

## 7. Caveats

- Every result in §3–4 is **one seed (0)**. The 5-seed final sweep is the
  evidence for the final results; the tables above explain the configuration choice.
- Within-run noise under the bare configuration is large: N=2 moved between 1218
  and 3065 across evaluations of a single run. Compare means and spread across
  seeds, not single checkpoints.
- The configuration differs between environments. That is a deviation from
  using one setup throughout, justified by §3–4.

## 8. Run ids

| what | run ids |
|---|---|
| original configuration, 5 seeds | `{env}-N{1,2,4,8}-s{0..4}` |
| bare Hopper agent sweep | `hopper-bare-N{1,2,4,8}` |
| isolation | `test-hopper-iso-{linesearch,obsnorm,rewnorm,advnorm,orthoinit,logstdclamp}` |
| cross-environment checks | deleted 2026-10-07; numbers recorded in §4 |
| refuted fixes | `hopper-fix-{critic,accept,eta}`, `diag-lam-N{1,2,8}`, `hopper-budget-N*` |
| **final runs** | Hopper: `final-hopper-N{1,2,4,8}-s{0..4}`; Swimmer, Humanoid: `{swimmer,humanoid}-N{1,2,4,8}-s{0..4}` |
| **final baselines** | Hopper: `final-hopper-{fednpg-standard,fedppo}-N8-s{0..4}`; Swimmer, Humanoid: `{swimmer,humanoid}-{fednpg-standard,fedppo}-N8-s{0..4}` |

## 9. Final five-seed results (2026-10-08)

Paper metric: mean of each seed's last 100 iterations of `mean_episode_return`,
then mean ± sample sd over 5 seeds (same as the paper's Table 2).

| Hopper-v4 | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| Hopper configuration (`final-hopper-*`) | 1941 ± 494 | 1636 ± 282 | 2536 ± 116 | 2656 ± 286 |
| original configuration (deleted runs) | 2497 ± 635 | 2140 ± 386 | 1658 ± 273 | 1605 ± 159 |
| paper | 1473 ± 383 | 2384 ± 371 | 2507 ± 230 | 2719 ± 173 |

N=4 and N=8 match the paper within one sd. Learning speed increases with N
early on (smoothed training return at iteration 100: 598 / 638 / 986 / 1092).
Not strictly monotone: N=2 finishes below N=1 and N=4 briefly leads N=8
around iteration 200, both within seed spread. The five seeds resolve two
groups, {1, 2} and {4, 8}, ~700 apart at the end.

Baselines at N=8 (Hopper under the Hopper configuration for all three):

| | Swimmer | Hopper | Humanoid |
|---|---|---|---|
| FedNPG-ADMM | 128.4 ± 0.8 | 2656 ± 286 | 6329 ± 334 |
| standard-average FedNPG | 127.1 ± 0.5 | 2494 ± 677 | 4699 ± 298 |
| FedPPO | 129.8 ± 0.7 | 1571 ± 588 | 5780 ± 158 |

Open: on Humanoid standard FedNPG (exact pooled direction) finishes well below
FedNPG-ADMM; cause not established. At equal cumulative uplink, FedPPO beats
FedNPG-ADMM on Humanoid and loses on Hopper.

The old Hopper baseline runs (`hopper-{fednpg-standard,fedppo}-N8-s*`) are
superseded by `final-hopper-*`.
