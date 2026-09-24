"""
CLI entry point for FedNPG-ADMM training: fedrl-train-admm.

Orchestrates, per iteration k:
  1. broadcast theta to every agent (set_flat_params)                 -- Algo 1 line 3
  2. each agent collects a fixed-length rollout under the shared theta
  3. GAE + a critic-fitting step (NOT part of Algorithm 1 -- see the
     TRPO-vs-NPG discussion: the paper never describes how the critic is
     trained in a federated setting at all; this pools every agent's data
     into one shared critic update, which is a choice, not a paper fact)
  4. each agent runs agent_update() (Algo 1 lines 5-9)
  5. y_i, g_i go through a Channel (byte-accounted, quantization's future hook)
  6. aggregate into a new theta, y                                    -- Algo 1 lines 11-13,
     by default wrapped in a TRPO-style backtracking line search (NOT in Algorithm 1's
     pseudocode -- see server.py's module docstring for why); --no-line-search falls
     back to the closed-form step applied unconditionally
  7. metrics get logged via common.results.ResultsWriter
"""

from __future__ import annotations

import argparse
import random
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common.comms import make_channels_for_agents
from common.device import resolve_device
from common.envs import NormalizingVectorEnv, get_env_spec, make_env, make_vector_env_for_agents
from common.evaluation import evaluate_policy, save_checkpoint
from common.networks import GaussianMLPPolicy, ValueMLP, get_flat_params, num_params, set_flat_params
from common.results import ResultsWriter
from common.rollout import (
    collect_rollout_for_agents,
    compute_episode_returns,
    compute_gae,
    explained_variance,
    fit_value_function,
    normalize_advantages,
)

from .agent import AgentState, agent_update
from .hyperparams import DEFAULT_HYPERPARAMS
from .server import compute_candidate_step, line_search_step


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FedNPG-ADMM (Algorithm 1)")
    parser.add_argument("--env", type=str, default="swimmer", choices=list(DEFAULT_HYPERPARAMS))
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument(
        "--critic-epochs",
        type=int,
        default=10,
        help=(
            "Passes over a freshly-shuffled permutation of the pooled critic batch "
            "per training iteration (default: 10, matching stable-baselines3's PPO "
            "default -- the paper's own stated practical-settings basis). NOT from "
            "Table 4, which doesn't specify critic training at all. Units changed "
            "2026-09: this used to mean 'N full-batch gradient steps on the "
            "identical data' (diagnosed as causing overfitting -- near-zero "
            "critic_loss alongside a policy that kept oscillating -- once pushed "
            "well above 5); now each epoch is a genuine pass over shuffled "
            "minibatches of size --critic-batch-size, so old --critic-epochs "
            "values (e.g. 80, 100) are NOT directly comparable to the new ones."
        ),
    )
    parser.add_argument(
        "--stats-window-size",
        type=int,
        default=100,
        help=(
            "Number of most-recent EPISODES averaged into the logged "
            "`ep_rew_mean` column (default: 100, matching stable-baselines3's "
            "stats_window_size, whose rollout/ep_rew_mean is the trailing mean "
            "it actually reports). Distinct from mean_episode_return, which "
            "covers only the episodes finishing inside THIS iteration's "
            "timestep budget -- a count that collapses to 1-2 on Hopper once "
            "episodes grow long, making that column extremely noisy through no "
            "fault of training. Both are logged."
        ),
    )
    parser.add_argument(
        "--critic-batch-size",
        type=int,
        default=64,
        help="Minibatch size for critic fitting (default: 64, matching stable-baselines3's "
        "PPO default). Not from the paper, which doesn't specify this at all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device: 'cpu', 'mps', or 'cuda:N'. See --help output for what's actually available.",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Don't suppress warnings (Gymnasium deprecation notices, etc.) -- "
        "on by default they're silenced so they don't interleave with the progress bar.",
    )
    parser.add_argument(
        "--on-invalid-step",
        choices=["skip", "raise"],
        default="skip",
        help=(
            "What to do when (sum g_i)^T y <= 0 makes the trust-region step size "
            "undefined -- a real failure mode of Algorithm 1 as literally written "
            "(no line search, no fallback). 'skip' (default): keep theta unchanged "
            "this iteration, log a warning, continue -- a practical safeguard that "
            "is NOT part of the paper's Algorithm 1, needed to produce complete "
            "training runs at all. 'raise': fail immediately, matching Algorithm 1 "
            "literally -- useful for documenting how often this happens unmitigated."
        ),
    )

    parser.add_argument(
        "--advantage-normalization-scope",
        choices=["per-agent", "pooled"],
        default="per-agent",
        help=(
            "Whether each agent normalizes its OWN advantages (default, what "
            "this project has always done) or all agents share one mean/std "
            "computed over the pooled batch. Not a Table 4 quantity -- the "
            "paper never mentions advantage normalization -- so this is "
            "entirely an implementation choice.\n\n"
            "Why it may matter at num_agents > 1: per-agent scaling divides "
            "each g_i by its own batch's advantage std, so an agent whose "
            "rollout carried strong signal and one whose rollout was flat "
            "both contribute unit-scale gradients, and Algorithm 1 line 13's "
            "sum_i g_i weights them equally regardless of how informative "
            "they were. At num_agents=1 this is a single global rescale that "
            "the trust-region step size absorbs, which is exactly why it has "
            "been invisible until now. 'pooled' keeps the relative weighting "
            "between agents intact and matches stable-baselines3, which "
            "normalizes across the whole rollout buffer."
        ),
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Run deterministic evaluation every N iterations (0 = off). Uses the "
        "MEAN action with no exploration noise on a SEPARATE env, and reports raw "
        "un-normalized returns -- the quantity published benchmarks report. The "
        "always-logged ep_rew_mean is instead the stochastic on-policy training "
        "return, which on Hopper is systematically lower because the exploration "
        "noise is what makes it fall over.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Write policy/critic/obs-normalizer to <results>/<run_id>/checkpoint.pt "
        "at the end, plus best.pt whenever deterministic eval improves. Without this "
        "a finished run cannot be re-evaluated, replayed or rendered afterwards.",
    )

    # Everything below defaults from the paper's Table 4 (see hyperparams.py)
    # for --env, but can be overridden -- e.g. for a quick smoke test.
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=None,
        help="Initial log std (default -0.5, std ~0.61 -- tianshou's and Spinning Up's "
        "value; SB3/baselines/CleanRL use 0.0, std 1.0). Not in Table 4. Andrychowicz "
        "et al. 2020 (arXiv:2006.05990) find initial action std ~0.5 best on most "
        "MuJoCo envs but explicitly note HOPPER prefers higher values.",
    )
    parser.add_argument(
        "--log-std-min",
        type=float,
        default=None,
        help=(
            "Floor on the policy's log_std (default: GaussianMLPPolicy's own default, "
            "-2.0 -- std ~0.135). NOT from the paper: Algorithm 1 has no entropy term "
            "or bounds, and log_std is part of theta, so the NPG/ADMM step is free to "
            "drive it toward -inf with nothing to stop it (observed with num_agents=8, "
            "where ADMM averaging smooths out single-round noise into a slow, "
            "one-directional collapse) -- see docs/npg_admm_theory_and_implementation.md."
        ),
    )
    parser.add_argument(
        "--log-std-max",
        type=float,
        default=None,
        help=(
            "Ceiling on the policy's log_std (default: GaussianMLPPolicy's own default, "
            "0.5 -- std ~1.65). Same missing-bound problem as --log-std-min but in the "
            "other direction -- observed with num_agents=1 (no cross-agent averaging to "
            "cancel a noisy round): log_std for one action dim ran away upward, reward "
            "collapsed into noise and never recovered. See "
            "docs/npg_admm_theory_and_implementation.md."
        ),
    )
    parser.add_argument(
        "--no-obs-normalization",
        dest="normalize_obs",
        action="store_false",
        default=True,
        help=(
            "Disable running observation normalization (default: enabled). NOT from "
            "Algorithm 1 -- raw MuJoCo obs have wildly different per-dimension scales, "
            "which most real NPG/TRPO/PPO reproductions correct for (e.g. VecNormalize, "
            "tianshou's VectorEnvNormObs) even though the paper's pseudocode never "
            "mentions it. Pass this flag to run the paper-literal, unnormalized path."
        ),
    )
    parser.add_argument(
        "--no-reward-normalization",
        dest="normalize_reward",
        action="store_false",
        default=True,
        help=(
            "Disable running reward normalization (default: enabled). NOT from "
            "Algorithm 1 -- mirrors stable-baselines3's VecNormalize (the paper's own "
            "cited practical basis), which divides reward by a running std of the "
            "discounted return (no mean-centering, so sign/ordering is preserved) "
            "rather than the raw per-step reward. Added because envs with a heavy-"
            "tailed return distribution (e.g. Hopper -- mostly hops the full episode, "
            "but sometimes falls over early) give the critic a much harder-to-fit "
            "raw-scale regression target than Swimmer's narrower one ever was. "
            "Reported metrics (mean_reward, mean_episode_return) always use the TRUE "
            "unnormalized reward regardless of this flag -- see RolloutBatch.raw_rewards."
        ),
    )
    parser.add_argument(
        "--no-ortho-init",
        dest="ortho_init",
        action="store_false",
        default=True,
        help=(
            "Disable orthogonal weight initialization (default: enabled). NOT from "
            "Table 4, which specifies layer sizes/activations only -- this matches "
            "stable-baselines3's ActorCriticPolicy default (ortho_init=True: gain "
            "sqrt(2) hidden, 0.01 on the policy's action head, 1.0 on the critic's "
            "value head, biases zeroed), the paper's own cited practical basis. Pass "
            "this flag to fall back to PyTorch's default Kaiming-uniform init."
        ),
    )
    parser.add_argument(
        "--no-advantage-normalization",
        dest="advantage_normalization",
        action="store_false",
        default=True,
        help=(
            "Disable per-agent advantage normalization (default: enabled, mean 0 std 1 "
            "over each agent's own local batch). NOT from Algorithm 1, but standard in "
            "practical NPG/TRPO implementations (e.g. tianshou defaults this on). Pass "
            "this flag to run the paper-literal, unnormalized path."
        ),
    )
    parser.add_argument(
        "--no-line-search",
        dest="line_search",
        action="store_false",
        default=True,
        help=(
            "Disable the TRPO-style backtracking line search around the server's "
            "closed-form theta step (default: enabled). NOT in Algorithm 1's "
            "pseudocode, but the paper's own Setup section states 'We follow the "
            "practical settings in TRPO with line search' -- i.e. the paper's "
            "reported curves were themselves produced with this safeguard. Pass this "
            "flag to apply Algorithm 1 line 13's closed-form step unconditionally "
            "instead (the paper-literal, pseudocode-only path)."
        ),
    )
    parser.add_argument(
        "--line-search-max-backtracks",
        type=int,
        default=10,
        help="Max backtracking attempts before giving up and skipping theta's update "
        "for this iteration (default: 10, same conventional value TRPO implementations use).",
    )
    parser.add_argument(
        "--line-search-backtrack-coeff",
        type=float,
        default=0.8,
        help="Shrink factor applied to the candidate step size after each failed "
        "backtracking attempt (default: 0.8, same conventional value TRPO implementations use).",
    )
    return parser


def resolve_hyperparams(args: argparse.Namespace):
    defaults = DEFAULT_HYPERPARAMS[args.env]
    return {
        "iterations": args.iterations if args.iterations is not None else defaults.iterations,
        "timesteps": args.timesteps if args.timesteps is not None else defaults.timesteps,
        "rho": args.rho if args.rho is not None else defaults.rho,
        "delta": args.delta if args.delta is not None else defaults.delta,
        "gamma": args.gamma if args.gamma is not None else defaults.gamma,
        "gae_lambda": args.gae_lambda if args.gae_lambda is not None else defaults.gae_lambda,
        "critic_lr": args.critic_lr if args.critic_lr is not None else defaults.critic_lr,
        "eta": args.eta if args.eta is not None else defaults.eta,
    }


def train(args: argparse.Namespace) -> str:
    """Returns the run_id, so callers (and tests) can locate the results directory."""
    if not args.show_warnings:
        warnings.filterwarnings("ignore")

    # `args.seed` was ONLY ever reaching the environment's own RNG (via
    # make_env's env.reset(seed=...)/env.action_space.seed(...)) -- torch's
    # global RNG (policy/critic weight init via nn.Linear's default init,
    # and every dist.sample()/rsample() call during rollout collection)
    # was never seeded at all, so every run had a genuinely different,
    # unreproducible policy initialization and action-sampling stream
    # regardless of --seed. Only the env's dynamics (episode timing, reset
    # noise) were actually deterministic. Matches npg_reference_project's
    # own set_seed() (random.seed/np.random.seed/torch.manual_seed) so a
    # given --seed now means the same thing in both projects.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    hp = resolve_hyperparams(args)
    spec = get_env_spec(args.env)

    # A single throwaway env, just to read observation/action space specs
    # for from_env() below — the actual per-agent envs live inside
    # `vector_env`'s subprocesses (see make_vector_env_for_agents), and a
    # batched vector env's own .observation_space/.action_space are the
    # BATCHED spaces (shape (num_agents, ...)), not the single-agent ones
    # from_env() expects.
    spec_env = make_env(args.env, seed=args.seed)

    # Create Actor (shared policy)
    policy_kwargs = {"ortho_init": args.ortho_init}
    if args.log_std_init is not None:
        policy_kwargs["log_std_init"] = args.log_std_init
    if args.log_std_min is not None:
        policy_kwargs["log_std_min"] = args.log_std_min
    if args.log_std_max is not None:
        policy_kwargs["log_std_max"] = args.log_std_max
    policy = GaussianMLPPolicy.from_env(spec_env, hidden_sizes=spec.hidden_sizes, **policy_kwargs).to(device)

    # Create Critic (using GAE) with optimizer
    value_fn = ValueMLP.from_env(spec_env, hidden_sizes=spec.hidden_sizes, ortho_init=args.ortho_init).to(device)
    spec_env.close()

    # One env per agent, each stepped in its own OS subprocess so all
    # num_agents agents' MuJoCo physics run in parallel across CPU cores
    # instead of one agent at a time in this process — see
    # make_vector_env_for_agents and collect_rollout_for_agents for the
    # autoreset/bootstrap-correctness details this depends on.
    vector_env = make_vector_env_for_agents(args.env, args.num_agents, base_seed=args.seed)
    if args.normalize_obs or args.normalize_reward:
        vector_env = NormalizingVectorEnv(
            vector_env,
            gamma=hp["gamma"],
            normalize_obs=args.normalize_obs,
            normalize_reward=args.normalize_reward,
        )
    # None when obs normalization is off -- evaluate_policy then feeds raw
    # observations, which is correct for a policy trained on raw ones.
    obs_normalizer = getattr(vector_env, "obs_rms", None) if args.normalize_obs else None

    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=hp["critic_lr"])

    # Separate env for deterministic evaluation -- never the training env:
    # sharing it would corrupt the running normalizer state mid-iteration,
    # and its reward normalization would make the number incomparable to
    # published benchmarks. Offset seed so eval episodes aren't the same
    # trajectories the agents just trained on.
    eval_env = make_env(args.env, seed=args.seed + 10_000) if args.eval_interval > 0 else None
    best_eval_return = float("-inf")


    # ADMM state: theta, y, and per-agent dual variables y_i, g_i
    d = num_params(policy)
    theta = get_flat_params(policy)  # already on `device` -- policy was moved there first
    y_global = torch.zeros(d, device=device)
    agent_states = [AgentState.initial(d, device=device) for _ in range(args.num_agents)]
    channels = make_channels_for_agents(args.num_agents)
    # Per-agent reward accumulated so far for whatever episode is still in
    # progress at the end of that agent's most recent batch -- threaded
    # into compute_episode_returns exactly like `obs` is threaded into
    # collect_rollout, and for the same reason: without it, an episode
    # spanning an iteration boundary gets its reported return silently
    # truncated to only the fraction inside whichever iteration's batch it
    # happened to end in. See compute_episode_returns's docstring.
    episode_return_carry = [0.0] * args.num_agents
    # Trailing window of the most recent completed episodes' returns,
    # pooled across agents -- the basis for the `ep_rew_mean` column.
    # See common.results.smoothing for why this is the metric worth
    # reading, and stable-baselines3's stats_window_size for precedent.
    episode_return_window: deque[float] = deque(maxlen=args.stats_window_size)

    run_id = args.run_id or f"fednpg_admm_{args.env}_{args.num_agents}agents_{datetime.now():%Y%m%d_%H%M%S}"
    # after run_id: an auto-generated id must land in its own directory,
    # not the results root.
    checkpoint_dir = Path(args.results_path) / run_id
    writer = ResultsWriter(
        args.results_path,
        run_id=run_id,
        metadata={
            "algorithm": "fednpg_admm",
            "env": args.env,
            "num_agents": args.num_agents,
            "num_params": d,
            "seed": args.seed,
            "device": str(device),
            "ortho_init": args.ortho_init,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            # The EFFECTIVE clamp, read off the policy rather than off args
            # (which are None when not overridden) -- these bounds change
            # training materially and were previously unrecorded, so a run's
            # results couldn't be attributed to them after the fact.
            "log_std_init": float(policy.log_std.detach().mean()),
            "log_std_min": policy.log_std_min,
            "log_std_max": policy.log_std_max,
            "normalize_obs": args.normalize_obs,
            "normalize_reward": args.normalize_reward,
            "advantage_normalization": args.advantage_normalization,
            "advantage_normalization_scope": args.advantage_normalization_scope,
            "cg_iters": args.cg_iters,
            "line_search": args.line_search,
            "line_search_max_backtracks": args.line_search_max_backtracks,
            "line_search_backtrack_coeff": args.line_search_backtrack_coeff,
            "critic_epochs": args.critic_epochs,
            "critic_batch_size": args.critic_batch_size,
            "stats_window_size": args.stats_window_size,
            **hp,
        },
    )

    obs = None
    progress = tqdm(range(1, hp["iterations"] + 1), desc=run_id, unit="it", dynamic_ncols=True)
    for k in progress:
        set_flat_params(policy, theta)  # Algo 1 line 3 (theta half)

        # Diagnostic only, not part of Algorithm 1: log_std is part of theta
        # (it's an nn.Parameter of `policy`, so NPG steps move it same as
        # any other weight) -- tracking it exposes exploration/variance
        # collapse (std shrinking toward 0 with nothing to counteract it,
        # since Algorithm 1 has no entropy term) as a distinct failure mode
        # from the ones server_update already guards against. Uses
        # policy.std() (not raw log_std) so this reflects the EFFECTIVE,
        # floored std actually used for sampling -- see policy.py's
        # log_std_min docstring.
        with torch.no_grad():
            action_std = policy.std()

        batches, obs = collect_rollout_for_agents(
            vector_env, policy, value_fn, hp["timesteps"], obs=obs
        )

        # GAE per agent. compute_gae resolves every bootstrap value
        # internally (including mid-batch episode boundaries) -- no
        # last_value bookkeeping needed here; see gae.py's docstring.
        for batch in batches:
            compute_gae(batch, gamma=hp["gamma"], lam=hp["gae_lambda"])
        if args.advantage_normalization:
            if args.advantage_normalization_scope == "pooled":
                # One mean/std over every agent's advantages at once, so the
                # RELATIVE magnitude between agents survives into sum_i g_i.
                # See --advantage-normalization-scope.
                pooled = torch.cat([b.advantages for b in batches])
                mean, std = pooled.mean(), pooled.std()
                for b in batches:
                    b.advantages = (b.advantages - mean) / (std + 1e-8)
            else:
                for batch in batches:
                    normalize_advantages(batch)

        # Critic: pooled across all agents' data. The paper does not
        # describe how the critic is trained federatedly at all -- pooling
        # is a choice made here, not a transcription of anything in
        # Algorithm 1 (which only covers the actor's update).
        all_obs = torch.cat([b.obs for b in batches])
        all_returns = torch.cat([b.returns for b in batches])
        # Explained variance BEFORE this iteration's fit, against the
        # PRE-update values (batch.values, frozen at collection time) --
        # the standard convention: how good was the value function GAE
        # actually used this round, not how well the critic just memorized
        # what it was fit to.
        all_values = torch.cat([b.values for b in batches])
        critic_ev = explained_variance(all_returns, all_values)
        critic_losses = fit_value_function(
            value_fn, all_obs, all_returns, value_optimizer,
            epochs=args.critic_epochs, batch_size=args.critic_batch_size,
        )

        # Algo 1 lines 5-9, per agent.
        y_list, g_list = [], []
        for i, (batch, channel) in enumerate(zip(batches, channels)):
            y_i, g_i, agent_states[i] = agent_update(
                policy, batch, agent_states[i], y_prev=y_global,
                rho=hp["rho"], cg_iters=args.cg_iters,
            )
            y_list.append(channel.send("y_i", y_i, iteration=k))
            g_list.append(channel.send("g_i", g_i, iteration=k))

        # Consensus health, free to compute (no Fisher products): if the
        # agents' y_i are mutually near-orthogonal, their mean is short and
        # poorly aligned with the direction the pooled data supports, and
        # more agents actively hurt. This is the metric that diagnosed the
        # num_agents=8 shortfall, so it's logged permanently rather than
        # living in a throwaway script.
        consensus_cos, consensus_spread = float("nan"), float("nan")
        if args.num_agents > 1:
            with torch.no_grad():
                ys = torch.stack([t.detach() for t in y_list])
                unit = ys / ys.norm(dim=1, keepdim=True).clamp_min(1e-12)
                off_diag = ~torch.eye(args.num_agents, dtype=torch.bool, device=ys.device)
                consensus_cos = (unit @ unit.T)[off_diag].mean().item()
                y_mean = ys.mean(dim=0)
                consensus_spread = ((ys - y_mean).norm(dim=1).mean() / y_mean.norm().clamp_min(1e-12)).item()

        # Algo 1 lines 11-13, optionally wrapped in a TRPO-style
        # backtracking line search (see server.py's module docstring for
        # why -- the paper's own Setup section says it followed TRPO's
        # practical line-search settings, even though Algorithm 1's
        # pseudocode applies the closed-form step unconditionally).
        step_skipped = False
        line_search_backtracks = 0
        # The KL the accepted step actually moved, and the step size that
        # produced it. line_search_step already computes the former and it
        # was being discarded, which left the ADMM runs with no way to
        # check the thing Algorithm 1's closed-form step size is supposed
        # to control -- unlike npg_trpo, which logs `kl` and `step_size`.
        # NaN (not 0.0) when no step was taken: "no step" and "a step that
        # moved zero KL" are different outcomes and must not be averaged
        # together by a downstream reader.
        achieved_kl = float("nan")
        closed_form_step = float("nan")
        applied_step = float("nan")
        try:
            y_global, step_size = compute_candidate_step(
                y_list, g_list, delta=hp["delta"], eta=hp["eta"]
            )
            closed_form_step = step_size.item()
        except ValueError as e:
            if args.on_invalid_step == "raise":
                raise
            step_skipped = True
            # theta is left unchanged; y (the consensus variable) still
            # advances, since it's just an average and always well-defined
            # -- only the theta STEP SIZE was undefined, not y itself. This
            # keeps next round's dual-variable update (line 6) consistent
            # rather than freezing the whole ADMM state.
            y_global = torch.stack(y_list).mean(dim=0)
            tqdm.write(f"[{run_id}] iter {k}: step skipped ({e})")
        else:
            if args.line_search:
                # Pooled across all agents' data, same choice already made
                # for critic fitting above -- the paper doesn't specify how
                # a federated line search should evaluate KL/improvement.
                all_actions = torch.cat([b.actions for b in batches])
                all_advantages = torch.cat([b.advantages for b in batches])
                theta, line_search_backtracks, achieved_kl = line_search_step(
                    policy,
                    theta,
                    y_global,
                    step_size,
                    all_obs,
                    all_actions,
                    all_advantages,
                    delta=hp["delta"],
                    max_backtracks=args.line_search_max_backtracks,
                    backtrack_coeff=args.line_search_backtrack_coeff,
                )
                if line_search_backtracks < args.line_search_max_backtracks:
                    applied_step = closed_form_step * (
                        args.line_search_backtrack_coeff**line_search_backtracks
                    )
                if line_search_backtracks >= args.line_search_max_backtracks:
                    tqdm.write(
                        f"[{run_id}] iter {k}: line search rejected all "
                        f"{args.line_search_max_backtracks} candidates, theta unchanged"
                    )
            else:
                theta = theta + step_size * y_global
                applied_step = closed_form_step

        # raw_rewards, not rewards -- the latter is reward-NORMALIZED when
        # --no-reward-normalization isn't passed, and this is a human-facing
        # reported metric, not something GAE/the critic consume.
        mean_rewards = torch.tensor([b.raw_rewards.mean().item() for b in batches])
        # Episode returns -- comparable to the paper's Fig. 2 y-axis, unlike
        # mean_rewards above which is a per-TIMESTEP average and will look
        # nothing like the paper's numbers by construction (their plots are
        # per-episode totals). Can be empty if no agent completed a full
        # episode within this iteration's timestep budget -- a real
        # possibility, not an error, so this is reported as NaN rather than
        # silently omitted or defaulted to 0 (which would fake a bad score).
        all_episode_returns: list[float] = []
        for i, b in enumerate(batches):
            ep_returns, episode_return_carry[i] = compute_episode_returns(b, carry_over=episode_return_carry[i])
            all_episode_returns.extend(ep_returns)
        mean_episode_return = (
            sum(all_episode_returns) / len(all_episode_returns) if all_episode_returns else float("nan")
        )
        episode_return_window.extend(all_episode_returns)
        ep_rew_mean = (
            sum(episode_return_window) / len(episode_return_window)
            if episode_return_window
            else float("nan")
        )
        comm_bytes_this_iter = sum(ch.meter.bytes_by_iteration().get(k, 0.0) for ch in channels)
        cumulative_comm_bytes = sum(ch.meter.total_bytes for ch in channels)

        # Deterministic evaluation. The keys are logged on EVERY iteration
        # (NaN when this isn't an eval step) because log_iteration fixes the
        # CSV columns on its first call and rejects a changed key set.
        eval_metrics = {
            "eval_ep_rew_mean": float("nan"),
            "eval_ep_rew_std": float("nan"),
            "eval_ep_len_mean": float("nan"),
        }
        if eval_env is not None and (k % args.eval_interval == 0 or k == hp["iterations"]):
            set_flat_params(policy, theta)
            result = evaluate_policy(
                policy, eval_env, num_episodes=args.eval_episodes,
                obs_rms=obs_normalizer, device=device,
            )
            eval_metrics = result.as_metrics
            progress.set_postfix({"eval": f"{result.mean_return:.0f}"})
            if args.save_checkpoint and result.mean_return > best_eval_return:
                best_eval_return = result.mean_return
                save_checkpoint(
                    checkpoint_dir / "best.pt", policy, value_fn, obs_normalizer,
                    extra={"iteration": k, "eval_return": result.mean_return},
                )

        writer.log_iteration(
            k,
            mean_reward=mean_rewards.mean().item(),
            reward_std=mean_rewards.std().item() if args.num_agents > 1 else 0.0,
            mean_episode_return=mean_episode_return,
            ep_rew_mean=ep_rew_mean,
            num_completed_episodes=len(all_episode_returns),
            critic_loss=critic_losses[-1],
            critic_explained_variance=critic_ev,
            comm_bytes_this_iteration=comm_bytes_this_iter,
            cumulative_comm_bytes=cumulative_comm_bytes,
            step_skipped=int(step_skipped),
            line_search_backtracks=line_search_backtracks,
            kl=achieved_kl,
            step_size=applied_step,
            closed_form_step_size=closed_form_step,
            consensus_cos=consensus_cos,
            consensus_spread=consensus_spread,
            **eval_metrics,
            mean_action_std=action_std.mean().item(),
            min_action_std=action_std.min().item(),
            max_action_std=action_std.max().item(),
        )

        progress.set_postfix(
            {
                "ep_ret": f"{mean_episode_return:.3f}" if all_episode_returns else "n/a"
            }
        )

    if args.save_checkpoint:
        save_checkpoint(
            checkpoint_dir / "checkpoint.pt", policy, value_fn, obs_normalizer,
            extra={"iteration": hp["iterations"], "ep_rew_mean": ep_rew_mean},
        )
    writer.finalize(final_mean_reward=mean_rewards.mean().item())
    vector_env.close()
    if eval_env is not None:
        eval_env.close()
    return run_id


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()