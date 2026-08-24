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
  6. server_update() aggregates into a new theta, y                   -- Algo 1 lines 11-13
  7. metrics get logged via common.results.ResultsWriter
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime

import torch
from tqdm import tqdm

from common.comms import make_channels_for_agents
from common.device import resolve_device
from common.envs import get_env_spec, make_envs_for_agents
from common.networks import GaussianMLPPolicy, ValueMLP, get_flat_params, num_params, set_flat_params
from common.results import ResultsWriter
from common.rollout import collect_rollout_for_agents, compute_episode_returns, compute_gae, fit_value_function

from .agent import AgentState, agent_update
from .hyperparams import DEFAULT_HYPERPARAMS
from .server import server_update


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FedNPG-ADMM (Algorithm 1)")
    parser.add_argument("--env", type=str, default="swimmer", choices=list(DEFAULT_HYPERPARAMS))
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument("--critic-epochs", type=int, default=5)
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

    device = resolve_device(args.device)
    hp = resolve_hyperparams(args)
    spec = get_env_spec(args.env)

    # Create an env for each agent
    envs = make_envs_for_agents(args.env, args.num_agents, base_seed=args.seed)

    # Create Actor (shared policy)
    policy = GaussianMLPPolicy.from_env(envs[0], hidden_sizes=spec.hidden_sizes).to(device)

    # Create Critic (using GAE) with optimizer
    value_fn = ValueMLP.from_env(envs[0], hidden_sizes=spec.hidden_sizes).to(device)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=hp["critic_lr"])

    # ADMM state: theta, y, and per-agent dual variables y_i, g_i
    d = num_params(policy)
    theta = get_flat_params(policy)  # already on `device` -- policy was moved there first
    y_global = torch.zeros(d, device=device)
    agent_states = [AgentState.initial(d, device=device) for _ in range(args.num_agents)]
    channels = make_channels_for_agents(args.num_agents)

    run_id = args.run_id or f"fednpg_admm_{args.env}_{args.num_agents}agents_{datetime.now():%Y%m%d_%H%M%S}"
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
            **hp,
        },
    )

    obs_list = None
    progress = tqdm(range(1, hp["iterations"] + 1), desc=run_id, unit="it", dynamic_ncols=True)
    for k in progress:
        set_flat_params(policy, theta)  # Algo 1 line 3 (theta half)

        # Diagnostic only, not part of Algorithm 1: log_std is part of theta
        # (it's an nn.Parameter of `policy`, so NPG steps move it same as
        # any other weight) -- tracking it exposes exploration/variance
        # collapse (std shrinking toward 0 with nothing to counteract it,
        # since Algorithm 1 has no entropy term) as a distinct failure mode
        # from the ones server_update already guards against.
        action_std = torch.exp(policy.log_std.detach())

        batches, obs_list = collect_rollout_for_agents(
            envs, policy, value_fn, hp["timesteps"], obs_list=obs_list
        )

        # GAE per agent. compute_gae resolves every bootstrap value
        # internally (including mid-batch episode boundaries) -- no
        # last_value bookkeeping needed here; see gae.py's docstring.
        for batch in batches:
            compute_gae(batch, gamma=hp["gamma"], lam=hp["gae_lambda"])

        # Critic: pooled across all agents' data. The paper does not
        # describe how the critic is trained federatedly at all -- pooling
        # is a choice made here, not a transcription of anything in
        # Algorithm 1 (which only covers the actor's update).
        all_obs = torch.cat([b.obs for b in batches])
        all_returns = torch.cat([b.returns for b in batches])
        critic_losses = fit_value_function(value_fn, all_obs, all_returns, value_optimizer, epochs=args.critic_epochs)

        # Algo 1 lines 5-9, per agent
        y_list, g_list = [], []
        for i, (batch, channel) in enumerate(zip(batches, channels)):
            y_i, g_i, agent_states[i] = agent_update(
                policy, batch, agent_states[i], y_prev=y_global, rho=hp["rho"], cg_iters=args.cg_iters
            )
            y_list.append(channel.send("y_i", y_i, iteration=k))
            g_list.append(channel.send("g_i", g_i, iteration=k))

        # Algo 1 lines 11-13
        step_skipped = False
        try:
            theta, y_global = server_update(theta, y_list, g_list, delta=hp["delta"], eta=hp["eta"])
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

        mean_rewards = torch.tensor([b.rewards.mean().item() for b in batches])
        # Episode returns -- comparable to the paper's Fig. 2 y-axis, unlike
        # mean_rewards above which is a per-TIMESTEP average and will look
        # nothing like the paper's numbers by construction (their plots are
        # per-episode totals). Can be empty if no agent completed a full
        # episode within this iteration's timestep budget -- a real
        # possibility, not an error, so this is reported as NaN rather than
        # silently omitted or defaulted to 0 (which would fake a bad score).
        all_episode_returns = [r for b in batches for r in compute_episode_returns(b)]
        mean_episode_return = (
            sum(all_episode_returns) / len(all_episode_returns) if all_episode_returns else float("nan")
        )
        comm_bytes_this_iter = sum(ch.meter.bytes_by_iteration().get(k, 0.0) for ch in channels)
        cumulative_comm_bytes = sum(ch.meter.total_bytes for ch in channels)

        writer.log_iteration(
            k,
            mean_reward=mean_rewards.mean().item(),
            reward_std=mean_rewards.std().item() if args.num_agents > 1 else 0.0,
            mean_episode_return=mean_episode_return,
            num_completed_episodes=len(all_episode_returns),
            critic_loss=critic_losses[-1],
            comm_bytes_this_iteration=comm_bytes_this_iter,
            cumulative_comm_bytes=cumulative_comm_bytes,
            step_skipped=int(step_skipped),
            mean_action_std=action_std.mean().item(),
            min_action_std=action_std.min().item(),
            max_action_std=action_std.max().item(),
        )

        progress.set_postfix(
            {
                "ep_ret": f"{mean_episode_return:.3f}" if all_episode_returns else "n/a"
            }
        )

    writer.finalize(final_mean_reward=mean_rewards.mean().item())
    for env in envs:
        env.close()
    return run_id


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()