"""
CLI entry point for the baseline algorithms: fedrl-train-baseline.

Currently implements `--algo fednpg-standard` (paper eq. 12, the O(d^2)
comparison); `fedppo` is still a stub.

This loop deliberately mirrors fednpg_admm/train.py step for step -- same
environments, same normalization, same GAE, same pooled critic, same
TRPO-style line search, same logged columns. That duplication is the point:
the two arms must differ ONLY in how the server turns per-agent gradients
into a step direction, or Figure 3's comparison measures the surrounding
machinery instead of the algorithms. Sibling packages never import each
other (see CLAUDE.md), and everything genuinely shareable already lives in
`common` -- including common.optim, which both arms use for CG and the
Fisher-vector product.
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
from common.networks.utils import get_flat_grad
from common.results import ResultsWriter
from common.rollout import (
    collect_rollout_for_agents,
    compute_episode_returns,
    compute_gae,
    explained_variance,
    fit_value_function,
    normalize_advantages,
)

from common.hyperparams import DEFAULT_HYPERPARAMS
from common.optim import line_search_step

from .fedppo import local_ppo_update
from .fednpg_standard import standard_average_step


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a FedRL baseline for comparison against FedNPG-ADMM")
    p.add_argument("--algo", choices=["fednpg-standard", "fedppo"], default="fednpg-standard")
    p.add_argument("--env", type=str, default="hopper", choices=list(DEFAULT_HYPERPARAMS))
    p.add_argument("--num-agents", type=int, default=8,
                   help="Default 8: the paper fixes N=8 for the Figure 3 comparison.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-path", type=str, default="results")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ppo-clip", type=float, default=0.2,
                   help="PPO clipping parameter (default 0.2, the value the paper states "
                        "for its Figure 3 FedPPO comparison). --algo fedppo only.")
    p.add_argument("--ppo-epochs", type=int, default=10,
                   help="Local PPO epochs per agent per round before parameter averaging "
                        "(--algo fedppo only). The paper does not specify this; 10 matches "
                        "stable-baselines3's PPO default, which its Setup section cites.")
    p.add_argument("--ppo-lr", type=float, default=3e-4,
                   help="Local Adam learning rate for the FedPPO actor (default 3e-4).")
    p.add_argument("--ppo-batch-size", type=int, default=64)
    p.add_argument("--cg-iters", type=int, default=50,
                   help="CG iterations for the pooled solve (default 50, higher than the ADMM "
                        "path's per-agent 10: this solve is the reference, not an approximation).")
    p.add_argument("--critic-epochs", type=int, default=10)
    p.add_argument("--critic-batch-size", type=int, default=64)
    p.add_argument("--stats-window-size", type=int, default=100)
    p.add_argument("--line-search-max-backtracks", type=int, default=10)
    p.add_argument("--line-search-backtrack-coeff", type=float, default=0.8)
    p.add_argument("--no-line-search", dest="line_search", action="store_false", default=True)
    p.add_argument("--no-obs-normalization", dest="normalize_obs", action="store_false", default=True)
    p.add_argument("--no-reward-normalization", dest="normalize_reward", action="store_false", default=True)
    p.add_argument("--no-advantage-normalization", dest="advantage_normalization",
                   action="store_false", default=True)
    p.add_argument("--no-ortho-init", dest="ortho_init", action="store_false", default=True)
    p.add_argument("--eval-interval", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--on-invalid-step", choices=["skip", "raise"], default="skip")
    p.add_argument("--show-warnings", action="store_true")
    # Table 4 overrides
    for flag in ("--iterations", "--timesteps"):
        p.add_argument(flag, type=int, default=None)
    for flag in ("--rho", "--delta", "--gamma", "--gae-lambda", "--critic-lr"):
        p.add_argument(flag, type=float, default=None)
    return p


def resolve_hyperparams(args) -> dict:
    hp = DEFAULT_HYPERPARAMS[args.env]
    out = {k: getattr(hp, k) for k in ("rho", "delta", "gamma", "gae_lambda",
                                       "timesteps", "iterations", "critic_lr")}
    for k in out:
        v = getattr(args, k, None)
        if v is not None:
            out[k] = v
    return out


def train(args: argparse.Namespace) -> str:
    if not args.show_warnings:
        warnings.filterwarnings("ignore")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    hp = resolve_hyperparams(args)
    spec = get_env_spec(args.env)

    spec_env = make_env(args.env, seed=args.seed)
    policy = GaussianMLPPolicy.from_env(
        spec_env, hidden_sizes=spec.hidden_sizes, ortho_init=args.ortho_init).to(device)
    value_fn = ValueMLP.from_env(
        spec_env, hidden_sizes=spec.hidden_sizes, ortho_init=args.ortho_init).to(device)
    spec_env.close()

    vector_env = make_vector_env_for_agents(args.env, args.num_agents, base_seed=args.seed)
    if args.normalize_obs or args.normalize_reward:
        vector_env = NormalizingVectorEnv(
            vector_env, gamma=hp["gamma"],
            normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    obs_normalizer = getattr(vector_env, "obs_rms", None) if args.normalize_obs else None
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=hp["critic_lr"])
    eval_env = make_env(args.env, seed=args.seed + 10_000) if args.eval_interval > 0 else None
    best_eval_return = float("-inf")

    d = num_params(policy)
    theta = get_flat_params(policy)
    channels = make_channels_for_agents(args.num_agents)
    episode_return_carry = [0.0] * args.num_agents
    episode_return_window: deque[float] = deque(maxlen=args.stats_window_size)

    run_id = args.run_id or f"{args.algo}_{args.env}_{args.num_agents}agents_{datetime.now():%Y%m%d_%H%M%S}"
    checkpoint_dir = Path(args.results_path) / run_id
    writer = ResultsWriter(
        args.results_path, run_id=run_id,
        metadata={
            "algorithm": "fednpg_standard" if args.algo == "fednpg-standard" else "fedppo", "env": args.env, "num_agents": args.num_agents,
            "num_params": d, "seed": args.seed, "device": str(device),
            "ortho_init": args.ortho_init, "log_std_init": float(policy.log_std.detach().mean()),
            "log_std_min": policy.log_std_min, "log_std_max": policy.log_std_max,
            "normalize_obs": args.normalize_obs, "normalize_reward": args.normalize_reward,
            "advantage_normalization": args.advantage_normalization,
            "cg_iters": args.cg_iters, "line_search": args.line_search,
            **({"ppo_clip": args.ppo_clip, "ppo_epochs": args.ppo_epochs,
                "ppo_lr": args.ppo_lr, "ppo_batch_size": args.ppo_batch_size,
                "fedppo_variant": "fedavg (local epochs + parameter averaging)"}
               if args.algo == "fedppo" else {}),
            "critic_epochs": args.critic_epochs, "critic_batch_size": args.critic_batch_size,
            "stats_window_size": args.stats_window_size,
            "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
            **hp,
        })

    obs = None
    progress = tqdm(range(1, hp["iterations"] + 1), desc=run_id, unit="it", dynamic_ncols=True)
    for k in progress:
        set_flat_params(policy, theta)
        with torch.no_grad():
            action_std = policy.std()

        batches, obs = collect_rollout_for_agents(
            vector_env, policy, value_fn, hp["timesteps"], obs=obs)
        for b in batches:
            compute_gae(b, gamma=hp["gamma"], lam=hp["gae_lambda"])
            if args.advantage_normalization:
                normalize_advantages(b)

        all_obs = torch.cat([b.obs for b in batches])
        all_returns = torch.cat([b.returns for b in batches])
        critic_ev = explained_variance(all_returns, torch.cat([b.values for b in batches]))
        critic_losses = fit_value_function(
            value_fn, all_obs, all_returns, value_optimizer,
            epochs=args.critic_epochs, batch_size=args.critic_batch_size)

        step_skipped = False
        backtracks = 0
        achieved_kl = closed_form_step = applied_step = float("nan")
        ppo_clip_fraction = float("nan")

        if args.algo == "fedppo":
            # Each agent runs local PPO epochs from the shared theta, then
            # uplinks its parameter delta (O(d)); the server averages them.
            # See fedppo.py for why this FedAvg reading was chosen over a
            # one-gradient-step FedSGD one.
            deltas, kls, clips = [], [], []
            for b, channel in zip(batches, channels):
                theta_i, st = local_ppo_update(
                    policy, theta, b.obs, b.actions, b.advantages.detach(),
                    lr=args.ppo_lr, clip=args.ppo_clip,
                    epochs=args.ppo_epochs, batch_size=args.ppo_batch_size)
                deltas.append(channel.send("delta_theta", theta_i - theta, iteration=k))
                kls.append(st.approx_kl)
                clips.append(st.clip_fraction)
            theta = theta + torch.stack(deltas).mean(dim=0)
            achieved_kl = float(sum(kls) / len(kls))
            ppo_clip_fraction = float(sum(clips) / len(clips))
            applied_step = 1.0  # averaging applies the full averaged delta
        else:
            # Each agent's score-function gradient g_i, then the O(d^2)
            # uplink: g_i is a real tensor through the Channel, H_i is
            # charged at its true d*d size without being built (see
            # fednpg_standard's module docstring).
            g_list = []
            for b, channel in zip(batches, channels):
                lp = policy.log_prob(b.obs, b.actions)
                g_i = get_flat_grad((lp * b.advantages.detach()).mean(), policy, retain_graph=True)
                g_list.append(channel.send("g_i", g_i, iteration=k))
                channel.record_unmaterialized_send("H_i", num_elements=d * d, iteration=k)

            try:
                direction, step_size = standard_average_step(
                    policy, all_obs, torch.stack(g_list).sum(dim=0),
                    num_agents=args.num_agents, delta=hp["delta"],
                    damping=hp["rho"], cg_iters=args.cg_iters)
                closed_form_step = step_size.item()
            except ValueError as e:
                if args.on_invalid_step == "raise":
                    raise
                step_skipped = True
                tqdm.write(f"[{run_id}] iter {k}: step skipped ({e})")
            else:
                if args.line_search:
                    theta, backtracks, achieved_kl = line_search_step(
                        policy, theta, direction, step_size, all_obs,
                        torch.cat([b.actions for b in batches]),
                        torch.cat([b.advantages for b in batches]),
                        delta=hp["delta"], max_backtracks=args.line_search_max_backtracks,
                        backtrack_coeff=args.line_search_backtrack_coeff)
                    if backtracks < args.line_search_max_backtracks:
                        applied_step = closed_form_step * (
                            args.line_search_backtrack_coeff**backtracks)
                else:
                    theta = theta + step_size * direction
                    applied_step = closed_form_step

        mean_rewards = torch.tensor([b.raw_rewards.mean().item() for b in batches])
        all_eps: list[float] = []
        for i, b in enumerate(batches):
            eps, episode_return_carry[i] = compute_episode_returns(
                b, carry_over=episode_return_carry[i])
            all_eps.extend(eps)
        mean_episode_return = sum(all_eps) / len(all_eps) if all_eps else float("nan")
        episode_return_window.extend(all_eps)
        ep_rew_mean = (sum(episode_return_window) / len(episode_return_window)
                       if episode_return_window else float("nan"))

        eval_metrics = {"eval_ep_rew_mean": float("nan"), "eval_ep_rew_std": float("nan"),
                        "eval_ep_len_mean": float("nan")}
        if eval_env is not None and (k % args.eval_interval == 0 or k == hp["iterations"]):
            set_flat_params(policy, theta)
            res = evaluate_policy(policy, eval_env, num_episodes=args.eval_episodes,
                                  obs_rms=obs_normalizer, device=device)
            eval_metrics = res.as_metrics
            if args.save_checkpoint and res.mean_return > best_eval_return:
                best_eval_return = res.mean_return
                save_checkpoint(checkpoint_dir / "best.pt", policy, value_fn, obs_normalizer,
                                extra={"iteration": k, "eval_return": res.mean_return})

        writer.log_iteration(
            k,
            mean_reward=mean_rewards.mean().item(),
            reward_std=mean_rewards.std().item() if args.num_agents > 1 else 0.0,
            mean_episode_return=mean_episode_return,
            ep_rew_mean=ep_rew_mean,
            num_completed_episodes=len(all_eps),
            critic_loss=critic_losses[-1],
            critic_explained_variance=critic_ev,
            comm_bytes_this_iteration=sum(c.meter.bytes_by_iteration().get(k, 0.0) for c in channels),
            cumulative_comm_bytes=sum(c.meter.total_bytes for c in channels),
            step_skipped=int(step_skipped),
            line_search_backtracks=backtracks,
            kl=achieved_kl,
            step_size=applied_step,
            closed_form_step_size=closed_form_step,
            ppo_clip_fraction=ppo_clip_fraction,
            mean_action_std=action_std.mean().item(),
            min_action_std=action_std.min().item(),
            max_action_std=action_std.max().item(),
            **eval_metrics,
        )
        progress.set_postfix({"ep_ret": f"{mean_episode_return:.1f}" if all_eps else "n/a"})

    if args.save_checkpoint:
        save_checkpoint(checkpoint_dir / "checkpoint.pt", policy, value_fn, obs_normalizer,
                        extra={"iteration": hp["iterations"]})
    writer.finalize(final_mean_reward=mean_rewards.mean().item())
    vector_env.close()
    if eval_env is not None:
        eval_env.close()
    return run_id


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
