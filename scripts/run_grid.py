"""
Grid of FedNPG-ADMM runs: seeds x num_agents x quantization bit-widths, for
one environment, run sequentially in a single process.

Run ids (also the results/ directory names):
    <env>-N<agents>-s<seed>             unquantized (always used for N=1)
    <env>-N<agents>-s<seed>-q<bits>     --scheme stochastic
    <env>-N<agents>-s<seed>-fp<bits>    --scheme float

With a single agent there is no agent->server uplink worth quantizing, so
N=1 gets exactly one unquantized run per seed, whatever --bits says. Put
`none` in --bits to also get the unquantized baseline for N>1.

Any flag not listed below is forwarded verbatim to fedrl-train-admm (e.g.
--device cuda:0 --iterations 500 --eval-interval 25 --save-checkpoint).

A run whose metadata.json already has `final_mean_reward` is complete and
is skipped, so an interrupted grid can simply be relaunched. A partial run
(directory exists, never finalized) is skipped with a warning -- ResultsWriter
appends to an existing metrics.csv, so rerunning into it would mix two runs --
unless --rerun-partial is given, which deletes it first.

Examples:
    python scripts/run_grid.py --env hopper --num-seeds 5 --agents 1,4,8 --bits 5,6,7,8 --device cuda:0
    python scripts/run_grid.py --env swimmer --num-seeds 5 --agents 4,8 --bits none,4,8 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

from fednpg_admm import DEFAULT_HYPERPARAMS
from fednpg_admm.train import build_arg_parser as fednpg_build_arg_parser
from fednpg_admm.train import train as fednpg_train
from quantization.quantizers import FLOAT_ALLOWED_BITS, SCHEMES, build_transform, make_channel_factory
from quantization.sweep import _annotate_metadata

# Set per run by this script; forwarding them too would be ambiguous.
_RESERVED_TRAIN_FLAGS = ("--env", "--seed", "--num-agents", "--run-id", "--results-path")


def _int_list(raw: str, flag: str) -> "list[int]":
    try:
        values = sorted({int(x) for x in raw.split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"{flag}: expected comma-separated integers, got {raw!r}") from None
    if not values:
        raise SystemExit(f"{flag}: empty list")
    return values


def _bits_list(raw: str, scheme: str) -> "list[int | None]":
    """None = unquantized baseline."""
    out: "list[int | None]" = []
    for chunk in (c.strip().lower() for c in raw.split(",")):
        if not chunk:
            continue
        if chunk == "none":
            out.append(None)
            continue
        try:
            b = int(chunk)
        except ValueError:
            raise SystemExit(f"--bits: expected integers or 'none', got {chunk!r}") from None
        if scheme == "stochastic" and b < 1:
            raise SystemExit(f"--bits: stochastic needs bits >= 1, got {b}")
        if scheme == "float" and b not in FLOAT_ALLOWED_BITS:
            raise SystemExit(f"--bits: float supports only {sorted(FLOAT_ALLOWED_BITS)}, got {b}")
        out.append(b)
    if not out:
        raise SystemExit("--bits: empty list")
    unique = list(dict.fromkeys(out))
    return sorted(unique, key=lambda b: -1 if b is None else b)


def _run_id(env: str, agents: int, seed: int, scheme: str, bits: "int | None") -> str:
    base = f"{env}-N{agents}-s{seed}"
    if bits is None:
        return base
    return f"{base}-q{bits}" if scheme == "stochastic" else f"{base}-fp{bits}"


def _run_status(run_dir: Path) -> str:
    """'new', 'done' or 'partial'."""
    if not run_dir.exists():
        return "new"
    try:
        metadata = json.loads((run_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "partial"
    return "done" if "final_mean_reward" in metadata else "partial"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid of FedNPG-ADMM runs over seeds x agents x bit-widths. "
        "Unknown flags are forwarded to fedrl-train-admm.",
        allow_abbrev=False,
    )
    parser.add_argument("--env", required=True, choices=list(DEFAULT_HYPERPARAMS))
    parser.add_argument("--num-seeds", type=int, default=5, help="Seeds seed-start .. seed-start+num-seeds-1.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--agents", type=str, default="1,4,8", help="Comma-separated, e.g. 1,4,8.")
    parser.add_argument("--bits", type=str, required=True, help="Comma-separated, e.g. 5,6,7,8 ('none' = unquantized).")
    parser.add_argument("--scheme", type=str, default="stochastic", choices=list(SCHEMES))
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--rerun-partial", action="store_true", help="Delete and rerun partial runs.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned runs.")
    args, train_flags = parser.parse_known_args()

    for flag in train_flags:
        if flag.split("=")[0] in _RESERVED_TRAIN_FLAGS:
            parser.error(f"{flag} is set by this script per run; use the grid flags instead")
    # Validate forwarded flags once, up front, rather than failing on run 1 after a dry-run looked fine.
    fednpg_build_arg_parser().parse_args(train_flags)

    if args.num_seeds < 1:
        parser.error("--num-seeds must be >= 1")
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    agents_list = _int_list(args.agents, "--agents")
    bits_list = _bits_list(args.bits, args.scheme)

    # Seed outermost: after each seed completes, every configuration has
    # one more sample, so a grid stopped halfway is still usable.
    plan: "list[tuple[int, int, int | None]]" = []
    for seed in seeds:
        for agents in agents_list:
            for bits in [None] if agents == 1 else bits_list:
                plan.append((seed, agents, bits))

    results_root = Path(args.results_path)
    print(f"[grid] {len(plan)} runs, env={args.env}, scheme={args.scheme}, forwarded: {' '.join(train_flags) or '-'}")
    for seed, agents, bits in plan:
        run_id = _run_id(args.env, agents, seed, args.scheme, bits)
        print(f"  {run_id:<28} [{_run_status(results_root / run_id)}]")
    if args.dry_run:
        return

    completed, skipped, failed = [], [], []
    for i, (seed, agents, bits) in enumerate(plan, start=1):
        run_id = _run_id(args.env, agents, seed, args.scheme, bits)
        run_dir = results_root / run_id
        status = _run_status(run_dir)
        if status == "done":
            print(f"[grid] ({i}/{len(plan)}) {run_id}: already complete, skipping")
            skipped.append(run_id)
            continue
        if status == "partial":
            if not args.rerun_partial:
                print(f"[grid] ({i}/{len(plan)}) {run_id}: PARTIAL run exists, skipping (delete it or use --rerun-partial)")
                skipped.append(run_id)
                continue
            print(f"[grid] ({i}/{len(plan)}) {run_id}: deleting partial run and restarting")
            shutil.rmtree(run_dir)

        train_args = fednpg_build_arg_parser().parse_args(train_flags)
        train_args.env = args.env
        train_args.seed = seed
        train_args.num_agents = agents
        train_args.results_path = args.results_path
        train_args.run_id = run_id

        print(f"[grid] ({i}/{len(plan)}) starting {run_id}")
        try:
            if bits is None:
                fednpg_train(train_args)
            else:
                # Fresh transform per run: the stochastic scheme's generator is stateful.
                transform = build_transform(args.scheme, bits, seed=seed)
                fednpg_train(train_args, channel_factory=make_channel_factory(transform))
                _annotate_metadata(args.results_path, run_id, args.scheme, bits)
        except KeyboardInterrupt:
            print(f"\n[grid] interrupted during {run_id} (left as partial)")
            raise
        except Exception:
            traceback.print_exc()
            print(f"[grid] {run_id} FAILED, continuing with the next run")
            failed.append(run_id)
            continue
        completed.append(run_id)

    print(f"[grid] done: {len(completed)} completed, {len(skipped)} skipped, {len(failed)} failed")
    for run_id in failed:
        print(f"  FAILED: {run_id}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
