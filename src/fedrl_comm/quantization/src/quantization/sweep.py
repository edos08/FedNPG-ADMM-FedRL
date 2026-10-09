"""
CLI entry point for the quantization bit-width sweep: fedrl-quantize-sweep.

Runs fednpg_admm.train.train() once per (scheme, bit-width) pair, each
time with the agent<->server channels wired to a quantizing Transform
(quantizers.py) instead of the default identity passthrough -- via
train()'s channel_factory parameter, so this file never touches
fednpg_admm's algorithm code (agent.py/server.py/fisher.py/cg.py).

Currently only --algo fednpg-admm actually runs anything -- fednpg-standard
is a real planned choice, not silently accepted: argparse's `choices`
rejects it with a clear error until baselines/fednpg_standard.py exists
and exposes an equivalent train(args, channel_factory=...).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from fednpg_admm import DEFAULT_HYPERPARAMS
from fednpg_admm.train import build_arg_parser as fednpg_build_arg_parser
from fednpg_admm.train import train as fednpg_train

from .quantizers import FLOAT_ALLOWED_BITS, SCHEMES, build_transform, make_channel_factory

SUPPORTED_ALGOS = ["fednpg-admm"]  # add "fednpg-standard" once baselines/fednpg_standard.py exists


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep communication bit-width for the agent->server channel; log reward/bytes tradeoffs."
    )
    parser.add_argument("--algo", type=str, default="fednpg-admm", choices=SUPPORTED_ALGOS)
    parser.add_argument("--env", type=str, default="swimmer", choices=list(DEFAULT_HYPERPARAMS))
    parser.add_argument("--scheme", type=str, default="stochastic", choices=list(SCHEMES))
    parser.add_argument(
        "--bits",
        type=str,
        required=True,
        help=(
            "Comma-separated bit-widths, e.g. 1,2,4,8,32. "
            f"--scheme float only supports {sorted(FLOAT_ALLOWED_BITS)}."
        ),
    )
    parser.add_argument("--num-agents", type=int, default=None, help="Default: fednpg_admm's own default (8).")
    parser.add_argument("--iterations", type=int, default=None, help="Override; default: per-env paper value.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override; default: per-env paper value.")
    parser.add_argument("--device", type=str, default=None, help="Default: fednpg_admm's own default (cpu).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-path", type=str, default="results")
    return parser


class _BitsError(ValueError):
    pass


def _parse_bits(raw: str, scheme: str) -> "list[int]":
    """
    Pure validation logic, kept separate from argparse's error
    presentation (main() below translates a _BitsError into
    parser.error()) so this is unit-testable without constructing an
    ArgumentParser.
    """
    try:
        bits_list = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    except ValueError:
        raise _BitsError(f"--bits must be a comma-separated list of integers, got {raw!r}") from None
    if not bits_list:
        raise _BitsError("--bits must contain at least one value")
    if scheme == "stochastic":
        bad = [b for b in bits_list if b < 1]
        if bad:
            raise _BitsError(f"--scheme stochastic requires bit-widths >= 1, got invalid values {bad}")
    else:  # scheme == "float", the only other choices= value
        bad = [b for b in bits_list if b not in FLOAT_ALLOWED_BITS]
        if bad:
            raise _BitsError(
                f"--scheme float only supports bit-widths in {sorted(FLOAT_ALLOWED_BITS)}, got invalid values {bad}"
            )
    deduped = sorted(set(bits_list))
    if deduped != sorted(bits_list):
        print(f"[fedrl-quantize-sweep] note: duplicate --bits values collapsed to {deduped}")
    return deduped


def _make_run_id(scheme: str, bits: int, env: str, num_agents: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"fednpg_admm_quant_{scheme}_{bits}bit_{env}_{num_agents}agents_{timestamp}"


def _annotate_metadata(results_path: str, run_id: str, scheme: str, bits: int) -> None:
    """
    train() builds metadata.json entirely internally (see
    fednpg_admm/train.py) with no hook for a caller to inject extra keys
    at construction time, and doesn't expose its ResultsWriter instance --
    so quant_scheme/quant_bits are added by reading the file back and
    merging, strictly AFTER train() (and its own internal
    writer.finalize() call) has already returned, so there is no
    concurrent-write race. Safe because metadata.json's location and
    freeform-dict format are a stable public contract (see
    common/results/writer.py's module docstring).
    """
    metadata_path = Path(results_path) / run_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["quant_scheme"] = scheme
    metadata["quant_bits"] = bits
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        bits_list = _parse_bits(args.bits, args.scheme)
    except _BitsError as e:
        parser.error(str(e))
        return  # unreachable -- parser.error() exits(2)

    if args.algo != "fednpg-admm":
        # Unreachable today: choices=SUPPORTED_ALGOS already rejected
        # anything else before main() ran. Kept as a guard, not a TODO,
        # so adding "fednpg-standard" later is an elif, not a rewrite.
        raise AssertionError(f"unhandled --algo {args.algo!r}")

    base_args = fednpg_build_arg_parser().parse_args([])
    base_args.env = args.env
    base_args.seed = args.seed
    base_args.results_path = args.results_path
    if args.num_agents is not None:
        base_args.num_agents = args.num_agents
    if args.iterations is not None:
        base_args.iterations = args.iterations
    if args.timesteps is not None:
        base_args.timesteps = args.timesteps
    if args.device is not None:
        base_args.device = args.device

    for bits in bits_list:
        base_args.run_id = _make_run_id(args.scheme, bits, base_args.env, base_args.num_agents)
        channel_factory = make_channel_factory(build_transform(args.scheme, bits, seed=args.seed))
        run_id = fednpg_train(base_args, channel_factory=channel_factory)
        _annotate_metadata(base_args.results_path, run_id, args.scheme, bits)
        print(f"[fedrl-quantize-sweep] completed scheme={args.scheme} bits={bits} -> {run_id}")


if __name__ == "__main__":
    main()
