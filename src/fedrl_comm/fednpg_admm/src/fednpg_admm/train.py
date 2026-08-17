"""
CLI entry point for FedNPG-ADMM training.

TODO: argparse for --env, --num-agents, --iterations, --penalty (rho),
      --radius (delta), --step-size (eta), --seed, --results-path, etc.
      (see paper Table 4 for per-env defaults).
      Orchestrates: common.envs -> common.rollout -> agent.py -> common.comms
      -> server.py -> broadcast, logging results via common.results.
"""


def main() -> None:
    raise NotImplementedError(
        "fedrl-train-admm is not implemented yet — this is task (ii)."
    )


if __name__ == "__main__":
    main()
