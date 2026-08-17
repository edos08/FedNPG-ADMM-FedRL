"""
CLI entry point for the quantization bit-width sweep.

TODO: argparse for --algo {fednpg-admm, fednpg-standard}, --env, --bits
      (comma-separated list), --iterations, --seed, --results-path.
      Runs the training loop once per bit-width, wrapping common.comms
      with a quantizing channel, logging reward curves + comm bytes via
      common.results.
"""


def main() -> None:
    raise NotImplementedError(
        "fedrl-quantize-sweep is not implemented yet — this is task (iv)."
    )


if __name__ == "__main__":
    main()
