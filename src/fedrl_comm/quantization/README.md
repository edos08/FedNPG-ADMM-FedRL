# quantization

Quantization of the two vectors (`y_i`, `g_i`) each FedNPG-ADMM agent sends to
the server, and sweeps of performance against bit width. This is not part of
the original paper, which cites Q-GADMM as related work but runs no
quantization experiment.

## Contents

```
src/quantization/
├── quantizers.py  # the schemes, as Transforms for common.comms.Channel
└── sweep.py       # CLI: fedrl-quantize-sweep (one run per bit width)
```

`scripts/run_grid.py` at the repository root runs a full grid of seeds x
agent counts x bit widths and names runs `<env>-N<n>-s<seed>-q<bits>`.

## Schemes

- `stochastic`: uniform stochastic quantization in the style of QSGD and
  Q-GADMM, any bit width >= 1. Values are rounded up or down at random with
  the probability that makes the quantizer unbiased, so the server receives the
  correct direction in expectation. Uses its own random generator, so runs at
  different bit widths see the same action-sampling sequence.
- `float`: reduced-precision floats (native fp16 and fp8, simulated fp4),
  bit widths in {4, 8, 16, 32}.

## How it plugs in

`fednpg_admm.train()` takes an optional `channel_factory`. This package passes
one that builds `Channel`s with a quantizing `Transform`, which quantizes and
dequantizes in one call and declares the bit width actually needed on the wire.
The server therefore aggregates the degraded vectors, and the byte meter
counts `d * b / 8` bytes per send. No file of the algorithm itself imports this
package.

## Usage

```bash
fedrl-quantize-sweep --env swimmer --scheme stochastic --bits 4,6,8,32
fedrl-quantize-sweep --env swimmer --scheme float --bits 4,8,16,32

python scripts/run_grid.py --env humanoid --num-seeds 5 --agents 2,4,8 --bits 6,8
python scripts/run_grid.py --env hopper --num-seeds 1 --agents 8 --bits none,8 --dry-run
```

`run_grid.py` forwards any other flag to `fedrl-train-admm`, so a grid can run
in a non-default configuration (Hopper's, for example: see
`final_sweep.sh hopper-quant`).

## Byte accounting: per-tensor metadata is not counted

Dequantizing on the server needs per-tensor parameters that a real deployment
would send with every transmission: the minimum and the scale for `stochastic`
(2 x float32, 64 bits per send) and the scale for the simulated fp4 (32 bits).
The logged bytes are `d * b / 8` per send and exclude this metadata on purpose:
it adds `64 / (d * b)` of the payload, about 1-3% in the worst case here (1 bit,
`d` in the thousands) and under 0.5% at 8 bits, which does not change the
comparison across bit widths. Details: `docs/quantization_report.md` §4.1.

## Installation

Installed together with the other packages from the repository root; see the
main [README](../../../README.md#installation).
