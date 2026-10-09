# quantization

This is task **(iv)** — the one part of this project that isn't in the
paper (the paper cites Q-GADMM [9] as related work but runs no
quantization experiment of its own), so there's no Figure to reproduce
and check your work against.

The paper's own limitations section admits it: no communication-efficiency
trick beyond ADMM itself was explored. This package is that trick.

## Planned contents

```
src/quantization/
├── quantizers.py  # quantization schemes applied to whatever tensor a
│                  # `common.comms.Channel` sends (y_i, g_i for ADMM;
│                  # H_i, g_i for the standard baseline). Start with
│                  # something simple and standard — e.g. uniform stochastic
│                  # quantization a la QSGD (the paper cites Q-GADMM [9],
│                  # which does exactly this for ADMM — worth reading before
│                  # implementing) — before reaching for anything fancier.
└── sweep.py       # CLI entry point: fedrl-quantize-sweep
                    # runs the same training loop at multiple bit-widths
                    # (e.g. 1, 2, 4, 8, 32-bit / no quantization) and
                    # records: final reward, convergence curve, and actual
                    # communication bytes at each resolution.
```

## Design constraint

This package needs only one small, optional, backward-compatible hook into
`fednpg_admm`: `train()` accepts a `channel_factory` parameter (defaulting
to the existing, unquantized `common.comms.make_channels_for_agents`),
used to build the per-agent Channels instead of a hardcoded call.
Everything about HOW quantization actually works lives entirely in this
package: it wraps `common.comms.Channel` by passing it a quantizing
`Transform` — quantize AND dequantize in one round trip, inside the
transform passed to `Channel(transform=...)`. `Channel.send()` calls that
transform and returns the (possibly degraded) tensor directly; there is
no separate `.receive()` step to hook into (see
`common/comms/channel.py`'s own docstring). If you find yourself editing
`agent.py`, `server.py`, `fisher.py`, or `cg.py` in `fednpg_admm` to make
quantization work, the abstraction wasn't used right — fix that instead
of special-casing here.

## What "study the performance" should produce for the report

At minimum, for each algorithm (FedNPG-ADMM, and ideally the standard
baseline too, for a fair comparison):
- reward-vs-iterations curves at each bit-width, overlaid
- final reward vs. bit-width (single plot, x-axis = bits, y-axis = reward)
- communication bytes vs. bit-width (to show what you're actually buying
  with the accuracy loss)

## Installation

```bash
cd quantization
pip install -e .
```

## Usage

```bash
fedrl-quantize-sweep --algo fednpg-admm --env swimmer --scheme stochastic --bits 1,2,4,8,32
fedrl-quantize-sweep --algo fednpg-admm --env swimmer --scheme float --bits 4,8,16,32
```

`--scheme` selects the quantization family (see `quantizers.py`):
`stochastic` (QSGD/Q-GADMM-style uniform stochastic quantization, unbiased,
any bit-width >= 1) or `float` (real reduced-precision float dtypes —
native fp16/fp8, a simulated fp4 — restricted to bits in {4, 8, 16, 32}).

## Byte accounting: per-tensor metadata is not counted

Dequantizing on the server side needs per-tensor parameters that, in a real
deployment, would travel with every `g_i` / `y_i` transmission: `t_min` and
`scale` for `stochastic` (2 × float32 = 64 bits per send), and `scale` for the
simulated fp4 (32 bits per send). fp16, fp8 and 32-bit sends carry no metadata.

The quantizing transforms report only `bits_per_element = b` to the
`ByteMeter`, so the logged comm bytes are `d · b / 8` per send and **exclude
this metadata**. This is deliberate. The overhead is `64 / (d · b)` of the
payload, about 1–3% at worst (1 bit, `d` in the thousands) and under 0.5% at
8 bits, so it does not change the bytes-vs-bits comparison. See
`docs/quantization_report.md` §4.1 for details.
