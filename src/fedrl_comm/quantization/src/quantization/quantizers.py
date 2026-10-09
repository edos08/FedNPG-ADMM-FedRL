"""
Quantization schemes applied to whatever tensor a common.comms.Channel
sends (y_i, g_i for FedNPG-ADMM). Two independent scheme families, both
exposed as plain common.comms.channel.Transform callables so Channel.send()
can use either with zero changes to Channel itself:

  - stochastic: uniform stochastic quantization (QSGD / Q-GADMM-style,
    paper ref [9]) -- affine min/max mapping to integer levels, unbiased
    via Bernoulli rounding.
  - float: reduced-precision float dtypes (fp16/fp8 native, fp4 simulated)
    -- deterministic round-to-nearest-representable-value, exponential
    (non-uniform) spacing. A genuinely different quantization philosophy
    from `stochastic`, not the same algorithm relabeled.

Every public *_transform function returns a Transform matching Channel's
contract exactly: (tensor, bits_per_element) in ONE call, already
dequantized -- there is no Channel.receive() to split quantize/dequantize
across (see common/comms/channel.py's module docstring).
"""

from __future__ import annotations

from typing import Callable

import torch

from common.comms.channel import Channel, Transform

SCHEMES = ("stochastic", "float")
FLOAT_ALLOWED_BITS = frozenset({4, 8, 16, 32})
_NATIVE_FLOAT_DTYPES = {16: torch.float16, 8: torch.float8_e4m3fn}


# --------------------------------------------------------------------------
# stochastic scheme
# --------------------------------------------------------------------------


def stochastic_quantize(
    tensor: torch.Tensor, num_bits: int, generator: "torch.Generator | None" = None
) -> "tuple[torch.Tensor, dict]":
    """
    Uniform stochastic quantization. Unbiased in expectation --
    E[dequantize(*stochastic_quantize(x))] == x -- because each element
    rounds to its nearest integer level via a Bernoulli draw weighted by
    fractional distance, not round-to-nearest (which is biased whenever x
    doesn't land exactly on a level). This is the standard reason this
    family is preferred in the distributed-optimization quantization
    literature (QSGD, Q-GADMM).

    num_bits=1 is a real, meaningful edge case, not an error: levels=1, so
    every element becomes a Bernoulli choice between t_min and t_max.

    `generator`: an isolated torch.Generator to draw the Bernoulli bits
    from, so this doesn't consume the global torch RNG shared with action
    sampling during rollout collection (see quantization.sweep). None
    falls back to the global RNG.

    Returns (quantized, meta): `quantized` is an integer-valued float
    tensor in [0, 2**num_bits - 1]; meta = {"t_min": ..., "scale": ...} is
    everything dequantize() needs to invert this.
    """
    if num_bits < 1:
        raise ValueError(f"num_bits must be >= 1, got {num_bits}")

    tensor = tensor.detach()
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max == t_min:
        # Degenerate: every element identical (e.g. an all-zero gradient).
        # scale=0 makes dequantize return t_min for every element --
        # zero reconstruction error, since there's nothing to quantize.
        return torch.zeros_like(tensor), {"t_min": t_min, "scale": torch.zeros_like(t_min)}

    levels = 2**num_bits - 1
    scale = (t_max - t_min) / levels
    normalized = (tensor - t_min) / scale
    floor = torch.floor(normalized)
    # clamp guards a real (empirically confirmed) crash: float32 rounding
    # can push `frac` a hair above 1.0, and torch.bernoulli hard-errors on
    # p > 1 (RuntimeError: "Expected p_in >= 0 && p_in <= 1"), not just
    # produces a wrong answer.
    frac = (normalized - floor).clamp(0.0, 1.0)
    quantized = (floor + torch.bernoulli(frac, generator=generator)).clamp(0, levels)
    return quantized, {"t_min": t_min, "scale": scale}


def dequantize(quantized: torch.Tensor, meta: "dict") -> torch.Tensor:
    return meta["t_min"] + quantized * meta["scale"]


def make_quantizing_transform(num_bits: int, seed: "int | None" = None) -> Transform:
    """
    Composes stochastic_quantize + dequantize into ONE round trip, per
    Channel's contract: the returned tensor is a plain, already-degraded
    float32 tensor ready for immediate downstream use.

    If `seed` is given, builds ONE torch.Generator (not one per call), so
    successive calls draw a genuinely-advancing, independent but fully
    reproducible sequence -- isolated from the global torch RNG that
    action sampling uses, so quantizing at different bit-widths never
    perturbs which actions get sampled during rollout collection.

    The generator is built lazily, on the first call, on whatever device
    that first tensor lives on (torch.bernoulli requires its generator
    and its input tensor to be on the same device) -- train.py resolves
    --device before ever calling this transform, so this correctly
    matches CPU or CUDA runs without sweep.py needing to know the device
    up front. Every tensor a Channel sends within one run lives on the
    same device, so building it once, on the first call, is sufficient.
    """
    generator_holder: "list[torch.Generator | None]" = [None]

    def transform(tensor: torch.Tensor) -> "tuple[torch.Tensor, int]":
        if seed is not None and generator_holder[0] is None:
            generator_holder[0] = torch.Generator(device=tensor.device).manual_seed(seed)
        quantized, meta = stochastic_quantize(tensor, num_bits, generator=generator_holder[0])
        return dequantize(quantized, meta), num_bits

    return transform


# --------------------------------------------------------------------------
# float scheme
# --------------------------------------------------------------------------


def _minifloat_levels(exponent_bits: int, mantissa_bits: int, exponent_bias: int) -> torch.Tensor:
    """
    All non-negative representable magnitudes of a minifloat with the
    given exponent/mantissa widths: (1 + mantissa/2**mantissa_bits) *
    2**(exponent - exponent_bias), for exponent in [1, 2**exponent_bits -
    1] (exponent code 0 is reserved and flushed to zero here rather than
    modeled as a subnormal -- a standard minifloat simplification).
    Sorted ascending, includes 0.0.
    """
    levels = [0.0]
    for exponent in range(1, 2**exponent_bits):
        for mantissa in range(2**mantissa_bits):
            frac = mantissa / (2**mantissa_bits)
            levels.append((1.0 + frac) * (2.0 ** (exponent - exponent_bias)))
    return torch.tensor(sorted(levels), dtype=torch.float32)


# E2M1-*like* level set (sign handled separately via torch.sign) --
# exponent_bits=2, mantissa_bits=1, bias=1: same bit widths and bias as
# real E2M1 (used in e.g. NVIDIA NVFP4 / the OCP Microscaling spec), but
# NOT bit-exact -- see _simulate_fp4's docstring for what's approximated.
_FP4_LEVELS = _minifloat_levels(exponent_bits=2, mantissa_bits=1, exponent_bias=1)
_FP4_MAX_LEVEL = float(_FP4_LEVELS.max())


def _simulate_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Manually simulated 4-bit minifloat (sign + 2 exponent bits + 1
    mantissa bit): torch has no native 4-bit float dtype as of 2.11.
    Loosely modeled on E2M1 but explicitly NOT bit-exact -- subnormals
    flush to zero, no inf/NaN encoding. Exactness isn't required for
    studying the reward/bytes tradeoff, only a genuine,
    monotonically-coarser-than-fp8 precision loss.

    A single per-tensor scale (the same spirit as a hardware block-scale
    factor, and as the stochastic scheme's own per-tensor min/max) maps
    the tensor's max magnitude onto the format's largest representable
    value, so the fixed level set is actually used regardless of the
    tensor's own scale. Without this, any small-magnitude tensor (typical
    for g_i/y_i) would collapse entirely to zero -- E2M1's representable
    magnitudes without a scale are {0, 1, 1.5, 2, 3, 4, 6}, none close to
    a typical gradient's scale -- which would be a correctness bug, not a
    compression tradeoff.
    """
    tensor = tensor.detach()
    sign = torch.sign(tensor)
    mag = tensor.abs()
    max_mag = mag.max()
    if max_mag == 0:
        return tensor.clone()
    scale = max_mag / _FP4_MAX_LEVEL
    normalized = mag / scale
    levels = _FP4_LEVELS.to(tensor.device)
    nearest_idx = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return sign * levels[nearest_idx] * scale


def float_quantize_transform(num_bits: int) -> Transform:
    """
    bits=32: uncompressed baseline, returned AS-IS with bits_per_element
    explicitly 32 -- deliberately NOT common.comms.identity_transform,
    whose bits_per_element=None falls back to Channel's 64-bit default
    (the paper's own fp64 REPORTING convention; unrelated to this
    scheme's literal "32"). Using identity_transform here would put the
    bits=32 point of a comm-bytes-vs-bits plot at 64, silently.
    bits=16 / bits=8: native torch.float16 / torch.float8_e4m3fn, cast
    down and back up to float32.
    bits=4: no native torch dtype -- see _simulate_fp4.
    """
    if num_bits not in FLOAT_ALLOWED_BITS:
        raise ValueError(f"--scheme float only supports bits in {sorted(FLOAT_ALLOWED_BITS)}, got {num_bits}")

    if num_bits == 32:

        def transform(tensor: torch.Tensor) -> "tuple[torch.Tensor, int]":
            return tensor, 32

        return transform

    if num_bits == 4:

        def transform(tensor: torch.Tensor) -> "tuple[torch.Tensor, int]":
            return _simulate_fp4(tensor), 4

        return transform

    dtype = _NATIVE_FLOAT_DTYPES[num_bits]

    def transform(tensor: torch.Tensor) -> "tuple[torch.Tensor, int]":
        return tensor.detach().to(dtype).to(torch.float32), num_bits

    return transform


# --------------------------------------------------------------------------
# Channel adapter + scheme dispatcher
# --------------------------------------------------------------------------


def make_channel_factory(transform: Transform) -> "Callable[[int], list[Channel]]":
    """
    Adapts one Transform into the channel_factory shape
    fednpg_admm.train.train() accepts (Callable[[int], list[Channel]] --
    same shape as common.comms.make_channels_for_agents, which this
    replaces). All num_agents Channels share this same transform (a pure
    function of one tensor -- safe to share). meter=None on each Channel
    is deliberate: every agent gets its OWN fresh ByteMeter, preserving
    the paper's per-agent byte-accounting convention.
    """

    def factory(num_agents: int) -> "list[Channel]":
        return [Channel(transform=transform) for _ in range(num_agents)]

    return factory


def build_transform(scheme: str, num_bits: int, seed: "int | None" = None) -> Transform:
    """Dispatches to the requested scheme's Transform-constructing function."""
    if scheme == "stochastic":
        return make_quantizing_transform(num_bits, seed=seed)
    if scheme == "float":
        return float_quantize_transform(num_bits)
    raise ValueError(f"unknown scheme {scheme!r}, expected one of {SCHEMES}")
