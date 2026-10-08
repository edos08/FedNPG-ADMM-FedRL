"""
Agent<->server communication channel — the one abstraction every algorithm
(fednpg_admm, baselines) routes tensor transmissions through, so that:

  1. byte counts are measured identically across algorithms — the O(d) vs
     O(d^2) comparison only means something if every algorithm counts
     bytes the same way, not by each computing its own estimate by hand.
  2. `quantization` can later wrap a Channel's transform without
     fednpg_admm or baselines importing quantization or knowing it exists.

This is a SIMULATED channel: federated training here runs as one Python
process with N agent objects, not N machines talking over a real network.
send() does three things in one call: applies an optional transform
(identity by default), records the resulting size on a ByteMeter, and
returns the (possibly transformed) tensor.

IMPORTANT: the RETURNED tensor is what "arrives" at the other end. Once a
tensor has gone through send(), downstream code must use the return value,
not the original argument — if a transform is lossy (quantization will be),
the server has to aggregate the degraded version, or the whole quantization
experiment measures nothing.
"""

from __future__ import annotations

from typing import Callable

import torch

from .meter import ByteMeter

# A transform takes a tensor and returns (possibly-modified tensor,
# bits_per_element used to represent it -- or None to fall back to the
# meter's 64-bit default). The identity transform below is a no-op with no
# opinion on bits_per_element.
Transform = Callable[[torch.Tensor], "tuple[torch.Tensor, int | None]"]


def identity_transform(tensor: torch.Tensor) -> "tuple[torch.Tensor, None]":
    return tensor, None


class Channel:
    """One agent's link to the server. See module docstring."""

    def __init__(self, meter: "ByteMeter | None" = None, transform: Transform = identity_transform) -> None:
        self.meter = meter if meter is not None else ByteMeter()
        self.transform = transform

    def send(self, name: str, tensor: torch.Tensor, iteration: "int | None" = None) -> torch.Tensor:
        """
        Send `tensor` through this channel. Returns what "arrives" at the
        other end — identical to `tensor` unless self.transform is lossy.
        Always use the RETURN VALUE downstream, never the original
        `tensor` argument, once it has gone through send().
        """
        out_tensor, bits_per_element = self.transform(tensor)
        if bits_per_element is None:
            bits_per_element = 64
        self.meter.record(name, out_tensor, bits_per_element=bits_per_element, iteration=iteration)
        return out_tensor


    def record_unmaterialized_send(
        self, name: str, num_elements: int, iteration: "int | None" = None
    ) -> None:
        """
        Record the cost of transmitting an object the simulation
        deliberately never builds, and return nothing -- because nothing
        arrives: the receiver reconstructs the same information locally.

        One caller: the standard-average FedNPG baseline (paper eq. 12),
        where each agent conceptually uplinks its full d x d Fisher H_i so
        the server can solve (sum H_i)^-1 sum g_i. That O(d^2) cost is
        exactly what FedNPG-ADMM exists to avoid, so it MUST appear in the
        byte accounting -- but materializing H_i is impossible at
        Humanoid's d = 727,074 (2.1 PB), and unnecessary: the simulation
        has every agent's data in-process and gets the identical direction
        from a pooled Fisher-vector product.

        Do NOT use this to avoid routing a real tensor through send() -- a
        transmission whose tensor exists must go through send(), or
        quantization silently stops applying to it.
        """
        self.meter.record(
            name, torch.empty(0), bits_per_element=64, iteration=iteration,
            num_elements=num_elements,
        )


def make_channels_for_agents(num_agents: int) -> list[Channel]:
    """
    One independent Channel (and one independent ByteMeter) per agent,
    matching the paper's convention of reporting overhead per agent, not
    summed across the federation (Fig. 3 caption).
    """
    return [Channel() for _ in range(num_agents)]
