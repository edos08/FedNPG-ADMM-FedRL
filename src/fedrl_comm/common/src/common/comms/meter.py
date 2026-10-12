"""
Byte accounting for the O(d) vs O(d^2) communication comparison the
project's results hinge on.

Counts are per-agent, matching the paper: Figure 3's caption says overhead
is "measured by the transmitted bytes in each agent" — not summed across
the federation. Each agent gets its own meter (see Channel /
make_channels_for_agents), not one shared global counter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TransmissionRecord:
    name: str
    iteration: "int | None"
    num_elements: int
    bits_per_element: int

    @property
    def num_bytes(self) -> float:
        return self.num_elements * self.bits_per_element / 8


class ByteMeter:
    """
    Accumulates communication cost as (element count, bits/element) pairs —
    deliberately NOT read off torch.Tensor.element_size(), for two reasons:

    1. The paper's own convention (Fig. 3 caption): overhead is "measured
       by the number of transmitted parameters with double precision" —
       8 bytes/element, 64 bits — even though nothing in this project is
       actually stored as float64. Default bits_per_element=64 exists to
       match the PAPER's reporting convention, not our training dtype.
    2. Quantization needs to report a DIFFERENT bits_per_element per
       transmission (e.g. 4) while the underlying tensor is still a
       perfectly ordinary float32 torch.Tensor in memory — that number is
       a bookkeeping fact about how it WOULD be encoded on the wire, not
       something torch can infer from the tensor itself.
    """

    def __init__(self) -> None:
        self._log: list[TransmissionRecord] = []

    def record(
        self,
        name: str,
        tensor: torch.Tensor,
        bits_per_element: int = 64,
        iteration: "int | None" = None,
        num_elements: "int | None" = None,
    ) -> TransmissionRecord:
        """
        `num_elements` overrides tensor.numel() for the one case where an
        algorithm's transmitted object is deliberately never materialized:
        the standard-average FedNPG baseline (paper eq. 12) uplinks each
        agent's d x d Fisher, which is 0.095 GB on Swimmer, 1.36 GB on
        Hopper and 2.1 PB on Humanoid -- the last being four orders of
        magnitude beyond any machine, so the baseline could not exist at
        all for the paper's largest environment. Same spirit as
        bits_per_element: a bookkeeping fact about the wire format, not a
        property of any tensor in memory. See baselines/fednpg_standard.py.
        """
        rec = TransmissionRecord(
            name=name,
            iteration=iteration,
            num_elements=tensor.numel() if num_elements is None else num_elements,
            bits_per_element=bits_per_element,
        )
        self._log.append(rec)
        return rec

    @property
    def total_bytes(self) -> float:
        return sum(r.num_bytes for r in self._log)

    def bytes_by_name(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self._log:
            out[r.name] = out.get(r.name, 0.0) + r.num_bytes
        return out

    def bytes_by_iteration(self) -> dict[int, float]:
        """Per-iteration totals, the raw data behind the
        comm-overhead-vs-iteration curves (see results_webapp)."""
        out: dict[int, float] = {}
        for r in self._log:
            if r.iteration is None:
                continue
            out[r.iteration] = out.get(r.iteration, 0.0) + r.num_bytes
        return out

    @property
    def log(self) -> list[TransmissionRecord]:
        return list(self._log)

    def reset(self) -> None:
        self._log.clear()
