"""
Device resolution: turn a CLI-provided string ("cpu", "mps", "cuda:0", ...)
into a validated torch.device, failing loudly if the requested backend
isn't actually available rather than silently falling back to CPU.

Two things worth knowing before assuming --device cuda/mps is automatically
a win, neither of which this module can fix -- they're properties of how
this project's training loop is structured, not of device selection itself:

1. Environment stepping (MuJoCo physics, via env.step()) is always
   CPU-bound through numpy, regardless of this setting. Nothing about GPU
   selection speeds up simulation -- only the neural network math does.

2. Rollout collection here calls the policy/critic ONE observation at a
   time (batch size 1 per env.step). GPU/MPS dispatch overhead can make
   that SLOWER than CPU, not faster, for a small network -- there's no
   batching to amortize the transfer/launch cost against. The parts that
   actually benefit from a faster device are the BATCHED operations
   (agent_update's Fisher-vector products over a whole rollout, and
   critic fitting), and the benefit scales with network size: negligible
   for Swimmer's ~4.9K-parameter policy, real for Humanoid's ~727K-
   parameter one. Benchmark before assuming GPU is faster for a small env.
"""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    """
    Parse and validate a device string. Raises ValueError with a specific,
    actionable message if the requested backend isn't available on this
    machine -- silently substituting CPU instead would hide a real
    misconfiguration (e.g. a typo, or code that assumes a GPU that isn't
    actually being used) behind results that are merely slower than
    expected, which is a much harder bug to notice than a crash.
    """
    device = torch.device(name)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{name}', but torch.cuda.is_available() is False "
                "on this machine -- no CUDA GPU visible to PyTorch. Use --device cpu "
                "or --device mps (Apple Silicon) instead."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested '{name}', but this machine only has "
                f"{torch.cuda.device_count()} CUDA device(s) available "
                f"(valid indices: 0..{torch.cuda.device_count() - 1})."
            )
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "Requested device 'mps', but torch.backends.mps.is_available() is "
                "False on this machine -- either this isn't Apple Silicon, or this "
                "torch build doesn't have MPS support. Use --device cpu instead."
            )

    return device