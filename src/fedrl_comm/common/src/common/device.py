"""
Device resolution: turn a CLI-provided string ("cpu", "mps", "cuda:0", ...)
into a validated torch.device.
"""

from __future__ import annotations

import torch


def list_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def resolve_device(name: str) -> torch.device:
    """
    Validate a device string against what's actually available, then
    confirm it actually works with one real op -- membership in
    list_available_devices() only proves a device is VISIBLE, not that
    this torch build has compiled kernels for its specific architecture.
    A GPU older/newer than the torch build supports can pass every
    availability check and still fail on the first real compute op with
    "no kernel image is available for execution on the device" -- this
    catches that here, at startup, instead of wherever training happens
    to first hit a non-trivial op.
    """
    devices = list_available_devices()
    if name not in devices:
        raise ValueError(f"Invalid device '{name}'. Available: {devices}")

    device = torch.device(name)
    if device.type != "cpu":
        try:
            torch.ones(1, device=device) + torch.ones(1, device=device)
        except Exception as e:
            raise ValueError(f"Device '{name}' is visible but not usable: {e}") from e

    return device