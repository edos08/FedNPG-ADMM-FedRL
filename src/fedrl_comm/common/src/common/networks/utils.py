"""
Flat-parameter utilities.

The paper's whole formalism (NPG, ADMM, Fisher-vector products) treats the
policy's parameters as a single flat vector theta in R^d. These helpers
convert between that view and a real nn.Module's per-layer parameters, so
fednpg_admm / baselines don't each reimplement this — and so both algorithms
are guaranteed to agree on what "d" and "flatten order" mean.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def get_flat_params(module: nn.Module) -> torch.Tensor:
    """Concatenate every parameter tensor into one flat vector, in
    `module.parameters()` iteration order. detached — this is a snapshot,
    not something you can backprop through; use get_flat_grad for that."""
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def set_flat_params(module: nn.Module, flat_params: torch.Tensor) -> None:
    """Inverse of get_flat_params: write a flat vector back into the
    module's parameters in place, using the same iteration order."""
    pointer = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat_params[pointer : pointer + n].view_as(p))
        pointer += n
    if pointer != flat_params.numel():
        raise ValueError(
            f"flat_params has {flat_params.numel()} elements but module "
            f"has {pointer} — shape mismatch, refusing to silently truncate."
        )


def get_flat_grad(
    loss: torch.Tensor,
    module: nn.Module,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Flat gradient of a scalar `loss` w.r.t. every parameter in `module`, in
    the same order as get_flat_params/set_flat_params.

    Set create_graph=True if you need to backprop through this gradient
    again afterward — that's what Hessian-vector products (needed for the
    ADMM local solve, Algorithm 1 line 8) require: differentiate once to
    get a flat gradient-times-vector scalar, then differentiate that again.
    """
    grads = torch.autograd.grad(
        loss,
        list(module.parameters()),
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=False,
    )
    return torch.cat([g.reshape(-1) for g in grads])
