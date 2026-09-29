"""
Conjugate gradient: solve A x = b given only a function Avp(v) = A @ v,
never A itself. This is the entire mechanism that lets NPG/ADMM avoid
materializing a d x d matrix — standard CG (see e.g. Shewchuk's
"An Introduction to the Conjugate Gradient Method Without the Agonizing
Pain"), with no dependency on policy.py, torch.nn, or anything RL-specific.
It would work identically solving a linear system that has nothing to do
with a neural network.
"""

from __future__ import annotations

from typing import Callable

import torch


def conjugate_gradient(
    Avp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    num_iters: int = 10,
    tol: float = 1e-10,
) -> torch.Tensor:
    """
    Approximately solve A x = b for x, where `Avp` computes A @ v for any
    vector v (e.g. a Fisher-vector product, or that plus a damping term).

    num_iters=10 is a conventional default in TRPO implementations (the
    matrix is never formed, so more iterations just cost more Hessian-
    vector products, not more memory) -- not a value derived from this
    paper, which doesn't discuss CG tolerance/iteration count at all.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rs_old = torch.dot(r, r)

    for _ in range(num_iters):
        Ap = Avp(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-10)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if rs_new.item() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x
