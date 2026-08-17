"""
Standard-average FedNPG — eq. (12) in the paper. The O(d^2) baseline.

TODO:
    def server_update(theta, H_list, g_list, delta) -> theta_new:
        H <- mean(H_list)   # full d x d matrices, transmitted in full
        g <- mean(g_list)
        theta_new <- theta + sqrt(2*delta / (g^T H^-1 g)) * H^-1 @ g

Note: this one is SUPPOSED to materialize and invert H (or at least solve
a linear system with it) — that's the expensive thing FedNPG-ADMM avoids.
Don't "optimize" this into a Hessian-free version; the whole comparison
depends on this baseline actually paying the O(d^2) cost.
"""
