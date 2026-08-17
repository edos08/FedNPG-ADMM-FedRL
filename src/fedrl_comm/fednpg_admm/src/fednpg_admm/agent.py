"""
Per-agent ADMM update — Algorithm 1, lines 5-9.

TODO:
    def agent_update(theta, y_prev, lam, trajectories, rho) -> (y_i, g_i):
        1. lam <- lam + rho * (y_i_prev - y_prev)
        2. g_i <- policy gradient estimate from `trajectories`
        3. y_i <- conjugate_gradient_solve(H_i + rho*I, g_i - lam + rho*y_prev)
           (H_i applied as a Hessian-vector product, never materialized)
        4. return y_i, g_i  (these get sent through common.comms.Channel)
"""
