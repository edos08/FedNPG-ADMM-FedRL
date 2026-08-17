"""
Server-side aggregation — Algorithm 1, lines 11-13.

TODO:
    def server_update(theta, y_list, g_list, delta, eta) -> theta_new:
        1. y <- mean(y_list)
        2. step_size <- eta * sqrt(2 * N * delta / (sum(g_list) . y))
        3. theta_new <- theta + step_size * y
        4. return theta_new
"""
