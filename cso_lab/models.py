"""
Market models and stochastic process implementations.
"""

import numpy as np

def generate_correlated_paths(mu1, mu2, sigma1, sigma2, rho, S1, S2, T, steps, n):
    dt = T / steps
    cov = rho * sigma1 * sigma2
    cov_matrix = [[sigma1**2, cov], [cov, sigma2**2]]
    chol = np.linalg.cholesky(cov_matrix)

    paths = np.zeros((n, 2))
    for i in range(n):
        z = np.random.normal(size=2)
        correlated = chol.dot(z)
        S1_T = S1 * np.exp((mu1 - 0.5 * sigma1**2) * T + correlated[0] * np.sqrt(T))
        S2_T = S2 * np.exp((mu2 - 0.5 * sigma2**2) * T + correlated[1] * np.sqrt(T))
        paths[i] = [S1_T, S2_T]

    return paths


class GeometricBrownianMotion:
    """Geometric Brownian Motion model."""

    def __init__(self, S0, mu, sigma):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T, n_steps, n_paths=1):
        """Simulate price paths."""
        pass
