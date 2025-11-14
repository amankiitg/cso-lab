"""
CSO pricing methods.
"""
import numpy as np
from .models import generate_correlated_paths

def kirk_cso_price(F1, F2, K, sigma1, sigma2, rho, T):
    sigma_kirk = np.sqrt(
        sigma1**2
        - 2 * rho * sigma1 * sigma2 * F2 / (F1 + F2)
        + sigma2**2 * (F2 / (F1 + F2))**2
    )
    d1 = (np.log((F1 + F2) / K) + 0.5 * sigma_kirk**2 * T) / (sigma_kirk * np.sqrt(T))
    d2 = d1 - sigma_kirk * np.sqrt(T)
    return (F1 + F2) * norm.cdf(d1) - K * norm.cdf(d2)

def mc_cso_price(F1, F2, K, sigma1, sigma2, rho, T, steps=50, n=10000):
    paths = generate_correlated_paths(0, 0, sigma1, sigma2, rho, F1, F2, T, steps, n)
    payoff = np.maximum(paths[:, 0] - paths[:, 1] - K, 0)
    return payoff.mean()
