# python
import numpy as np
from scipy.stats import norm

from .models import generate_correlated_paths

def kirk_cso_price(F1, F2, K, sigma1, sigma2, rho, T, eps=1e-12):
    """
    Kirk approximation for a spread option on S1 - S2 with strike K:
    Uses b = F2 / (F2 + K) and treats the option as Black on F1 with strike (F2 + K).
    Returns non-negative price and guards against pathological inputs.
    """
    denom = (F2 + K)
    if denom <= 0:
        # invalid configuration for Kirk; fallback to intrinsic / zero
        return max(F1 - F2 - K, 0.0)

    b = F2 / denom
    sigma_kirk_sq = sigma1**2 - 2.0 * rho * sigma1 * sigma2 * b + (sigma2**2) * (b**2)
    # numerical safety
    sigma_kirk_sq = max(sigma_kirk_sq, 0.0)
    sigma_kirk = np.sqrt(sigma_kirk_sq)

    # if volatility is essentially zero, use the deterministic payoff (forward)
    if sigma_kirk < eps or T <= 0.0:
        return max(F1 - denom, 0.0)

    d1 = (np.log(F1 / denom) + 0.5 * sigma_kirk_sq * T) / (sigma_kirk * np.sqrt(T))
    d2 = d1 - sigma_kirk * np.sqrt(T)
    price = F1 * norm.cdf(d1) - denom * norm.cdf(d2)
    return float(max(price, 0.0))


def mc_cso_price(F1, F2, K, sigma1, sigma2, rho, T, steps=1, n=10000, return_se=False):
    """
    Monte Carlo price for the same payoff used in kirk_cso_price:
    payoff = max(S1_T - S2_T - K, 0)
    Uses generate_correlated_paths with zero drift (forwards).
    Returns mean payoff (and optional standard error).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")

    paths = generate_correlated_paths(0.0, 0.0, sigma1, sigma2, rho, F1, F2, T, steps, n)
    payoffs = np.maximum(paths[:, 0] - paths[:, 1] - K, 0.0)
    mean = float(payoffs.mean())
    if return_se:
        se = float(payoffs.std(ddof=1) / np.sqrt(n))
        return mean, se
    return mean
