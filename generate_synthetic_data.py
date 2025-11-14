import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

def ornstein_uhlenbeck(T, kappa, mu, sigma, x0):
    n = T
    dt = 1
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + kappa * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return x

def generate_spot_series():
    dates = pd.date_range("2015-01-01", periods=3000, freq="D")

    hh = ornstein_uhlenbeck(len(dates), kappa=0.15, mu=3.0, sigma=0.6, x0=3.5)
    wti = ornstein_uhlenbeck(len(dates), kappa=0.08, mu=65.0, sigma=3.0, x0=60.0)

    hh += np.sin(np.linspace(0, 12 * np.pi, len(dates))) * 0.3
    wti += np.sin(np.linspace(0, 5 * np.pi, len(dates))) * 2.0

    hh += (np.random.randn(len(dates)) * 0.3)
    wti += (np.random.randn(len(dates)) * 1.5)

    df = pd.DataFrame({"date": dates, "hh_spot": hh, "wti_spot": wti})
    df.to_csv(OUTPUT_DIR / "spot_series.csv", index=False)
    print("Saved spot series.")

def generate_futures_curves():
    dates = pd.date_range("2015-01-01", periods=2000)

    tenors = np.array([1, 2, 3, 6, 12])  # months

    base_hh = ornstein_uhlenbeck(len(dates), 0.12, 3.1, 0.5, 3.3)
    base_wti = ornstein_uhlenbeck(len(dates), 0.06, 63.0, 2.0, 60.0)

    futures_hh = {}
    futures_wti = {}
    for t in tenors:
        futures_hh[f"tenor_{t}m"] = base_hh + np.random.randn(len(dates)) * (0.05 * t)
        futures_wti[f"tenor_{t}m"] = base_wti + np.random.randn(len(dates)) * (0.3 * t)

    df_hh = pd.DataFrame({"date": dates, **futures_hh})
    df_wti = pd.DataFrame({"date": dates, **futures_wti})

    df_hh.to_csv(OUTPUT_DIR / "hh_futures_curves.csv", index=False)
    df_wti.to_csv(OUTPUT_DIR / "wti_futures_curves.csv", index=False)

    print("Saved futures curves.")

def generate_correlation():
    dates = pd.date_range("2015-01-01", periods=1500)

    rho = np.sin(np.linspace(0, 3 * np.pi, len(dates))) * 0.2 + 0.3
    rho += np.random.randn(len(dates)) * 0.05
    rho = np.clip(rho, -0.1, 0.9)

    df = pd.DataFrame({"date": dates, "corr_hh_wti": rho})
    df.to_csv(OUTPUT_DIR / "correlation.csv", index=False)
    print("Saved correlation.")

def generate_vol_surface(name):
    tenors = np.array([0.25, 0.5, 1.0])
    strikes = np.array([-0.5, 0.0, 0.5, 1.0])
    rows = []

    for T in tenors:
        for k in strikes:
            if name == "hh":
                base = 0.6
            else:
                base = 0.3

            vol = base + 0.1 * k + 0.05 * np.random.randn()
            vol = max(0.05, vol)
            rows.append([T, k, vol])

    df = pd.DataFrame(rows, columns=["tenor", "strike", "vol"])
    df.to_csv(OUTPUT_DIR / f"vol_surface_{name}.csv", index=False)
    print(f"Saved {name} vol surface.")

def generate_cso_chain():
    rows = []
    tenors = [0.25, 0.5, 1.0]
    spreads = [-1.0, -0.5, 0.0, 0.5, 1.0]

    for T in tenors:
        for K in spreads:
            F1 = 3.0
            F2 = 65.0
            vol1 = 0.6
            vol2 = 0.3
            rho = 0.25

            theo = (F1 - F2 - K) * np.exp(-0.5 * rho * T)
            theo += np.random.randn() * 0.2

            rows.append([T, K, theo])

    df = pd.DataFrame(rows, columns=["tenor", "strike", "theoretical"])
    df.to_csv(OUTPUT_DIR / "cso_chain.csv", index=False)
    print("Saved CSO chain.")

def main():
    generate_spot_series()
    generate_futures_curves()
    generate_correlation()
    generate_vol_surface("hh")
    generate_vol_surface("wti")
    generate_cso_chain()
    print("All synthetic data generated.")

if __name__ == "__main__":
    main()
