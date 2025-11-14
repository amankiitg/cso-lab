# run_pipeline.py
"""
Pipeline to ingest synthetic data, build vol surfaces, price CSOs, compute realized correlation,
run a simple relative value backtest and export results.

Expected input files (data/synthetic):
 - spot_series.csv
 - hh_futures_curves.csv
 - wti_futures_curves.csv
 - correlation.csv
 - vol_surface_hh.csv
 - vol_surface_wti.csv
 - cso_chain.csv

Outputs (data/outputs):
 - priced_cso_kirk.csv
 - priced_cso_mc.csv
 - backtest_pnl.csv
 - realized_corr.csv
 - some quick plots in data/outputs/plots/
"""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- paths ---
ROOT = Path(__file__).parent
SYN_DIR = ROOT / "data" / "synthetic"
OUT_DIR = ROOT / "data" / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- helpers ---

def safe_read_csv(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)

# --- Vol surface helper ---

@dataclass
class VolSurface:
    tenor_arr: np.ndarray
    strike_arr: np.ndarray
    vol_arr: np.ndarray
    interpolator: LinearNDInterpolator

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        # df expected columns: tenor, strike, vol
        points = list(zip(df["tenor"].values, df["strike"].values))
        vals = df["vol"].values
        interp = LinearNDInterpolator(points, vals)
        return cls(df["tenor"].values, df["strike"].values, vals, interp)

    def vol(self, tenor: float, strike: float) -> float:
        v = self.interpolator(tenor, strike)
        if v is None or np.isnan(v):
            # fallback to nearest point
            d = (self.tenor_arr - tenor)**2 + (self.strike_arr - strike)**2
            i = int(np.argmin(d))
            return float(self.vol_arr[i])
        return float(v)

# --- pricing functions ---

def kirk_cso_price(F1, F2, K, sigma1, sigma2, rho, T):
    # Implementation adapted from classic Kirk style approximation.
    # Add guards for edge cases.
    if T <= 0:
        return max(F1 - F2 - K, 0.0)
    # Avoid division by zero
    denom = (F1 + F2) if (F1 + F2) != 0 else 1e-8
    try:
        sigma_kirk_sq = (
            sigma1**2
            - 2 * rho * sigma1 * sigma2 * (F2 / denom)
            + sigma2**2 * (F2 / denom)**2
        )
        sigma_kirk_sq = max(sigma_kirk_sq, 1e-12)
        sigma_kirk = math.sqrt(sigma_kirk_sq)
        d1 = (math.log((F1 + F2) / (K if K > 0 else 1e-8)) + 0.5 * sigma_kirk_sq * T) / (sigma_kirk * math.sqrt(T))
        d2 = d1 - sigma_kirk * math.sqrt(T)
        price = (F1 + F2) * norm.cdf(d1) - K * norm.cdf(d2)
        return max(price, 0.0)
    except Exception as e:
        # fallback to intrinsic
        return max(F1 - F2 - K, 0.0)

def generate_correlated_gbm_paths(S1, S2, mu1, mu2, sigma1, sigma2, rho, T, steps, n_sims, rng=None):
    # returns (n_sims, 2) terminal prices using correlated GBM
    if rng is None:
        rng = np.random.default_rng(0)
    dt = T / steps
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    S1_paths = np.full(n_sims, S1, dtype=float)
    S2_paths = np.full(n_sims, S2, dtype=float)
    for _ in range(steps):
        z = rng.standard_normal((n_sims, 2))
        corr_z = z @ chol.T  # shape (n_sims, 2)
        S1_paths *= np.exp((mu1 - 0.5 * sigma1**2) * dt + sigma1 * np.sqrt(dt) * corr_z[:, 0])
        S2_paths *= np.exp((mu2 - 0.5 * sigma2**2) * dt + sigma2 * np.sqrt(dt) * corr_z[:, 1])
    return np.vstack([S1_paths, S2_paths]).T

def mc_cso_price(F1, F2, K, sigma1, sigma2, rho, T, steps=50, n_sims=20000):
    if T <= 0:
        return max(F1 - F2 - K, 0.0)
    sims = generate_correlated_gbm_paths(F1, F2, 0.0, 0.0, sigma1, sigma2, rho, T, steps, n_sims)
    payoff = np.maximum(sims[:,0] - sims[:,1] - K, 0.0)
    return float(payoff.mean())

# --- core pipeline functions ---

def load_all():
    print("Loading synthetic data from", SYN_DIR)
    spot_df = safe_read_csv(SYN_DIR / "spot_series.csv")
    hh_fut = safe_read_csv(SYN_DIR / "hh_futures_curves.csv")
    wti_fut = safe_read_csv(SYN_DIR / "wti_futures_curves.csv")
    corr = safe_read_csv(SYN_DIR / "correlation.csv")
    vol_hh = safe_read_csv(SYN_DIR / "vol_surface_hh.csv")
    vol_wti = safe_read_csv(SYN_DIR / "vol_surface_wti.csv")
    cso_chain = safe_read_csv(SYN_DIR / "cso_chain.csv")
    return spot_df, hh_fut, wti_fut, corr, vol_hh, vol_wti, cso_chain

def build_vol_surfaces(vol_hh_df, vol_wti_df):
    vs_hh = VolSurface.from_df(vol_hh_df)
    vs_wti = VolSurface.from_df(vol_wti_df)
    return vs_hh, vs_wti

def price_cso_grid(cso_chain_df, vol_hh: VolSurface, vol_wti: VolSurface, method="kirk"):
    rows = []
    for idx, r in cso_chain_df.iterrows():
        T = float(r["tenor"])
        K = float(r["strike"])
        # for synthetic data we take spot proxies, but here we use fixed forwards from chain example
        # estimate F1 and F2 from sample: use last known spot from spot_series if present
        # fallback constants
        F1 = 3.0
        F2 = 65.0
        # choose strike mapping for vol query: use K as moneyness proxy
        vol1 = vol_hh.vol(T, K) if isinstance(vol_hh, VolSurface) else float(r.get("vol1", 0.6))
        vol2 = vol_wti.vol(T, K) if isinstance(vol_wti, VolSurface) else float(r.get("vol2", 0.3))
        rho = 0.25  # default, can be replaced by time series lookup
        if method == "kirk":
            price = kirk_cso_price(F1, F2, K, vol1, vol2, rho, T)
        else:
            price = mc_cso_price(F1, F2, K, vol1, vol2, rho, T, steps=50, n_sims=5000)
        rows.append({"tenor": T, "strike": K, "method": method, "price": price, "vol1": vol1, "vol2": vol2, "rho": rho})
    return pd.DataFrame(rows)

def compute_realized_corr(spot_df, window=60):
    # Use log returns on spot series if available
    df = spot_df.copy()
    # identify columns containing hh and wti spot names
    hh_col = next((c for c in df.columns if "hh" in c.lower()), None)
    wti_col = next((c for c in df.columns if "wti" in c.lower()), None)
    if hh_col is None or wti_col is None:
        raise ValueError("Spot columns not found in spot_series.csv")
    df["r1"] = np.log(df[hh_col]).diff()
    df["r2"] = np.log(df[wti_col]).diff()
    df["realized_corr"] = df["r1"].rolling(window).corr(df["r2"])
    out = df[["date", "realized_corr"]].dropna().reset_index(drop=True)
    return out

def generate_rv_signal_and_backtest(priced_df):
    # naive relative value strategy:
    # signal = theoretical - market_price. Here we treat kirk price as theoretical and use cso_chain 'theoretical' as proxy market
    # For synthetic data cso_chain had 'theoretical' column. We'll align by tenor and strike.
    price_kirk = priced_df[priced_df["method"] == "kirk"].copy()
    # load actual synthetic market from cso_chain
    market = safe_read_csv(SYN_DIR / "cso_chain.csv")
    merged = price_kirk.merge(market, on=["tenor", "strike"], how="left", suffixes=("_kirk", "_market"))
    merged["market_price"] = merged["theoretical"].fillna(merged["price"])
    merged["signal"] = - merged["price"] + merged["market_price"]  # positive means overpriced theoretical
    # simple backtest over synthetic dates, treat each row as a trade instance
    # we create synthetic PnL by assuming we trade at market_price, close at next synthetic repricing which we simulate as small move
    pnl_rows = []
    rng = np.random.default_rng(123)
    for i, r in merged.iterrows():
        # position = -signal (if price > market, short the CSO)
        pos = -r["signal"]
        # simulate daily return of underlying spread via small random shock
        spread_move = rng.normal(0, 0.2)  # synthetic move
        # approximate PnL = pos * spread_move
        pnl = pos * spread_move
        pnl_rows.append({
            "tenor": r["tenor"],
            "strike": r["strike"],
            "pos": pos,
            "pnl": pnl,
            "price_kirk": r["price"],
            "market_price": r["market_price"]
        })
    pnl_df = pd.DataFrame(pnl_rows)
    # stats
    total = pnl_df["pnl"].sum()
    vol = pnl_df["pnl"].std()
    sharpe = (pnl_df["pnl"].mean() / (vol if vol > 0 else 1e-8)) * math.sqrt(252)
    stats = {"total_pnl": float(total), "vol": float(vol), "sharpe": float(sharpe)}
    return pnl_df, stats

def quick_plots(priced_kirk, priced_mc, realized_corr_df, pnl_df, stats):
    # price grid scatter
    fig, ax = plt.subplots(figsize=(6,4))
    pk = priced_kirk
    ax.scatter(pk["strike"], pk["price"], label="kirk price")
    pm = priced_mc
    ax.scatter(pm["strike"], pm["price"], marker='x', label="mc price")
    ax.set_xlabel("strike")
    ax.set_ylabel("price")
    ax.legend()
    fig.suptitle("CSO Prices by Strike")
    fig.savefig(PLOTS_DIR / "cso_prices_by_strike.png")
    plt.close(fig)

    # realized corr plot
    fig, ax = plt.subplots(figsize=(8,3))
    rc = realized_corr_df.copy()
    rc["date"] = pd.to_datetime(rc["date"])
    ax.plot(rc["date"], rc["realized_corr"])
    ax.set_title("Realized Correlation")
    ax.set_xlabel("date")
    ax.set_ylim(-1,1)
    fig.savefig(PLOTS_DIR / "realized_corr.png")
    plt.close(fig)

    # pnl histogram
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(pnl_df["pnl"], bins=30)
    ax.set_title(f"PnL Distribution, Sharpe {stats['sharpe']:.2f}")
    fig.savefig(PLOTS_DIR / "pnl_hist.png")
    plt.close(fig)

# --- main ---

def main():
    print("Starting pipeline")
    spot_df, hh_fut, wti_fut, corr_df, vol_hh_df, vol_wti_df, cso_chain = load_all()
    vs_hh, vs_wti = build_vol_surfaces(vol_hh_df, vol_wti_df)

    print("Pricing CSO grid with Kirk")
    priced_kirk = price_cso_grid(cso_chain, vs_hh, vs_wti, method="kirk")
    priced_kirk.to_csv(OUT_DIR / "priced_cso_kirk.csv", index=False)

    print("Pricing CSO grid with Monte Carlo")
    priced_mc = price_cso_grid(cso_chain, vs_hh, vs_wti, method="mc")
    priced_mc.to_csv(OUT_DIR / "priced_cso_mc.csv", index=False)

    print("Computing realized correlation")
    realized_corr_df = compute_realized_corr(spot_df, window=60)
    realized_corr_df.to_csv(OUT_DIR / "realized_corr.csv", index=False)

    print("Running simple relative value backtest")
    pnl_df, stats = generate_rv_signal_and_backtest(pd.concat([priced_kirk, priced_mc], ignore_index=True))
    pnl_df.to_csv(OUT_DIR / "backtest_pnl.csv", index=False)
    # write quick summary
    with open(OUT_DIR / "backtest_stats.txt", "w") as f:
        f.write(str(stats))

    print("Generating quick plots")
    quick_plots(priced_kirk, priced_mc, realized_corr_df, pnl_df, stats)

    print("Pipeline complete. Outputs in", OUT_DIR)
    print("Backtest stats:", stats)

if __name__ == "__main__":
    main()
