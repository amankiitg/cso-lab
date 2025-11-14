"""
Data loading and preprocessing utilities.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data" / "processed"

def load_spot(name):
    file = DATA_DIR / f"{name}_spot.csv"
    return pd.read_csv(file, parse_dates=["date"])

def load_futures(name):
    file = DATA_DIR / f"{name}_futures_curves.csv"
    return pd.read_csv(file, parse_dates=["date"])

def load_vol_surface(name):
    file = Path(__file__).parents[1] / "data" / "synthetic" / f"vol_surface_{name}.csv"
    return pd.read_csv(file)

def load_correlation():
    file = DATA_DIR / "corr_history.csv"
    return pd.read_csv(file, parse_dates=["date"])

