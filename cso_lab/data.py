"""
Data loading and preprocessing utilities.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIRS = [
    _PROJECT_ROOT / "data" / "synthetic",
    _PROJECT_ROOT / "data" / "processed",
    _PROJECT_ROOT / "data" / "raw",
]


def _find_csv(name: str) -> Path:
    fname = name if name.lower().endswith(".csv") else f"{name}.csv"
    for d in _DATA_DIRS:
        p = d / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {fname} in {[_d.as_posix() for _d in _DATA_DIRS]}")


def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
    # find likely date column
    candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "timestamp"))]
    if candidates:
        col = candidates[0]
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].notna().any():
            df = df.sort_values(col).reset_index(drop=True)
            df.set_index(col, inplace=True)
    return df


def _ensure_numeric_column(df: pd.DataFrame, possible_names: List[str], standard_name: str) -> pd.DataFrame:
    for n in possible_names:
        if n in df.columns:
            df = df.rename(columns={n: standard_name})
            df[standard_name] = pd.to_numeric(df[standard_name], errors="coerce")
            return df
    # fall back: pick first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df = df.rename(columns={numeric_cols[0]: standard_name})
        return df
    raise ValueError(f"No numeric column found to use as `{standard_name}`. Expected one of {possible_names}.")


def _basic_clean(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    # drop exact duplicates and reset index if index is integer sequence
    df = df.drop_duplicates()
    missing = [c for c in required if c not in df.columns and c != "index"]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_spot(name: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load a spot time series CSV (e.g. `spot_series.csv`).
    Returns a DataFrame indexed by parsed date (if available) and normalizes the price column to `spot`.
    """
    p = _find_csv(name)
    df = pd.read_csv(p)
    df = _parse_date(df)
    df = _ensure_numeric_column(df, ["spot", "price", "value"], "spot")
    df = _basic_clean(df, required=["spot"])
    if verbose:
        print(f"Loaded {p} -> {df.shape[0]} rows, columns: {list(df.columns)}")
        print("Missing values per column:")
        print(df.isna().sum())
    return df


# python
def load_futures(name, verbose: bool = False):
    p = _find_csv(name)
    df = pd.read_csv(p)
    # normalize/parse date column (accept Date, DATE, etc.)
    date_cols = [c for c in df.columns if c.lower() == "date"]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        if date_cols[0] != "date":
            df = df.rename(columns={date_cols[0]: "date"})

    # detect wide tenor-style columns like `tenor_1m`, `_2m`, `3m`, etc.
    import re
    tenor_re = re.compile(r"^(?:tenor_)?_?(\d+)m$", re.IGNORECASE)
    tenor_cols = [c for c in df.columns if tenor_re.match(c)]

    if tenor_cols:
        # melt wide -> long
        df_long = df.melt(id_vars=["date"], value_vars=tenor_cols, var_name="tenor", value_name="price")
        # normalize tenor -> e.g. 'tenor_1m' or '_2m' -> '1m'
        df_long["contract"] = df_long["tenor"].str.replace(r"^(?:tenor_)?_?", "", regex=True).str.lower()
        # ensure trailing 'm'
        df_long["contract"] = df_long["contract"].apply(lambda s: s if s.endswith("m") else f"{s}m")
        df = df_long[["date", "contract", "price"]]
    else:
        # try common alternative column names for contract column
        if "contract" not in df.columns:
            for alt in ("symbol", "ticker", "contract_id"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "contract"})
                    break

    # final basic clean and required columns
    df = _basic_clean(df, required=["date", "price", "contract"])
    if verbose:
        print(f"Loaded {p} -> {df.shape[0]} rows, columns: {list(df.columns)}")
    return df



def load_vol_surface(name: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load a vol-surface CSV. Expected columns: date, strike, expiry/tenor, vol/implied_vol/iv.
    Normalizes vol column to `iv`.
    """
    p = _find_csv(name)
    df = pd.read_csv(p)
    df = _parse_date(df)

    # strike
    strike_cols = [c for c in df.columns if "strike" in c.lower()]
    if strike_cols:
        df = df.rename(columns={strike_cols[0]: "strike"})
    else:
        raise ValueError("Vol surface requires a `strike` column.")

    # expiry/tenor column
    expiry_cols = [c for c in df.columns if any(k in c.lower() for k in ("expiry", "tenor", "maturity"))]
    if expiry_cols:
        df = df.rename(columns={expiry_cols[0]: "expiry"})

    df = _ensure_numeric_column(df, ["iv", "implied_vol", "vol", "impliedVol", "sigma"], "iv")
    required = ["strike", "iv"]
    if "expiry" not in df.columns:
        required.append("expiry")
    df = _basic_clean(df, required=required)
    if verbose:
        print(f"Loaded {p} -> {df.shape[0]} rows, columns: {list(df.columns)}")
        print("Missing values per column:")
        print(df.isna().sum())
    return df


def load_correlation(name: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load correlation timeseries or matrix CSV. If a `date`/time column exists it will be parsed and used as index.
    Returns a DataFrame (timeseries) suitable for quick checks.
    """
    p = _find_csv(name)
    df = pd.read_csv(p)
    df = _parse_date(df)

    # check at least one numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Correlation file contains no numeric columns.")
    df = _basic_clean(df, required=[])
    if verbose:
        print(f"Loaded {p} -> {df.shape[0]} rows, numeric cols: {numeric_cols[:10]}")
        print("Missing values per column:")
        print(df.isna().sum())
    return df
