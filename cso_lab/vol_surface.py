# python
"""
Volatility surface modeling and calibration.
Provides VolSurface.from_df(...) and vol(tenor, strike).
"""
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class VolSurface:
    """Interpolate implied vols by (tenor, strike) with nearest-neighbor fallback."""

    def __init__(self, linear_interp: LinearNDInterpolator, nearest_interp: NearestNDInterpolator):
        self.linear = linear_interp
        self.nearest = nearest_interp

    @classmethod
    def from_df(cls, df: pd.DataFrame, tenor_col: str = "tenor", strike_col: str = "strike", vol_col: str = "vol"):
        """
        Build VolSurface from a DataFrame with columns for tenor, strike and vol.
        Example columns: 'tenor' (in months or years), 'strike', 'vol'.
        """
        if tenor_col not in df.columns or strike_col not in df.columns or vol_col not in df.columns:
            raise ValueError(f"DataFrame must contain columns: {tenor_col}, {strike_col}, {vol_col}")

        pts = np.vstack([df[tenor_col].to_numpy(dtype=float), df[strike_col].to_numpy(dtype=float)]).T
        vals = df[vol_col].to_numpy(dtype=float)

        linear = LinearNDInterpolator(pts, vals, fill_value=np.nan)
        nearest = NearestNDInterpolator(pts, vals)
        return cls(linear, nearest)

    def vol(self, tenor, strike):
        """
        Query implied vol at (tenor, strike).
        Accepts scalars or array-like inputs (broadcastable). Returns scalar for scalar inputs,
        or ndarray with broadcast shape for array inputs.
        """
        # Broadcast inputs to common shape
        t_arr, k_arr = np.broadcast_arrays(np.asarray(tenor), np.asarray(strike))
        orig_shape = t_arr.shape

        # Build query points (N,2)
        pts = np.column_stack((t_arr.ravel().astype(float), k_arr.ravel().astype(float)))

        # Linear interpolation (may return nan for OOB)
        v = self.linear(pts)
        v = np.asarray(v).ravel()

        # Replace NaNs with nearest-neighbor values
        nan_mask = np.isnan(v)
        if nan_mask.any():
            v[nan_mask] = self.nearest(pts[nan_mask])

        # Return scalar if inputs were scalars
        if orig_shape == ():
            return float(v[0])
        return v.reshape(orig_shape)
