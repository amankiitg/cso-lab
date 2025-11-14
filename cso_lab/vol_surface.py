"""
Volatility surface modeling and calibration.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

class VolSurface:
    def __init__(self, df):
        self.interpolator = LinearNDInterpolator(
            list(zip(df["tenor"], df["strike"])),
            df["vol"]
        )

    def vol(self, tenor, strike):
        return float(self.interpolator(tenor, strike))
