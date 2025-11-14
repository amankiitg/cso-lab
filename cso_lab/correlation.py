"""
Correlation estimation and forecasting.
"""

import numpy as np
import pandas as pd

def compute_rolling_corr(series1, series2, window=60):
    return series1.rolling(window).corr(series2)


def rolling_correlation(returns1, returns2, window=30):
    """Calculate rolling correlation between two return series."""
    pass


def ewma_correlation(returns1, returns2, lambda_=0.94):
    """Calculate EWMA correlation."""
    pass


def forecast_correlation(historical_corr, horizon=1):
    """Forecast future correlation."""
    pass
