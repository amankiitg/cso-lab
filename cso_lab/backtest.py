"""
Strategy backtesting framework.
"""
import numpy as np
import pandas as pd

class Backtest:
    def __init__(self, prices, fair_values):
        self.prices = prices
        self.fair_values = fair_values

    def run(self):
        signal = self.prices - self.fair_values
        pnl = (signal.shift(1) * (self.prices.diff())).fillna(0)
        stats = {
            "total_return": pnl.sum(),
            "vol": pnl.std() * (252 ** 0.5),
            "sharpe": pnl.mean() / pnl.std() * np.sqrt(252)
        }
        return pnl, stats
