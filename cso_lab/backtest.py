"""
Strategy backtesting framework for relative value trading strategies.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List


class Backtest:
    def __init__(self, prices: pd.Series, fair_values: pd.Series):
        """
        Initialize the backtest with market prices and fair values.

        Args:
            prices: Series of market prices with datetime index
            fair_values: Series of model-implied fair values with same index as prices
        """
        self.prices = prices
        self.fair_values = fair_values
        self.positions = None
        self.pnl = None
        self.stats = None
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input data consistency."""
        if not isinstance(self.prices, pd.Series) or not isinstance(self.fair_values, pd.Series):
            raise ValueError("Both prices and fair_values must be pandas Series")
        if len(self.prices) != len(self.fair_values):
            raise ValueError("prices and fair_values must have the same length")
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            self.prices.index = pd.to_datetime(self.prices.index)
        if not isinstance(self.fair_values.index, pd.DatetimeIndex):
            self.fair_values.index = pd.to_datetime(self.fair_values.index)

    def generate_signals(self, z_score_threshold: float = 1.0) -> pd.Series:
        """
        Generate trading signals based on the difference between market prices and fair values.

        Args:
            z_score_threshold: Number of standard deviations for signal thresholding

        Returns:
            Series of positions (-1, 0, 1)
        """
        # Calculate mispricing
        mispricing = self.prices - self.fair_values

        # Calculate z-scores of mispricing
        rolling_mean = mispricing.rolling(window=20, min_periods=10).mean()
        rolling_std = mispricing.rolling(window=20, min_periods=10).std()
        z_scores = (mispricing - rolling_mean) / rolling_std

        # Generate positions
        positions = pd.Series(0, index=self.prices.index)
        positions[z_scores > z_score_threshold] = -1  # Overpriced: short
        positions[z_scores < -z_score_threshold] = 1  # Underpriced: long

        # Ensure no new positions in the last period (can't exit)
        positions.iloc[-1] = 0

        self.positions = positions
        return positions

    def calculate_pnl(self, transaction_costs: float = 0.0) -> pd.Series:
        """
        Calculate mark-to-market PnL based on positions and price changes.

        Args:
            transaction_costs: Per-trade transaction cost as a fraction of notional

        Returns:
            Series of daily PnL
        """
        if self.positions is None:
            raise ValueError("Positions not generated. Call generate_signals() first.")

        # Calculate price returns
        returns = self.prices.pct_change()

        # Calculate PnL from position changes and price moves
        position_changes = self.positions.diff().abs()
        trade_costs = position_changes * self.prices * transaction_costs

        # PnL from holding positions and paying transaction costs
        self.pnl = (self.positions.shift(1) * returns * self.prices) - trade_costs
        self.pnl = self.pnl.fillna(0)

        return self.pnl

    def calculate_stats(self) -> Dict[str, float]:
        """Calculate performance statistics."""
        if self.pnl is None:
            raise ValueError("PnL not calculated. Call calculate_pnl() first.")

        # Calculate returns
        returns = self.pnl / self.prices.shift(1)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) < 2:
            return {"error": "Insufficient data for statistics"}

        # Calculate cumulative PnL
        cum_pnl = self.pnl.cumsum()
        total_return = cum_pnl.iloc[-1]

        # Calculate drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak

        # Calculate statistics
        self.stats = {
            "total_return": float(total_return),
            "total_return_pct": float(total_return / self.prices.iloc[0] * 100) if self.prices.iloc[0] != 0 else 0,
            "annualized_return": float(returns.mean() * 252),
            "annualized_vol": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0,
            "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0,
            "max_drawdown": float(drawdown.min()),
            "win_rate": float((returns > 0).mean()) if len(returns) > 0 else 0,
            "profit_factor": float(-returns[returns > 0].sum() / returns[returns < 0].sum()) if (returns < 0).any() and returns[returns < 0].sum() != 0 else np.inf,
            "num_trades": int((self.positions.diff().abs() > 0).sum() / 2)  # Count round trips
        }

        return self.stats

    def run(self, z_score_threshold: float = 1.0, transaction_costs: float = 0.001) -> Tuple[pd.Series, Dict]:
        """
        Run the complete backtest.

        Args:
            z_score_threshold: Number of standard deviations for signal thresholding
            transaction_costs: Per-trade transaction cost as a fraction of notional

        Returns:
            Tuple of (pnl_series, stats_dict)
        """
        self.generate_signals(z_score_threshold=z_score_threshold)
        self.calculate_pnl(transaction_costs=transaction_costs)
        stats = self.calculate_stats()

        return self.pnl, stats