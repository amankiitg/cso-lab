"""
Backtest view component.
"""
import panel as pn
import hvplot.pandas
from cso_lab.backtest import Backtest

import numpy as np
import pandas as pd

dates = pd.date_range("2020", periods=400)
prices = pd.Series(np.random.randn(400).cumsum() + 100, index=dates)
fair = pd.Series(prices.rolling(20).mean().fillna(method="bfill"), index=dates)

bt = Backtest(prices, fair)
pnl, stats = bt.run()

backtest_view = pn.Column(
    "## Backtest Viewer",
    pnl.hvplot(title="Strategy PnL"),
    pn.pane.Markdown(f"Sharpe: {round(stats['sharpe'], 2)}")
)
