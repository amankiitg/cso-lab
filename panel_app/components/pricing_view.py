"""
CSO pricing view component.
"""

import panel as pn
import numpy as np
from cso_lab.pricing import kirk_cso_price

underlyings = ["hh", "wti"]

F1 = pn.widgets.FloatInput(name="F1", value=3.0)
F2 = pn.widgets.FloatInput(name="F2", value=70.0)
K = pn.widgets.FloatInput(name="K", value=0.0)
sigma1 = pn.widgets.FloatInput(name="Vol 1", value=0.6)
sigma2 = pn.widgets.FloatInput(name="Vol 2", value=0.3)
rho = pn.widgets.FloatInput(name="Correlation", value=0.2)
T = pn.widgets.FloatInput(name="Tenor", value=0.5)

@pn.depends(F1, F2, K, sigma1, sigma2, rho, T)
def price_output(F1, F2, K, sigma1, sigma2, rho, T):
    price = kirk_cso_price(F1, F2, K, sigma1, sigma2, rho, T)
    return pn.pane.Markdown(f"### CSO Price: {round(price, 4)}")

pricing_view = pn.Column(
    "## Pricing Explorer",
    F1, F2, K, sigma1, sigma2, rho, T,
    price_output
)
