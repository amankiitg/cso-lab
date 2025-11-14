"""
Main Panel dashboard application.
"""
import panel as pn
from components.pricing_view import pricing_view
from components.vol_surface_view import vol_surface_view
from components.backtest_view import backtest_view
from components.layout import create_layout

pn.extension(sizing_mode="stretch_width")

tabs = pn.Tabs(
    ("Pricing", pricing_view),
    ("Vol Surfaces", vol_surface_view),
    ("Backtest", backtest_view)
)

app = create_layout(tabs)
app.servable()
