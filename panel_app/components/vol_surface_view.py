"""
Volatility surface view component.
"""
import panel as pn
import hvplot.pandas
from cso_lab.vol_surface import VolSurface
from cso_lab.data import load_vol_surface

vol_df = load_vol_surface("hh")
surface = VolSurface(vol_df)

vol_surface_view = pn.Column(
    "## Vol Surface Viewer",
    vol_df.hvplot.scatter(x="tenor", y="strike", c="vol", size=60, title="Vol Surface Points")
)
