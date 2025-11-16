# panel_app/components/vol_surface_view.py
import panel as pn
import numpy as np
import holoviews as hv
import hvplot.pandas
from pathlib import Path
import pandas as pd


class VolSurfaceView:
    def __init__(self, data_dir="data/synthetic"):
        self.data_dir = Path(data_dir)
        self.vol_data = None
        self._create_widgets()
        self._load_vol_data()
        self.layout = self._create_layout()

    def _create_widgets(self):
        """Create interactive widgets."""
        self.asset_selector = pn.widgets.Select(
            name='Underlying Asset',
            options=['HH', 'WTI'],
            value='HH'
        )
        self.surface_type = pn.widgets.Select(
            name='Surface Type',
            options=['Implied Volatility', 'Local Volatility'],
            value='Implied Volatility'
        )
        self.slice_tenor = pn.widgets.FloatSlider(
            name='Slice Tenor (years)',
            start=0.1, end=2.0, step=0.1, value=0.5
        )
        self.slice_strike = pn.widgets.FloatSlider(
            name='Slice Strike (%)',
            start=70, end=130, step=5, value=100
        )

    def _load_vol_data(self, asset='HH'):
        """Load or generate sample volatility surface data."""
        # In a real app, you would load this from a file
        tenors = np.linspace(0.1, 2.0, 20)
        strikes = np.linspace(70, 130, 20)
        tt, kk = np.meshgrid(tenors, strikes)

        # Simple parametric vol surface
        atm_vol = 0.3
        skew = -0.2
        term = 0.1
        vol_surface = atm_vol + skew * (kk / 100 - 1) + term * np.sqrt(tt)

        self.vol_data = pd.DataFrame({
            'tenor': tt.ravel(),
            'strike': kk.ravel(),
            'vol': np.maximum(0.1, np.minimum(1.0, vol_surface.ravel()))
        })

    def _create_3d_surface(self):
        """Create 3D surface plot using heatmap instead of surface."""
        if self.vol_data is None:
            return hv.Curve([]).opts(title="No data available")

        return self.vol_data.hvplot.heatmap(
            x='strike',
            y='tenor',
            C='vol',
            cmap='viridis',
            title=f"{self.asset_selector.value} Volatility Surface",
            height=400,
            width=600
        ).opts(
            xlabel='Strike',
            ylabel='Tenor (years)',
            colorbar=True,
            tools=['hover']
        )

    def _create_term_structure(self):
        """Create term structure plot."""
        if self.vol_data is None:
            return hv.Curve([]).opts(title="No data available")

        strike_pct = self.slice_strike.value
        df = self.vol_data[
            (self.vol_data['strike'] >= strike_pct * 0.99) &
            (self.vol_data['strike'] <= strike_pct * 1.01)
            ]

        if df.empty:
            return hv.Curve([]).opts(title="No data for selected strike")

        term_structure = df.groupby('tenor')['vol'].mean().reset_index()
        return term_structure.hvplot.line(
            x='tenor',
            y='vol',
            title=f'Term Structure at {strike_pct}% Strike',
            line_width=2,
            color='blue'
        ).opts(
            xlabel='Tenor (years)',
            ylabel='Volatility',
            ylim=(0, None)
        )

    def _create_vol_smile(self):
        """Create volatility smile plot."""
        if self.vol_data is None:
            return hv.Curve([]).opts(title="No data available")

        tenor = self.slice_tenor.value
        df = self.vol_data[
            (self.vol_data['tenor'] >= tenor * 0.95) &
            (self.vol_data['tenor'] <= tenor * 1.05)
            ]

        if df.empty:
            return hv.Curve([]).opts(title="No data for selected tenor")

        vol_smile = df.groupby('strike')['vol'].mean().reset_index()
        return vol_smile.hvplot.line(
            x='strike',
            y='vol',
            title=f'Volatility Smile at {tenor:.1f} Years',
            line_width=2,
            color='red'
        ).opts(
            xlabel='Strike',
            ylabel='Volatility',
            ylim=(0, None)
        )

    def _create_layout(self):
        """Create the dashboard layout."""
        # 3D surface plot
        surface_plot = pn.pane.HoloViews(
            self._create_3d_surface(),
            sizing_mode='stretch_both'
        )

        # Slices
        term_structure = pn.pane.HoloViews(
            self._create_term_structure(),
            sizing_mode='stretch_both'
        )

        vol_smile = pn.pane.HoloViews(
            self._create_vol_smile(),
            sizing_mode='stretch_both'
        )

        # Controls
        controls = pn.Card(
            self.asset_selector,
            self.surface_type,
            self.slice_tenor,
            self.slice_strike,
            title="Controls",
            collapsed=False
        )

        # Wire up callbacks for interactive updates
        self.slice_tenor.param.watch(
            lambda e: vol_smile.object.update(self._create_vol_smile()), 'value'
        )
        self.slice_strike.param.watch(
            lambda e: term_structure.object.update(self._create_term_structure()), 'value'
        )
        self.asset_selector.param.watch(
            lambda e: (
                self._load_vol_data(asset=e.new),
                surface_plot.object.update(self._create_3d_surface()),
                term_structure.object.update(self._create_term_structure()),
                vol_smile.object.update(self._create_vol_smile())
            ), 'value'
        )

        # Layout
        return pn.Row(
            controls,
            pn.Column(
                surface_plot,
                pn.Row(
                    term_structure,
                    vol_smile,
                    sizing_mode='stretch_both'
                ),
                sizing_mode='stretch_both'
            ),
            sizing_mode='stretch_both'
        )


# For testing the view standalone
if __name__ == '__main__':
    pn.extension(sizing_mode="stretch_width")
    app = VolSurfaceView()
    app.layout.servable()