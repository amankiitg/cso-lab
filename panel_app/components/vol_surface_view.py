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
            start=70, end=130, step=5, value=95
        )

    def _load_vol_data(self, asset='HH'):
        """Load or generate sample volatility surface data."""
        tenors = np.linspace(0.1, 2.0, 20)
        strikes = np.linspace(70, 130, 20)
        tt, kk = np.meshgrid(tenors, strikes)

        # Base parameters - different for WTI and HH
        if asset == 'WTI':
            atm_vol = 0.35  # Higher base vol for WTI
            skew = -0.25  # More negative skew for WTI
            term = 0.12  # Stronger term structure for WTI
        else:  # HH
            atm_vol = 0.28  # Lower base vol for HH
            skew = -0.15  # Less negative skew for HH
            term = 0.08  # Weaker term structure for HH

        if self.surface_type.value == 'Implied Volatility':
            # Flatter surface for implied vol
            vol_surface = atm_vol + skew * 0.7 * (kk / 100 - 1) + term * np.sqrt(tt)
        else:
            # More pronounced surface for local vol
            vol_surface = atm_vol + skew * 1.3 * (kk / 100 - 1) + term * 1.5 * np.sqrt(tt)

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

    def _update_plots(self):
        """Update all plots when parameters change."""
        self.surface_plot.object = self._create_3d_surface()
        self.term_structure.object = self._create_term_structure()
        self.vol_smile.object = self._create_vol_smile()

    def _create_layout(self):
        """Create the dashboard layout."""
        # Create plot containers
        self.surface_plot = pn.pane.HoloViews(
            self._create_3d_surface(),
            sizing_mode='stretch_both'
        )

        self.term_structure = pn.pane.HoloViews(
            self._create_term_structure(),
            sizing_mode='stretch_both'
        )

        self.vol_smile = pn.pane.HoloViews(
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
        for widget in [self.slice_tenor, self.slice_strike, self.asset_selector, self.surface_type]:
            widget.param.watch(lambda _: self._update_plots(), 'value')

        # Layout
        return pn.Row(
            controls,
            pn.Column(
                self.surface_plot,
                pn.Row(
                    self.term_structure,
                    self.vol_smile,
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