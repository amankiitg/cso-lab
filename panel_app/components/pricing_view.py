# panel_app/components/pricing_view.py
import panel as pn
import numpy as np
import holoviews as hv
from holoviews import opts
from cso_lab.pricing import kirk_cso_price

# Enable Bokeh and Panel extensions
pn.extension()
hv.extension('bokeh')


class PricingView:
    def __init__(self):
        # Default parameters
        self.default_params = {
            'F1':66.0,  # Forward price asset 1
            'F2': 45.0,  # Forward price asset 2
            'K': 5.0,  # Strike price
            'sigma1': 0.3,  # Volatility asset 1
            'sigma2': 0.25,  # Volatility asset 2
            'rho': 0.2,  # Correlation
            'T': 1.0,  # Time to maturity (years)
        }

        # Create widgets and layout
        self._create_widgets()
        self._setup_callbacks()
        self.layout = self._create_layout()

    def _create_widgets(self):
        """Create interactive widgets."""
        # Sliders for option parameters
        self.F1_slider = pn.widgets.FloatSlider(
            name='F1 (Asset 1 Forward Price)',
            start=10.0, end=100.0, step=1.0,
            value=self.default_params['F1']
        )
        self.F2_slider = pn.widgets.FloatSlider(
            name='F2 (Asset 2 Forward Price)',
            start=5.0, end=90.0, step=1.0,
            value=self.default_params['F2']
        )
        self.K_slider = pn.widgets.FloatSlider(
            name='K (Strike Price)',
            start=0.1, end=20.0, step=0.5,
            value=self.default_params['K']
        )
        self.sigma1_slider = pn.widgets.FloatSlider(
            name='σ₁ (Volatility Asset 1)',
            start=0.05, end=1.0, step=0.05,
            value=self.default_params['sigma1']
        )
        self.sigma2_slider = pn.widgets.FloatSlider(
            name='σ₂ (Volatility Asset 2)',
            start=0.05, end=1.0, step=0.05,
            value=self.default_params['sigma2']
        )
        self.rho_slider = pn.widgets.FloatSlider(
            name='ρ (Correlation)',
            start=-0.99, end=0.99, step=0.05,
            value=self.default_params['rho']
        )
        self.T_slider = pn.widgets.FloatSlider(
            name='T (Time to Maturity)',
            start=0.1, end=5.0, step=0.1,
            value=self.default_params['T']
        )

        # Output displays
        self.price_display = pn.indicators.Number(
            name='Kirk Option Price',
            value=0.0,
            format='{value:.4f}',
            colors=[(0.8, 'red'), (1, 'green')]
        )
        self.intrinsic_display = pn.indicators.Number(
            name='Intrinsic Value',
            value=0.0,
            format='{value:.4f}',
            colors=[(0.8, 'red'), (1, 'green')]
        )
        self.time_value_display = pn.indicators.Number(
            name='Time Value',
            value=0.0,
            format='{value:.4f}',
            colors=[(0.5, 'orange'), (1, 'blue')]
        )

        # Sensitivity plot
        self.sensitivity_plot = hv.Curve([]).opts(
            title='Price Sensitivity',
            xlabel='Parameter Value',
            ylabel='Option Price',
            width=600,
            height=400
        )

    def _setup_callbacks(self):
        """Set up widget callbacks."""
        # Update on any slider change
        for widget in [self.F1_slider, self.F2_slider, self.K_slider,
                       self.sigma1_slider, self.sigma2_slider,
                       self.rho_slider, self.T_slider]:
            widget.param.watch(self._update, 'value')

        # Initial update
        self._update()

    def _calculate_price(self, **params):
        """Calculate option price with given parameters."""
        try:
            return kirk_cso_price(
                F1=params['F1'],
                F2=params['F2'],
                K=params['K'],
                sigma1=params['sigma1'],
                sigma2=params['sigma2'],
                rho=params['rho'],
                T=params['T']
            )
        except Exception as e:
            print(f"Error calculating price: {e}")
            return 0.0

    def _calculate_intrinsic(self, F1, F2, K):
        """Calculate intrinsic value of the spread option."""
        return max(F1 - F2 - K, 0)

    def _create_sensitivity_plot(self, base_params, param_name, param_range):
        """Create sensitivity plot for a parameter."""
        prices = []
        for val in param_range:
            params = base_params.copy()
            params[param_name] = val
            prices.append(self._calculate_price(**params))

        return hv.Curve(
            (param_range, prices),
            param_name,
            'Option Price'
        ).opts(
            title=f'Price Sensitivity to {param_name}',
            xlabel=param_name,
            ylabel='Option Price',
            line_width=2,
            color='#1f77b4'
        )

    def _update(self, *events):
        """Update displays and plots when parameters change."""
        # Get current parameters
        params = {
            'F1': self.F1_slider.value,
            'F2': self.F2_slider.value,
            'K': self.K_slider.value,
            'sigma1': self.sigma1_slider.value,
            'sigma2': self.sigma2_slider.value,
            'rho': self.rho_slider.value,
            'T': self.T_slider.value
        }

        # Calculate and update prices
        price = self._calculate_price(**params)
        intrinsic = self._calculate_intrinsic(params['F1'], params['F2'], params['K'])
        time_value = max(0, price - intrinsic)

        self.price_display.value = price
        self.intrinsic_display.value = intrinsic
        self.time_value_display.value = time_value

        # Update sensitivity plot
        param_name = 'rho'  # Default sensitivity parameter
        param_range = np.linspace(-0.99, 0.99, 50)

        if events:
            # If a slider was changed, show sensitivity to that parameter
            widget = events[0].obj
            if widget == self.F1_slider:
                param_name = 'F1'
                param_range = np.linspace(10, 100, 50)
            elif widget == self.F2_slider:
                param_name = 'F2'
                param_range = np.linspace(5, 90, 50)
            elif widget == self.K_slider:
                param_name = 'K'
                param_range = np.linspace(0.1, 20, 50)
            elif widget == self.sigma1_slider:
                param_name = 'sigma1'
                param_range = np.linspace(0.05, 1.0, 50)
            elif widget == self.sigma2_slider:
                param_name = 'sigma2'
                param_range = np.linspace(0.05, 1.0, 50)
            elif widget == self.rho_slider:
                param_name = 'rho'
                param_range = np.linspace(-0.99, 0.99, 50)
            elif widget == self.T_slider:
                param_name = 'T'
                param_range = np.linspace(0.1, 5.0, 50)

        self.sensitivity_plot = self._create_sensitivity_plot(
            params, param_name, param_range
        )

    def _create_layout(self):
        """Create the dashboard layout."""
        # Parameter sliders in a card
        params_card = pn.Card(
            pn.Column(
                self.F1_slider,
                self.F2_slider,
                self.K_slider,
                self.sigma1_slider,
                self.sigma2_slider,
                self.rho_slider,
                self.T_slider,
                sizing_mode='stretch_width'
            ),
            title='Option Parameters',
            collapsed=False
        )

        # Price display in a card
        price_card = pn.Card(
            pn.Column(
                self.price_display,
                self.intrinsic_display,
                self.time_value_display
            ),
            title='Option Valuation',
            collapsed=False
        )

        # Main layout
        return pn.Row(
            pn.Column(
                pn.Row(
                    params_card,
                    price_card,
                    sizing_mode='stretch_width'
                ),
                pn.Row(
                    pn.pane.HoloViews(
                        self.sensitivity_plot,
                        sizing_mode='stretch_both'
                    )
                ),
                sizing_mode='stretch_both'
            ),
            sizing_mode='stretch_both'
        )


# For testing the view standalone
if __name__ == '__main__':
    pn.extension(sizing_mode="stretch_width")
    app = PricingView()
    app.layout.servable()