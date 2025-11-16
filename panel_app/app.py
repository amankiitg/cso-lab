# panel_app/app.py
import panel as pn
from pathlib import Path

# Import views
from panel_app.components.pricing_view import PricingView
from panel_app.components.vol_surface_view import VolSurfaceView
from panel_app.components.backtest_view import BacktestView
from panel_app.components.layout import create_layout


def initialize_application():
    """Initialize the Panel application with all components."""
    # Initialize Panel with required extensions
    pn.extension(
        'tabulator',
        'plotly',
        'vega',
        'katex',
        'mathjax',
        sizing_mode="stretch_width",
        notifications=True
    )

    # Initialize views
    views = {
        'pricing': PricingView(),
        'vol_surface': VolSurfaceView(),
        'backtest': BacktestView()
    }

    # Create and return the configured template
    return create_layout(views)


def serve_application(port: int = 5006, show: bool = True, **kwargs):
    """Serve the Panel application.

    Args:
        port: Port to serve the application on
        show: Whether to open the app in a browser
        **kwargs: Additional arguments to pass to pn.serve
    """
    app = initialize_application()

    # Use a different template to avoid the template_resources error
    template = pn.template.FastListTemplate(
        title='CSO Analytics Dashboard',
        theme='default'
    )
    template.main.append(app)

    pn.serve(
        template,
        port=port,
        show=show,
        **{
            'title': 'CSO Analytics Dashboard',
            **kwargs
        }
    )


# For production with gunicorn or similar
# app = initialize_application()

# For development
if __name__ == '__main__':
    serve_application(autoreload=True)