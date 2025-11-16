# panel_app/app.py
import os
import sys
from pathlib import Path
import panel as pn

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import views using absolute imports
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
    # Initialize the application
    app = initialize_application()
    
    # Create and configure the template
    template = pn.template.FastListTemplate(
        title='CSO Analytics Dashboard',
        theme='default'
    )
    
    # Add the app to the main area of the template
    template.main.append(app)
    
    # Mark the template as servable
    template.servable()
    
    # For development, we'll also return the template
    if pn.state.served:
        return template
        
    # For panel serve
    return pn.panel(template).servable()


# For gunicorn or similar
app = initialize_application()

if __name__.startswith('bokeh_app') or __name__ == "__main__":
    # For development and when run directly
    if __name__ == "__main__":
        serve_application(show=True, autoreload=True)
    # For bokeh serve
    else:
        app.servable()