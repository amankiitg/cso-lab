#!/usr/bin/env python3
"""
CSO Lab Project Structure Generator
Creates the complete directory structure and initial files for the CSO Lab project.
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create all directories for the CSO Lab project."""

    directories = [
        # Data directories
        'data/raw',
        'data/processed',
        'data/synthetic',

        # Notebooks
        'notebooks',

        # Main package
        'cso_lab',

        # Panel app
        'panel_app/components',
        'panel_app/assets',
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}/")


def create_root_files():
    """Create root-level configuration files."""

    # README.md
    readme_content = """# CSO Lab

A comprehensive toolkit for Commodity Spread Option (CSO) analysis, pricing, and trading strategy development.

## Overview

This project provides tools for analyzing and pricing CSOs between Henry Hub Natural Gas and WTI Crude Oil, including:

- Volatility surface modeling and calibration
- Multiple CSO pricing methods (Kirk's approximation, Monte Carlo, numerical integration)
- Correlation analysis and forecasting
- Strategy backtesting framework
- Interactive dashboard for real-time analysis

## Project Structure

```
cso-lab/
├── data/              # Raw, processed, and synthetic data
├── notebooks/         # Jupyter notebooks for research and analysis
├── cso_lab/          # Core Python package
└── panel_app/        # Interactive web dashboard
```

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- Panel (for dashboard)

## Usage

### Python API

```python
from cso_lab import pricing, vol_surface

# Price a CSO
option_price = pricing.kirk_approximation(
    S1=3.50, S2=75.0, K=0, T=0.25,
    sigma1=0.30, sigma2=0.25, rho=0.40, r=0.05
)
```

### Dashboard

```bash
panel serve panel_app/app.py
```

## License

MIT License
"""

    # requirements.txt
    requirements_content = """numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyterlab>=3.0.0
panel>=0.14.0
hvplot>=0.8.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
"""

    # setup.py
    setup_content = """from setuptools import setup, find_packages

setup(
    name='cso-lab',
    version='0.1.0',
    description='Commodity Spread Option Analysis and Pricing Toolkit',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'panel>=0.14.0',
        'hvplot>=0.8.0',
        'scikit-learn>=1.0.0',
        'statsmodels>=0.13.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
"""

    # .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/
.eggs/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
*.csv
*.xlsx
*.parquet
!data/README.md

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
"""

    files = {
        'README.md': readme_content,
        'requirements.txt': requirements_content,
        'setup.py': setup_content,
        '.gitignore': gitignore_content,
    }

    for filename, content in files.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✓ Created file: {filename}")


def create_data_placeholders():
    """Create placeholder README files in data directories."""

    data_readme = """# Data Directory

## Raw Data
Place raw market data files here:
- `hh_spot_raw.csv`: Henry Hub spot prices
- `wti_spot_raw.csv`: WTI spot prices
- `hh_futures_raw.csv`: Henry Hub futures curves
- `wti_futures_raw.csv`: WTI futures curves
- `correlation_raw.csv`: Historical correlation data

## Processed Data
Cleaned and processed data will be stored here.

## Synthetic Data
Generated synthetic data for testing and development.
"""

    for subdir in ['raw', 'processed', 'synthetic']:
        readme_path = Path('data') / subdir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(data_readme)
        print(f"✓ Created: {readme_path}")


def create_notebooks():
    """Create notebook placeholders with basic structure."""

    notebooks = {
        '01_data_cleaning.ipynb': 'Data Cleaning and Preprocessing',
        '02_vol_surface_fit.ipynb': 'Volatility Surface Fitting',
        '03_cso_pricing_methods.ipynb': 'CSO Pricing Methods Comparison',
        '04_correlation_research.ipynb': 'Correlation Analysis and Forecasting',
        '05_strategy_backtest.ipynb': 'Strategy Backtesting',
    }

    for filename, title in notebooks.items():
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {title}\n\n"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "\n",
                        "from cso_lab import *\n",
                        "\n",
                        "%matplotlib inline\n",
                        "sns.set_style('whitegrid')"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        import json
        notebook_path = Path('notebooks') / filename
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        print(f"✓ Created: {notebook_path}")


def create_package_files():
    """Create Python package files with basic structure."""

    package_files = {
        '__init__.py': '''"""
CSO Lab - Commodity Spread Option Analysis Toolkit
"""

__version__ = '0.1.0'

from . import data
from . import models
from . import pricing
from . import vol_surface
from . import correlation
from . import backtest
from . import plotting
from . import utils
''',

        'data.py': '''"""
Data loading and preprocessing utilities.
"""

import pandas as pd
from pathlib import Path


def load_spot_data(commodity: str) -> pd.DataFrame:
    """Load spot price data for a commodity."""
    pass


def load_futures_data(commodity: str) -> pd.DataFrame:
    """Load futures curve data for a commodity."""
    pass


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw market data."""
    pass
''',

        'models.py': '''"""
Market models and stochastic process implementations.
"""

import numpy as np


class GeometricBrownianMotion:
    """Geometric Brownian Motion model."""

    def __init__(self, S0, mu, sigma):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T, n_steps, n_paths=1):
        """Simulate price paths."""
        pass
''',

        'pricing.py': '''"""
CSO pricing methods.
"""

import numpy as np
from scipy import stats


def kirk_approximation(S1, S2, K, T, sigma1, sigma2, rho, r=0):
    """
    Price a CSO using Kirk's approximation.

    Parameters
    ----------
    S1, S2 : float
        Current prices of underlying assets
    K : float
        Strike price (spread)
    T : float
        Time to maturity (years)
    sigma1, sigma2 : float
        Volatilities
    rho : float
        Correlation
    r : float
        Risk-free rate

    Returns
    -------
    float
        Option premium
    """
    pass


def monte_carlo_pricing(S1, S2, K, T, sigma1, sigma2, rho, r=0, n_sims=10000):
    """Price CSO using Monte Carlo simulation."""
    pass
''',

        'vol_surface.py': '''"""
Volatility surface modeling and calibration.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator


class VolatilitySurface:
    """Volatility surface representation and interpolation."""

    def __init__(self):
        self.strikes = None
        self.maturities = None
        self.vols = None

    def fit(self, strikes, maturities, vols):
        """Fit surface to market data."""
        pass

    def get_vol(self, strike, maturity):
        """Interpolate volatility at given strike and maturity."""
        pass
''',

        'correlation.py': '''"""
Correlation estimation and forecasting.
"""

import numpy as np
import pandas as pd


def rolling_correlation(returns1, returns2, window=30):
    """Calculate rolling correlation between two return series."""
    pass


def ewma_correlation(returns1, returns2, lambda_=0.94):
    """Calculate EWMA correlation."""
    pass


def forecast_correlation(historical_corr, horizon=1):
    """Forecast future correlation."""
    pass
''',

        'backtest.py': '''"""
Strategy backtesting framework.
"""

import pandas as pd
import numpy as np


class Backtester:
    """Backtesting engine for CSO strategies."""

    def __init__(self, data):
        self.data = data
        self.positions = None
        self.pnl = None

    def run(self, strategy):
        """Execute backtest."""
        pass

    def calculate_metrics(self):
        """Calculate performance metrics."""
        pass
''',

        'plotting.py': '''"""
Visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_vol_surface(surface, title="Volatility Surface"):
    """Plot 3D volatility surface."""
    pass


def plot_correlation_history(corr_series, title="Correlation History"):
    """Plot correlation time series."""
    pass


def plot_pnl(pnl_series, title="Strategy P&L"):
    """Plot strategy P&L."""
    pass
''',

        'utils.py': '''"""
Utility functions.
"""

import numpy as np


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    pass


def implied_volatility(price, S, K, T, r):
    """Calculate implied volatility from option price."""
    pass


def annualize_returns(returns, periods_per_year=252):
    """Annualize return series."""
    pass
''',
    }

    for filename, content in package_files.items():
        filepath = Path('cso_lab') / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Created: {filepath}")


def create_panel_app():
    """Create Panel dashboard application files."""

    app_files = {
        'app.py': '''"""
Main Panel dashboard application.
"""

import panel as pn
from components import pricing_view, vol_surface_view, backtest_view, layout

pn.extension(sizing_mode="stretch_width")

# Create tabs
pricing_tab = pricing_view.create_pricing_view()
vol_surface_tab = vol_surface_view.create_vol_surface_view()
backtest_tab = backtest_view.create_backtest_view()

# Main layout
tabs = pn.Tabs(
    ("Pricing", pricing_tab),
    ("Volatility Surface", vol_surface_tab),
    ("Backtest", backtest_tab),
)

app = layout.create_layout(tabs)

app.servable()
''',

        'components/pricing_view.py': '''"""
CSO pricing view component.
"""

import panel as pn


def create_pricing_view():
    """Create the pricing view tab."""

    # Placeholder
    return pn.Column(
        "# CSO Pricing",
        "Interactive pricing calculator coming soon..."
    )
''',

        'components/vol_surface_view.py': '''"""
Volatility surface view component.
"""

import panel as pn


def create_vol_surface_view():
    """Create the volatility surface view tab."""

    # Placeholder
    return pn.Column(
        "# Volatility Surface",
        "Interactive volatility surface visualization coming soon..."
    )
''',

        'components/backtest_view.py': '''"""
Backtest view component.
"""

import panel as pn


def create_backtest_view():
    """Create the backtest view tab."""

    # Placeholder
    return pn.Column(
        "# Strategy Backtest",
        "Interactive backtesting framework coming soon..."
    )
''',

        'components/layout.py': '''"""
Dashboard layout and styling.
"""

import panel as pn


def create_layout(content):
    """Create the main dashboard layout."""

    header = pn.Row(
        pn.pane.Markdown("# CSO Lab Dashboard", sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    )

    return pn.Column(
        header,
        content,
        sizing_mode="stretch_width"
    )
''',

        'assets/styles.css': '''/* Custom styles for Panel dashboard */

:root {
    --primary-color: #1f77b4;
    --secondary-color: #ff7f0e;
    --background-color: #f8f9fa;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.bk-root h1 {
    color: var(--primary-color);
}
''',
    }

    for filename, content in app_files.items():
        filepath = Path('panel_app') / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Created: {filepath}")


def main():
    """Generate the complete CSO Lab project structure."""

    print("=" * 60)
    print("CSO Lab Project Structure Generator")
    print("=" * 60)
    print()

    print("Creating directory structure...")
    create_directory_structure()
    print()

    print("Creating root files...")
    create_root_files()
    print()

    print("Creating data placeholders...")
    create_data_placeholders()
    print()

    print("Creating notebooks...")
    create_notebooks()
    print()

    print("Creating package files...")
    create_package_files()
    print()

    print("Creating Panel app...")
    create_panel_app()
    print()

    print("=" * 60)
    print("✓ Project structure generated successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. cd into the project directory")
    print("  2. Initialize git: git init")
    print("  3. Install dependencies: pip install -e .")
    print("  4. Start developing!")
    print()


if __name__ == '__main__':
    main()