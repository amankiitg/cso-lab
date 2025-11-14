# CSO Lab

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
