# CSO Lab

[![Live Demo](https://img.shields.io/badge/View-Live%20Demo-brightgreen)](https://cso-lab.onrender.com/app)

A comprehensive toolkit for Commodity Spread Option (CSO) analysis, pricing, and trading strategy development with an interactive web interface.

## 🚀 Live Demo

Access the live application at: [https://cso-lab.onrender.com/app](https://cso-lab.onrender.com/app) It may take some time to load if you are loading for the first time.

![Screen Recording 2025-11-20 at 6 50 46 PM](https://github.com/user-attachments/assets/34f204c7-1ccd-4055-abbc-198c952050ef)


## 📊 Overview

This project provides tools for analyzing and pricing CSOs between Henry Hub (HH) Natural Gas and WTI Crude Oil, including:

- **Interactive Dashboard** with real-time visualization
- **Volatility Surface Modeling** for both HH and WTI
- **Pricing Methods**:
  - Kirk's approximation
  - Monte Carlo simulation
  - Numerical integration
- **Strategy Analysis**:
  - Backtesting framework
  - Performance metrics (PnL, Drawdown, Sharpe ratio)
  - Trade visualization
  - Parameter sensitivity analysis

## Project Structure

```
cso-lab/
├── data/                  # Data storage
│   ├── raw/              # Raw market data
│   ├── processed/        # Processed data files
│   └── outputs/          # Backtest results and plots
├── notebooks/            # Jupyter notebooks for research and analysis
├── cso_lab/             # Core Python package
│   ├── pricing.py       # Pricing models
│   ├── backtest.py      # Backtesting framework
│   └── data_utils.py    # Data processing utilities
└── panel_app/           # Interactive web dashboard
    ├── app.py           # Main application
    ├── components/      # UI components
    │   ├── backtest_view.py
    │   ├── pricing_view.py
    │   └── vol_surface_view.py
    └── layout.py        # Application layout
```

## Features

### 1. Interactive Dashboard
- **Pricing View**:
  - Real-time CSO pricing with interactive sliders
  - Sensitivity analysis for all parameters
  - Visual representation of option value components

- **Volatility Surface**:
  - 3D visualization of implied and local volatility surfaces
  - Term structure and volatility smile analysis
  - Interactive slicing and inspection

- **Backtesting**:
  - Multi-strategy performance comparison
  - Detailed trade analysis
  - Risk metrics and performance attribution

### 2. Core Functionality
- Efficient CSO pricing with multiple methods
- Volatility surface calibration
- Correlation modeling between HH and WTI
- Backtesting framework for strategy development

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cso-lab.git
   cd cso-lab
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Usage

### Running the Dashboard
```bash
panel serve panel_app/app.py --show
```

### Python API

```python
from cso_lab.pricing import kirk_cso_price

# Price a CSO using Kirk's approximation
price = kirk_cso_price(
    F1=66.0,        # Forward price asset 1 (HH)
    F2=45.0,        # Forward price asset 2 (WTI)
    K=5.0,          # Strike price
    sigma1=0.30,    # Volatility asset 1
    sigma2=0.25,    # Volatility asset 2
    rho=0.20,       # Correlation
    T=1.0,          # Time to maturity (years)
    r=0.05          # Risk-free rate
)
```

## Requirements

- Python 3.8+
- Core Dependencies:
  - numpy
  - pandas
  - scipy
  - numba (for performance optimization)
- Visualization:
  - panel
  - holoviews
  - bokeh
  - matplotlib
  - seaborn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

---
*Note: This project is for educational and research purposes only. Not financial advice.*
