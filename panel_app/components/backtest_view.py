# panel_app/components/backtest_view.py
import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from pathlib import Path
from datetime import datetime


class BacktestView:
    def __init__(self, data_dir=None):
        # If data_dir is not provided, look in the parent directory's data/outputs
        if data_dir is None:
            current_dir = Path(__file__).parent
            self.data_dir = current_dir.parent.parent / "data" / "outputs"
        else:
            self.data_dir = Path(data_dir)

        self.pnl_data = None
        self.stats = {}
        self.plots_dir = self.data_dir / "plots"

        print(f"Using data directory: {self.data_dir.absolute()}")

        # Create widgets
        self._create_widgets()
        self._load_backtest_data()

        # Create layout
        self.layout = self._create_layout()

    def _create_widgets(self):
        """Create interactive widgets."""
        self.backtest_selector = pn.widgets.Select(
            name='Select Backtest',
            options=self._get_available_backtests(),
            value=None
        )
        self.rolling_window = pn.widgets.IntSlider(
            name='Rolling Window (days)',
            start=5, end=252, step=5, value=21
        )

        # Wire up callbacks
        self.backtest_selector.param.watch(self._update_backtest, 'value')

    def _get_available_backtests(self):
        """Get list of available backtest results."""
        try:
            pnl_file = self.data_dir / 'backtest_pnl.csv'
            if pnl_file.exists():
                return ['Backtest Results']
            return ['No backtest results found']
        except Exception as e:
            print(f"Error finding backtest files: {e}")
            return ['No backtest results found']

    def _load_backtest_data(self):
        """Load backtest results from the data/outputs directory."""
        try:
            print(f"Looking for data in: {self.data_dir.absolute()}")

            # Check if directory exists
            if not self.data_dir.exists():
                print(f"Directory does not exist: {self.data_dir.absolute()}")
                self.pnl_data = self._generate_sample_data()
                self.stats = self._generate_sample_stats()
                return

            # List all files in the directory for debugging
            print("Files in directory:")
            for f in self.data_dir.glob('*'):
                print(f" - {f.name}")

            # Load PnL data
            pnl_file = self.data_dir / 'backtest_pnl.csv'
            print(f"Looking for PnL file at: {pnl_file.absolute()}")

            if pnl_file.exists():
                print("Found PnL file, loading...")
                self.pnl_data = pd.read_csv(pnl_file, parse_dates=['date'], index_col='date')
                print(f"Loaded PnL data with shape: {self.pnl_data.shape if self.pnl_data is not None else 'None'}")
            else:
                print("PnL file not found")

            # Load statistics
            stats_file = self.data_dir / 'backtest_stats.txt'
            print(f"Looking for stats file at: {stats_file.absolute()}")

            if stats_file.exists():
                print("Found stats file, loading...")
                self.stats = self._parse_stats_file(stats_file)
                print(f"Loaded {len(self.stats)} stats")
            else:
                print("Stats file not found")

            # Check plots directory
            plots_dir = self.data_dir / 'plots'
            if plots_dir.exists():
                print("Plots directory exists")
                print("Available plots:")
                for p in plots_dir.glob('*.png'):
                    print(f" - {p.name}")
            else:
                print(f"Plots directory not found at: {plots_dir.absolute()}")

        except Exception as e:
            print(f"Error loading backtest data: {str(e)}")
            import traceback
            traceback.print_exc()
            self.pnl_data = self._generate_sample_data()
            self.stats = self._generate_sample_stats()

    def _parse_stats_file(self, file_path):
        """Parse stats from text file."""
        stats = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        try:
                            # Try to convert to float, if possible
                            stats[key] = float(value.strip())
                        except ValueError:
                            # If not a number, store as string
                            stats[key] = value.strip()
        except Exception as e:
            print(f"Error parsing stats file: {e}")
        return stats

    def _generate_sample_data(self):
        """Generate sample PnL data for demo purposes."""
        date_range = pd.date_range(end=pd.Timestamp.today(), periods=252)
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, len(date_range))
        cum_returns = np.cumprod(1 + returns) - 1

        return pd.DataFrame({
            'pnl': returns * 10000,  # Scale to bps
            'cumulative_pnl': cum_returns * 10000,
            'position': np.random.choice([-1, 0, 1], size=len(date_range), p=[0.2, 0.6, 0.2])
        }, index=date_range)

    def _generate_sample_stats(self):
        """Generate sample performance statistics."""
        return {
            'Total Return': 1250.50,
            'Annualized Return': 0.1875,
            'Annualized Volatility': 0.1562,
            'Sharpe Ratio': 1.2012,
            'Max Drawdown': -0.1245,
            'Win Rate': 0.5567,
            'Profit Factor': 1.4532,
            'Number of Trades': 42
        }

    def _create_pnl_plot(self):
        """Create cumulative PnL plot."""
        if self.pnl_data is None or 'cumulative_pnl' not in self.pnl_data.columns:
            # Try to load the plot from file if PnL data is not available
            png_file = self.plots_dir / 'cumulative_pnl.png'
            if png_file.exists():
                return pn.pane.PNG(png_file, sizing_mode='scale_both')
            return hv.Curve([]).opts(title="No PnL data available")

        # Create interactive plot if data is available
        plot = self.pnl_data['cumulative_pnl'].hvplot.line(
            title='Cumulative PnL',
            line_width=2,
            color='green',
            yformatter='%.0f bps',
            xlabel='Date',
            ylabel='Cumulative PnL',
            grid=True,
            responsive=True,
            height=400
        )

        # Add zero line
        zero_line = hv.HLine(0).opts(color='black', line_dash='dashed', alpha=0.5)
        return plot * zero_line

    def _create_drawdown_plot(self):
        """Create drawdown plot."""
        if self.pnl_data is None or 'cumulative_pnl' not in self.pnl_data.columns:
            # Try to load the plot from file if PnL data is not available
            png_file = self.plots_dir / 'drawdown.png'
            if png_file.exists():
                return pn.pane.PNG(png_file, sizing_mode='scale_both')
            return hv.Curve([]).opts(title="No drawdown data available")

        # Calculate drawdown from cumulative PnL
        cum_pnl = self.pnl_data['cumulative_pnl']
        cum_returns = 1 + cum_pnl / 10000  # Convert to simple returns
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max

        # Create interactive plot
        plot = drawdown.hvplot.area(
            title='Drawdown',
            color='red',
            alpha=0.3,
            ylim=(-1, 0.1),
            xlabel='Date',
            ylabel='Drawdown',
            grid=True,
            responsive=True,
            height=300
        )

        # Add zero line
        zero_line = hv.HLine(0).opts(color='black', line_dash='dashed', alpha=0.5)
        return plot * zero_line

    def _create_rolling_sharpe(self):
        """Create rolling Sharpe ratio plot."""
        if self.pnl_data is None or 'pnl' not in self.pnl_data.columns:
            return hv.Curve([]).opts(title="No PnL data available")

        window = self.rolling_window.value
        if window >= len(self.pnl_data):
            return hv.Curve([]).opts(title="Not enough data for selected window")

        returns = self.pnl_data['pnl'] / 10000  # Convert to decimal returns
        rolling_mean = returns.rolling(window=window).mean() * 252  # Annualize
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)  # Annualize
        rolling_sharpe = rolling_mean / rolling_std
        rolling_sharpe = rolling_sharpe.dropna()

        # Create the plot with Bokeh-compatible options
        plot = rolling_sharpe.hvplot.line(
            title=f'Rolling {window}-Day Sharpe Ratio',
            line_width=2,
            color='purple',
            xlabel='Date',
            ylabel='Sharpe Ratio',
            grid=True,
            responsive=True,
            height=300
        )

        return plot

    def _create_metrics_table(self):
        """Create metrics table."""
        if not self.stats:
            return pn.pane.Markdown("No metrics available")

        # Create a list of (display_name, stat_key, format_str) tuples
        stat_definitions = [
            ("Total Return", "Total Return",
             lambda x: f"{float(x):,.2f} bps" if isinstance(x, (int, float)) else str(x)),
            ("Annualized Return", "Annualized Return",
             lambda x: f"{float(x):.2%}" if isinstance(x, (int, float)) else str(x)),
            ("Annualized Volatility", "Annualized Volatility",
             lambda x: f"{float(x):.2%}" if isinstance(x, (int, float)) else str(x)),
            ("Sharpe Ratio", "Sharpe Ratio",
             lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else str(x)),
            ("Max Drawdown", "Max Drawdown",
             lambda x: f"{float(x):.2%}" if isinstance(x, (int, float)) else str(x)),
            ("Win Rate", "Win Rate",
             lambda x: f"{float(x):.1%}" if isinstance(x, (int, float)) else str(x)),
            ("Profit Factor", "Profit Factor",
             lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else str(x)),
            ("Number of Trades", "Number of Trades",
             lambda x: f"{int(x):,}" if str(x).isdigit() else str(x))
        ]

        # Create metrics list with formatted values
        metrics = []
        for display_name, stat_key, formatter in stat_definitions:
            if stat_key in self.stats:
                metrics.append((display_name, formatter(self.stats[stat_key])))

        if not metrics:
            return pn.pane.Markdown("No valid metrics found in stats")

        # Create a DataFrame for display
        df = pd.DataFrame(metrics, columns=['Metric', 'Value'])

        # Use a styled DataFrame for better appearance
        return pn.pane.DataFrame(
            df,
            sizing_mode='stretch_width',
            index=False,
            width=400
        )

    def _update_backtest(self, event):
        """Update backtest data when selection changes."""
        self._load_backtest_data()

    def _create_trades_plot(self):
        """Create plot showing price and trades."""
        png_file = self.plots_dir / 'price_chart_with_trades.png'
        if png_file.exists():
            return pn.pane.PNG(png_file, sizing_mode='scale_both')
        return pn.pane.Markdown("Trades plot not available")

    def _create_layout(self):
        """Create the dashboard layout with a 2x2 grid for plots and metrics."""
        # Create the main plots with increased height
        pnl_plot = pn.Card(
            pn.pane.HoloViews(
                self._create_pnl_plot(),
                sizing_mode='stretch_both',
                min_height=400
            ),
            title="Cumulative PnL",
            sizing_mode='stretch_both',
            margin=(10, 10, 10, 10)
        )
        
        drawdown_plot = pn.Card(
            pn.pane.HoloViews(
                self._create_drawdown_plot(),
                sizing_mode='stretch_both',
                min_height=400
            ),
            title="Drawdown",
            sizing_mode='stretch_both',
            margin=(10, 10, 10, 10)
        )
        
        sharpe_plot = pn.Card(
            pn.pane.HoloViews(
                self._create_rolling_sharpe(),
                sizing_mode='stretch_both',
                min_height=400
            ),
            title="Rolling Sharpe Ratio",
            sizing_mode='stretch_both',
            margin=(10, 10, 10, 10)
        )
        
        # Create metrics card with increased height to match plots
        metrics_card = pn.Card(
            self._create_metrics_table(),
            title="Performance Metrics",
            sizing_mode='stretch_both',
            margin=(10, 10, 10, 10),
            min_height=400
        )
        
        # Create trades plot card
        trades_plot = pn.Card(
            self._create_trades_plot(),
            title="Trades",
            sizing_mode='stretch_both',
            margin=(10, 10, 10, 10),
            min_height=400
        )
        
        # Create a 2x2 grid for the main plots and metrics
        grid = pn.GridBox(
            pnl_plot,
            metrics_card,
            drawdown_plot,
            sharpe_plot,
            ncols=2,
            nrows=2,
            sizing_mode='stretch_both',
            margin=(0, 10, 10, 10)
        )
        
        # Controls
        controls = pn.Card(
            self.backtest_selector,
            self.rolling_window,
            title="Backtest Controls",
            collapsed=False,
            width=300,
            margin=(10, 10, 10, 10)
        )
        
        # Wire up callbacks for interactive updates
        self.rolling_window.param.watch(
            lambda e: sharpe_plot[0].object.update(self._create_rolling_sharpe()), 'value'
        )
        
        # Main layout with tabs for different views
        tabs = pn.Tabs(
            ('Performance', pn.Column(
                grid,
                sizing_mode='stretch_both'
            )),
            ('Trades', trades_plot),
            sizing_mode='stretch_both'
        )
        
        # Main layout
        return pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    "## Backtest Results",
                    margin=(10, 0, 10, 0)
                ),
                sizing_mode='stretch_width'
            ),
            pn.layout.Divider(),
            pn.Row(
                controls,
                tabs,
                sizing_mode='stretch_both',
                min_height=800
            ),
            sizing_mode='stretch_both',
            min_height=900
        )


# For testing the view standalone
if __name__ == '__main__':
    pn.extension(sizing_mode="stretch_width")
    app = BacktestView()
    app.layout.servable()