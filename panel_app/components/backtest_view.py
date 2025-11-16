import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from pathlib import Path
from datetime import datetime
import os


class BacktestView:
    def __init__(self, data_dir=None):
        # If data_dir is not provided, look in the parent directory's data/outputs
        if data_dir is None:
            current_dir = Path(__file__).parent
            self.data_dir = current_dir.parent.parent / "data" / "outputs"
        else:
            self.data_dir = Path(data_dir)

        self.strategies = {}
        self.current_strategy = None
        self.pnl_data = None
        self.stats = {}
        self.sensitivity_data = None
        self.plots_dir = None

        print(f"Using data directory: {self.data_dir.absolute()}")

        # Initialize panel extension
        pn.extension('tabulator', 'plotly')

        # Scan for available strategies
        self._scan_strategies()

        # Create widgets
        self._create_widgets()

        # Load initial data if strategies exist
        if self.strategies:
            self._update_backtest(None)

        # Create layout
        self.layout = self._create_layout()

    def _scan_strategies(self):
        """Scan the data directory for available strategies."""
        strategy_dirs = [d for d in self.data_dir.glob('strategy*') if d.is_dir()]
        for strategy_dir in strategy_dirs:
            strategy_name = strategy_dir.name
            self.strategies[strategy_name] = {
                'path': strategy_dir,
                'pnl_file': strategy_dir / 'backtest_pnl.csv',
                'stats_file': strategy_dir / 'backtest_stats.txt',
                'plots_dir': strategy_dir / 'plots',
                'sensitivity_dir': strategy_dir / 'sensitivity'
            }

    def _create_widgets(self):
        """Create interactive widgets."""
        self.strategy_selector = pn.widgets.Select(
            name='Select Strategy',
            options=list(self.strategies.keys()) or ['No strategies found'],
            value=list(self.strategies.keys())[0] if self.strategies else None
        )

        self.rolling_window = pn.widgets.IntSlider(
            name='Rolling Window (days)',
            start=5, end=252, step=5, value=21
        )

        # Wire up callbacks
        self.strategy_selector.param.watch(self._update_backtest, 'value')
        self.rolling_window.param.watch(lambda e: self._update_plots(), 'value')

    def _update_backtest(self, event):
        """Update backtest data when strategy changes."""
        strategy_name = self.strategy_selector.value
        if not strategy_name or strategy_name not in self.strategies:
            return

        self.current_strategy = strategy_name
        strategy = self.strategies[strategy_name]
        self.plots_dir = strategy['plots_dir']

        print(f"Loading data for strategy: {strategy_name}")

        try:
            # Load PnL data
            if strategy['pnl_file'].exists():
                self.pnl_data = pd.read_csv(strategy['pnl_file'], parse_dates=['date'], index_col='date')
                print(f"Loaded PnL data with shape: {self.pnl_data.shape}")

            # Load statistics
            if strategy['stats_file'].exists():
                self.stats = self._parse_stats_file(strategy['stats_file'])
                print(f"Loaded {len(self.stats)} stats")

            # Update all plots
            self._update_plots()

        except Exception as e:
            print(f"Error updating backtest: {str(e)}")
            import traceback
            traceback.print_exc()

    def _update_plots(self):
        """Update all plots when data changes."""
        if hasattr(self, 'pnl_plot') and self.pnl_plot:
            self.pnl_plot[0].object = self._create_pnl_plot()
        if hasattr(self, 'drawdown_plot') and self.drawdown_plot:
            self.drawdown_plot[0].object = self._create_drawdown_plot()
        if hasattr(self, 'sharpe_plot') and self.sharpe_plot:
            self.sharpe_plot[0].object = self._create_rolling_sharpe()
        if hasattr(self, 'metrics_card') and self.metrics_card:
            self.metrics_card[0].object = self._create_metrics_table()
        if hasattr(self, 'trades_plot') and self.trades_plot:
            self.trades_plot[0].object = self._create_trades_plot()
        if hasattr(self, 'sensitivity_plot') and self.sensitivity_plot:
            self._update_sensitivity_plot()

    def _update_sensitivity_plot(self):
        """Update sensitivity plot when data changes."""
        if hasattr(self, 'sensitivity_plot') and self.sensitivity_plot:
            self.sensitivity_plot[0].object = self._create_sensitivity_plot()

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

    def _create_pnl_plot(self):
        """Create cumulative PnL plot."""
        if self.pnl_data is None or 'cumulative_pnl' not in self.pnl_data.columns:
            png_file = self.plots_dir / 'cumulative_pnl.png'
            if png_file.exists():
                return pn.pane.PNG(png_file, sizing_mode='scale_both')
            return hv.Curve([]).opts(title="No PnL data available")

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
        zero_line = hv.HLine(0).opts(color='black', line_dash='dashed', alpha=0.5)
        return plot * zero_line

    def _create_drawdown_plot(self):
        """Create drawdown plot."""
        if self.pnl_data is None or 'cumulative_pnl' not in self.pnl_data.columns:
            png_file = self.plots_dir / 'drawdown.png'
            if png_file.exists():
                return pn.pane.PNG(png_file, sizing_mode='scale_both')
            return hv.Curve([]).opts(title="No drawdown data available")

        cum_pnl = self.pnl_data['cumulative_pnl']
        cum_returns = 1 + cum_pnl / 10000
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max

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
        zero_line = hv.HLine(0).opts(color='black', line_dash='dashed', alpha=0.5)
        return plot * zero_line

    def _create_rolling_sharpe(self):
        """Create rolling Sharpe ratio plot."""
        if self.pnl_data is None or 'pnl' not in self.pnl_data.columns:
            return hv.Curve([]).opts(title="No PnL data available")

        window = self.rolling_window.value
        if window >= len(self.pnl_data):
            return hv.Curve([]).opts(title="Not enough data for selected window")

        returns = self.pnl_data['pnl'] / 10000
        rolling_mean = returns.rolling(window=window).mean() * 252
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std
        rolling_sharpe = rolling_sharpe.dropna()

        return rolling_sharpe.hvplot.line(
            title=f'Rolling {window}-Day Sharpe Ratio',
            line_width=2,
            color='purple',
            xlabel='Date',
            ylabel='Sharpe Ratio',
            grid=True,
            responsive=True,
            height=300
        )

    def _create_metrics_table(self):
        """Create metrics table."""
        if not self.stats:
            return pn.pane.Markdown("No metrics available")

        stat_definitions = [
            ("Total Return", "Total Return", "{:,.2f} bps"),
            ("Annualized Return", "Annualized Return", "{:.2%}"),
            ("Annualized Volatility", "Annualized Volatility", "{:.2%}"),
            ("Sharpe Ratio", "Sharpe Ratio", "{:.2f}"),
            ("Max Drawdown", "Max Drawdown", "{:.2%}"),
            ("Win Rate", "Win Rate", "{:.1%}"),
            ("Profit Factor", "Profit Factor", "{:.2f}"),
            ("Number of Trades", "Number of Trades", "{:,.0f}")
        ]

        metrics = []
        for display_name, stat_key, fmt in stat_definitions:
            if stat_key in self.stats:
                try:
                    value = float(self.stats[stat_key])
                    formatted_value = fmt.format(value)
                    metrics.append((display_name, formatted_value))
                except (ValueError, TypeError):
                    metrics.append((display_name, str(self.stats[stat_key])))

        if not metrics:
            return pn.pane.Markdown("No valid metrics found in stats")

        return pn.widgets.Tabulator(
            pd.DataFrame(metrics, columns=['Metric', 'Value']),
            show_index=False,
            disabled=True,
            layout='fit_data_stretch',
            sizing_mode='stretch_both'
        )

    def _create_trades_plot(self):
        """Create plot showing price and trades from saved PNG."""
        if not hasattr(self, 'plots_dir') or self.plots_dir is None:
            return pn.pane.Markdown("No plots directory found")

        png_file = self.plots_dir / 'price_chart_with_trades.png'
        if png_file.exists():
            return pn.pane.PNG(
                png_file,
                sizing_mode='scale_width',  # Changed from 'scale_both'
                min_height=500,
                max_height=800  # Add max height constraint
            )
        return pn.pane.Markdown("Trades plot not found")

    def _create_sensitivity_plot(self):
        """Show sensitivity plot from saved PNG."""
        if not hasattr(self, 'current_strategy') or not self.current_strategy:
            return pn.pane.Markdown("No strategy selected")

        strategy = self.strategies.get(self.current_strategy, {})
        sensitivity_dir = strategy.get('sensitivity_dir')

        if not sensitivity_dir or not sensitivity_dir.exists():
            return pn.pane.Markdown("Sensitivity data not available")

        png_file = sensitivity_dir / 'sensitivity_analysis.png'
        if png_file.exists():
            return pn.pane.PNG(
                png_file,
                sizing_mode='scale_width',  # Changed from 'scale_both'
                min_height=500,
                max_height=800  # Add max height constraint
            )
        return pn.pane.Markdown("Sensitivity plot not found")

    def _create_layout(self):
        """Create the dashboard layout with tabs for different views."""
        # Create plot cards
        self.pnl_plot = pn.Card(
            pn.pane.HoloViews(
                self._create_pnl_plot(),
                sizing_mode='stretch_both',
                min_height=400
            ),
            title="Cumulative PnL",
            sizing_mode='stretch_both'
        )

        self.drawdown_plot = pn.Card(
            pn.pane.HoloViews(
                self._create_drawdown_plot(),
                sizing_mode='stretch_both',
                min_height=300
            ),
            title="Drawdown",
            sizing_mode='stretch_both'
        )

        self.sharpe_plot = pn.Card(
            pn.pane.HoloViews(
                self._create_rolling_sharpe(),
                sizing_mode='stretch_both',
                min_height=300
            ),
            title="Rolling Sharpe Ratio",
            sizing_mode='stretch_both'
        )

        self.metrics_card = pn.Card(
            self._create_metrics_table(),
            title="Performance Metrics",
            sizing_mode='stretch_both',
            min_height=400
        )

        # In the _create_layout method, update the cards:
        self.trades_plot = pn.Card(
            self._create_trades_plot(),
            title="Trades",
            sizing_mode='stretch_width',  # Changed from 'stretch_both'
            height=600,  # Set fixed height
            min_height=500
        )

        self.sensitivity_plot = pn.Card(
            self._create_sensitivity_plot(),
            title="Parameter Sensitivity",
            sizing_mode='stretch_width',  # Changed from 'stretch_both'
            height=600,  # Set fixed height
            min_height=500
        )

        # Create a 2x2 grid for the main plots
        grid = pn.GridBox(
            self.pnl_plot,
            self.metrics_card,
            self.drawdown_plot,
            self.sharpe_plot,
            ncols=2,
            nrows=2,
            sizing_mode='stretch_both'
        )

        # Controls
        controls = pn.Card(
            self.strategy_selector,
            self.rolling_window,
            title="Backtest Controls",
            collapsed=False,
            width=300
        )

        # Main layout with tabs
        tabs = pn.Tabs(
            ('Performance', pn.Column(grid, sizing_mode='stretch_both')),
            ('Trades', self.trades_plot),
            ('Sensitivity', self.sensitivity_plot),
            sizing_mode='stretch_both'
        )

        # Main layout
        return pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    "## Strategy Backtest Analysis",
                    margin=(10, 0, 10, 0)
                ),
                sizing_mode='stretch_width'
            ),
            pn.layout.Divider(),
            pn.Row(
                controls,
                tabs,
                sizing_mode='stretch_both',
                min_height=900
            ),
            sizing_mode='stretch_both'
        )


# For testing the view standalone
if __name__ == '__main__':
    pn.extension('tabulator', 'plotly', sizing_mode="stretch_width")
    app = BacktestView()
    app.layout.servable()