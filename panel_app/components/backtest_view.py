# panel_app/components/backtest_view.py
import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from pathlib import Path
from datetime import datetime

class BacktestView:
    def __init__(self, data_dir="data/outputs"):
        self.data_dir = Path(data_dir)
        self.pnl_data = None
        self.stats = None
        
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
            output_dir = Path(self.data_dir)
            pnl_files = list(output_dir.glob('*pnl*.csv'))
            return [f.stem for f in pnl_files] or ['No backtests found']
        except Exception:
            return ['No backtests found']
    
    def _load_backtest_data(self):
        """Load backtest results."""
        try:
            if self.backtest_selector.value and self.backtest_selector.value != 'No backtests found':
                # Load PnL data
                pnl_file = self.data_dir / f"{self.backtest_selector.value}.csv"
                if pnl_file.exists():
                    self.pnl_data = pd.read_csv(pnl_file, parse_dates=['date'], index_col='date')
                
                # Load stats
                stats_file = self.data_dir / f"{self.backtest_selector.value}_stats.txt"
                if stats_file.exists():
                    self.stats = self._parse_stats_file(stats_file)
        except Exception as e:
            print(f"Error loading backtest data: {e}")
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
                        stats[key.strip()] = float(value.strip())
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
            return hv.Curve([]).opts(title="No PnL data available")
            
        return self.pnl_data['cumulative_pnl'].hvplot.line(
            title='Cumulative PnL',
            line_width=2,
            color='green'
        ).opts(
            yformatter='%.0f bps',
            xlabel='Date',
            ylabel='Cumulative PnL (bps)',
            grid=True
        )
    
    def _create_drawdown_plot(self):
        """Create drawdown plot."""
        if self.pnl_data is None or 'cumulative_pnl' not in self.pnl_data.columns:
            return hv.Curve([]).opts(title="No PnL data available")
            
        cum_pnl = self.pnl_data['cumulative_pnl'] / 10000  # Convert back to decimal
        rolling_max = cum_pnl.cummax()
        drawdown = (cum_pnl - rolling_max) / (1 + rolling_max)
        
        return drawdown.hvplot.area(
            title='Drawdown',
            color='red',
            alpha=0.3,
            ylim=(-0.5, 0.1)
        ).opts(
            yformatter='%.1%',
            xlabel='Date',
            ylabel='Drawdown',
            grid=True
        )
    
    def _create_rolling_sharpe(self):
        """Create rolling Sharpe ratio plot."""
        if self.pnl_data is None or 'pnl' not in self.pnl_data.columns:
            return hv.Curve([]).opts(title="No PnL data available")
            
        window = self.rolling_window.value
        if window >= len(self.pnl_data):
            return hv.Curve([]).opts(title="Not enough data for selected window")
            
        returns = self.pnl_data['pnl'] / 10000  # Convert back to decimal
        rolling_mean = returns.rolling(window=window).mean() * 252  # Annualize
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)  # Annualize
        rolling_sharpe = rolling_mean / rolling_std
        rolling_sharpe = rolling_sharpe.dropna()
        
        return rolling_sharpe.hvplot.line(
            title=f'Rolling {window}-Day Sharpe Ratio',
            line_width=2,
            color='purple'
        ).opts(
            xlabel='Date',
            ylabel='Sharpe Ratio',
            grid=True
        )
    
    def _create_metrics_table(self):
        """Create metrics table."""
        if not self.stats:
            return pn.pane.Markdown("No metrics available")
            
        metrics = [
            ("Total Return", f"{self.stats.get('Total Return', 0):.2f} bps"),
            ("Annualized Return", f"{self.stats.get('Annualized Return', 0):.2%}"),
            ("Annualized Volatility", f"{self.stats.get('Annualized Volatility', 0):.2%}"),
            ("Sharpe Ratio", f"{self.stats.get('Sharpe Ratio', 0):.2f}"),
            ("Max Drawdown", f"{self.stats.get('Max Drawdown', 0):.2%}"),
            ("Win Rate", f"{self.stats.get('Win Rate', 0):.1%}"),
            ("Profit Factor", f"{self.stats.get('Profit Factor', 0):.2f}"),
            ("Number of Trades", f"{int(self.stats.get('Number of Trades', 0)):,}")
        ]
        
        return pn.widgets.DataFrame(
            pd.DataFrame(metrics, columns=['Metric', 'Value']),
            sizing_mode='stretch_width',
            disabled=True
        )
    
    def _update_backtest(self, event):
        """Update backtest data when selection changes."""
        self._load_backtest_data()
    
    def _create_layout(self):
        """Create the dashboard layout."""
        # Plots
        pnl_plot = pn.pane.HoloViews(
            self._create_pnl_plot(),
            sizing_mode='stretch_width'
        )
        
        drawdown_plot = pn.pane.HoloViews(
            self._create_drawdown_plot(),
            sizing_mode='stretch_width'
        )
        
        sharpe_plot = pn.pane.HoloViews(
            self._create_rolling_sharpe(),
            sizing_mode='stretch_width'
        )
        
        metrics_table = self._create_metrics_table()
        
        # Controls
        controls = pn.Card(
            self.backtest_selector,
            self.rolling_window,
            title="Controls",
            collapsed=False
        )
        
        # Wire up callbacks for interactive updates
        self.rolling_window.param.watch(
            lambda e: sharpe_plot.object.update(self._create_rolling_sharpe()), 'value'
        )
        
        # Layout
        return pn.Row(
            controls,
            pn.Column(
                pn.Row(
                    pn.Card(metrics_table, title="Performance Metrics", width=400),
                    pn.Spacer(width=20),
                    pn.Column(
                        pn.Card(pnl_plot, title="Cumulative PnL"),
                        pn.Card(drawdown_plot, title="Drawdown"),
                        pn.Card(sharpe_plot, title="Rolling Sharpe Ratio")
                    )
                ),
                sizing_mode='stretch_both'
            ),
            sizing_mode='stretch_both'
        )

# For testing the view standalone
if __name__ == '__main__':
    pn.extension(sizing_mode="stretch_width")
    app = BacktestView()
    app.layout.servable()