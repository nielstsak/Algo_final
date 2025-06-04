# src/backtesting/visualizations.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import vectorbt as vbt
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
import warnings
from scipy import stats

# Attempt to import PerformanceMetrics, handle if not found gracefully for standalone use
try:
    from src.backtesting.performance_metrics import PerformanceMetrics
except ImportError:
    PerformanceMetrics = None # Allow basic plotting even if metrics module is missing

# Default Plotly template
pio.templates.default = "plotly_dark"

class BacktestVisualizer:
    """
    Creates various interactive visualizations for backtesting results using Plotly.
    """

    def __init__(self, theme: str = 'dark', default_colors: Optional[Dict] = None):
        """
        Initializes the BacktestVisualizer.

        Args:
            theme: Plotly theme to use ('dark' or 'light').
            default_colors: Optional dictionary to override default plot colors.
                            Example: {'equity': 'blue', 'drawdown': 'rgba(255,0,0,0.3)'}
        """
        self.theme = theme
        if theme == 'light':
            pio.templates.default = "plotly_white"
        else:
            pio.templates.default = "plotly_dark"

        self.colors = {
            'equity': '#1f77b4',  # Muted blue
            'drawdown_area': 'rgba(239, 83, 80, 0.3)', # Light red with alpha
            'benchmark': '#ff7f0e', # Orange
            'signal_entry_long': 'green',
            'signal_exit_long': 'darkgreen',
            'signal_entry_short': 'red',
            'signal_exit_short': 'darkred',
            'primary_metric': '#17becf', # Cyan
            'secondary_metric': '#7f7f7f', # Grey
            'trade_win': 'rgba(76, 175, 80, 0.7)', # Green
            'trade_loss': 'rgba(244, 67, 54, 0.7)', # Red
            'candlestick_increasing': '#00CC96',
            'candlestick_decreasing': '#EF553B',
        }
        if default_colors:
            self.colors.update(default_colors)

    def _apply_theme(self, fig: go.Figure) -> go.Figure:
        """Applies the selected theme and common layout properties to a figure."""
        fig.update_layout(
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            transition_duration=300
        )
        return fig

    def plot_equity_curve(
        self,
        portfolio: vbt.Portfolio,
        benchmark_rets: Optional[pd.Series] = None,
        title: str = "Equity Curve and Drawdowns",
        show_drawdown_area: bool = True
    ) -> go.Figure:
        """
        Plots the equity curve with optional drawdown area and benchmark.

        Args:
            portfolio: vectorbt.Portfolio object.
            benchmark_rets: Optional Series of benchmark returns.
            title: Title of the plot.
            show_drawdown_area: Whether to show the drawdown area.

        Returns:
            Plotly Figure object.
        """
        fig = go.Figure()

        # Equity curve
        equity = portfolio.value()
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity,
            mode='lines',
            name='Strategy Equity',
            line=dict(color=self.colors['equity'], width=2)
        ))

        # Drawdown area
        if show_drawdown_area:
            drawdown = portfolio.drawdown() * equity.iloc[0] # Scale drawdown to equity
            underwater_equity = (1 - drawdown / equity.iloc[0]) * equity
            
            # Find peak equity before drawdown starts
            peak_equity = equity.cummax()
            
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=peak_equity,
                fill=None,
                mode='lines',
                line_color='rgba(255,255,255,0)', # Transparent line
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity, # Drawdown relative to peak
                fill='tonexty',
                mode='lines',
                line_color='rgba(255,255,255,0)', # Transparent line
                fillcolor=self.colors['drawdown_area'],
                name='Drawdown'
            ))


        # Benchmark
        if benchmark_rets is not None and not benchmark_rets.empty:
            benchmark_equity = (1 + benchmark_rets).cumprod() * portfolio.init_cash
            fig.add_trace(go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity,
                mode='lines',
                name='Benchmark Equity',
                line=dict(color=self.colors['benchmark'], dash='dash')
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            yaxis_type="log" # Log scale often useful for equity curves
        )
        
        # Annotations for key metrics
        if PerformanceMetrics:
            total_return_val = PerformanceMetrics.total_return(portfolio)
            annual_return_val = PerformanceMetrics.annualized_return(portfolio)
            max_drawdown_val = PerformanceMetrics.max_drawdown(portfolio)

            annotations = [
                dict(xref='paper', yref='paper', x=0.02, y=0.95,
                     text=f"Total Return: {total_return_val:.2%}", showarrow=False, font=dict(size=10)),
                dict(xref='paper', yref='paper', x=0.02, y=0.90,
                     text=f"Annualized Return: {annual_return_val:.2%}", showarrow=False, font=dict(size=10)),
                dict(xref='paper', yref='paper', x=0.02, y=0.85,
                     text=f"Max Drawdown: {max_drawdown_val:.2%}", showarrow=False, font=dict(size=10)),
            ]
            fig.update_layout(annotations=annotations)

        return self._apply_theme(fig)

    def plot_returns_distribution(
        self,
        portfolio: vbt.Portfolio,
        period: str = 'D', # 'D' for daily, 'M' for monthly, etc.
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """
        Plots the histogram and QQ-plot of returns.

        Args:
            portfolio: vectorbt.Portfolio object.
            period: Resampling period for returns ('D', 'W', 'M').
            title: Title of the plot.

        Returns:
            Plotly Figure object.
        """
        returns = portfolio.returns().resample(period).sum() # Sum for log returns, or use .apply(lambda x: (1+x).prod()-1) for simple
        if returns.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (No returns data)")
            return self._apply_theme(fig)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "QQ-Plot vs Normal"))

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns,
            name='Returns',
            marker_color=self.colors['primary_metric'],
            nbinsx=50
        ), row=1, col=1)

        # QQ-plot
        qq_data = stats.probplot(returns.dropna(), dist="norm")
        theoretical_q = qq_data[0][0]
        ordered_vals = qq_data[0][1]

        fig.add_trace(go.Scatter(
            x=theoretical_q,
            y=ordered_vals,
            mode='markers',
            name='Data Quantiles',
            marker=dict(color=self.colors['primary_metric'])
        ), row=1, col=2)

        # Add theoretical line for QQ-plot
        min_val = min(theoretical_q.min(), ordered_vals.min())
        max_val = max(theoretical_q.max(), ordered_vals.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Normal Line',
            line=dict(color=self.colors['secondary_metric'], dash='dash')
        ), row=1, col=2)

        fig.update_layout(title_text=title, showlegend=False)
        fig.update_xaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles (Normal)", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

        return self._apply_theme(fig)

    def plot_monthly_heatmap(
        self,
        portfolio: vbt.Portfolio,
        title: str = "Monthly Returns Heatmap"
    ) -> go.Figure:
        """
        Plots a heatmap of monthly returns.

        Args:
            portfolio: vectorbt.Portfolio object.

        Returns:
            Plotly Figure object.
        """
        monthly_rets = portfolio.returns().resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        if monthly_rets.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (No returns data)")
            return self._apply_theme(fig)
            
        monthly_rets_table = monthly_rets.to_frame(name='returns').pivot_table(
            values='returns',
            index=monthly_rets.index.year,
            columns=monthly_rets.index.month_name()
        )
        # Order columns by month
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        monthly_rets_table = monthly_rets_table.reindex(columns=month_order)


        fig = go.Figure(data=go.Heatmap(
            z=monthly_rets_table.values,
            x=monthly_rets_table.columns,
            y=monthly_rets_table.index,
            colorscale='RdYlGn', # Red-Yellow-Green
            text=monthly_rets_table.applymap(lambda x: f"{x:.2f}%" if not pd.isna(x) else ""),
            texttemplate="%{text}",
            hoverongaps=False
        ))
        fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Year")
        return self._apply_theme(fig)

    def plot_trade_analysis(
        self,
        portfolio: vbt.Portfolio,
        title: str = "Trade Analysis (Duration vs. P&L)"
    ) -> go.Figure:
        """
        Scatter plot of trades (duration vs. profit/loss).

        Args:
            portfolio: vectorbt.Portfolio object.

        Returns:
            Plotly Figure object.
        """
        trades = portfolio.trades.records_readable
        if trades.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (No trades to analyze)")
            return self._apply_theme(fig)

        trades['Duration_days'] = (trades['Exit Timestamp'] - trades['Entry Timestamp']).dt.total_seconds() / (24 * 3600)
        
        # Separate winning and losing trades for different colors
        winning_trades = trades[trades['Return'] > 0]
        losing_trades = trades[trades['Return'] <= 0]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=winning_trades['Duration_days'],
            y=winning_trades['Return'] * 100, # P&L as percentage
            mode='markers',
            name='Winning Trades',
            marker=dict(color=self.colors['trade_win'], size=8, opacity=0.7,
                        line=dict(width=1, color='DarkSlateGrey')),
            text=winning_trades.apply(lambda r: f"Entry: {r['Entry Timestamp']:%Y-%m-%d}<br>Exit: {r['Exit Timestamp']:%Y-%m-%d}<br>PnL: {r['Return']:.2%}", axis=1),
            hoverinfo='text+x+y'
        ))
        
        fig.add_trace(go.Scatter(
            x=losing_trades['Duration_days'],
            y=losing_trades['Return'] * 100, # P&L as percentage
            mode='markers',
            name='Losing Trades',
            marker=dict(color=self.colors['trade_loss'], size=8, opacity=0.7,
                        line=dict(width=1, color='DarkSlateGrey')),
            text=losing_trades.apply(lambda r: f"Entry: {r['Entry Timestamp']:%Y-%m-%d}<br>Exit: {r['Exit Timestamp']:%Y-%m-%d}<br>PnL: {r['Return']:.2%}", axis=1),
            hoverinfo='text+x+y'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Trade Duration (Days)",
            yaxis_title="Trade P&L (%)"
        )
        return self._apply_theme(fig)

    def plot_drawdown_analysis(
        self,
        portfolio: vbt.Portfolio,
        top_n: int = 5,
        title: str = "Top Drawdown Periods"
    ) -> go.Figure:
        """
        Plots the periods and depth of the top N drawdowns.

        Args:
            portfolio: vectorbt.Portfolio object.
            top_n: Number of top drawdowns to display.
            title: Title of the plot.

        Returns:
            Plotly Figure object.
        """
        drawdowns = portfolio.drawdowns.records_readable
        if drawdowns.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (No drawdowns recorded)")
            return self._apply_theme(fig)

        top_drawdowns = drawdowns.sort_values(by='Drawdown', ascending=True).head(top_n) # Drawdown is negative

        fig = go.Figure()
        
        # Plot on equity curve
        equity = portfolio.value()
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity,
            mode='lines',
            name='Equity Curve',
            line=dict(color=self.colors['equity'], width=1.5)
        ))

        for i, dd_info in top_drawdowns.iterrows():
            peak_date = dd_info['Peak Date']
            valley_date = dd_info['Valley Date']
            recovery_date = dd_info['Recovery Date'] # Can be NaT if not recovered

            # Highlight drawdown period
            fig.add_vrect(
                x0=peak_date, x1=valley_date, 
                fillcolor=self.colors['drawdown_area'], opacity=0.5, layer="below", line_width=0,
            )
            if pd.notna(recovery_date):
                 fig.add_vrect(
                    x0=valley_date, x1=recovery_date, 
                    fillcolor=self.colors['drawdown_area'], opacity=0.3, layer="below", line_width=0,
                )

            # Annotate drawdown depth
            fig.add_annotation(
                x=valley_date, y=equity.loc[valley_date],
                text=f"{dd_info['Drawdown']:.2%}",
                showarrow=True, arrowhead=1, ax=0, ay=-40,
                bgcolor="rgba(0,0,0,0.6)" if self.theme == 'dark' else "rgba(255,255,255,0.6)"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value"
        )
        return self._apply_theme(fig)


    def plot_rolling_metrics(
        self,
        portfolio: vbt.Portfolio,
        metrics: List[str] = ['sharpe_ratio', 'sortino_ratio'],
        window: int = 100, # Default window for rolling calculation
        title: str = "Rolling Performance Metrics"
    ) -> go.Figure:
        """
        Plots rolling performance metrics like Sharpe or Sortino ratio.

        Args:
            portfolio: vectorbt.Portfolio object.
            metrics: List of metrics to plot (e.g., ['sharpe_ratio', 'sortino_ratio']).
            window: Rolling window size.
            title: Title of the plot.

        Returns:
            Plotly Figure object.
        """
        fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True,
                            subplot_titles=[m.replace('_', ' ').title() for m in metrics])

        for i, metric_name in enumerate(metrics):
            rolling_metric_series = None
            if metric_name == 'sharpe_ratio':
                rolling_metric_series = portfolio.rolling_sharpe_ratio(window=window)
            elif metric_name == 'sortino_ratio':
                rolling_metric_series = portfolio.rolling_sortino_ratio(window=window)
            # Add more metrics here if needed, e.g., rolling Calmar
            else:
                warnings.warn(f"Metric '{metric_name}' not supported for rolling plot. Skipping.")
                continue
            
            if rolling_metric_series is not None and not rolling_metric_series.empty:
                fig.add_trace(go.Scatter(
                    x=rolling_metric_series.index,
                    y=rolling_metric_series,
                    mode='lines',
                    name=metric_name.replace('_', ' ').title(),
                    line=dict(color=self.colors['primary_metric'] if i % 2 == 0 else self.colors['secondary_metric'])
                ), row=i+1, col=1)
                fig.update_yaxes(title_text=metric_name.replace('_', ' ').title(), row=i+1, col=1)

        fig.update_layout(title_text=title, showlegend=False)
        fig.update_xaxes(title_text="Date", row=len(metrics), col=1)
        return self._apply_theme(fig)

    def plot_entry_exit_signals(
        self,
        portfolio: vbt.Portfolio, # Or just price data and trades
        price_data: Optional[pd.DataFrame] = None, # OHLC data
        trades: Optional[pd.DataFrame] = None, # From portfolio.trades.records_readable
        plot_type: str = 'candlestick', # 'candlestick' or 'ohlc' or 'line'
        title: str = "Price Chart with Entry/Exit Signals"
    ) -> go.Figure:
        """
        Plots price (candlestick/OHLC) with entry and exit signal markers.

        Args:
            portfolio: vectorbt.Portfolio object. Used if price_data and trades are None.
            price_data: DataFrame with 'Open', 'High', 'Low', 'Close' columns.
            trades: DataFrame of trades (from portfolio.trades.records_readable).
            plot_type: Type of price plot ('candlestick', 'ohlc', 'line').
            title: Title of the plot.

        Returns:
            Plotly Figure object.
        """
        if price_data is None:
            if portfolio is None:
                raise ValueError("Either portfolio or price_data must be provided.")
            # Assuming portfolio.data contains the OHLC data used for backtest
            # This might need adjustment based on how portfolio.data is structured by user
            if 'close' in portfolio.data: # Check if 'close' is a direct attribute
                 price_data = portfolio.data # This assumes portfolio.data is a group like object
            else: # Try to get it from the wrapper if it's a single asset portfolio
                price_data = pd.DataFrame({
                    'Open': portfolio.open(),
                    'High': portfolio.high(),
                    'Low': portfolio.low(),
                    'Close': portfolio.close(),
                })
                if 'Volume' in portfolio.wrapper.columns_: # if wrapper has volume
                    price_data['Volume'] = portfolio.volume()


        if trades is None:
            if portfolio is None:
                raise ValueError("Either portfolio or trades must be provided.")
            trades = portfolio.trades.records_readable
        
        if price_data.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (No price data)")
            return self._apply_theme(fig)

        fig = go.Figure()

        # Price plot
        if plot_type == 'candlestick':
            fig.add_trace(go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price',
                increasing_line_color=self.colors['candlestick_increasing'],
                decreasing_line_color=self.colors['candlestick_decreasing']
            ))
        elif plot_type == 'ohlc':
            fig.add_trace(go.Ohlc(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price',
                increasing_line_color=self.colors['candlestick_increasing'],
                decreasing_line_color=self.colors['candlestick_decreasing']
            ))
        else: # Line plot of close price
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['equity'])
            ))

        # Entry/Exit signals
        if not trades.empty:
            entry_trades = trades[trades['Side'] == vbt.records.trades.TradeSide.Buy] # Long entries
            exit_trades = trades[trades['Side'] == vbt.records.trades.TradeSide.Sell] # Long exits or Short entries
            
            # More robustly:
            long_entries = trades[trades['Size'] > 0] # Positive size = buy
            short_entries = trades[trades['Size'] < 0] # Negative size = sell to open short

            # For simplicity, let's use Entry Timestamp and Exit Timestamp
            # This plot is for visualization, not for precise signal generation logic
            
            fig.add_trace(go.Scatter(
                x=trades['Entry Timestamp'],
                y=trades['Entry Price'],
                mode='markers',
                name='Entry',
                marker=dict(
                    color=trades['Size'].apply(lambda s: self.colors['signal_entry_long'] if s > 0 else self.colors['signal_entry_short']),
                    size=10,
                    symbol=trades['Size'].apply(lambda s: 'triangle-up' if s > 0 else 'triangle-down'),
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertext=trades.apply(lambda r: f"Entry {r['Side'].name}<br>Size: {r['Size']:.2f}<br>Price: {r['Entry Price']:.2f}", axis=1),
                hoverinfo='text'
            ))
            fig.add_trace(go.Scatter(
                x=trades['Exit Timestamp'],
                y=trades['Exit Price'],
                mode='markers',
                name='Exit',
                marker=dict(
                    color=trades['Size'].apply(lambda s: self.colors['signal_exit_long'] if s > 0 else self.colors['signal_exit_short']), # Exit color opposite of entry logic
                    size=10,
                    symbol=trades['Size'].apply(lambda s: 'triangle-down' if s > 0 else 'triangle-up'), # Exit symbol opposite of entry
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertext=trades.apply(lambda r: f"Exit {r['Side'].name}<br>Size: {r['Size']:.2f}<br>Price: {r['Exit Price']:.2f}<br>P&L: {r['Return']:.2%}", axis=1),
                hoverinfo='text'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False # Common for financial charts
        )
        return self._apply_theme(fig)

    def plot_portfolio_composition(
        self,
        portfolio: vbt.Portfolio,
        title: str = "Portfolio Composition (Cash vs. Assets)"
    ) -> go.Figure:
        """
        Plots the evolution of portfolio composition (cash vs. asset value).

        Args:
            portfolio: vectorbt.Portfolio object.
            title: Title of the plot.

        Returns:
            Plotly Figure object.
        """
        cash = portfolio.cash()
        asset_value = portfolio.asset_value() # Value of assets held
        total_value = portfolio.value() # cash + asset_value

        if cash.empty or asset_value.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (No composition data)")
            return self._apply_theme(fig)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=cash.index,
            y=cash,
            mode='lines',
            name='Cash',
            stackgroup='one', # For stacked area chart
            line=dict(color=self.colors['secondary_metric'])
        ))
        fig.add_trace(go.Scatter(
            x=asset_value.index,
            y=asset_value,
            mode='lines',
            name='Asset Value',
            stackgroup='one',
            line=dict(color=self.colors['primary_metric'])
        ))
        
        # Optional: Plot total value as a line on top for reference
        fig.add_trace(go.Scatter(
            x=total_value.index,
            y=total_value,
            mode='lines',
            name='Total Equity',
            line=dict(color=self.colors['equity'], width=1, dash='dot'),
            opacity=0.7
        ))


        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x unified"
        )
        return self._apply_theme(fig)

    def create_full_report(
        self,
        portfolio: vbt.Portfolio,
        benchmark_rets: Optional[pd.Series] = None,
        price_data_for_signals: Optional[pd.DataFrame] = None,
        report_title: str = "Backtest Performance Report",
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        rolling_window: int = 100
    ) -> Optional[str]:
        """
        Generates a full HTML report dashboard with all relevant plots.

        Args:
            portfolio: vectorbt.Portfolio object.
            benchmark_rets: Optional Series of benchmark returns.
            price_data_for_signals: OHLC data for the signal plot.
            report_title: Title for the HTML report.
            output_path: Path to save the HTML report. If None, not saved.
            show: Whether to open the report in a browser.
            rolling_window: Window for rolling metrics plots.

        Returns:
            HTML string of the report if output_path is None and show is False.
            Otherwise, path to the saved file or None.
        """
        figs = {}
        
        # Generate all plots
        with warnings.catch_warnings(): # Suppress warnings during plot generation for report
            warnings.simplefilter("ignore")
            try:
                figs['equity_curve'] = self.plot_equity_curve(portfolio, benchmark_rets)
            except Exception as e: figs['equity_curve'] = self._error_figure(f"Equity Curve: {e}")
            
            try:
                figs['returns_dist'] = self.plot_returns_distribution(portfolio)
            except Exception as e: figs['returns_dist'] = self._error_figure(f"Returns Distribution: {e}")

            try:
                figs['monthly_heatmap'] = self.plot_monthly_heatmap(portfolio)
            except Exception as e: figs['monthly_heatmap'] = self._error_figure(f"Monthly Heatmap: {e}")

            try:
                figs['trade_analysis'] = self.plot_trade_analysis(portfolio)
            except Exception as e: figs['trade_analysis'] = self._error_figure(f"Trade Analysis: {e}")

            try:
                figs['drawdown_analysis'] = self.plot_drawdown_analysis(portfolio)
            except Exception as e: figs['drawdown_analysis'] = self._error_figure(f"Drawdown Analysis: {e}")
            
            try:
                figs['rolling_metrics'] = self.plot_rolling_metrics(portfolio, window=rolling_window)
            except Exception as e: figs['rolling_metrics'] = self._error_figure(f"Rolling Metrics: {e}")

            try:
                figs['signals_chart'] = self.plot_entry_exit_signals(portfolio, price_data=price_data_for_signals)
            except Exception as e: figs['signals_chart'] = self._error_figure(f"Signals Chart: {e}")
            
            try:
                figs['portfolio_comp'] = self.plot_portfolio_composition(portfolio)
            except Exception as e: figs['portfolio_comp'] = self._error_figure(f"Portfolio Composition: {e}")

        # Create HTML content
        html_content = f"<html><head><title>{report_title}</title>"
        html_content += "<style>"
        html_content += "body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #1e1e1e; color: #d4d4d4; }" if self.theme == 'dark' \
            else "body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #ffffff; color: #333333; }"
        html_content += """
            .container { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; padding: 20px; }
            .plot-container { border: 1px solid #444; border-radius: 5px; padding: 10px; }
            h1 { text-align: center; padding: 20px; }
            h1, h2 { color: #e0e0e0; }
        """  if self.theme == 'dark' else """
            .container { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; padding: 20px; }
            .plot-container { border: 1px solid #ddd; border-radius: 5px; padding: 10px; box-shadow: 2px 2px 5px #eee; }
            h1 { text-align: center; padding: 20px; }
            h1, h2 { color: #333; }
        """
        html_content += "</style></head><body>"
        html_content += f"<h1>{report_title}</h1>"
        html_content += "<div class='container'>"

        # Add performance metrics table at the top
        if PerformanceMetrics:
            try:
                all_metrics = PerformanceMetrics.calculate_all_metrics(portfolio, benchmark_rets, monte_carlo_sims=0) # No MC for table
                metrics_df = PerformanceMetrics.metrics_to_dataframe(all_metrics)
                # Style the table for better readability
                styled_table = metrics_df.style.set_table_attributes('style="width:100%; border-collapse: collapse;"') \
                                            .set_properties(**{'border': '1px solid #555' if self.theme == 'dark' else '1px solid #ccc',
                                                               'padding': '8px',
                                                               'text-align': 'left'}) \
                                            .set_caption("Key Performance Indicators") \
                                            .to_html()
                html_content += f"<div class='plot-container' style='grid-column: 1 / -1;'><h2>Summary Metrics</h2>{styled_table}</div>" # Full width
            except Exception as e:
                html_content += f"<div class='plot-container' style='grid-column: 1 / -1;'><h2>Summary Metrics</h2><p>Error generating metrics table: {e}</p></div>"


        plot_order = ['equity_curve', 'signals_chart', 'returns_dist', 'monthly_heatmap', 
                      'trade_analysis', 'drawdown_analysis', 'rolling_metrics', 'portfolio_comp']
        
        for plot_name in plot_order:
            if plot_name in figs:
                fig = figs[plot_name]
                plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                html_content += f"<div class='plot-container'>{plot_html}</div>"

        html_content += "</div></body></html>"

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            if show:
                import webbrowser
                webbrowser.open(f"file://{output_path.resolve()}")
            return str(output_path.resolve())
        elif show:
            # Save to temp file and open
            import tempfile
            import webbrowser
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as tmpfile:
                tmpfile.write(html_content)
                filepath = tmpfile.name
            webbrowser.open(f"file://{Path(filepath).resolve()}")
            return None # Or return filepath if needed
        else:
            return html_content # Return HTML string

    def _error_figure(self, error_message: str) -> go.Figure:
        """Creates a Plotly figure displaying an error message."""
        fig = go.Figure()
        fig.update_layout(
            title="Plot Generation Error",
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[
                {
                    "text": f"Error: {error_message}",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14}
                }
            ]
        )
        return self._apply_theme(fig)

# Example Usage (comment out when used as a module):
# if __name__ == '__main__':
#     # Create dummy data for demonstration
#     n_days = 500
#     price_index = pd.date_range("2020-01-01", periods=n_days, freq='D')
#     price = pd.Series(np.random.randn(n_days).cumsum() + 100, index=price_index)
#     price_data_df = pd.DataFrame({
#         'Open': price - np.random.rand(n_days) * 0.5,
#         'High': price + np.random.rand(n_days) * 0.5,
#         'Low': price - np.random.rand(n_days) * 0.5 - 0.1, # Ensure low is lower
#         'Close': price,
#         'Volume': np.random.randint(100, 1000, n_days)
#     })
#     price_data_df['Low'] = price_data_df[['Open', 'High', 'Low', 'Close']].min(axis=1) # Ensure Low is min
#     price_data_df['High'] = price_data_df[['Open', 'High', 'Low', 'Close']].max(axis=1) # Ensure High is max


#     entries = pd.Series(False, index=price_index)
#     entries.iloc[::30] = True # Enter every 30 days
#     exits = pd.Series(False, index=price_index)
#     exits.iloc[15::30] = True # Exit 15 days after entry

#     # Ensure exits don't happen before entries if simple slicing is used
#     last_entry_idx = -1
#     for i in range(len(entries)):
#         if entries.iloc[i]:
#             last_entry_idx = i
#         if exits.iloc[i] and i <= last_entry_idx :
#             exits.iloc[i] = False # Basic conflict resolution


#     portfolio = vbt.Portfolio.from_signals(
#         close=price_data_df['Close'],
#         entries=entries,
#         exits=exits,
#         init_cash=100000,
#         freq='D', # Important for vbt
#         # Pass ohlc data for candlestick plot if portfolio.data is not set up this way
#         # This example relies on portfolio.open(), portfolio.high() etc. which are fine with just close
#     )
#     # For plot_entry_exit_signals to work with candlestick, it needs OHLC.
#     # We can pass it explicitly or hope portfolio.data is structured.
#     # Let's assume portfolio.data is not automatically OHLC for this example.
    
#     benchmark_returns_series = (price_data_df['Close'].pct_change().fillna(0))


#     visualizer = BacktestVisualizer(theme='dark') # or 'light'

#     if portfolio.trades.count() > 0:
#         print("Portfolio has trades, generating plots...")
#         # fig_equity = visualizer.plot_equity_curve(portfolio, benchmark_rets=benchmark_returns_series)
#         # fig_equity.show()

#         # fig_returns_dist = visualizer.plot_returns_distribution(portfolio)
#         # fig_returns_dist.show()

#         # fig_monthly_heatmap = visualizer.plot_monthly_heatmap(portfolio)
#         # fig_monthly_heatmap.show()

#         # fig_trade_analysis = visualizer.plot_trade_analysis(portfolio)
#         # fig_trade_analysis.show()
        
#         # fig_drawdown_analysis = visualizer.plot_drawdown_analysis(portfolio)
#         # fig_drawdown_analysis.show()

#         # fig_rolling = visualizer.plot_rolling_metrics(portfolio, metrics=['sharpe_ratio', 'sortino_ratio'], window=60)
#         # fig_rolling.show()

#         # fig_signals = visualizer.plot_entry_exit_signals(portfolio, price_data=price_data_df)
#         # fig_signals.show()
        
#         # fig_composition = visualizer.plot_portfolio_composition(portfolio)
#         # fig_composition.show()

#         # Generate full report
#         report_path = visualizer.create_full_report(
#             portfolio,
#             benchmark_rets=benchmark_returns_series,
#             price_data_for_signals=price_data_df, # Pass OHLC data here
#             report_title="Sample Backtest Report",
#             output_path="sample_backtest_report.html",
#             show=True
#         )
#         print(f"Full report generated at: {report_path}")

#     else:
#         print("Dummy portfolio has no trades. Cannot generate meaningful plots.")

