# src/backtesting/visualization/charts.py

"""
Provides a library of plotting functions for visualizing backtest results.

This module uses the `plotly` library to create interactive and high-quality
charts. Each function is designed to visualize a specific aspect of the
backtest, such as the equity curve or drawdowns.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtesting.engine.base import BacktestResult


def plot_equity_curve(
    result: BacktestResult,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Equity Curve"
) -> go.Figure:
    """
    Generates an interactive plot of the portfolio's equity curve over time.

    Args:
        result (BacktestResult): The result object from a backtest run.
        benchmark (Optional[pd.Series]): An optional benchmark series (e.g., S&P 500)
                                          to plot for comparison.
        title (str): The title of the chart.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()

    # Plot Equity Curve
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        )
    )

    # Plot Benchmark if provided
    if benchmark is not None:
        # Normalize benchmark to start at the same value as the portfolio
        initial_value = result.equity_curve.iloc[0]
        normalized_benchmark = initial_value * (1 + benchmark.pct_change().cumsum()).fillna(0)
        
        fig.add_trace(
            go.Scatter(
                x=normalized_benchmark.index,
                y=normalized_benchmark,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', dash='dash')
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        legend=dict(x=0.01, y=0.99, bordercolor="black", borderwidth=1),
        template="plotly_white"
    )
    return fig


def plot_drawdowns(
    result: BacktestResult,
    title: str = "Portfolio Drawdowns"
) -> go.Figure:
    """
    Generates an interactive plot of the portfolio's drawdown periods.

    Args:
        result (BacktestResult): The result object from a backtest run.
        title (str): The title of the chart.

    Returns:
        go.Figure: A Plotly figure object.
    """
    equity_curve = result.equity_curve
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max

    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill='tozeroy',
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1)
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".2%",
        legend=dict(x=0.01, y=0.99, bordercolor="black", borderwidth=1),
        template="plotly_white"
    )
    return fig
