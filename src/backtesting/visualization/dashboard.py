# src/backtesting/visualization/dashboards.py

"""
Provides functions to create and export standalone dashboards for backtest results.

This module combines various charts and metrics into a single, interactive
HTML file for easy analysis and sharing.
"""

from typing import Dict, Any, Optional

import pandas as pd
from plotly.graph_objects import Figure

from src.backtesting.analysis.analyzer import BacktestAnalyzer
from src.backtesting.engine.base import BacktestResult
from src.backtesting.visualization.charts import plot_equity_curve, plot_drawdowns


def create_standalone_html_report(
    result: BacktestResult,
    metrics_report: pd.Series,
    report_title: str = "Backtest Performance Report",
    output_filename: str = "backtest_report.html",
    benchmark: Optional[pd.Series] = None
) -> None:
    """
    Generates a full standalone HTML report with charts and metrics.

    Args:
        result (BacktestResult): The result object from a backtest run.
        metrics_report (pd.Series): A series containing calculated metrics.
        report_title (str): The main title for the HTML report.
        output_filename (str): The name of the file to save the report to.
        benchmark (Optional[pd.Series]): An optional benchmark series for comparison.
    """
    # 1. Generate figures
    equity_fig = plot_equity_curve(result, benchmark=benchmark)
    drawdown_fig = plot_drawdowns(result)

    # 2. Convert metrics Series to HTML table
    metrics_html = metrics_report.to_frame(name='Value').to_html()

    # 3. Assemble the HTML document
    html_content = f"""
    <html>
    <head>
        <title>{report_title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            .container {{ display: flex; flex-direction: column; gap: 40px; }}
            .chart {{ border: 1px solid #ddd; padding: 10px; }}
            .metrics {{ border: 1px solid #ddd; padding: 10px; }}
            table {{ border-collapse: collapse; width: 400px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{report_title}</h1>

            <div class="chart">
                <h2>Equity Curve & Drawdowns</h2>
                {equity_fig.to_html(full_html=False, include_plotlyjs='cdn')}
                {drawdown_fig.to_html(full_html=False, include_plotlyjs=False)}
            </div>

            <div class="metrics">
                <h2>Performance Metrics</h2>
                {metrics_html}
            </div>
        </div>
    </body>
    </html>
    """

    # 4. Save to file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

