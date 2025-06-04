# src/cli/commands/backtest.py
import click
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger
import time # Keep for potential synchronous operations or logging
import asyncio 
from typing import List, Optional, Dict, Any 

from src.core.config import settings 
from src.core.logging_config import setup_logging
from src.core.exceptions import (
    AlgoBotException, ConfigurationError, DataError, 
    StrategyError, BacktestError, StrategyLoadError
)
from src.data.data_manager import DataManager 
from src.strategies.strategy_loader import StrategyLoader 
from src.backtesting.vectorbt_engine import VectorBTEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.visualizations import BacktestVisualizer
from src.core.constants import Kline 

# Setup logging - This should ideally be done once at the application entry point.
# If this script is an entry point, this is okay.
try:
    if not logger._core.handlers: 
        setup_logging(settings) 
except Exception as e:
    print(f"Error setting up logging from backtest CLI: {e}. Using default logger.")
    if not logger._core.handlers: # Ensure logger is minimally configured
        logger.add(lambda _: None) 


class CommaSeparatedPairs(click.ParamType):
    name = "comma_separated_pairs"

    def convert(self, value, param, ctx):
        if value is None:
            return []
        if isinstance(value, list): # Already a list (e.g., from default)
            return value
        if isinstance(value, str):
            pairs = [p.strip().upper() for p in value.split(',') if p.strip()]
            if not pairs:
                self.fail(f"'{value}' is not a valid comma-separated list of pairs.", param, ctx)
            return pairs
        self.fail(f"Expected a string for pairs, got {type(value)}.", param, ctx)


DEFAULT_OUTPUT_DIR_BASE = Path.cwd() / "backtest_results"


async def async_backtest_logic(
    strategy_name: str,
    pairs: List[str],
    start_date: datetime, 
    end_date: datetime,   
    timeframe: str,
    strategy_params_override: Dict[str, Any], 
    initial_capital: float,
    position_size: float,
    size_type: str,
    fees: Optional[float],
    slippage: Optional[float],
    current_run_output_dir: Path, 
    no_visualizations: bool,
    theme: str
):
    """
    Main asynchronous logic for running the backtest.
    """
    data_manager = DataManager()
    # StrategyLoader is instantiated here as its methods are synchronous
    # and might be needed before full async execution for some checks,
    # though create_strategy is called within the async flow.
    strategy_loader_instance = StrategyLoader() 

    if not strategy_loader_instance.loaded_strategies:
        logger.error("StrategyLoader failed to load any strategies within async logic.")
        raise StrategyLoadError("No strategies loaded by StrategyLoader in async_backtest_logic.")

    try:
        await data_manager.initialize()
        logger.info(f"Async backtest logic started for {strategy_name} on {', '.join(pairs)}...")

        vbt_engine = VectorBTEngine(
            initial_capital=initial_capital,
            commission=fees,
            slippage=slippage,
            freq=timeframe, # This is the kline interval for VBT
            default_size=position_size,
            size_type=size_type,
        )
        
        visualizer = None
        if not no_visualizations:
            visualizer = BacktestVisualizer(theme=theme)

        all_results_summary = []

        for pair_symbol in pairs:
            logger.info(f"Processing pair: {pair_symbol} in async logic")
            pair_output_dir = current_run_output_dir / pair_symbol
            pair_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Fetching data for {pair_symbol} from {start_date} to {end_date} (timeframe: {timeframe})...")
            
            historical_data_ohlcv = await data_manager.get_klines(
                pair=pair_symbol,
                interval=timeframe, # Requesting data at the specified CLI timeframe
                start_time=start_date,
                end_time=end_date
            )

            if historical_data_ohlcv is None or historical_data_ohlcv.empty:
                logger.warning(f"No data found for {pair_symbol} for timeframe {timeframe} in the period. Skipping.")
                all_results_summary.append({"pair": pair_symbol, "status": "No Data"})
                continue
            
            # Rename columns for strategy and VectorBT compatibility
            rename_map_std = {
                'open_price': 'open', 'high_price': 'high', 'low_price': 'low',
                'close_price': 'close', 'base_asset_volume': 'volume'
            }
            data_for_strategy_and_vbt = historical_data_ohlcv.rename(
                columns={k: v for k, v in rename_map_std.items() if k in historical_data_ohlcv.columns}
            )

            # Ensure essential OHLC columns exist, VBT needs them
            for col in ['open', 'high', 'low', 'close']:
                if col not in data_for_strategy_and_vbt.columns:
                    logger.error(f"Essential column '{col}' missing for {pair_symbol} after rename. Skipping pair.")
                    all_results_summary.append({"pair": pair_symbol, "status": f"Missing Column: {col}"})
                    # This is a critical data issue, so we skip this pair.
                    continue 

            logger.info(f"Loading strategy '{strategy_name}' for {pair_symbol} and generating signals...")
            strategy_instance = strategy_loader_instance.create_strategy(
                strategy_name, 
                params=strategy_params_override,
                pair_symbol=pair_symbol 
            )
            
            # The strategy receives klines at the `timeframe` specified by the CLI.
            # It's the strategy's `calculate_indicators` job to use this data,
            # potentially resample it if its internal `indicator_frequency` params differ.
            klines_for_calc_indicators = {timeframe: data_for_strategy_and_vbt.copy()}
            
            indicators_dict_by_tf = strategy_instance.calculate_indicators(klines_for_calc_indicators)
            
            # The strategy's signals should be generated based on its configured 'indicator_frequency'.
            # This DataFrame should be the one containing all necessary indicators.
            primary_indicator_freq_for_signals = strategy_instance.get_param('indicator_frequency', timeframe)
            
            if primary_indicator_freq_for_signals not in indicators_dict_by_tf or \
               indicators_dict_by_tf[primary_indicator_freq_for_signals] is None:
                logger.error(f"Indicators for strategy's primary frequency '{primary_indicator_freq_for_signals}' "
                             f"not found after calculate_indicators for {pair_symbol}. Skipping.")
                all_results_summary.append({"pair": pair_symbol, "status": "Indicator Data Missing"})
                continue
                
            df_for_signal_generation = indicators_dict_by_tf[primary_indicator_freq_for_signals]

            # Ensure 'close' column is present for VBT price, and OHLC for strategy signals if needed
            if 'close' not in df_for_signal_generation.columns:
                 logger.error(f"DataFrame for strategy signals (freq: {primary_indicator_freq_for_signals}) is missing 'close' column for {pair_symbol}. Skipping.")
                 all_results_summary.append({"pair": pair_symbol, "status": "Close Column Missing in Indicator DF"})
                 continue

            signals_df_raw = strategy_instance.generate_signals(indicators_dict_by_tf)

            if signals_df_raw is None or not isinstance(signals_df_raw, pd.DataFrame) or \
               not all(col in signals_df_raw.columns for col in ['entry_long', 'exit_long', 'entry_short', 'exit_short']):
                logger.error(f"Strategy '{strategy_name}' did not return a valid signals DataFrame for {pair_symbol}. Skipping.")
                all_results_summary.append({"pair": pair_symbol, "status": "Invalid Signals DF"})
                continue
            
            # Align signals_df_raw index with the price data index for VectorBT
            # The price data for VBT is df_for_signal_generation (which is at primary_indicator_freq_for_signals)
            if not signals_df_raw.index.equals(df_for_signal_generation.index):
                logger.warning(f"Index of raw signals DataFrame and indicator DataFrame for {pair_symbol} do not match. Reindexing signals.")
                signals_df_for_vbt = signals_df_raw.reindex(df_for_signal_generation.index, fill_value=False)
                # Ensure boolean columns remain boolean after reindex
                for bool_col in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
                    if bool_col in signals_df_for_vbt.columns:
                        signals_df_for_vbt[bool_col] = signals_df_for_vbt[bool_col].fillna(False).astype(bool)
            else:
                signals_df_for_vbt = signals_df_raw

            logger.info(f"Running backtest for {pair_symbol} with VectorBT...")
            # VectorBT needs the price data (df_for_signal_generation) and signals (signals_df_for_vbt)
            # Ensure df_for_signal_generation has 'open', 'high', 'low', 'close' columns correctly named.
            portfolio = vbt_engine.run_backtest(
                data=df_for_signal_generation, 
                signals=signals_df_for_vbt,    
                symbol=pair_symbol
            )

            if portfolio is None:
                logger.error(f"VectorBTEngine.run_backtest returned None for {pair_symbol}. Skipping metrics.")
                all_results_summary.append({"pair": pair_symbol, "status": "VBT Portfolio None"})
                continue
            
            if portfolio.trades.count() == 0:
                logger.warning(f"No trades were executed for {pair_symbol} by VectorBT.")
            
            logger.info(f"Calculating performance metrics for {pair_symbol}...")
            # Benchmark returns should be based on the same frequency as the portfolio/signals
            benchmark_returns_series = df_for_signal_generation['close'].pct_change().fillna(0)
            
            calculated_metrics = PerformanceMetrics.calculate_all_metrics(
                portfolio, 
                benchmark_returns=benchmark_returns_series,
            )
            calculated_metrics['pair'] = pair_symbol 
            all_results_summary.append(calculated_metrics) # Store full metrics dict

            click.echo(f"\n--- Results for {pair_symbol} ({strategy_name}) ---")
            key_metrics_to_display = {
                "general_start_date": "Start Date", "general_end_date": "End Date",
                "general_duration_days": "Duration (Days)", "general_initial_capital": "Initial Capital",
                "general_final_value": "Final Value", "returns_total_return_pct": "Total Return %",
                "returns_annualized_return_pct": "Annualized Return %",
                "drawdown_max_drawdown_pct": "Max Drawdown %", "risk_ratios_sharpe_ratio": "Sharpe Ratio",
                "risk_ratios_sortino_ratio": "Sortino Ratio", "trade_stats_total_trades": "Total Trades",
                "trade_stats_win_rate_pct": "Win Rate %", "trade_stats_profit_factor": "Profit Factor",
            }
            # Add benchmark return from VBT portfolio stats if available
            vbt_portfolio_stats = portfolio.stats() # Get stats once
            if 'Benchmark Return [%]' in vbt_portfolio_stats: 
                calculated_metrics['vbt_benchmark_return_%'] = vbt_portfolio_stats['Benchmark Return [%]']
                key_metrics_to_display["vbt_benchmark_return_%"] = "Benchmark Return %"

            for key, name_display in key_metrics_to_display.items():
                value = calculated_metrics.get(key) 
                if value is None or (isinstance(value, float) and np.isnan(value)): value = 'N/A'
                
                if isinstance(value, float) and value != 'N/A':
                    format_str = ".2f"
                    click.echo(f"{name_display:<25}: {value:{format_str}}")
                elif isinstance(value, (datetime, pd.Timestamp)):
                    click.echo(f"{name_display:<25}: {value.strftime('%Y-%m-%d')}")
                else:
                    click.echo(f"{name_display:<25}: {value}")
            
            metrics_df_to_save = PerformanceMetrics.metrics_to_dataframe(calculated_metrics)
            metrics_csv_path = pair_output_dir / f"performance_metrics_{pair_symbol}.csv"
            metrics_df_to_save.to_csv(metrics_csv_path)
            logger.info(f"Performance metrics saved to: {metrics_csv_path}")

            # Ensure metrics are JSON serializable for saving
            json_serializable_metrics = PerformanceMetrics.convert_metrics_to_json_serializable(calculated_metrics)
            with open(pair_output_dir / f"performance_metrics_{pair_symbol}.json", 'w') as f_json:
                json.dump(json_serializable_metrics, f_json, indent=4)
            logger.info(f"Detailed metrics (JSON) saved to: {pair_output_dir / f'performance_metrics_{pair_symbol}.json'}")

            if not no_visualizations and visualizer:
                logger.info(f"Generating visualizations for {pair_symbol}...")
                report_html_path = visualizer.create_full_report(
                    portfolio=portfolio, 
                    benchmark_rets=benchmark_returns_series, 
                    price_data_for_signals=df_for_signal_generation, 
                    report_title=f"Backtest Report: {strategy_name} on {pair_symbol}",
                    output_path=pair_output_dir / f"full_report_{pair_symbol}.html",
                    show=False 
                )
                if report_html_path: logger.info(f"Full HTML report saved to: {report_html_path}")
                else: logger.warning(f"HTML report generation failed for {pair_symbol}.")
            
            if portfolio.trades.count() > 0:
                trades_df_to_save = portfolio.trades.records_readable
                trades_csv_path = pair_output_dir / f"trades_{pair_symbol}.csv"
                trades_df_to_save.to_csv(trades_csv_path, index=False)
                logger.info(f"Trades data saved to: {trades_csv_path}")

        # After processing all pairs, generate overall summary
        if len(all_results_summary) > 0 : # Check if there's anything to summarize
            # Filter out entries that only have "status" (i.e., pairs that failed)
            successful_results = [res for res in all_results_summary if "status" not in res]
            if len(successful_results) > 1: # Only print summary if more than one successful pair
                click.echo("\n\n--- Overall Summary (Successful Pairs) ---")
                summary_df_data = []
                for res in successful_results:
                    summary_df_data.append({
                        "Pair": res.get('pair', 'N/A'),
                        "Total Return %": res.get('returns_total_return_pct', np.nan),
                        "Annualized Return %": res.get('returns_annualized_return_pct', np.nan),
                        "Benchmark Return %": res.get('vbt_benchmark_return_%', np.nan), # Ensure this key exists
                        "Max Drawdown %": res.get('drawdown_max_drawdown_pct', np.nan),
                        "Sharpe Ratio": res.get('risk_ratios_sharpe_ratio', np.nan),
                        "Total Trades": res.get('trade_stats_total_trades', 0),
                        "Win Rate %": res.get('trade_stats_win_rate_pct', np.nan)
                    })
                summary_df = pd.DataFrame(summary_df_data)
                click.echo(summary_df.to_string(index=False, float_format="%.2f"))
                summary_df.to_csv(current_run_output_dir / "overall_summary_successful.csv", index=False)
                logger.info(f"Overall summary for successful pairs saved to: {current_run_output_dir / 'overall_summary_successful.csv'}")
            elif len(successful_results) == 1:
                logger.info("Only one pair processed successfully, overall summary not generated.")
            else:
                logger.warning("No pairs processed successfully, overall summary cannot be generated.")


        logger.success(f"Async backtest logic for {strategy_name} completed.")

    except ConfigurationError as e_conf: # Specific catch for ConfigurationError
        logger.critical(f"Configuration error during async backtest: {e_conf}", exc_info=True)
        raise 
    except DataError as e_data: # Specific catch for DataError
        logger.critical(f"Data error during async backtest: {e_data}", exc_info=True)
        raise
    except StrategyError as e_strat: # Includes StrategyLoadError
        logger.critical(f"Strategy error during async backtest: {e_strat}", exc_info=True)
        raise
    except BacktestError as e_bt: # Specific catch for BacktestError
        logger.critical(f"Backtest execution error during async backtest: {e_bt}", exc_info=True)
        raise
    except AlgoBotException as e_algo: # Catch other AlgoBot exceptions
        logger.critical(f"An application error occurred in async backtest: {e_algo}", exc_info=True)
        raise
    except Exception as e_unexp: # Catch any other unexpected error
        logger.critical(f"An unexpected error occurred during async backtest: {e_unexp}", exc_info=True)
        raise
    finally:
        if data_manager: 
            await data_manager.close()


@click.command()
@click.option('--strategy', '-s', "strategy_name_cli", required=True, type=str, help="Name of the strategy to backtest (as defined in strategy_class.name).")
@click.option('--pair', '-p', 'pairs', required=True, type=CommaSeparatedPairs(), help="Comma-separated trading pair(s) (e.g., BTCUSDT,ETHUSDT).")
@click.option('--start-date', '-sd', required=True, type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]), help="Start date for backtesting (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).")
@click.option('--end-date', '-ed', required=True, type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]), help="End date for backtesting (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).")
@click.option('--timeframe', '-tf', default='1h', type=str, help="Timeframe for kline data (e.g., 1m, 5m, 1h, 1d). Default: 1h.")
@click.option('--params-file', '-pf', type=click.Path(exists=True, dir_okay=False, resolve_path=True), help="Path to JSON file with strategy parameters.")
@click.option('--initial-capital', '-ic', default=10000.0, type=float, show_default=True, help="Initial capital for the backtest.")
@click.option('--position-size', '-ps', default=0.1, type=float, show_default=True, help="Default position size (e.g., 0.1 for 10% of capital).")
@click.option('--size-type', default='percent', type=click.Choice(['percent', 'amount', 'value'], case_sensitive=False), show_default=True, help="Type of position sizing for VectorBTEngine.")
@click.option('--fees', type=float, default=None, help="Fixed commission fee per trade (e.g., 0.001 for 0.1%). Overrides dynamic fees.")
@click.option('--slippage', type=float, default=None, help="Fixed slippage per trade (e.g., 0.0005 for 0.05%). Overrides dynamic slippage.")
@click.option('--output-dir', '-o', "output_dir_base_cli", type=click.Path(file_okay=False, writable=True, resolve_path=True), default=str(DEFAULT_OUTPUT_DIR_BASE), show_default=True, help="Base directory to save backtest results.")
@click.option('--no-visualizations', '-nv', is_flag=True, help="Disable generation of visualizations and HTML report.")
@click.option('--dry-run', '-dr', is_flag=True, help="Validate inputs and setup without running the full backtest.")
@click.option('--theme', default='dark', type=click.Choice(['dark', 'light'], case_sensitive=False), show_default=True, help="Theme for visualizations.")
def backtest(
    strategy_name_cli: str,
    pairs: List[str], 
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
    params_file: Optional[str],
    initial_capital: float,
    position_size: float,
    size_type: str,
    fees: Optional[float],
    slippage: Optional[float],
    output_dir_base_cli: str, 
    no_visualizations: bool,
    dry_run: bool,
    theme: str
):
    """
    CLI entry point to run a trading strategy backtest using historical market data.
    """
    strategy_name_cleaned = strategy_name_cli.strip() 

    run_id_str = f"{strategy_name_cleaned}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir_base_path_obj = Path(output_dir_base_cli)
    current_run_output_dir_path_obj = output_dir_base_path_obj / run_id_str
    try:
        current_run_output_dir_path_obj.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Failed to create output directory {current_run_output_dir_path_obj}: {e}")
        click.echo(f"Error: Could not create output directory. {e}", err=True)
        return

    # Initial logging (synchronous part)
    logger.info(f"Preparing backtest run: {run_id_str}")
    logger.info(f"Strategy: {strategy_name_cleaned}, Pairs: {', '.join(pairs)}, Timeframe: {timeframe}")
    logger.info(f"Period: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Initial Capital: {initial_capital}, Position Size: {position_size} ({size_type})")
    logger.info(f"Output Directory: {current_run_output_dir_path_obj}")

    # Ensure dates are timezone-aware (UTC)
    start_date_utc = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date.astimezone(timezone.utc)
    end_date_utc = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date.astimezone(timezone.utc)

    # Load strategy parameters (synchronous part)
    strategy_params_override_dict = {}
    if params_file:
        try:
            with open(params_file, 'r') as f_params:
                strategy_params_override_dict = json.load(f_params)
            logger.info(f"Loaded strategy parameters from: {params_file}")
        except Exception as e_params:
            logger.error(f"Failed to load strategy parameters from {params_file}: {e_params}")
            click.echo(f"Error: Could not load strategy parameters. {e_params}", err=True)
            return

    # Dry run logic (synchronous part)
    if dry_run:
        logger.info("Dry run mode: Validating strategy loading...")
        strategy_loader_dry_run = StrategyLoader() # Instantiate for dry run
        if not strategy_loader_dry_run.loaded_strategies:
            logger.error("Dry run: StrategyLoader failed to load any strategies.")
            click.echo("Error: No strategies loaded by StrategyLoader (dry run). Aborting.", err=True)
            return
        try:
            # Use the first pair for dry run instantiation check, or a placeholder
            temp_pair_for_dry_run = pairs[0] if pairs else "DRY_RUN_PAIR_PLACEHOLDER"
            _ = strategy_loader_dry_run.create_strategy(
                strategy_name_cleaned, 
                params=strategy_params_override_dict,
                pair_symbol=temp_pair_for_dry_run 
            )
            logger.success(f"Dry run: Strategy '{strategy_name_cleaned}' loaded successfully with params: {strategy_params_override_dict}.")
            click.echo("Dry run completed successfully. Inputs appear valid.")
            return # End of dry run
        except Exception as e_dry_load:
            logger.error(f"Dry run: Failed to load or instantiate strategy '{strategy_name_cleaned}': {e_dry_load}", exc_info=True)
            click.echo(f"Error: Failed to load strategy during dry run. {e_dry_load}", err=True)
            return

    # Prepare arguments for the asynchronous logic
    async_run_args = {
        "strategy_name": strategy_name_cleaned,
        "pairs": pairs,
        "start_date": start_date_utc,
        "end_date": end_date_utc,
        "timeframe": timeframe,
        "strategy_params_override": strategy_params_override_dict,
        "initial_capital": initial_capital,
        "position_size": position_size,
        "size_type": size_type,
        "fees": fees,
        "slippage": slippage,
        # "output_dir_base": output_dir_base_path_obj, # Argument supprimé
        "current_run_output_dir": current_run_output_dir_path_obj, 
        "no_visualizations": no_visualizations,
        "theme": theme,
    }

    try:
        asyncio.run(async_backtest_logic(**async_run_args))
        click.echo(f"\nBacktest run {run_id_str} completed. Results saved in: {current_run_output_dir_path_obj}")
    except AlgoBotException as e_algo_final: # Catch specific application exceptions
        logger.critical(f"Backtest run {run_id_str} failed with an application error: {e_algo_final}", exc_info=True)
        click.echo(f"Application Error: {e_algo_final}", err=True)
    except Exception as e_final: # Catch any other unexpected errors from async execution
        logger.critical(f"Backtest run {run_id_str} failed with an unexpected error: {e_final}", exc_info=True)
        click.echo(f"Unexpected Error: {e_final}. Check logs at {current_run_output_dir_path_obj} for details.", err=True)

if __name__ == '__main__':
    backtest()
