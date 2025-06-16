# src/cli/commands/backtest.py
import click
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from loguru import logger
import time
import asyncio
from typing import List, Optional, Dict, Any, Union

from src.core.config import get_settings
from src.core.logging_config import setup_logging
from src.core.exceptions import (
    AlgoBotException, ConfigurationError, DataError,
    StrategyError, BacktestError, StrategyLoadError, IndicatorCalculationError,
    SignalGenerationError
)
from src.data.data_manager import DataManager
from src.data.enriched_dataframe import EnrichedDataFrame
from src.strategies.strategy_loader import StrategyLoader
from src.strategies.base_strategy import BaseStrategy
from src.backtesting.vectorbt_engine import VectorBTEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.visualizations import BacktestVisualizer
from src.utils.exchange_utils import normalize_pair_symbol

# Configuration du logging
try:
    if not getattr(logger, 'level', None):
        setup_logging(get_settings())
except Exception as e:
    print(f"Error setting up logging from backtest CLI: {e}. Using default logger.")
    if not getattr(logger, 'level', None):
        logger.add(lambda _: None)

class CommaSeparatedPairs(click.ParamType):
    name = "comma_separated_pairs"

    def convert(self, value, param, ctx):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            pairs = [normalize_pair_symbol(p) for p in value.split(',') if p.strip()] # Use utility
            if not pairs:
                self.fail(f"'{value}' is not a valid comma-separated list of pairs.", param, ctx)
            return pairs
        self.fail(f"Expected a string for pairs, got {type(value)}.", param, ctx)

DEFAULT_OUTPUT_DIR_BASE = Path.cwd() / "backtest_results"

def convert_metrics_to_json_serializable(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convertit un dictionnaire de métriques pour qu'il soit sérialisable en JSON."""
    serializable_dict = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (datetime, pd.Timestamp)):
            serializable_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            serializable_dict[key] = value.total_seconds()
        elif isinstance(value, (np.integer, np.int64)):
            serializable_dict[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            serializable_dict[key] = float(value) if not np.isnan(value) else None
        elif isinstance(value, np.bool_):
            serializable_dict[key] = bool(value)
        elif pd.isna(value):
            serializable_dict[key] = None
        else:
            try:
                json.dumps(value)
                serializable_dict[key] = value
            except TypeError:
                serializable_dict[key] = str(value)
    return serializable_dict

async def async_backtest_logic(
    strategy_name: str,
    pairs: List[str],
    start_date: datetime,
    end_date: datetime,
    strategy_params_override: Dict[str, Any],
    initial_capital: float,
    position_size: float,
    size_type: str,
    fees: Optional[float],
    slippage: Optional[float],
    current_run_output_dir: Path,
    no_visualizations: bool,
    theme: str,
    use_enriched_data: bool
):
    """Logique principale du backtest, supportant les données brutes et enrichies."""
    data_manager = DataManager()
    strategy_loader_instance = StrategyLoader()

    if not strategy_loader_instance.loaded_strategies:
        raise StrategyLoadError("Aucune stratégie n'a été chargée par le StrategyLoader.")

    try:
        await data_manager.initialize()
        logger.info(f"Logique de backtest asynchrone démarrée pour {strategy_name} sur {', '.join(pairs)}...")

        execution_freq = '1m' if use_enriched_data else strategy_params_override.get('indicator_frequency', '1h')

        vbt_engine = VectorBTEngine(
            initial_capital=initial_capital, commission=fees, slippage=slippage,
            freq=execution_freq, default_size=position_size, size_type=size_type
        )
        visualizer = BacktestVisualizer(theme=theme) if not no_visualizations else None
        all_results_summary = []

        for pair_symbol in pairs:
            logger.info(f"Traitement de la paire : {pair_symbol}")
            pair_output_dir = current_run_output_dir / pair_symbol
            pair_output_dir.mkdir(parents=True, exist_ok=True)

            strategy_instance: BaseStrategy = strategy_loader_instance.create_strategy(
                strategy_name, params=strategy_params_override, pair_symbol=pair_symbol
            )

            klines_data: Optional[Union[EnrichedDataFrame, pd.DataFrame]] = None
            if use_enriched_data:
                logger.info(f"Chargement des données enrichies pour {pair_symbol}...")
                klines_data = await data_manager.get_enriched_klines(
                    pair=pair_symbol, start_time=start_date, end_time=end_date
                )
                if klines_data is None:
                    logger.warning(f"Aucune donnée enrichie trouvée pour {pair_symbol}. Passage au suivant.")
                    continue
            else: # Mode legacy
                logger.info(f"Chargement des données brutes (legacy) pour {pair_symbol}...")
                required_frequencies = set(strategy_instance.required_timeframes)
                required_frequencies.add(strategy_instance.get_param('indicator_frequency', '1h'))
                
                # En mode legacy, on ne charge que la fréquence principale des indicateurs pour le backtest
                main_freq = strategy_instance.get_param('indicator_frequency', '1h')
                klines_data = await data_manager.get_klines(
                    pair=pair_symbol, interval=main_freq, start_time=start_date, end_time=end_date
                )
                if klines_data is None or klines_data.empty:
                    logger.warning(f"Aucune donnée trouvée pour {pair_symbol} pour le timeframe {main_freq}.")
                    continue


            # Calcul des indicateurs et génération des signaux
            try:
                logger.info(f"La stratégie '{strategy_name}' calcule ses indicateurs...")
                indicators_df = strategy_instance.calculate_indicators(klines_data)

                logger.info(f"Génération des signaux pour {strategy_name} sur {pair_symbol}...")
                signals_df_raw = strategy_instance.generate_signals(indicators_df)

            except (IndicatorCalculationError, SignalGenerationError) as e:
                logger.error(f"Erreur lors de la préparation de la stratégie pour {pair_symbol}: {e}", exc_info=True)
                continue
            
            # Sauvegarde pour le débogage
            try:
                debug_df_with_signals = indicators_df.join(signals_df_raw, how='left')
                debug_file_path = pair_output_dir / f"indicators_and_signals_{pair_symbol}.csv"
                debug_df_with_signals.to_csv(debug_file_path)
                logger.info(f"Fichier de débogage sauvegardé sur : {debug_file_path}")
            except Exception as e_debug:
                logger.error(f"Échec de la création du fichier de débogage : {e_debug}")

            if signals_df_raw is None or signals_df_raw.empty:
                logger.warning(f"La stratégie n'a généré aucun signal pour {pair_symbol}. Passage au suivant.")
                all_results_summary.append({"pair": pair_symbol, "status": "Aucun signal"})
                continue
            
            # Lancer le backtest
            logger.info(f"Lancement du backtest pour {pair_symbol} avec VectorBT...")
            
            portfolio = vbt_engine.run_backtest(
                data=klines_data if isinstance(klines_data, EnrichedDataFrame) else indicators_df, 
                signals=signals_df_raw, 
                symbol=pair_symbol
            )
            
            if portfolio is None or portfolio.trades.count() == 0:
                logger.warning(f"Aucun trade n'a été exécuté pour {pair_symbol} pendant le backtest.")
                all_results_summary.append({"pair": pair_symbol, "status": "Aucun trade exécuté"})
                continue

            # Calcul et sauvegarde des métriques et visualisations
            logger.info(f"Calcul des métriques de performance pour {pair_symbol}...")
            
            benchmark_data = klines_data.get_view('1m') if use_enriched_data else indicators_df
            benchmark_returns = benchmark_data['close'].pct_change().fillna(0)
            
            calculated_metrics = PerformanceMetrics.calculate_all_metrics(portfolio, benchmark_returns)
            calculated_metrics['pair'] = pair_symbol
            all_results_summary.append(calculated_metrics)
            
            metrics_df = PerformanceMetrics.metrics_to_dataframe(calculated_metrics)
            metrics_df.to_csv(pair_output_dir / f"performance_metrics_{pair_symbol}.csv")
            json_metrics = convert_metrics_to_json_serializable(calculated_metrics)
            with open(pair_output_dir / f"performance_metrics_{pair_symbol}.json", 'w') as f:
                json.dump(json_metrics, f, indent=4)

            if visualizer:
                logger.info(f"Génération des visualisations pour {pair_symbol}...")
                price_data_for_plot = klines_data.get_view('1m') if use_enriched_data else indicators_df
                report_path = visualizer.create_full_report(
                    portfolio=portfolio, benchmark_rets=benchmark_returns,
                    price_data_for_signals=price_data_for_plot,
                    report_title=f"Backtest: {strategy_name} on {pair_symbol}",
                    output_path=pair_output_dir / f"full_report_{pair_symbol}.html",
                    show=False
                )
                if report_path: logger.info(f"Rapport HTML complet sauvegardé sur : {report_path}")

        # Résumé global
        if all_results_summary:
            successful_results = [res for res in all_results_summary if "status" not in res]
            if len(successful_results) > 1:
                summary_df = pd.DataFrame([{
                    "Paire": res.get('pair', 'N/A'),
                    "Rendement total %": res.get('returns_total_return_pct', np.nan),
                    "Max Drawdown %": res.get('drawdown_max_drawdown_pct', np.nan),
                    "Ratio de Sharpe": res.get('risk_ratios_sharpe_ratio', np.nan),
                    "Trades totaux": res.get('trade_stats_total_trades', 0)
                } for res in successful_results])
                click.echo("\n--- Résumé global ---")
                click.echo(summary_df.to_string(index=False, float_format="%.2f"))
                summary_df.to_csv(current_run_output_dir / "overall_summary.csv", index=False)

    except Exception as e:
        logger.critical(f"Une erreur inattendue est survenue dans la logique de backtest : {e}", exc_info=True)
        raise
    finally:
        if data_manager:
            await data_manager.close()

@click.command()
@click.option('--strategy', '-s', "strategy_name_cli", required=True, type=str, help="Name of the strategy to backtest.")
@click.option('--pair', '-p', 'pairs', required=True, type=CommaSeparatedPairs(), help="Comma-separated trading pair(s).")
@click.option('--start-date', '-sd', required=True, type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]), help="Start date (YYYY-MM-DD).")
@click.option('--end-date', '-ed', required=True, type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]), help="End date (YYYY-MM-DD).")
@click.option('--params-file', '-pf', type=click.Path(exists=True, dir_okay=False, resolve_path=True), help="Path to JSON file with strategy parameters.")
@click.option('--initial-capital', '-ic', default=10000.0, type=float, show_default=True, help="Initial capital.")
@click.option('--position-size', '-ps', default=0.1, type=float, show_default=True, help="Default position size.")
@click.option('--size-type', default='percent', type=click.Choice(['percent', 'amount', 'value'], case_sensitive=False), show_default=True, help="Position sizing type.")
@click.option('--fees', type=float, default=None, help="Fixed commission fee per trade.")
@click.option('--slippage', type=float, default=None, help="Fixed slippage per trade.")
@click.option('--output-dir', '-o', "output_dir_base_cli", type=click.Path(file_okay=False, writable=True, resolve_path=True), default=str(DEFAULT_OUTPUT_DIR_BASE), show_default=True, help="Base directory for results.")
@click.option('--no-visualizations', '-nv', is_flag=True, help="Disable visualizations and HTML report.")
@click.option('--theme', default='dark', type=click.Choice(['dark', 'light'], case_sensitive=False), show_default=True, help="Theme for visualizations.")
@click.option('--use-enriched-data', is_flag=True, default=False, help="Use enriched data format (K_*) instead of legacy raw data.")
@click.option('--dry-run', '-dr', is_flag=True, help="Validate inputs without running the backtest.")
def backtest(
    strategy_name_cli: str, pairs: List[str], start_date: datetime, end_date: datetime,
    params_file: Optional[str], initial_capital: float,
    position_size: float, size_type: str, fees: Optional[float], slippage: Optional[float],
    output_dir_base_cli: str, no_visualizations: bool, theme: str, use_enriched_data: bool,
    dry_run: bool
):
    """Point d'entrée CLI pour exécuter un backtest de stratégie de trading."""
    strategy_name_cleaned = strategy_name_cli.strip()
    run_id_str = f"{strategy_name_cleaned}_{'enriched' if use_enriched_data else 'raw'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir_base_path_obj = Path(output_dir_base_cli)
    current_run_output_dir_path_obj = output_dir_base_path_obj / run_id_str
    try:
        current_run_output_dir_path_obj.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Impossible de créer le répertoire de sortie {current_run_output_dir_path_obj}: {e}")
        return

    logger.info(f"Préparation du backtest : {run_id_str}")
    logger.info(f"Stratégie: {strategy_name_cleaned}, Paires: {', '.join(pairs)}, Mode: {'Enriched Data' if use_enriched_data else 'Legacy Raw Data'}")

    start_date_utc = start_date.astimezone(timezone.utc) if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    
    # Correction : La date de fin doit inclure toute la journée.
    # On ajoute 1 jour et on retire 1 microseconde pour couvrir jusqu'à 23:59:59.999999
    end_date_inclusive = end_date + timedelta(days=1, microseconds=-1)
    end_date_utc = end_date_inclusive.astimezone(timezone.utc) if end_date_inclusive.tzinfo else end_date_inclusive.replace(tzinfo=timezone.utc)
    
    logger.info(f"Plage de dates du backtest (inclusive) : {start_date_utc} à {end_date_utc}")


    strategy_params_override_dict = {}
    if params_file:
        try:
            with open(params_file, 'r') as f:
                strategy_params_override_dict = json.load(f)
        except Exception as e:
            logger.error(f"Échec du chargement des paramètres de stratégie : {e}")
            return

    if dry_run:
        logger.info("Mode Dry run : Validation des entrées terminée.")
        return

    async_run_args = {
        "strategy_name": strategy_name_cleaned, "pairs": pairs, "start_date": start_date_utc,
        "end_date": end_date_utc, "strategy_params_override": strategy_params_override_dict,
        "initial_capital": initial_capital, "position_size": position_size, "size_type": size_type,
        "fees": fees, "slippage": slippage, "current_run_output_dir": current_run_output_dir_path_obj,
        "no_visualizations": no_visualizations, "theme": theme, "use_enriched_data": use_enriched_data
    }

    try:
        asyncio.run(async_backtest_logic(**async_run_args))
        click.echo(f"\nBacktest {run_id_str} terminé. Résultats sauvegardés dans : {current_run_output_dir_path_obj}")
    except AlgoBotException as e:
        logger.critical(f"Le backtest a échoué avec une erreur applicative : {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Le backtest a échoué avec une erreur inattendue : {e}", exc_info=True)

if __name__ == '__main__':
    backtest()
