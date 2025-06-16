"""
Ce module est le cœur du moteur de backtesting, utilisant la bibliothèque `vectorbt`.
Il a été entièrement refactorisé pour être piloté par la nouvelle structure de
configuration Pydantic (`SimulationConfig`) et pour intégrer les modèles
dynamiques de calcul de frais et de slippage.
"""

import logging
import pandas as pd
import numpy as np
import vectorbt as vbt # type: ignore # type: ignore
from typing import Type, Optional, Dict, Any, Union, Tuple, List


# --- Imports Locaux ---
from src.strategies.base_strategy import BaseStrategy
from src.strategies.signal_formatter import SignalFormatter
from src.optimization.config import SimulationConfig
from src.backtesting.fee_calculator import get_fee_calculator
from src.backtesting.slippage_model import get_slippage_model
from src.core.constants import Kline # Ajouté pour Kline.INTERVAL_1MINUTE
from src.core.exceptions import BacktestFailureError

logger = logging.getLogger(__name__)

class VectorBTEngine:
    """
    Moteur de backtesting vectorisé qui orchestre l'exécution d'une stratégie
    sur des données de marché en utilisant vectorbt.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategy: BaseStrategy,
                 config: SimulationConfig):
        """
        Initialise le moteur de backtesting.

        Args:
            data: DataFrame contenant les données de marché (OHLCV).
            strategy: Une instance de la stratégie à exécuter.
            config: Un objet Pydantic `SimulationConfig` contenant tous les
                    paramètres de la simulation.
        """
        self.data = data
        self.strategy = strategy
        self.config = config
        
        self.fee_calculator = get_fee_calculator(self.config.fee_config)
        self.slippage_model = get_slippage_model(self.config.slippage_config)
        
        logger.info("VectorBTEngine initialisé avec la nouvelle configuration de simulation.")

    def _sanitize_series(self, series: Optional[pd.Series], series_name: str) -> Optional[pd.Series]:
        """Assainit une série pour s'assurer qu'elle est numérique et scalaire."""
        if series is None:
            return None
        
        if not pd.api.types.is_numeric_dtype(series.dtype):
            logger.warning(
                f"La série '{series_name}' (dtype: {series.dtype}) pour {self.strategy.pair_symbol} "
                f"n'est pas de type numérique. Tentative de conversion/assainissement."
            )
            
            def _try_extract_scalar(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    return x[0] if len(x) > 0 and isinstance(x[0], (int, float, np.number)) else np.nan
                return x

            sanitized_values = series.apply(_try_extract_scalar)
            series = pd.to_numeric(sanitized_values, errors='coerce')
            logger.info(f"Série '{series_name}' pour {self.strategy.pair_symbol} assainie. Nouveau dtype: {series.dtype}")
        return series

    def _prepare_signals(self) -> pd.DataFrame:
        """
        Prépare les signaux de trading en exécutant la logique de la stratégie.
        """
        logger.debug("Génération des indicateurs et des signaux de la stratégie...")
        indicators = self.strategy.calculate_indicators(self.data)
        raw_signals = self.strategy.generate_signals(indicators)
        
        return raw_signals

    def run(self) -> vbt.Portfolio:
        """
        Exécute le backtest complet.

        Returns:
            Un objet `vbt.Portfolio` contenant les résultats du backtest.

        Raises:
            BacktestFailureError: Si le backtest échoue ou ne produit aucun trade.
        """
        try:
            signals = self._prepare_signals()

            close_col_name = next((c for c in self.data.columns if c.endswith('_close')), None)
            if close_col_name is None:
                if 'close' in self.data.columns:
                    close_col_name = 'close'
                else:
                    raise BacktestFailureError(
                        f"Impossible de trouver une colonne de prix de clôture ('close' ou '*_close'). "
                        f"Colonnes disponibles: {self.data.columns.tolist()}"
                    )
            logger.debug(f"Colonne de clôture identifiée pour le backtest : '{close_col_name}'")
            close_prices = self.data[close_col_name]

            # Assainissement de close_prices
            close_prices = self._sanitize_series(close_prices, f"close_prices ({close_col_name})")
            if close_prices is None or close_prices.isnull().all():
                raise BacktestFailureError(f"Les prix de clôture pour {self.strategy.pair_symbol} sont tous NaN après assainissement.")

            fees = self.config.fee_config.value if self.config.fee_config.method == 'PERCENTAGE' else 0.0
            if self.config.fee_config.method != 'PERCENTAGE':
                logger.warning(f"La méthode de frais '{self.config.fee_config.method}' n'est pas directement supportée. Les frais ne seront pas appliqués.")

            # Slippage series est basée sur close_prices, qui est maintenant assaini.
            slippage_series = self.slippage_model.generate_slippage_series(close_prices)

            # Préparer sl_stop et tp_stop
            sl_series = signals.get('sl')
            sl_series = self._sanitize_series(sl_series, "sl_stop")
            if sl_series is not None and sl_series.isnull().all():
                logger.debug("La série sl_stop ne contient que des NaN, passage de None à vectorbt.")
                sl_series = None

            tp_series = signals.get('tp')
            tp_series = self._sanitize_series(tp_series, "tp_stop")
            if tp_series is not None and tp_series.isnull().all():
                logger.debug("La série tp_stop ne contient que des NaN, passage de None à vectorbt.")
                tp_series = None
            logger.debug("Exécution du backtest avec vectorbt...")
            
            backtest_frequency = self.strategy.get_param('indicator_frequency')
            
            # --- CORRECTION ---
            # Suppression des paramètres de callback (stop_func, call_seq) qui ne sont
            # pas supportés par cette fonction et causaient le crash.
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=signals['entry_long'],
                exits=signals['exit_long'],
                short_entries=signals['entry_short'],
                short_exits=signals['exit_short'],
                freq=backtest_frequency,
                init_cash=self.config.initial_capital,
                fees=fees,
                slippage=slippage_series,
                sl_stop=sl_series,
                tp_stop=tp_series
            )
            
            if portfolio.trades.count() == 0:
                logger.warning("Le backtest s'est terminé sans aucune transaction.")
                raise BacktestFailureError("Aucun trade exécuté.")

            logger.info(f"Backtest réussi. Nombre de trades: {portfolio.trades.count()}.")
            return portfolio

        except Exception as e:
            logger.error(f"Une erreur est survenue durant l'exécution du backtest : {e}", exc_info=True)
            raise BacktestFailureError(f"Échec du moteur vectorbt: {e}") from e
