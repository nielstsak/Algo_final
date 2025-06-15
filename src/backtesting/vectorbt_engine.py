"""
Ce module est le cœur du moteur de backtesting, utilisant la bibliothèque `vectorbt`.
Il a été entièrement refactorisé pour être piloté par la nouvelle structure de
configuration Pydantic (`SimulationConfig`) et pour intégrer les modèles
dynamiques de calcul de frais et de slippage.
"""

import logging
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Type

# --- Imports Locaux ---
from src.strategies.base_strategy import BaseStrategy
from src.strategies.signal_formatter import SignalFormatter
from src.optimization.config import SimulationConfig
from src.backtesting.fee_calculator import get_fee_calculator
from src.backtesting.slippage_model import get_slippage_model
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

            fees = self.config.fee_config.value if self.config.fee_config.method == 'PERCENTAGE' else 0.0
            if self.config.fee_config.method != 'PERCENTAGE':
                logger.warning(f"La méthode de frais '{self.config.fee_config.method}' n'est pas directement supportée. Les frais ne seront pas appliqués.")

            slippage_series = self.slippage_model.generate_slippage_series(close_prices)

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
                sl_stop=signals.get('sl'),
                tp_stop=signals.get('tp')
            )
            
            if portfolio.trades.count() == 0:
                logger.warning("Le backtest s'est terminé sans aucune transaction.")
                raise BacktestFailureError("Aucun trade exécuté.")

            logger.info(f"Backtest réussi. Nombre de trades: {portfolio.trades.count()}.")
            return portfolio

        except Exception as e:
            logger.error(f"Une erreur est survenue durant l'exécution du backtest : {e}", exc_info=True)
            raise BacktestFailureError(f"Échec du moteur vectorbt: {e}") from e
