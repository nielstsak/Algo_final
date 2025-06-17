# src/backtesting/vectorbt_engine.py
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.base import Portfolio
from src.core.constants import Kline
from src.core.exceptions import DataError
from src.core.config import Settings, get_settings
from src.core.exceptions import  ConfigurationError
from src.strategies.base import BaseStrategy
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.signal_adapter import SignalAdapter

logger = logging.getLogger(__name__)

# --- Classe VectorBTBacktestingEngine (Ajout) ---
# Cette classe est ajoutée pour répondre à l'ImportError.
# Elle intègre la logique de backtesting en utilisant le code existant.

class VectorBTBacktestingEngine:
    """
    Moteur de backtesting utilisant la bibliothèque VectorBT.

    Cette classe orchestre l'exécution d'un backtest vectoriel. Elle utilise
    SignalAdapter pour transformer les signaux de la stratégie, exécute le
    backtest via vectorbt, et calcule un ensemble complet de métriques de
    performance.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str,
        settings: Optional[Settings] = None,
        backtest_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise le moteur de backtesting.

        Args:
            strategy: L'instance de la stratégie à backtester.
            symbol: Le symbole de la paire de trading (ex: 'BTC/USDT').
            settings: L'objet de configuration global. Si None, il sera chargé.
            backtest_config: Dictionnaire de configuration spécifique au backtest.
        """
        self.strategy = strategy
        self.symbol = symbol
        self.settings = settings or get_settings()
        
        # Charge la configuration de backtest depuis les settings ou utilise celle fournie
        self.backtest_config = backtest_config or self._load_backtest_config()
        
        self.signal_adapter = SignalAdapter()

    def _load_backtest_config(self) -> Dict[str, Any]:
        """Charge la configuration de backtesting depuis l'objet settings."""
        try:
            # Idéalement, les paramètres de backtest seraient dans une section
            # dédiée de config.yaml. Pour l'instant, on utilise des valeurs
            # par défaut si elles ne sont pas trouvées.
            config_dict = self.settings.model_dump().get("backtesting", {})
            return {
                "initial_capital": config_dict.get("initial_capital", 10000.0),
                "commission_pct": config_dict.get("commission_pct", 0.001),
                "slippage_pct": config_dict.get("slippage_pct", 0.0005),
                "freq": config_dict.get("freq", "1D"),
            }
        except Exception as e:
            logger.warning(f"Impossible de charger la configuration de backtest. Utilisation des valeurs par défaut. Erreur: {e}")
            return {
                "initial_capital": 10000.0,
                "commission_pct": 0.001,
                "slippage_pct": 0.0005,
                "freq": "1D",
            }

    def run(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], Optional[Portfolio]]:
        """
        Exécute le backtest complet de la stratégie.

        Args:
            data: DataFrame pandas contenant les données de marché (OHLCV).

        Returns:
            Un tuple contenant :
            - Un dictionnaire des métriques de performance.
            - L'objet Portfolio de vectorbt (ou None si le backtest échoue).
        """
        if data.empty:
            logger.error("Les données fournies pour le backtest sont vides. Annulation.")
            raise DataError("Les données pour le backtest ne peuvent pas être vides.")

        try:
            logger.debug(f"Début du backtest pour la stratégie '{self.strategy.name}' sur '{self.symbol}'.")
            
            # 1. Calculer les indicateurs via la stratégie
            indicators_df = self.strategy.calculate_indicators(data)
            if indicators_df.empty:
                raise BacktestingError("Le calcul des indicateurs n'a retourné aucune donnée.")

            # 2. Générer les signaux de la stratégie
            signals_df = self.strategy.generate_signals(indicators_df)
            if signals_df.empty:
                logger.warning("Aucun signal généré. Le backtest résultera en une performance nulle.")
                return calculate_performance_metrics(None, self.backtest_config["initial_capital"]), None

            # 3. Adapter les signaux pour VectorBT
            vbt_signals = self.signal_adapter.adapt(signals_df)

            # 4. Exécuter le backtest avec VectorBT
            portfolio = Portfolio.from_signals(
                close=indicators_df['close'],
                entries=vbt_signals['entries'],
                exits=vbt_signals['exits'],
                sl_stop=vbt_signals.get('sl'),
                tp_stop=vbt_signals.get('tp'),
                init_cash=self.backtest_config["initial_capital"],
                fees=self.backtest_config["commission_pct"],
                slippage=self.backtest_config["slippage_pct"],
                freq=self.backtest_config["freq"],
            )

            # 5. Calculer les métriques de performance
            performance_stats = calculate_performance_metrics(
                portfolio, self.backtest_config["initial_capital"]
            )
            
            logger.info(f"Backtest terminé pour {self.symbol}. "
                        f"Rendement Total: {performance_stats.get('Total Return [%]', 'N/A'):.2f}%, "
                        f"Ratio Sharpe: {performance_stats.get('Sharpe Ratio', 'N/A'):.2f}")

            return performance_stats, portfolio

        except Exception as e:
            logger.exception(
                f"Erreur critique lors de l'exécution du backtest pour "
                f"'{self.strategy.name}' sur '{self.symbol}': {e}"
            )
            # Retourne des métriques vides mais ne crashe pas le programme
            return calculate_performance_metrics(None, self.backtest_config["initial_capital"]), None

