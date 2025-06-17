"""
Ce module définit la fonction objectif pour l'optimisation avec Optuna.
Il a été refactorisé pour supporter le multi-objectif et pour fournir
une méthode dédiée à la validation des paramètres sur les données Out-of-Sample (OOS).
"""

import logging
import optuna
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Type, Dict, Any, Union, Tuple, List

# --- Imports Locaux ---
from src.optimization.config import MainOptimizationConfig
from src.strategies.base import BaseStrategy
from src.strategies.params import suggest_params_from_config
from src.data.enriched_dataframe import EnrichedDataFrame
# MODIFIÉ: L'importation de VectorBTEngine est retirée pour éviter une dépendance circulaire.
# from src.backtesting.vectorbt_engine import VectorBTEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.core.exceptions import BacktestFailureError, ConfigurationError, InvalidStrategyParamsError

logger = logging.getLogger(__name__)


class Objective:
    """
    Classe représentant la fonction objectif pour une étude Optuna.
    """
    def __init__(self,
                 config: MainOptimizationConfig,
                 strategy_class: Type[BaseStrategy],
                 strategy_params_config: Dict[str, Any],
                 data: pd.DataFrame,
                 pair_symbol: str):
        self.config = config
        self.strategy_class = strategy_class
        self.strategy_params_config = strategy_params_config
        self.data = data
        self.pair_symbol = pair_symbol
        self.is_multi_objective = len(config.optuna_config.objectives) > 1

    def _get_params_for_trial(self, trial: optuna.Trial) -> Dict[str, Any]:
        return suggest_params_from_config(trial, self.strategy_params_config)

    def _run_backtest(self, params: Dict[str, Any]) -> vbt.Portfolio:
        """Exécute le backtest et retourne l'objet Portfolio complet."""
        # MODIFIÉ: L'importation est déplacée ici pour briser la dépendance circulaire.
        from src.backtesting.vectorbt_engine import VectorBTEngine

        try:
            hyperparams = params.copy()
            strategy_name = hyperparams.pop('strategy_name', self.strategy_class.__name__)

            strategy = self.strategy_class(
                pair_symbol=self.pair_symbol,
                strategy_name=strategy_name,
                **hyperparams
            )

            vbt_engine = VectorBTEngine(
                data=self.data,
                strategy=strategy,
                config=self.config.simulation_config
            )
            return vbt_engine.run()  # Retourne le portefeuille directement

        except Exception as e:
            logger.warning(f"Le backtest a échoué pour les paramètres {params}: {e}")
            raise BacktestFailureError(f"Backtest failed: {e}") from e

    def backtest_with_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Exécute un backtest de validation et retourne les métriques."""
        logger.debug(f"Exécution d'un backtest de validation avec les paramètres : {params}")
        try:  # self.data est le oos_df dans ce contexte
            if self.data.empty:
                logger.warning(f"Validation backtest with params {params}: Input data (OOS) is empty. Returning empty metrics.")
                return {}

            portfolio = self._run_backtest(params)

            if portfolio.value().min() <= 0:  # Vérifier la condition de ruine
                logger.warning(f"Validation backtest with params {params} resulted in ruin. Returning empty metrics.")
                return {}

            metrics = PerformanceMetrics.calculate_all_metrics(portfolio)
            if not metrics:  # Si calculate_all_metrics retourne vide ou None
                logger.warning(f"Validation backtest with params {params}: Metrics calculation returned empty. Returning empty metrics.")
                return {}
            return metrics
        except BacktestFailureError as e:
            logger.warning(f"Validation backtest failed for params {params}: {e}. Returning empty metrics.")
            return {}
        except Exception as e:  # Attraper toute autre erreur inattendue
            logger.error(f"Unexpected error during validation backtest with params {params}: {e}", exc_info=True)
            return {}

    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        try:
            params = self._get_params_for_trial(trial)

            # Vérifier si les données d'entrée (In-Sample) sont vides
            if self.data.empty:
                logger.warning(f"Trial {trial.number}: Input data for In-Sample backtest is empty. Penalizing trial.")
                raise BacktestFailureError("Input data for IS backtest is empty.")

            portfolio = self._run_backtest(params)

            # --- IMPLÉMENTATION DE LA CONDITION DE RUINE ---
            # Vérifie si la valeur du portefeuille est tombée à zéro ou en dessous.
            if (portfolio.value().min() <= 0):
                logger.warning(f"Essai {trial.number} a conduit à la ruine (capital <= 0). Pénalisation.")
                raise BacktestFailureError("Condition de ruine atteinte.")

            all_metrics = PerformanceMetrics.calculate_all_metrics(portfolio)
            if not all_metrics:  # Vérifier si le dictionnaire de métriques est vide
                logger.warning(f"Essai {trial.number}: Le calcul des métriques a retourné un résultat vide. Pénalisation.")
                raise BacktestFailureError("Le calcul des métriques a retourné un résultat vide.")

            objective_values = []
            for obj_config in self.config.optuna_config.objectives:
                metric_name = obj_config.name

                possible_keys = {
                    'sharpe_ratio': ['risk_ratios_sharpe_ratio', 'vbt_sharpe_ratio', 'sharpe_ratio'],
                    'max_drawdown': ['drawdown_max_drawdown_pct', 'vbt_max_drawdown_[%]', 'max_drawdown'],
                    'sortino_ratio': ['risk_ratios_sortino_ratio', 'vbt_sortino_ratio', 'sortino_ratio'],
                    'calmar_ratio': ['risk_ratios_calmar_ratio', 'vbt_calmar_ratio', 'calmar_ratio'],
                    'profit_factor': ['trade_stats_profit_factor', 'vbt_profit_factor', 'profit_factor'],
                }.get(metric_name, [metric_name])

                value_found = None
                for key in possible_keys:
                    if key in all_metrics:
                        value = all_metrics[key]
                        if pd.notna(value) and np.isfinite(value):
                            value_found = float(value)
                        else:
                            logger.warning(f"Métrique '{key}' invalide: {value}. Essai d'une clé alternative.")
                            value_found = None
                        break

                if value_found is None:
                    raise ConfigurationError(f"Métrique '{metric_name}' non trouvée ou invalide.")

                objective_values.append(value_found)

            if self.is_multi_objective:
                return tuple(objective_values)
            return objective_values[0]

        except Exception as e:
            logger.warning(f"Essai {trial.number} a échoué et sera pénalisé. Erreur : {e}")

            worst_values = []
            for obj in self.config.optuna_config.objectives:
                if obj.direction == 'maximize':
                    worst_values.append(-1e9)
                else:
                    worst_values.append(1e9)

            if self.is_multi_objective:
                return tuple(worst_values)
            return worst_values[0]
