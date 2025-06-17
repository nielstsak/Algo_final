from __future__ import annotations
import logging
import pandas as pd
import json
from typing import Tuple, List, Dict, Any, TYPE_CHECKING

from src.core.exceptions import OptimizationError, DataError
from src.core.config import get_settings
from src.data.data_manager import DataManager
from src.backtesting.vectorbt_engine import VectorBTBacktestingEngine
from src.strategies.strategy_loader import StrategyLoader

if TYPE_CHECKING:
    from src.optimization.optimizer import StrategyOptimizer

logger = logging.getLogger(__name__)


# MODIFIÉ : Le nom de la classe est rétabli à WFOptimizer pour correspondre
# à ce qui est attendu par le script de ligne de commande.
class WFOptimizer:
    """
    Moteur d'Optimisation Walk-Forward (WFO).

    Cette classe gère le processus WFO en divisant les données historiques
    en plusieurs "plis" (folds), chacun contenant une période d'entraînement
    et une période de test consécutive.
    """

    def __init__(self, optimizer: "StrategyOptimizer"):
        """
        Initialise le moteur WFO.

        Args:
            optimizer: Une instance de StrategyOptimizer, déjà configurée avec la stratégie,
                       le symbole et les configurations d'optimisation.
        """
        self.optimizer = optimizer
        self.settings = optimizer.settings
        self.symbol = optimizer.symbol
        self.strategy_name = optimizer.strategy_name

        self.wfo_config = self.optimizer.optimization_config.get("wfo_config")
        if not self.wfo_config or not isinstance(self.wfo_config, dict):
            raise OptimizationError(
                "La configuration WFO ('wfo_config') est manquante ou invalide dans le fichier de configuration de l'optimisation."
            )

        self.train_days = self.wfo_config.get("train_days")
        self.test_days = self.wfo_config.get("test_days")
        self.num_folds = self.wfo_config.get("num_folds")

        if not all([self.train_days, self.test_days, self.num_folds]):
            raise OptimizationError(
                "La configuration WFO doit contenir 'train_days', 'test_days' et 'num_folds'."
            )

        self.data_manager = DataManager(settings=self.settings)
        self.full_data = self._load_data()
        self.strategy_loader = StrategyLoader()

    def _load_data(self) -> pd.DataFrame:
        """Charge l'ensemble des données historiques nécessaires pour le WFO."""
        logger.info(f"Chargement des données pour le WFO sur le symbole : {self.symbol}")
        try:
            # Pour le calcul des plis, nous utilisons une résolution journalière
            data = self.data_manager.get_data(symbol=self.symbol, timeframe="1d")
            if data.empty:
                raise DataError(f"Aucune donnée trouvée pour le symbole {self.symbol}")
            return data
        except Exception as e:
            logger.exception(f"Échec du chargement des données pour le WFO : {e}")
            raise DataError(f"Échec du chargement des données pour le WFO : {e}") from e

    def _get_walk_forward_splits(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Génère les dates de début et de fin pour chaque pli d'entraînement et de test."""
        splits = []
        data_start_date = self.full_data.index.min()
        data_end_date = self.full_data.index.max()

        total_days_required = self.train_days + (self.num_folds * self.test_days)
        if (data_end_date - data_start_date).days < total_days_required:
            raise DataError(
                f"Données insuffisantes pour la configuration WFO. "
                f"Requis: {total_days_required} jours, Disponible: {(data_end_date - data_start_date).days} jours."
            )

        current_test_end = data_end_date
        for i in range(self.num_folds):
            test_end = current_test_end
            test_start = test_end - pd.Timedelta(days=self.test_days - 1)
            train_end = test_start - pd.Timedelta(days=1)
            train_start = train_end - pd.Timedelta(days=self.train_days - 1)

            if train_start < data_start_date:
                logger.warning(f"Le pli {self.num_folds - i} commencerait avant le début des données. Arrêt de la génération des plis.")
                break

            splits.append((train_start, train_end, test_start, test_end))
            current_test_end = train_end

        return list(reversed(splits))  # Retourner dans l'ordre chronologique

    def run(self) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
        """
        Exécute le processus complet d'Optimisation Walk-Forward.

        Returns:
            Un tuple contenant :
            - Un dictionnaire des meilleurs paramètres trouvés pour chaque pli.
            - Un DataFrame avec les résultats de performance agrégés de tous les plis de test.
        """
        splits = self._get_walk_forward_splits()
        if not splits:
            raise OptimizationError("Impossible de générer des plis WFO valides avec les données et la configuration fournies.")

        logger.info(f"Démarrage de l'Optimisation Walk-Forward avec {len(splits)} plis.")

        all_fold_results = []
        best_params_per_fold = {}

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            fold_num = i + 1
            logger.info(f"--- Traitement du Pli {fold_num}/{len(splits)} ---")
            logger.info(f"Période d'entraînement : {train_start.date()} à {train_end.date()}")
            logger.info(f"Période de test : {test_start.date()} à {test_end.date()}")

            # 1. Optimiser sur les données d'entraînement de ce pli
            train_data = self.full_data.loc[train_start:train_end]
            best_params, best_trial = self.optimizer.optimize(data=train_data)
            best_params_per_fold[fold_num] = best_params
            logger.info(f"Pli {fold_num} | Meilleurs paramètres (entraînement) : {best_params} | Valeur Objective : {best_trial.value:.4f}")

            # 2. Backtester sur les données de test avec les meilleurs paramètres
            test_data = self.full_data.loc[test_start:test_end]
            
            strategy_instance = self.strategy_loader.create_strategy(
                strategy_identifier=self.strategy_name,
                params=best_params,
                pair_symbol=self.symbol
            )
            
            backtesting_engine = VectorBTBacktestingEngine(
                strategy=strategy_instance,
                symbol=self.symbol,
                settings=self.settings
            )
            # La méthode run de VectorBTBacktestingEngine retourne (stats, portfolio)
            test_performance_stats, portfolio = backtesting_engine.run(data=test_data)
            
            # Stocker les résultats
            fold_result = {
                "fold": fold_num,
                "train_start": train_start.strftime('%Y-%m-%d'),
                "train_end": train_end.strftime('%Y-%m-%d'),
                "test_start": test_start.strftime('%Y-%m-%d'),
                "test_end": test_end.strftime('%Y-%m-%d'),
                **test_performance_stats,
                "best_params": json.dumps(best_params)
            }
            all_fold_results.append(fold_result)
            logger.info(f"Pli {fold_num} | Performance (test) : {test_performance_stats}")

        if not all_fold_results:
            raise OptimizationError("Le WFO s'est terminé mais aucun résultat n'a été généré.")
            
        wfo_results_df = pd.DataFrame(all_fold_results)
        
        logger.info("--- Résumé de l'Optimisation Walk-Forward ---")
        summary_df = wfo_results_df.drop(columns=['best_params'], errors='ignore')
        logger.info(f"\n{summary_df.to_string()}")
        
        return best_params_per_fold, wfo_results_df
