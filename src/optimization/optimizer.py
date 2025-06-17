"""
Ce module contient l'orchestrateur principal pour l'optimisation de stratégies,
capable de gérer à la fois des optimisations simples et des Walk-Forward Optimizations (WFO) complexes.
Il est conçu pour être entièrement piloté par la configuration Pydantic définie dans `optimization.config`.
"""

import logging
import yaml
import optuna
import pandas as pd
from pathlib import Path
from typing import Type, Dict, Any, List, Tuple, Union
import inspect

# --- Imports Optuna ---
from optuna.samplers import TPESampler, NSGAIISampler, CmaEsSampler, BaseSampler
from optuna.pruners import MedianPruner, HyperbandPruner, BasePruner
from optuna.trial import FrozenTrial, TrialState
from optuna.study import StudyDirection

# --- Imports Locaux ---
from src.core.config import get_settings
from src.optimization.config import MainOptimizationConfig, OptunaProfile
from src.optimization.objective import Objective
from src.optimization.wfo_engine import WFOptimizer 
from src.strategies.base_strategy import BaseStrategy
from src.data.enriched_dataframe import EnrichedDataFrame
from src.core.exceptions import OptimizationError, ConfigurationError

# --- Configuration du Logger ---
logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Orchestre le processus d'optimisation d'une stratégie de trading.
    """
    def __init__(self,
                 config: MainOptimizationConfig,
                 strategy_class: Type[BaseStrategy],
                 strategy_params_config: Dict[str, Any],
                 data: EnrichedDataFrame,
                 pair_symbol: str): # Ajout du symbole de la paire
        self.config = config
        self.strategy_class = strategy_class
        self.strategy_params_config = strategy_params_config
        self.data = data
        self.pair_symbol = pair_symbol # Stockage du symbole
        self.is_multi_objective = len(self.config.optuna_config.objectives) > 1
        logger.info(f"Optimiseur initialisé pour la stratégie '{self.strategy_class.__name__}' sur la paire '{self.pair_symbol}'.")
        logger.info(f"Optimisation {'multi-objectif' if self.is_multi_objective else 'mono-objectif'} configurée.")

    @classmethod
    def from_yaml(cls,
                  config_path: Union[str, Path],
                  strategy_class: Type[BaseStrategy],
                  strategy_params_config: Dict[str, Any],
                  data: EnrichedDataFrame,
                  pair_symbol: str) -> 'StrategyOptimizer': # Ajout du symbole ici aussi
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            main_config = MainOptimizationConfig.parse_obj(config_dict)
            logger.info(f"Configuration chargée et validée depuis '{config_path}'.")
            
            # Passe le symbole au constructeur de la classe
            return cls(main_config, strategy_class, strategy_params_config, data, pair_symbol)
        except FileNotFoundError:
            logger.error(f"Le fichier de configuration '{config_path}' n'a pas été trouvé.")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la validation ou du parsing de la configuration : {e}")
            raise ConfigurationError(f"Configuration invalide: {e}") from e

    def _get_sampler_and_pruner(self) -> Tuple[BaseSampler, BasePruner]:
        profile = self.config.optuna_config.profile
        sampler_params = self.config.optuna_config.sampler_params
        pruner_params = self.config.optuna_config.pruner_params

        def filter_params(cls, params):
            sig = inspect.signature(cls.__init__)
            valid_keys = sig.parameters.keys()
            return {k: v for k, v in params.items() if k in valid_keys}

        if profile == OptunaProfile.DEFAULT_TPE:
            sampler = TPESampler(**filter_params(TPESampler, sampler_params))
        elif profile == OptunaProfile.NSGAII_HYPERBAND:
            if not self.is_multi_objective:
                logger.warning("Le profil NSGAII est pour le multi-objectif. Utilisation de TPESampler.")
                sampler = TPESampler(**filter_params(TPESampler, sampler_params))
            else:
                sampler = NSGAIISampler(**filter_params(NSGAIISampler, sampler_params))
        elif profile == OptunaProfile.CMAES_MEDIAN:
            sampler = CmaEsSampler(**filter_params(CmaEsSampler, sampler_params))
        elif profile == OptunaProfile.CUSTOM:
            logger.info("Profil 'CUSTOM' sélectionné. Utilisation de TPESampler par défaut.")
            sampler = TPESampler(**filter_params(TPESampler, sampler_params))
        else:
            raise ConfigurationError(f"Profil Optuna non supporté : '{profile}'")

        if profile in [OptunaProfile.DEFAULT_TPE, OptunaProfile.CMAES_MEDIAN, OptunaProfile.CUSTOM]:
            pruner = MedianPruner(**filter_params(MedianPruner, pruner_params))
        elif profile == OptunaProfile.NSGAII_HYPERBAND:
            pruner = HyperbandPruner(**filter_params(HyperbandPruner, pruner_params))
        else:
            pruner = MedianPruner()

        logger.info(f"Utilisation du Sampler: {sampler.__class__.__name__}, Pruner: {pruner.__class__.__name__}")
        return sampler, pruner

    def _setup_study(self, study_name_suffix: str = "") -> optuna.Study:
        optuna_cfg = self.config.optuna_config
        directions = [obj.direction for obj in optuna_cfg.objectives]
        study_name = optuna_cfg.study_name or f"{self.strategy_class.__name__}-study"
        if study_name_suffix:
            study_name = f"{study_name}-{study_name_suffix}"
            
        sampler, pruner = self._get_sampler_and_pruner()
        
        storage_url = "sqlite:///optuna_studies.db"
        logger.info(f"Utilisation de SQLite pour le stockage Optuna : {storage_url}")

        return optuna.create_study(
            study_name=study_name, 
            directions=directions, 
            sampler=sampler, 
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True
        )

    def _select_best_trial(self, study: optuna.Study) -> FrozenTrial:
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            raise OptimizationError("Aucun essai complété avec succès n'a été trouvé.")

        if not self.is_multi_objective:
            return max(complete_trials, key=lambda t: t.value)

        try:
            from optuna.visualization import is_pareto_front
            pareto_mask = is_pareto_front(trials=complete_trials, directions=study.directions)
            pareto_trials = [t for i, t in enumerate(complete_trials) if pareto_mask[i]]
        except ImportError:
            pareto_trials = study.best_trials
        
        if not pareto_trials:
            raise OptimizationError("Aucun essai valide trouvé sur le front de Pareto.")

        selection_cfg = self.config.optuna_config.pareto_selection
        if not selection_cfg or selection_cfg.strategy != 'COMPOSITE_SCORE':
            logger.warning("Stratégie de sélection Pareto non spécifiée ou non supportée. Retour du premier essai.")
            return pareto_trials[0]

        weights = selection_cfg.weights
        if not weights:
            raise ConfigurationError("La stratégie 'COMPOSITE_SCORE' nécessite des poids ('weights').")
        
        best_trial = max(pareto_trials, key=lambda t: sum(
            (t.values[i] if study.directions[i] == StudyDirection.MAXIMIZE else -t.values[i]) * weights.get(obj.name, 0)
            for i, obj in enumerate(self.config.optuna_config.objectives)
        ))
        return best_trial

    def _run_optimization_on_period(self, data_period: pd.DataFrame, study_suffix: str) -> optuna.Study:
        study = self._setup_study(study_name_suffix=study_suffix)
        
        # --- CORRECTION: Passage du pair_symbol à l'objet Objective ---
        objective_func = Objective(
            config=self.config,
            strategy_class=self.strategy_class,
            strategy_params_config=self.strategy_params_config,
            data=data_period,
            pair_symbol=self.pair_symbol # Passage du symbole
        )
        # --- FIN DE LA CORRECTION ---
        
        n_jobs = self.config.optuna_config.n_jobs
        
        if "sqlite" in study._storage._backend.url and n_jobs != 4:
            logger.warning(
                f"Optuna utilise SQLite. Forçage de n_jobs=1 pour éviter les erreurs de verrouillage (n_jobs original était {n_jobs})."
            )
            n_jobs = 1

        study.optimize(
            objective_func,
            n_trials=self.config.optuna_config.n_trials,
            n_jobs=n_jobs,
            catch=(Exception,)
        )
        return study

    def _run_wfo(self) -> pd.DataFrame:
        logger.info("Démarrage du processus de Walk-Forward Optimization.")
        wfo_splitter = WFOptimizer(config=self.config.wfo_config)
        
        splits = wfo_splitter.split(self.data.df)

        all_oos_results = []
        n_best_to_validate = self.config.wfo_config.validation_config.n_best_trials_to_validate

        for i, (is_indices, oos_indices) in enumerate(splits):
            logger.info(f"--- WFO Split {i+1}/{len(splits)} ---")
            
            is_df = self.data.df.loc[is_indices]
            oos_df = self.data.df.loc[oos_indices]

            if is_df.empty:
                logger.warning(f"WFO Split {i+1}: Les données In-Sample sont vides après le découpage avec is_indices (longueur: {len(is_indices)}). "
                               f"Début IS: {is_indices.min() if not is_indices.empty else 'N/A'}, Fin IS: {is_indices.max() if not is_indices.empty else 'N/A'}. "
                               "Ce split sera ignoré.")
                continue
            logger.info(f"Optimisation In-Sample sur {len(is_df)} points...")
            is_study = self._run_optimization_on_period(is_df, f"IS-split{i+1}")

            try:
                best_is_trials = is_study.best_trials
            except ValueError:
                logger.warning(f"Aucun essai complété avec succès pour le split {i+1}. Passage au suivant.")
                continue

            best_is_trials = sorted(best_is_trials, key=lambda t: t.values, reverse=True)[:n_best_to_validate]

            logger.info(f"Validation de {len(best_is_trials)} essai(s) sur la période Out-of-Sample...")
            
            # --- CORRECTION: Passage du pair_symbol à l'objet Objective pour l'OOS ---
            objective_oos = Objective(
                config=self.config, strategy_class=self.strategy_class,
                strategy_params_config=self.strategy_params_config, data=oos_df,
                pair_symbol=self.pair_symbol # Passage du symbole
            )
            # --- FIN DE LA CORRECTION ---

            for rank, trial in enumerate(best_is_trials):
                try:
                    oos_metrics = objective_oos.backtest_with_params(trial.params)
                    result_record = {
                        'split': i + 1, 'is_trial_rank': rank + 1, 'is_trial_number': trial.number,
                        **{f"is_{obj.name}": val for obj, val in zip(self.config.optuna_config.objectives, trial.values)},
                        **{f"oos_{k}": v for k, v in oos_metrics.items()},
                        'params': trial.params
                    }
                    all_oos_results.append(result_record)
                except Exception as e:
                    logger.error(f"Échec de la validation OOS pour l'essai {trial.number}: {e}")

        return pd.DataFrame(all_oos_results)

    def _run_simple_optimization(self) -> Tuple[FrozenTrial, optuna.Study]:
        logger.info("Démarrage d'une optimisation simple sur l'ensemble des données.")
        data_for_opt = self.data.df

        if data_for_opt.empty:
            logger.error("Les données pour l'optimisation simple sont vides. Impossible de continuer.")
            # Vous pourriez vouloir retourner une structure indiquant l'échec ou lever une exception plus spécifique.
            raise OptimizationError("Les données d'entrée pour l'optimisation simple sont vides.")

        study = self._run_optimization_on_period(data_for_opt, "full-period")
        best_trial = self._select_best_trial(study)
        
        logger.info("--- Meilleur Essai Trouvé ---")
        logger.info(f"   Numéro: {best_trial.number}")
        if self.is_multi_objective:
            for i, obj in enumerate(self.config.optuna_config.objectives):
                logger.info(f"   Objectif '{obj.name}': {best_trial.values[i]:.4f}")
        else:
            logger.info(f"   Objectif: {best_trial.value:.4f}")
        logger.info("   Paramètres:")
        for key, value in best_trial.params.items():
            logger.info(f"     {key}: {value}")
        return best_trial, study

    def run(self) -> Union[Tuple[FrozenTrial, optuna.Study], pd.DataFrame]:
        if self.config.wfo_config.enabled:
            return self._run_wfo()
        else:
            return self._run_simple_optimization()
