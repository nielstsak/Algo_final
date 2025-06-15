# src/optimization/config.py

"""
Ce module définit la structure de configuration de l'optimisation à l'aide de modèles Pydantic.
Il remplace les configurations simples précédentes par une structure hiérarchique et détaillée,
capable de gérer des scénarios complexes de backtesting, de Walk-Forward Optimization (WFO)
et d'optimisation multi-objectif avec Optuna.

Cette nouvelle configuration sert de "contrat de données" pour l'ensemble du pipeline
d'optimisation, garantissant la validation, la clarté et la maintenabilité des paramètres.
"""

from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

# ==============================================================================
# SECTION 1: CONFIGURATION DE LA SIMULATION (BACKTESTING)
# ==============================================================================
# Modèles décrivant les paramètres fondamentaux de la simulation de backtesting,
# y compris le capital, l'effet de levier, et des modèles avancés pour les
# frais et le slippage.

class FeeConfig(BaseModel):
    """Configuration du calcul des frais de transaction."""
    method: Literal['PERCENTAGE', 'FIXED_PER_TRADE'] = Field(
        'PERCENTAGE',
        description="Méthode de calcul des frais ('PERCENTAGE' ou 'FIXED_PER_TRADE')."
    )
    value: float = Field(
        ...,
        description="Valeur numérique pour le calcul des frais (ex: 0.00075 pour 0.075%)."
    )

class SlippageConfig(BaseModel):
    """Configuration du modèle de slippage."""
    method: Literal['PERCENTAGE', 'FIXED_PER_TRADE', 'VOLATILITY_ADJUSTED'] = Field(
        'PERCENTAGE',
        description="Méthode de modélisation du slippage."
    )
    value: float = Field(
        ...,
        description="Valeur de base pour le slippage (ex: 0.0001 pour 0.01%)."
    )
    volatility_window: Optional[int] = Field(
        None,
        description="Fenêtre de calcul de la volatilité pour le modèle 'VOLATILITY_ADJUSTED'."
    )
    volatility_multiplier: Optional[float] = Field(
        None,
        description="Multiplicateur à appliquer à la volatilité pour le slippage."
    )

    @validator('volatility_window', 'volatility_multiplier')
    def _check_volatility_params(cls, v, values):
        if values.get('method') == 'VOLATILITY_ADJUSTED' and v is None:
            raise ValueError("volatility_window et volatility_multiplier sont requis pour la méthode 'VOLATILITY_ADJUSTED'.")
        return v

class SimulationConfig(BaseModel):
    """
    Configuration de base pour une session de backtesting.
    Ces paramètres sont utilisés dans chaque exécution de backtest au sein de l'optimisation.
    """
    initial_capital: float = Field(10000.0, gt=0, description="Capital initial pour la simulation.")
    leverage: float = Field(1.0, ge=1.0, description="Effet de levier à appliquer.")
    fee_config: FeeConfig = Field(..., description="Configuration des frais de transaction.")
    slippage_config: SlippageConfig = Field(..., description="Configuration du modèle de slippage.")


# ==============================================================================
# SECTION 2: CONFIGURATION DU WALK-FORWARD OPTIMIZATION (WFO)
# ==============================================================================
# Modèles pour une gestion avancée du WFO, incluant purging, embargo, et
# plusieurs méthodes de découpage des données (temporel, adaptatif).

class PurgeAndEmbargoConfig(BaseModel):
    """Configuration pour le purging et l'embargo afin d'éviter la fuite de données."""
    purging_days: int = Field(5, ge=0, description="Nombre de jours à purger à la fin de la période In-Sample.")
    embargo_days: int = Field(5, ge=0, description="Nombre de jours à ignorer au début de la période Out-of-Sample.")

class WFOValidationConfig(BaseModel):
    """Configuration pour la validation des meilleurs essais de la période In-Sample."""
    n_best_trials_to_validate: int = Field(
        1,
        gt=0,
        description="Nombre de meilleurs essais de la période IS à valider sur la période OOS."
    )

class FixedSplitConfig(BaseModel):
    """Configuration pour un découpage WFO basé sur des durées temporelles fixes."""
    type: Literal['FIXED_TIME'] = 'FIXED_TIME'
    is_days: Optional[int] = Field(
        None,
        gt=0,
        description="Durée en jours de la période In-Sample (IS). Calculée automatiquement à partir de la durée totale des données si non fournie."
    )
    oos_days: int = Field(30, gt=0, description="Durée en jours de la période Out-of-Sample (OOS).")
    n_splits: int = Field(10, gt=0, description="Nombre total de découpages à effectuer.")
    expanding_window: bool = Field(False, description="Utiliser une fenêtre IS expansive plutôt que glissante.")

class AdaptiveVolatilitySplitConfig(BaseModel):
    """Configuration pour un découpage WFO adaptatif basé sur la volatilité du marché."""
    type: Literal['ADAPTIVE_VOLATILITY'] = 'ADAPTIVE_VOLATILITY'
    volatility_window: int = Field(20, gt=0, description="Fenêtre pour le calcul de la volatilité.")
    target_volatility_multiplier: float = Field(1.5, gt=0, description="Multiplicateur de la volatilité moyenne pour définir la fin d'une période.")
    min_is_days: int = Field(60, gt=0, description="Taille minimale en jours pour une période In-Sample.")
    min_oos_days: int = Field(15, gt=0, description="Taille minimale en jours pour une période Out-of-Sample.")

class WFOConfig(BaseModel):
    """
    Configuration complète du processus de Walk-Forward Optimization (WFO).
    Permet de choisir entre différentes stratégies de découpage et de validation.
    """
    enabled: bool = Field(True, description="Active ou désactive le WFO.")
    split_config: Union[FixedSplitConfig, AdaptiveVolatilitySplitConfig] = Field(
        ...,
        discriminator='type',
        description="Configuration de la méthode de découpage des données."
    )
    purge_and_embargo: PurgeAndEmbargoConfig = Field(
        default_factory=PurgeAndEmbargoConfig,
        description="Configuration du purging et de l'embargo."
    )
    validation_config: WFOValidationConfig = Field(
        default_factory=WFOValidationConfig,
        description="Configuration de la validation des essais OOS."
    )
    # Le support pour la validation croisée combinatoire pourrait être ajouté ici comme un autre type.
    # validation_method: Literal['WALK_FORWARD', 'COMBINATORIAL_CV'] = 'WALK_FORWARD'


# ==============================================================================
# SECTION 3: CONFIGURATION DE L'OPTIMISATION (OPTUNA)
# ==============================================================================
# Modèles pour une configuration fine et flexible d'Optuna, supportant le
# multi-objectif, des profils prédéfinis et des contrôles avancés.

class OptunaObjectiveConfig(BaseModel):
    """Définit un objectif pour l'optimisation (potentiellement multi-objectif)."""
    name: str = Field(..., description="Nom de la métrique à optimiser (ex: 'sharpe_ratio').")
    direction: Literal['maximize', 'minimize'] = Field('maximize', description="Direction de l'optimisation.")

class ParetoSelectionConfig(BaseModel):
    """Stratégie pour sélectionner le meilleur essai depuis un front de Pareto (multi-objectif)."""
    strategy: Literal['COMPOSITE_SCORE', 'DISTANCE_METRIC'] = Field(
        'COMPOSITE_SCORE',
        description="Stratégie de sélection du meilleur compromis."
    )
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Poids pour chaque objectif si la stratégie est 'COMPOSITE_SCORE'."
    )

class OptunaProfile(str, Enum):
    """Profils prédéfinis de sampler/pruner pour des cas d'usage courants."""
    DEFAULT_TPE = "DEFAULT_TPE"  # TPE Sampler, Median Pruner
    NSGAII_HYPERBAND = "NSGAII_HYPERBAND"  # NSGA-II pour multi-objectif, avec HyperbandPruner
    CMAES_MEDIAN = "CMAES_MEDIAN"  # CMA-ES Sampler, Median Pruner
    CUSTOM = "CUSTOM"  # Configuration manuelle du sampler et pruner

class OptunaConfig(BaseModel):
    """
    Configuration complète pour une session d'optimisation avec Optuna.
    Supporte le multi-objectif et des profils de configuration avancés.
    """
    objectives: List[OptunaObjectiveConfig] = Field(
        ...,
        description="Liste des objectifs à optimiser. Une seule entrée pour du mono-objectif."
    )
    n_trials: int = Field(100, gt=0, description="Nombre total d'essais d'optimisation.")
    n_jobs: int = Field(1, ge=-1, description="Nombre de processus parallèles à utiliser (-1 pour tous les cœurs).")
    profile: OptunaProfile = Field(
        OptunaProfile.DEFAULT_TPE,
        description="Profil prédéfini pour le couple sampler/pruner."
    )
    pareto_selection: Optional[ParetoSelectionConfig] = Field(
        None,
        description="Stratégie de sélection sur le front de Pareto (pour le multi-objectif)."
    )
    sampler_params: Dict[str, Any] = Field(default_factory=dict, description="Paramètres spécifiques pour le sampler Optuna.")
    pruner_params: Dict[str, Any] = Field(default_factory=dict, description="Paramètres spécifiques pour le pruner Optuna.")
    early_stopping_rounds: Optional[int] = Field(
        None,
        description="Nombre de tours sans amélioration avant d'arrêter un essai (via callback)."
    )
    study_name: Optional[str] = Field(None, description="Nom de l'étude Optuna. Généré automatiquement si non fourni.")

    @validator('objectives')
    def _check_objectives_not_empty(cls, v):
        if not v:
            raise ValueError("La liste des objectifs ne peut pas être vide.")
        return v

    @validator('pareto_selection')
    def _check_pareto_selection(cls, v, values):
        objectives = values.get('objectives', [])
        if len(objectives) > 1 and v is None:
            raise ValueError("Une stratégie `pareto_selection` est requise pour l'optimisation multi-objectif.")
        if len(objectives) <= 1 and v is not None:
            raise ValueError("`pareto_selection` ne s'applique qu'à l'optimisation multi-objectif.")
        return v


# ==============================================================================
# SECTION 4: CONFIGURATION PRINCIPALE DE L'OPTIMISATION
# ==============================================================================
# Le modèle racine qui assemble toutes les pièces de la configuration.

class MainOptimizationConfig(BaseModel):
    """
    Modèle Pydantic racine qui agrège toutes les configurations nécessaires
    pour lancer une session complète d'optimisation de stratégie.
    """
    simulation_config: SimulationConfig = Field(..., description="Configuration de base pour chaque backtest.")
    wfo_config: WFOConfig = Field(..., description="Configuration du processus de Walk-Forward Optimization.")
    optuna_config: OptunaConfig = Field(..., description="Configuration de l'optimisation avec Optuna.")

# Exemple d'utilisation (peut être utilisé pour des tests ou comme référence)
if __name__ == '__main__':
    # Ceci est un exemple de dictionnaire qui pourrait être chargé depuis un fichier YAML.
    # Notez que `is_days` n'est plus spécifié ici.
    example_config_dict = {
        "simulation_config": {
            "initial_capital": 50000,
            "leverage": 5.0,
            "fee_config": {
                "method": "PERCENTAGE",
                "value": 0.001
            },
            "slippage_config": {
                "method": "PERCENTAGE",
                "value": 0.0005
            }
        },
        "wfo_config": {
            "enabled": True,
            "split_config": {
                "type": "FIXED_TIME",
                "is_days": None, # Ou simplement omettre la clé
                "oos_days": 40,
                "n_splits": 8,
                "expanding_window": False
            },
            "purge_and_embargo": {
                "purging_days": 7,
                "embargo_days": 3
            },
            "validation_config": {
                "n_best_trials_to_validate": 3
            }
        },
        "optuna_config": {
            "objectives": [
                {"name": "sharpe_ratio", "direction": "maximize"},
                {"name": "max_drawdown", "direction": "minimize"}
            ],
            "n_trials": 200,
            "n_jobs": -1,
            "profile": "NSGAII_HYPERBAND",
            "pareto_selection": {
                "strategy": "COMPOSITE_SCORE",
                "weights": {
                    "sharpe_ratio": 0.7,
                    "max_drawdown": 0.3
                }
            },
            "sampler_params": {"seed": 42},
            "pruner_params": {"n_warmup_steps": 10},
            "early_stopping_rounds": 50,
            "study_name": "multi-obj-study-btc"
        }
    }

    try:
        # Valider et parser le dictionnaire de configuration
        main_config = MainOptimizationConfig.parse_obj(example_config_dict)
        print("Configuration validée avec succès !")
        print(f"La valeur de is_days est: {main_config.wfo_config.split_config.is_days}")


    except Exception as e:
        print(f"Erreur de validation de la configuration : {e}")
