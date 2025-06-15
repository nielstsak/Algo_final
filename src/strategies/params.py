# src/strategies/params.py

from typing import Dict, Any, Type, Union
import optuna
from pydantic import BaseModel, Field, ValidationError
from loguru import logger

from src.core.exceptions import ConfigurationError

# --- Modèles Pydantic pour les Paramètres ---

class BaseFixedParams(BaseModel):
    """
    Classe de base pour les paramètres de stratégie qui ne sont pas optimisés.
    Valide que seuls les champs définis sont présents.
    """
    class Config:
        extra = 'forbid'

class BaseOptimizableParams(BaseModel):
    """
    Classe de base pour les paramètres de stratégie qui sont sujets à optimisation.
    Valide que seuls les champs définis sont présents.
    """
    class Config:
        extra = 'forbid'

# --- Fonctions de Suggestion pour Optuna ---

def suggest_params_from_config(trial: optuna.Trial, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggère un dictionnaire de paramètres pour un essai Optuna basé sur la configuration
    complète d'une stratégie.

    Args:
        trial: L'objet Trial d'Optuna.
        strategy_config: Le dictionnaire de configuration complet pour une stratégie,
                         contenant la section `optimizable_params`.

    Returns:
        Un dictionnaire de paramètres suggérés pour cet essai.

    Raises:
        ConfigurationError: Si le format de la configuration est invalide.
    """
    suggested_params = {}
    logger.debug(f"Début de la suggestion de paramètres pour le trial {trial.number}.")
    
    # --- CORRECTION ---
    # Extrait la section 'optimizable_params' de la configuration de la stratégie.
    # Cela rend la fonction robuste même si elle reçoit la configuration complète.
    params_config = strategy_config.get('optimizable_params')
    if not isinstance(params_config, dict):
        raise ConfigurationError(
            "La configuration de la stratégie doit contenir une section 'optimizable_params' sous forme de dictionnaire."
        )

    for param_name, config in params_config.items():
        
        if not isinstance(config, dict):
            raise ConfigurationError(
                f"La configuration pour le paramètre '{param_name}' doit être un dictionnaire."
            )
            
        param_type = config.get('type')
        if not param_type:
            raise ConfigurationError(
                f"Le type ('type') du paramètre '{param_name}' est manquant dans la configuration."
            )

        # Cas 1: Paramètre catégoriel
        if param_type == 'categorical':
            choices = config.get('choices')
            if not choices or not isinstance(choices, list):
                raise ConfigurationError(
                    f"Le paramètre catégoriel '{param_name}' doit avoir une liste de 'choices'."
                )
            value = trial.suggest_categorical(param_name, choices)
            logger.trace(f"   -> Paramètre '{param_name}': suggéré (catégoriel) = {value}")

        # Cas 2: Paramètre numérique (entier ou flottant)
        elif param_type in ['int', 'float']:
            low = config.get('low')
            high = config.get('high')
            step = config.get('step')
            log = config.get('log', False)
            
            if low is None or high is None:
                raise ConfigurationError(
                    f"Le paramètre numérique '{param_name}' doit avoir des bornes 'low' et 'high'."
                )
            
            if param_type == 'int':
                value = trial.suggest_int(param_name, low, high, step=step if step is not None else 1, log=log)
                logger.trace(f"   -> Paramètre '{param_name}': suggéré (int) = {value}")
            else: # float
                value = trial.suggest_float(param_name, low, high, step=step, log=log)
                logger.trace(f"   -> Paramètre '{param_name}': suggéré (float) = {value}")
        
        else:
            raise ConfigurationError(
                f"Type de paramètre non supporté '{param_type}' pour '{param_name}'."
            )
            
        suggested_params[param_name] = value

    logger.debug(f"Paramètres suggérés pour le trial {trial.number}: {suggested_params}")
    return suggested_params
