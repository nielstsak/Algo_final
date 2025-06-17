import logging
from typing import Dict, Any, Type, Optional
import optuna
from pydantic import BaseModel, Field, validator

# Logger
logger = logging.getLogger(__name__)

# --- Modèles de Paramètres de Base ---


class BaseStrategyParams(BaseModel):
    """Modèle de base pour tous les paramètres de stratégie."""

    class Config:
        validate_assignment = True
        extra = "forbid"


class BaseFixedParams(BaseStrategyParams):
    """Modèle de base pour les paramètres fixes de n'importe quelle stratégie."""

    pass


class BaseOptimizableParams(BaseStrategyParams):
    """Modèle de base pour les paramètres optimisables de n'importe quelle stratégie."""

    pass


class StopLossTakeProfitParams(BaseFixedParams):
    """Paramètres communs pour la gestion des risques (généralement fixes)."""

    sl_atr_mult: float = Field(
        ...,
        gt=0,
        description="Multiplicateur de l'ATR pour définir le Stop-Loss.",
    )
    tp_atr_mult: float = Field(
        ...,
        gt=0,
        description="Multiplicateur de l'ATR pour définir le Take-Profit.",
    )

# --- Paramètres pour BbandsVolumeRsiStrategy ---


class BbandsVolumeRsiStrategyFixedParams(StopLossTakeProfitParams):
    """Paramètres fixes pour la stratégie BbandsVolumeRsiStrategy."""

    pass  # Hérite déjà de sl_atr_mult et tp_atr_mult


class BbandsVolumeRsiStrategyOptimizableParams(BaseOptimizableParams):
    """Paramètres optimisables pour la stratégie BbandsVolumeRsiStrategy."""

    bbands_period: int = Field(..., gt=1, description="Période des Bandes de Bollinger.")
    bbands_std_dev: float = Field(
        ..., gt=0, description="Écart-type des Bandes de Bollinger."
    )
    volume_ma_period: int = Field(
        ..., gt=1, description="Période de la MM du volume."
    )
    rsi_period: int = Field(..., gt=1, description="Période du RSI.")
    rsi_buy_breakout_threshold: int = Field(
        ..., gt=50, lt=100, description="Seuil RSI pour cassure haussière."
    )
    rsi_sell_breakout_threshold: int = Field(
        ..., gt=0, lt=50, description="Seuil RSI pour cassure baissière."
    )
    atr_period: int = Field(..., gt=1, description="Période de l'ATR.")

    @validator("rsi_buy_breakout_threshold")
    def rsi_buy_must_be_greater(cls, v, values):
        if "rsi_sell_breakout_threshold" in values and v <= values["rsi_sell_breakout_threshold"]:
            raise ValueError(
                "rsi_buy_breakout_threshold doit être supérieur à rsi_sell_breakout_threshold"
            )
        return v

# --- Fonction pour Optuna ---


def suggest_params_from_config(
    trial: optuna.trial.Trial, params_config: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Suggère des hyperparamètres pour une étude Optuna à partir d'un dictionnaire de configuration.

    Args:
        trial: L'objet Trial d'Optuna pour l'itération en cours.
        params_config: Un dictionnaire où chaque clé est le nom du paramètre et
                       la valeur est un autre dictionnaire spécifiant le type de
                       suggestion et ses bornes (low, high, step, choices).

    Returns:
        Un dictionnaire contenant les paramètres suggérés pour cet essai.
    
    Raises:
        ValueError: Si un type de suggestion non supporté est rencontré.
    """
    suggested_params = {}
    for param_name, config in params_config.items():
        suggestion_type = config.get("type")

        if suggestion_type == "int":
            low = config["low"]
            high = config["high"]
            step = config.get("step", 1)
            suggested_params[param_name] = trial.suggest_int(
                param_name, low, high, step=step
            )
        elif suggestion_type == "float":
            low = config["low"]
            high = config["high"]
            step = config.get("step")  # Peut être None
            suggested_params[param_name] = trial.suggest_float(
                param_name, low, high, step=step
            )
        elif suggestion_type == "categorical":
            choices = config["choices"]
            suggested_params[param_name] = trial.suggest_categorical(
                param_name, choices
            )
        else:
            error_msg = f"Type de suggestion non supporté '{suggestion_type}' pour le paramètre '{param_name}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

    logger.debug(f"Paramètres suggérés pour l'essai {trial.number}: {suggested_params}")
    return suggested_params
