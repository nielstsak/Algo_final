"""
Ce module définit les modèles Pydantic pour les paramètres de chaque stratégie.
L'architecture sépare les paramètres en deux catégories :
- `BaseFixedParams`: Paramètres qui ne sont généralement pas optimisés (ex: noms de colonnes).
- `BaseOptimizableParams`: Paramètres destinés à l'optimisation (ex: périodes, seuils).
"""
from pydantic import BaseModel, Field


class BaseParams(BaseModel):
    """
    Classe de base pour tous les modèles de paramètres Pydantic.
    """
    class Config:
        extra = 'forbid'


class BaseFixedParams(BaseParams):
    """
    Classe de base pour les paramètres fixes d'une stratégie.
    """
    pass


class BaseOptimizableParams(BaseParams):
    """
    Classe de base pour les paramètres optimisables d'une stratégie.
    """
    pass

# --- Paramètres pour SMACrossStrategy ---
class SMACrossFixedParams(BaseFixedParams):
    pass

class SMACrossOptimizableParams(BaseOptimizableParams):
    fast_ma: int = Field(10, gt=0, description="Période de la moyenne mobile rapide.")
    slow_ma: int = Field(30, gt=0, description="Période de la moyenne mobile lente.")
    stop_loss_pct: float = Field(0.05, gt=0, lt=1, description="Pourcentage de perte pour le stop-loss.")
    take_profit_pct: float = Field(0.10, gt=0, lt=1, description="Pourcentage de gain pour le take-profit.")

# --- Paramètres pour PsarReversalOtocoStrategy ---
class PsarReversalOtocoFixedParams(BaseFixedParams):
    pass

class PsarReversalOtocoOptimizableParams(BaseOptimizableParams):
    psar_step: float = Field(0.02, gt=0, description="Le pas (step) pour l'indicateur PSAR.")
    psar_max_step: float = Field(0.2, gt=0, description="Le pas maximum (max step) pour l'indicateur PSAR.")
    risk_reward_ratio: float = Field(2.0, gt=0, description="Ratio risque/rendement.")

# --- Paramètres pour TripleMaAnticipationStrategy ---
class TripleMaAnticipationFixedParams(BaseFixedParams):
    pass

class TripleMaAnticipationOptimizableParams(BaseOptimizableParams):
    fast_ma_period: int = Field(5, gt=0, description="Période de la MA rapide.")
    medium_ma_period: int = Field(8, gt=0, description="Période de la MA intermédiaire.")
    slow_ma_period: int = Field(13, gt=0, description="Période de la MA lente.")
    atr_period_sl: int = Field(14, gt=0, description="Période de l'ATR pour le stop-loss.")
    atr_multiplier_sl: float = Field(2.0, gt=0, description="Multiplicateur de l'ATR pour le stop-loss.")
    risk_reward_ratio: float = Field(1.5, gt=0, description="Ratio risque/rendement.")

# --- Paramètres pour BbandsVolumeRsiStrategy (Version Breakout) ---
class BbandsVolumeRsiStrategyFixedParams(BaseFixedParams):
    """
    Paramètres fixes pour la stratégie. La plupart des paramètres de cette
    stratégie sont optimisables, donc cette classe est vide pour le moment.
    """
    pass

class BbandsVolumeRsiStrategyOptimizableParams(BaseOptimizableParams):
    """
    Paramètres optimisables pour la stratégie de cassure (breakout)
    basée sur les Bandes de Bollinger, le Volume et le RSI.
    """
    bbands_period: int = Field(20, gt=1, description="Période pour les Bandes de Bollinger.")
    bbands_std_dev: float = Field(2.0, gt=0, description="Écart-type pour les Bandes de Bollinger.")
    volume_ma_period: int = Field(20, gt=1, description="Période pour la moyenne mobile du volume.")
    rsi_period: int = Field(14, gt=1, description="Période pour le RSI.")
    rsi_buy_breakout_threshold: float = Field(60.0, ge=50, le=100, description="Seuil RSI pour confirmer une cassure haussière.")
    rsi_sell_breakout_threshold: float = Field(40.0, ge=0, le=50, description="Seuil RSI pour confirmer une cassure baissière.")
    atr_period: int = Field(14, gt=1, description="Période pour le calcul de l'ATR.")
    sl_atr_mult: float = Field(1.5, gt=0, description="Multiplicateur ATR pour le Stop-Loss.")
    tp_atr_mult: float = Field(3.0, gt=0, description="Multiplicateur ATR pour le Take-Profit.")
