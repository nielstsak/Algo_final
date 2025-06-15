# src/backtesting/slippage_model.py

"""
Ce module définit des modèles pour simuler le slippage (glissement de prix)
lors de l'exécution des trades. Il utilise une architecture flexible à base de
classes, similaire à celle du `fee_calculator`.

Chaque modèle de slippage est encapsulé dans sa propre classe, et une fonction
factory (`get_slippage_model`) sélectionne le modèle approprié en fonction
de la configuration `SlippageConfig`.
"""

from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np

# --- Imports Locaux ---
from src.optimization.config import SlippageConfig
from src.core.exceptions import ConfigurationError, DataError

logger = logging.getLogger(__name__)


class SlippageModel(ABC):
    """
    Classe de base abstraite pour tous les modèles de slippage.

    Chaque modèle doit être capable de générer une série de valeurs de slippage
    correspondant à une série de prix, car c'est ainsi que vectorbt l'applique.
    """
    @abstractmethod
    def generate_slippage_series(self, price_series: pd.Series) -> pd.Series:
        """
        Génère une série de valeurs de slippage.

        Args:
            price_series: La série de prix sur laquelle le slippage doit être calculé.
                          Typiquement la série des prix de clôture ('close').

        Returns:
            Une série pandas avec les valeurs de slippage (en pourcentage) pour chaque point de données.
        """
        pass

class PercentageSlippage(SlippageModel):
    """
    Modélise le slippage comme un pourcentage fixe du prix d'exécution.
    """
    def __init__(self, percentage: float):
        if not (0 <= percentage < 1):
            raise ValueError("Le pourcentage de slippage doit être compris entre 0 et 1.")
        self.percentage = percentage
        logger.debug(f"Initialisation de PercentageSlippage avec une valeur de {self.percentage:.4%}.")

    def generate_slippage_series(self, price_series: pd.Series) -> pd.Series:
        """
        Retourne une série où chaque valeur est le pourcentage de slippage constant.
        """
        return pd.Series(self.percentage, index=price_series.index)

class FixedPerTradeSlippage(SlippageModel):
    """
    Modélise le slippage comme un montant monétaire fixe par trade.
    Ce montant est ensuite converti en pourcentage du prix au moment du trade.
    """
    def __init__(self, fixed_amount: float):
        if fixed_amount < 0:
            raise ValueError("Le montant fixe de slippage ne peut pas être négatif.")
        self.fixed_amount = fixed_amount
        logger.debug(f"Initialisation de FixedPerTradeSlippage avec un montant de {self.fixed_amount}.")

    def generate_slippage_series(self, price_series: pd.Series) -> pd.Series:
        """
        Calcule le slippage en pourcentage en divisant le montant fixe par le prix.
        """
        # Éviter la division par zéro
        slippage_series = self.fixed_amount / price_series.replace(0, np.nan)
        return slippage_series.fillna(0)

class VolatilityAdjustedSlippage(SlippageModel):
    """
    Modélise le slippage comme une fonction de la volatilité du marché.
    Le slippage augmente lorsque le marché est plus volatil.
    """
    def __init__(self, value: float, volatility_window: int, volatility_multiplier: float):
        self.base_slippage = value
        self.volatility_window = volatility_window
        self.volatility_multiplier = volatility_multiplier
        logger.debug(f"Initialisation de VolatilityAdjustedSlippage (window={self.volatility_window}, mult={self.volatility_multiplier}).")

    def generate_slippage_series(self, price_series: pd.Series) -> pd.Series:
        """
        Calcule une série de slippage dynamique basée sur la volatilité.
        """
        daily_returns = price_series.pct_change()
        # Calcul de la volatilité glissante (écart-type des rendements)
        rolling_volatility = daily_returns.rolling(window=self.volatility_window).std()
        
        # Le slippage est le slippage de base plus un composant lié à la volatilité
        dynamic_slippage = self.base_slippage + (rolling_volatility * self.volatility_multiplier)
        
        # Remplacer les NaN (au début de la série) par le slippage de base
        return dynamic_slippage.fillna(self.base_slippage)


def get_slippage_model(config: SlippageConfig) -> SlippageModel:
    """
    Factory qui retourne une instance du modèle de slippage approprié
    en fonction de l'objet de configuration `SlippageConfig`.

    Args:
        config: Un objet `SlippageConfig`.

    Returns:
        Une instance d'une sous-classe de `SlippageModel`.

    Raises:
        ConfigurationError: Si la méthode de slippage n'est pas supportée.
    """
    method = config.method
    logger.info(f"Création d'un modèle de slippage de type '{method}'.")

    if method == 'PERCENTAGE':
        return PercentageSlippage(percentage=config.value)
    elif method == 'FIXED_PER_TRADE':
        return FixedPerTradeSlippage(fixed_amount=config.value)
    elif method == 'VOLATILITY_ADJUSTED':
        if config.volatility_window is None or config.volatility_multiplier is None:
            raise ConfigurationError("`volatility_window` et `volatility_multiplier` sont requis pour le slippage ajusté à la volatilité.")
        return VolatilityAdjustedSlippage(
            value=config.value,
            volatility_window=config.volatility_window,
            volatility_multiplier=config.volatility_multiplier
        )
    else:
        raise ConfigurationError(f"Méthode de slippage non supportée : '{method}'")


# --- Exemple d'utilisation ---
if __name__ == '__main__':
    # Créer un faux jeu de données
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200, freq='D'))
    price_data = pd.Series(
        np.random.randn(len(dates)).cumsum() + 100,
        index=dates
    )
    price_data.name = 'close'

    # Scénario 1: Slippage en pourcentage
    slippage_config_pct = SlippageConfig(method='PERCENTAGE', value=0.0005) # 0.05%
    model_pct = get_slippage_model(slippage_config_pct)
    series_pct = model_pct.generate_slippage_series(price_data)
    print("--- Modèle de Slippage en Pourcentage ---")
    print(f"Valeur constante attendue : {slippage_config_pct.value}")
    print(series_pct.head())

    # Scénario 2: Slippage ajusté à la volatilité
    slippage_config_vol = SlippageConfig(
        method='VOLATILITY_ADJUSTED',
        value=0.0001, # 0.01% de base
        volatility_window=10,
        volatility_multiplier=0.5
    )
    model_vol = get_slippage_model(slippage_config_vol)
    series_vol = model_vol.generate_slippage_series(price_data)
    print("\n--- Modèle de Slippage Ajusté à la Volatilité ---")
    print("Le slippage devrait varier avec la volatilité des prix :")
    print(series_vol.tail())
    series_vol.plot(title="Slippage Dynamique (Ajusté à la Volatilité)").get_figure()