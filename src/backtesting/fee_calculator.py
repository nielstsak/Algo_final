# src/backtesting/fee_calculator.py

"""
Ce module fournit une structure flexible pour le calcul des frais de transaction.
Il utilise un modèle de 'stratégie' (Strategy Pattern) où différentes méthodes de
calcul des frais sont encapsulées dans leurs propres classes.

Une fonction 'factory' (`get_fee_calculator`) est utilisée pour instancier
le calculateur approprié en fonction de la configuration fournie.
"""

from abc import ABC, abstractmethod
import logging

# Utilisation des modèles Pydantic pour une configuration robuste
# Note: Le chemin d'importation est ajusté pour pointer vers la config centralisée.
from src.optimization.config import FeeConfig
from src.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class FeeCalculator(ABC):
    """
    Classe de base abstraite pour tous les calculateurs de frais.
    Chaque calculateur doit implémenter une méthode `calculate`.
    """
    @abstractmethod
    def calculate(self, trade_value: float) -> float:
        """
        Calcule les frais pour un trade donné.

        Args:
            trade_value: La valeur monétaire totale du trade.

        Returns:
            Le montant des frais calculés.
        """
        pass

class PercentageFee(FeeCalculator):
    """
    Calcule les frais comme un pourcentage de la valeur du trade.
    C'est la méthode la plus courante pour les exchanges de cryptomonnaies.
    """
    def __init__(self, percentage: float):
        if not (0 <= percentage < 1):
            raise ValueError("Le pourcentage de frais doit être compris entre 0 et 1.")
        self.percentage = percentage
        logger.debug(f"Initialisation de PercentageFee avec un taux de {self.percentage:.4%}.")

    def calculate(self, trade_value: float) -> float:
        """
        Calcule les frais en appliquant le pourcentage stocké.
        """
        return trade_value * self.percentage

class FixedPerTradeFee(FeeCalculator):
    """
    Applique des frais fixes pour chaque transaction, indépendamment de sa valeur.
    """
    def __init__(self, fixed_amount: float):
        if fixed_amount < 0:
            raise ValueError("Le montant fixe des frais ne peut pas être négatif.")
        self.fixed_amount = fixed_amount
        logger.debug(f"Initialisation de FixedPerTradeFee avec un montant de {self.fixed_amount}.")

    def calculate(self, trade_value: float) -> float:
        """
        Retourne simplement le montant fixe des frais.
        `trade_value` est ignoré mais conservé pour la cohérence de l'interface.
        """
        return self.fixed_amount

def get_fee_calculator(config: FeeConfig) -> FeeCalculator:
    """
    Factory qui retourne une instance du calculateur de frais approprié
    en fonction de l'objet de configuration `FeeConfig`.

    Args:
        config: Un objet `FeeConfig` contenant la méthode et la valeur des frais.

    Returns:
        Une instance d'une sous-classe de `FeeCalculator`.

    Raises:
        ConfigurationError: Si la méthode de frais spécifiée n'est pas supportée.
    """
    method = config.method
    value = config.value

    logger.info(f"Création d'un calculateur de frais de type '{method}' avec la valeur '{value}'.")

    if method == 'PERCENTAGE':
        return PercentageFee(percentage=value)
    elif method == 'FIXED_PER_TRADE':
        return FixedPerTradeFee(fixed_amount=value)
    else:
        # Cette erreur ne devrait pas se produire si la validation Pydantic est correcte,
        # mais elle constitue une bonne sécurité.
        raise ConfigurationError(f"Méthode de calcul des frais non supportée : '{method}'")

# --- Exemple d'utilisation ---
if __name__ == '__main__':
    # Scénario 1: Frais en pourcentage
    fee_config_pct = FeeConfig(method='PERCENTAGE', value=0.00075) # 0.075%
    fee_calculator_pct = get_fee_calculator(fee_config_pct)
    trade_value_1 = 10000
    calculated_fee_1 = fee_calculator_pct.calculate(trade_value_1)
    print(f"Pour un trade de {trade_value_1} avec frais de {fee_config_pct.value:.4%}, les frais sont : {calculated_fee_1:.4f}")

    # Scénario 2: Frais fixes
    fee_config_fixed = FeeConfig(method='FIXED_PER_TRADE', value=1.5) # 1.5$ par trade
    fee_calculator_fixed = get_fee_calculator(fee_config_fixed)
    trade_value_2 = 500
    calculated_fee_2 = fee_calculator_fixed.calculate(trade_value_2)
    print(f"Pour un trade de {trade_value_2} avec frais fixes de {fee_config_fixed.value}, les frais sont : {calculated_fee_2:.4f}")