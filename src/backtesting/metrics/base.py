"""
Module contenant la classe de base pour les métriques de performance.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import vectorbt as vbt
import logging

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """
    Classe abstraite de base pour les métriques de performance.
    Toutes les métriques spécifiques doivent hériter de cette classe.
    """
    
    @property
    @abstractmethod
    def category(self) -> str:
        """
        Retourne la catégorie de la métrique.
        
        Returns:
            str: Catégorie de la métrique (e.g. 'Returns', 'Risk', 'Trade')
        """
        pass
    
    @abstractmethod
    def calculate(self, portfolio: vbt.Portfolio, **kwargs) -> Dict[str, Any]:
        """
        Calcule les métriques spécifiques pour un portfolio VectorBT.
        
        Args:
            portfolio: Portfolio VectorBT pour lequel calculer les métriques
            **kwargs: Arguments supplémentaires spécifiques à la métrique
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant les métriques calculées
        """
        pass
    
    def _handle_calculation_error(self, metric_name: str, error: Exception) -> Optional[float]:
        """
        Gère les erreurs lors du calcul des métriques.
        
        Args:
            metric_name: Nom de la métrique qui a échoué
            error: Exception levée pendant le calcul
            
        Returns:
            Optional[float]: None en cas d'erreur, pour être remplacé par NaN dans le résultat
        """
        logger.warning(f"Erreur lors du calcul de la métrique '{metric_name}': {error}")
        return None
    
    def _format_metric_name(self, base_name: str) -> str:
        """
        Formate le nom d'une métrique selon une convention standard.
        
        Args:
            base_name: Nom de base de la métrique
            
        Returns:
            str: Nom formaté de la métrique
        """
        category_prefix = self.category.lower()
        return f"{category_prefix}_{base_name}"
    
    def _safe_calculate(self, metric_func, metric_name: str, *args, **kwargs) -> Any:
        """
        Exécute une fonction de calcul de métrique avec gestion d'erreur.
        
        Args:
            metric_func: Fonction à exécuter
            metric_name: Nom de la métrique (pour le log en cas d'erreur)
            *args, **kwargs: Arguments à passer à la fonction
            
        Returns:
            Any: Résultat du calcul ou None en cas d'erreur
        """
        try:
            return metric_func(*args, **kwargs)
        except Exception as e:
            return self._handle_calculation_error(metric_name, e)