"""
Module de registre de stratégies, permettant l'enregistrement et la récupération
dynamique des stratégies de trading.
"""

from typing import Dict, Type, Any, Optional
import inspect
import logging

from src.strategies.base import BaseStrategy
from src.core.exceptions import StrategyLoadError

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Registre central pour toutes les stratégies de trading.
    Utilise le pattern Registry pour enregistrer les classes de stratégie et les instancier dynamiquement.
    """
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        Décorateur pour enregistrer une classe de stratégie.
        
        Args:
            name: Nom personnalisé pour la stratégie. Si non fourni, le nom de la classe sera utilisé.
            
        Returns:
            Le décorateur à appliquer à la classe de stratégie.
        
        Exemple:
            @StrategyRegistry.register("sma_cross")
            class SMACrossStrategy(BaseStrategy):
                pass
        """
        def decorator(strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
            if not inspect.isclass(strategy_class):
                raise TypeError(f"Impossible d'enregistrer {strategy_class}. Ce n'est pas une classe.")
                
            if not issubclass(strategy_class, BaseStrategy):
                raise TypeError(
                    f"Impossible d'enregistrer {strategy_class.__name__}. "
                    f"Les stratégies doivent hériter de BaseStrategy."
                )
                
            strategy_name = name or strategy_class.__name__
            cls._strategies[strategy_name] = strategy_class
            logger.debug(f"Stratégie '{strategy_name}' enregistrée: {strategy_class.__name__}")
            return strategy_class
            
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseStrategy:
        """
        Crée une instance de stratégie à partir de son nom.
        
        Args:
            name: Nom de la stratégie à instancier
            **kwargs: Arguments à passer au constructeur de la stratégie
            
        Returns:
            Une nouvelle instance de la stratégie demandée
            
        Raises:
            StrategyLoadError: Si la stratégie demandée n'est pas trouvée dans le registre
        """
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise StrategyLoadError(
                f"Stratégie inconnue: '{name}'. "
                f"Stratégies disponibles: {available}"
            )
            
        strategy_class = cls._strategies[name]
        try:
            instance = strategy_class(**kwargs)
            return instance
        except Exception as e:
            raise StrategyLoadError(
                f"Erreur lors de l'instanciation de la stratégie '{name}': {str(e)}"
            ) from e
    
    @classmethod
    def get_strategy_class(cls, name: str) -> Type[BaseStrategy]:
        """
        Récupère la classe de stratégie (non instanciée) par son nom.
        
        Args:
            name: Nom de la stratégie à récupérer
            
        Returns:
            La classe de stratégie
            
        Raises:
            StrategyLoadError: Si la stratégie demandée n'est pas trouvée dans le registre
        """
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise StrategyLoadError(
                f"Stratégie inconnue: '{name}'. "
                f"Stratégies disponibles: {available}"
            )
            
        return cls._strategies[name]
    
    @classmethod
    def list_strategies(cls) -> Dict[str, Dict[str, Any]]:
        """
        Liste toutes les stratégies enregistrées avec leurs métadonnées.
        
        Returns:
            Un dictionnaire contenant toutes les stratégies enregistrées et leurs métadonnées
        """
        result = {}
        for name, strategy_class in cls._strategies.items():
            # Extraire les métadonnées de la classe
            description = strategy_class.__doc__.strip() if strategy_class.__doc__ else "Pas de description"
            params_spec = {}
            
            # Analyser les paramètres du constructeur
            try:
                sig = inspect.signature(strategy_class.__init__)
                for param_name, param in sig.parameters.items():
                    # Ignorer self, pair_symbol et strategy_name
                    if param_name in ('self', 'pair_symbol', 'strategy_name'):
                        continue
                    
                    # Capturer les informations sur le paramètre
                    params_spec[param_name] = {
                        'default': None if param.default is inspect.Parameter.empty else param.default,
                        'required': param.default is inspect.Parameter.empty,
                        'type': str(param.annotation) if param.annotation is not inspect.Parameter.empty else None
                    }
            except Exception as e:
                logger.warning(f"Impossible d'analyser les paramètres de {name}: {e}")
            
            result[name] = {
                'class': strategy_class.__name__,
                'description': description,
                'params': params_spec
            }
            
        return result