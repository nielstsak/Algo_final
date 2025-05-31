# src/strategies/strategy_loader.py
import importlib
import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple
from loguru import logger

from src.strategies.base_strategy import BaseStrategy
from src.core.exceptions import StrategyLoadError, StrategyError


class StrategyLoader:
    """
    Gestionnaire pour charger dynamiquement les stratégies de trading.
    Permet de découvrir, charger et instancier les stratégies.
    """
    
    def __init__(self):
        """Initialise le chargeur de stratégies."""
        self.strategies_path = Path(__file__).parent / "implementations"
        self.loaded_strategies: Dict[str, Type[BaseStrategy]] = {}
        self.strategy_instances: Dict[str, BaseStrategy] = {}
        
        # Charger les stratégies built-in
        self._load_builtin_strategies()
        
        logger.info(f"StrategyLoader initialized with {len(self.loaded_strategies)} strategies")
        
    def _load_builtin_strategies(self):
        """Charge les stratégies intégrées connues."""
        builtin_strategies = [
            ("sma_cross", "SMACrossStrategy"),
            ("rsi_divergence", "RSIDivergenceStrategy"),
            ("adx_direction_otoco", "AdxDirectionOtocoStrategy"),
            ("psar_reversal_otoco", "PsarReversalOtocoStrategy"),
            ("triple_ma_anticipation_strategy", "TripleMAAnticipationStrategy")
        ]
        
        for module_name, class_name in builtin_strategies:
            try:
                self._load_strategy_module(f"implementations.{module_name}", class_name)
            except Exception as e:
                logger.warning(f"Failed to load builtin strategy {module_name}: {e}")
                
    def _load_strategy_module(self, module_path: str, class_name: Optional[str] = None):
        """
        Charge un module de stratégie spécifique.
        
        Args:
            module_path: Chemin relatif du module depuis src.strategies
            class_name: Nom de la classe à charger (optionnel)
        """
        try:
            # Construire le chemin complet du module
            full_module_path = f"src.strategies.{module_path}"
            module = importlib.import_module(full_module_path)
            
            # Si pas de nom de classe spécifié, chercher toutes les stratégies
            if class_name:
                if hasattr(module, class_name):
                    strategy_class = getattr(module, class_name)
                    if self._is_valid_strategy_class(strategy_class):
                        self._register_strategy(strategy_class)
                else:
                    logger.warning(f"Class {class_name} not found in module {module_path}")
            else:
                # Chercher toutes les classes qui héritent de BaseStrategy
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_valid_strategy_class(obj) and obj.__module__ == module.__name__:
                        self._register_strategy(obj)
                        
            logger.debug(f"Successfully loaded module {module_path}")
            
        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            raise StrategyLoadError(f"Cannot import strategy module: {e}", original_exception=e)
        except Exception as e:
            logger.error(f"Error loading strategy module {module_path}: {e}")
            raise StrategyLoadError(f"Error loading strategy: {e}", original_exception=e)
            
    def _is_valid_strategy_class(self, obj: Any) -> bool:
        """
        Vérifie si un objet est une classe de stratégie valide.
        
        Args:
            obj: Objet à vérifier
            
        Returns:
            True si c'est une stratégie valide
        """
        return (
            inspect.isclass(obj) and
            issubclass(obj, BaseStrategy) and
            obj is not BaseStrategy and
            not inspect.isabstract(obj)
        )
        
    def _register_strategy(self, strategy_class: Type[BaseStrategy]):
        """
        Enregistre une classe de stratégie.
        
        Args:
            strategy_class: Classe de stratégie à enregistrer
        """
        strategy_name = strategy_class.name
        
        if strategy_name in self.loaded_strategies:
            logger.warning(f"Strategy {strategy_name} already registered, replacing")
            
        self.loaded_strategies[strategy_name] = strategy_class
        logger.info(f"Registered strategy: {strategy_name} (v{strategy_class.version})")
        
    def load_custom_strategies(self, custom_path: Optional[Path] = None):
        """
        Charge les stratégies personnalisées depuis un répertoire.
        
        Args:
            custom_path: Chemin vers le répertoire des stratégies custom
        """
        if custom_path is None:
            custom_path = self.strategies_path / "custom"
            
        if not custom_path.exists():
            logger.debug(f"Custom strategies directory {custom_path} does not exist")
            return
            
        # Parcourir les fichiers Python dans le répertoire
        for file_path in custom_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
                
            module_name = file_path.stem
            try:
                # Calculer le chemin relatif depuis src.strategies
                relative_path = file_path.relative_to(self.strategies_path)
                module_path = str(relative_path.with_suffix("")).replace(os.sep, ".")
                
                self._load_strategy_module(f"implementations.{module_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load custom strategy from {file_path}: {e}")
                
    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Récupère une classe de stratégie par son nom.
        
        Args:
            name: Nom de la stratégie
            
        Returns:
            Classe de stratégie ou None si non trouvée
        """
        return self.loaded_strategies.get(name)
        
    def create_strategy(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseStrategy:
        """
        Crée une instance de stratégie.
        
        Args:
            name: Nom de la stratégie
            params: Paramètres de la stratégie
            **kwargs: Paramètres additionnels
            
        Returns:
            Instance de la stratégie
            
        Raises:
            StrategyLoadError: Si la stratégie n'existe pas
            StrategyError: Si erreur lors de l'instanciation
        """
        strategy_class = self.get_strategy_class(name)
        
        if not strategy_class:
            available = list(self.loaded_strategies.keys())
            raise StrategyLoadError(
                f"Strategy '{name}' not found. Available: {available}"
            )
            
        try:
            # Créer l'instance
            instance = strategy_class(params=params, **kwargs)
            
            # Optionnel: garder une référence
            instance_key = f"{name}_{id(instance)}"
            self.strategy_instances[instance_key] = instance
            
            logger.debug(f"Created strategy instance: {name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create strategy {name}: {e}")
            raise StrategyError(
                f"Cannot instantiate strategy '{name}': {e}",
                strategy_name=name,
                original_exception=e
            )
            
    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        Liste toutes les stratégies disponibles.
        
        Returns:
            Liste des informations sur les stratégies
        """
        strategies_info = []
        
        for name, strategy_class in self.loaded_strategies.items():
            info = {
                'name': name,
                'class_name': strategy_class.__name__,
                'version': strategy_class.version,
                'description': strategy_class.description,
                'required_timeframes': strategy_class.required_timeframes,
                'min_required_periods': strategy_class.min_required_periods,
                'default_params': strategy_class.default_params
            }
            strategies_info.append(info)
            
        return strategies_info
        
    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations détaillées sur une stratégie.
        
        Args:
            name: Nom de la stratégie
            
        Returns:
            Informations sur la stratégie ou None
        """
        strategy_class = self.get_strategy_class(name)
        
        if not strategy_class:
            return None
            
        return {
            'name': name,
            'class_name': strategy_class.__name__,
            'version': strategy_class.version,
            'description': strategy_class.description,
            'module': strategy_class.__module__,
            'required_timeframes': strategy_class.required_timeframes,
            'min_required_periods': strategy_class.min_required_periods,
            'default_params': strategy_class.default_params,
            'docstring': inspect.getdoc(strategy_class)
        }
        
    def reload_strategy(self, name: str) -> bool:
        """
        Recharge une stratégie (utile pour le développement).
        
        Args:
            name: Nom de la stratégie
            
        Returns:
            True si rechargée avec succès
        """
        strategy_class = self.get_strategy_class(name)
        
        if not strategy_class:
            logger.warning(f"Cannot reload unknown strategy: {name}")
            return False
            
        try:
            # Recharger le module
            module = importlib.import_module(strategy_class.__module__)
            importlib.reload(module)
            
            # Récupérer la classe rechargée
            reloaded_class = getattr(module, strategy_class.__name__)
            
            # Ré-enregistrer
            self._register_strategy(reloaded_class)
            
            logger.info(f"Successfully reloaded strategy: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload strategy {name}: {e}")
            return False
            
    def validate_strategy_params(
        self,
        name: str,
        params: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Valide les paramètres pour une stratégie.
        
        Args:
            name: Nom de la stratégie
            params: Paramètres à valider
            
        Returns:
            Tuple (is_valid, list_of_errors)
        """
        strategy_class = self.get_strategy_class(name)
        
        if not strategy_class:
            return False, [f"Strategy '{name}' not found"]
            
        errors = []
        
        try:
            # Créer une instance temporaire pour valider
            temp_instance = strategy_class(params=params)
            # Si on arrive ici, les paramètres sont valides
            return True, []
            
        except InvalidStrategyParamsError as e:
            errors.append(str(e))
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return False, errors
        
    def clear_instances(self):
        """Supprime toutes les instances de stratégies en mémoire."""
        self.strategy_instances.clear()
        logger.debug("Cleared all strategy instances")


# Singleton global
strategy_loader = StrategyLoader()


# Fonctions utilitaires pour un accès facile
def get_strategy(name: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
    """
    Crée une instance de stratégie.
    
    Args:
        name: Nom de la stratégie
        params: Paramètres de la stratégie
        
    Returns:
        Instance de la stratégie
    """
    return strategy_loader.create_strategy(name, params)


def list_available_strategies() -> List[str]:
    """
    Liste les noms des stratégies disponibles.
    
    Returns:
        Liste des noms
    """
    return list(strategy_loader.loaded_strategies.keys())


def reload_all_strategies():
    """Recharge toutes les stratégies."""
    for name in list(strategy_loader.loaded_strategies.keys()):
        strategy_loader.reload_strategy(name)
        

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les stratégies custom
    strategy_loader.load_custom_strategies()
    
    # Lister les stratégies
    print("Available strategies:")
    for info in strategy_loader.list_strategies():
        print(f"  - {info['name']} v{info['version']}: {info['description']}")
        
    # Créer une stratégie
    try:
        strategy = get_strategy("sma_cross", params={"fast_period": 10, "slow_period": 20})
        print(f"\nCreated strategy: {strategy}")
    except Exception as e:
        print(f"Error creating strategy: {e}")