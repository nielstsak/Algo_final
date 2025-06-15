# src/strategies/strategy_loader.py
import importlib
import inspect
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple
from loguru import logger

from src.strategies.base_strategy import BaseStrategy
from src.strategies.registry import StrategyRegistry
from src.core.exceptions import StrategyLoadError, StrategyError, InvalidStrategyParamsError

class StrategyLoader:
    """
    Gestionnaire pour charger dynamiquement les stratégies de trading.
    Permet de découvrir, charger et instancier les stratégies.
    """
    
    def __init__(self):
        """Initialise le chargeur de stratégies."""
        self.strategies_path = Path(__file__).parent # Points to src/strategies
        self.implementations_path = self.strategies_path / "implementations"
        self.loaded_strategies: Dict[str, Type[BaseStrategy]] = {}
        self.strategy_instances: Dict[str, BaseStrategy] = {}
        
        self._load_builtin_strategies()
        # Consider adding a method to discover strategies in implementations_path if not covered by builtin
        self._discover_strategies(self.implementations_path)
        
        logger.info(f"StrategyLoader initialized with {len(self.loaded_strategies)} strategies found.")
        if not self.loaded_strategies:
            logger.warning(f"No strategies were loaded. Check paths and implementations. Searched in: {self.implementations_path}")

    def _load_builtin_strategies(self):
        """Charge les stratégies intégrées connues."""
        # Updated module names and class names based on provided files
        # Assumes strategy files are directly under 'implementations'
        builtin_strategies = [
            ("sma_cross", "SMACrossStrategy"), # Assuming sma_cross.py and class SMACrossStrategy
            # ("rsi_divergence", "RSIDivergenceStrategy"), # File not provided, commented out
            ("psar_reversal_otoco_strategy_impl", "PsarReversalOtocoStrategy"),
            ("triple_ma_anticipation_strategy_impl", "TripleMAAnticipationStrategy"),
            ("bbands_volume_rsi_strategy_impl", "BbandsVolumeRsiStrategy") # Added target strategy
        ]
        
        for module_file_stem, class_name_to_load in builtin_strategies:
            try:
                # Module path is relative to src.strategies.implementations
                module_full_path = f"src.strategies.implementations.{module_file_stem}"
                self._load_strategy_module(module_full_path, class_name_to_load)
            except StrategyLoadError as e: # Catch specific load errors
                logger.warning(f"Failed to load builtin strategy from module '{module_file_stem}' (class '{class_name_to_load}'): {e}")
            except Exception as e: # Catch any other unexpected error during loading of a specific strategy
                logger.error(f"Unexpected error loading builtin strategy {module_file_stem} (class {class_name_to_load}): {e}", exc_info=True)

    def _discover_strategies(self, directory: Path):
        """
        Découvre et charge les stratégies à partir de fichiers Python dans un répertoire.
        """
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Strategy implementations directory not found or not a directory: {directory}")
            return

        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name == "base_strategy.py": # Ignore __init__.py, base_strategy.py
                continue
                
            module_file_stem = file_path.stem
            module_full_path = f"src.strategies.implementations.{module_file_stem}"
            
            # Avoid reloading if already loaded via builtin_strategies by checking strategy names later
            try:
                self._load_strategy_module(module_full_path) # Load all valid classes from the module
            except StrategyLoadError as e:
                logger.warning(f"Failed to discover/load strategy from file '{file_path.name}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error discovering strategy from file {file_path.name}: {e}", exc_info=True)


    def _load_strategy_module(self, module_full_path: str, specific_class_name: Optional[str] = None):
        """
        Charge un module de stratégie spécifique.
        
        Args:
            module_full_path: Chemin Python complet du module (e.g., src.strategies.implementations.my_strategy)
            specific_class_name: Nom de la classe spécifique à charger (optionnel, si None, charge toutes les classes valides)
        """
        try:
            module = importlib.import_module(module_full_path)
            
            classes_found_in_module = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_valid_strategy_class(obj) and obj.__module__ == module.__name__: # Ensure class is defined in this module
                    if specific_class_name and name == specific_class_name:
                        self._register_strategy(obj)
                        classes_found_in_module.append(name)
                        break # Found the specific class
                    elif not specific_class_name:
                        self._register_strategy(obj)
                        classes_found_in_module.append(name)
            
            if specific_class_name and not classes_found_in_module:
                 logger.warning(f"Specified class '{specific_class_name}' not found or not a valid strategy in module '{module_full_path}'.")
            elif not classes_found_in_module:
                logger.debug(f"No valid strategy classes found in module '{module_full_path}'.")
            else:
                logger.debug(f"Successfully processed module '{module_full_path}'. Found strategies: {classes_found_in_module}")

        except ImportError as e:
            # logger.error(f"Failed to import module {module_full_path}: {e}") # Already logged by caller usually
            raise StrategyLoadError(f"Cannot import strategy module: {module_full_path}. Error: {e}", original_exception=e)
        except Exception as e: # Catch other errors like Pydantic validation during class definition
            # logger.error(f"Error loading strategy module {module_full_path}: {e}")
            raise StrategyLoadError(f"Error loading strategy from module {module_full_path}. Error: {e}", original_exception=e)
            
    def _is_valid_strategy_class(self, obj: Any) -> bool:
        """
        Vérifie si un objet est une classe de stratégie valide.
        """
        return (
            inspect.isclass(obj) and
            issubclass(obj, BaseStrategy) and
            obj is not BaseStrategy and
            hasattr(obj, 'name') and isinstance(obj.name, str) and # Check for 'name' attribute
            not inspect.isabstract(obj)
        )
        
    def _register_strategy(self, strategy_class: Type[BaseStrategy]):
        """
        Enregistre une classe de stratégie.
        Utilise `strategy_class.name` comme clé et délègue au StrategyRegistry.
        """
        strategy_name_attr = getattr(strategy_class, 'name', None)
        if not strategy_name_attr:
            logger.warning(f"Strategy class {strategy_class.__name__} in module {strategy_class.__module__} is missing 'name' attribute. Skipping registration.")
            return

        # Ancien code pour gérer les conflits de noms - maintenant géré par le registre
        if strategy_name_attr in self.loaded_strategies:
            existing_class = self.loaded_strategies[strategy_name_attr]
            if existing_class.__module__ != strategy_class.__module__:
                logger.warning(
                    f"Strategy name '{strategy_name_attr}' from {strategy_class.__module__}.{strategy_class.__name__} "
                    f"conflicts with existing strategy from {existing_class.__module__}.{existing_class.__name__}. Replacing."
                )
            elif existing_class != strategy_class:
                logger.info(f"Reloading/replacing strategy: {strategy_name_attr} from {strategy_class.__module__}")
            else:
                logger.debug(f"Strategy {strategy_name_attr} from {strategy_class.__module__} already registered with the same class object.")
                return
        
        # Ajouter la stratégie au registre global
        decorator = StrategyRegistry.register(strategy_name_attr)
        decorator(strategy_class)
        
        # Maintenir également notre dictionnaire local pour la compatibilité
        self.loaded_strategies[strategy_name_attr] = strategy_class
        version_attr = getattr(strategy_class, 'version', 'N/A')
        logger.info(f"Registered strategy: {strategy_name_attr} (v{version_attr}) from {strategy_class.__module__}")
        
    def load_custom_strategies(self, custom_path: Optional[Path] = None):
        """
        Charge les stratégies personnalisées depuis un répertoire.
        DEPRECATED if _discover_strategies covers the main implementations path.
        Kept for potential separate custom folders.
        """
        target_path = custom_path if custom_path else self.implementations_path / "custom" # Example path
            
        if not target_path.exists() or not target_path.is_dir():
            logger.debug(f"Custom strategies directory {target_path} does not exist or not a dir.")
            return
        
        self._discover_strategies(target_path) # Reuse discovery logic
                
    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Récupère une classe de stratégie par son nom (attribut `name` de la classe).
        Utilise en priorité le registre global, puis le chargeur local comme fallback.
        """
        try:
            # Essayer d'abord de récupérer depuis le registre global
            return StrategyRegistry.get_strategy_class(name)
        except StrategyLoadError:
            # Si non trouvé dans le registre, tenter le dictionnaire local
            strategy_class = self.loaded_strategies.get(name)
            if strategy_class:
                # Si trouvé localement mais pas dans le registre, l'ajouter au registre
                warnings.warn(
                    f"Strategy '{name}' found in local loader but not in global registry. "
                    f"Adding it to registry for future lookups.",
                    DeprecationWarning
                )
                decorator = StrategyRegistry.register(name)
                decorator(strategy_class)
            return strategy_class
        
    def create_strategy(
        self,
        strategy_identifier: str, # This is the 'name' attribute of the strategy class
        params: Optional[Dict[str, Any]] = None,
        **kwargs # Pass other necessary args like pair_symbol, etc.
    ) -> BaseStrategy:
        try:
            # Utiliser directement la méthode du registre
            instance = StrategyRegistry.create(strategy_identifier, **kwargs)
            
            # Appliquer les paramètres si fournis
            if params:
                for key, value in params.items():
                    setattr(instance, key, value)
                
            # Mémoriser l'instance si nécessaire
            # instance_key = f"{strategy_identifier}_{id(instance)}"
            # self.strategy_instances[instance_key] = instance
            
            logger.debug(f"Created strategy instance: {strategy_identifier}")
            return instance
            
        except InvalidStrategyParamsError as e:
            logger.error(f"Invalid parameters for strategy {strategy_identifier}: {e}")
            raise
        except StrategyLoadError as e:
            # Si échec au niveau du registre, on peut tenter l'ancienne méthode
            logger.warning(f"Registry failed to create strategy: {e}. Trying legacy method...")
            
            strategy_class = self.get_strategy_class(strategy_identifier)
            if not strategy_class:
                available = list(self.loaded_strategies.keys())
                logger.error(f"Strategy '{strategy_identifier}' not found. Available strategies: {available}")
                raise StrategyLoadError(
                    f"Strategy '{strategy_identifier}' not found. Available: {available}"
                )
                
            try:
                instance = strategy_class(params=params, **kwargs)
                logger.debug(f"Created strategy instance (legacy method): {strategy_identifier}")
                return instance
            except Exception as e2:
                logger.error(f"Failed to create strategy instance (legacy) {strategy_identifier}: {e2}", exc_info=True)
                raise StrategyError(
                    f"Cannot instantiate strategy '{strategy_identifier}': {e2}",
                    strategy_name=strategy_identifier,
                    original_exception=e2
                )
        except Exception as e:
            logger.error(f"Failed to create strategy instance {strategy_identifier}: {e}", exc_info=True)
            raise StrategyError(
                f"Cannot instantiate strategy '{strategy_identifier}': {e}",
                strategy_name=strategy_identifier,
                original_exception=e
            )
            
    def list_strategies(self) -> List[Dict[str, Any]]:
        strategies_info = []
        for name, strategy_class in self.loaded_strategies.items():
            info = {
                'name': name, # This is strategy_class.name
                'class_name': strategy_class.__name__,
                'version': getattr(strategy_class, 'version', 'N/A'),
                'description': getattr(strategy_class, 'description', 'N/A'),
                'required_timeframes': getattr(strategy_class, 'required_timeframes', []),
                'min_required_periods': getattr(strategy_class, 'min_required_periods', 0),
                'default_params': getattr(strategy_class, 'default_params', {})
            }
            strategies_info.append(info)
        return strategies_info
        
    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        strategy_class = self.get_strategy_class(name)
        if not strategy_class: return None
        return {
            'name': name,
            'class_name': strategy_class.__name__,
            'version': getattr(strategy_class, 'version', 'N/A'),
            'description': getattr(strategy_class, 'description', 'N/A'),
            'module': strategy_class.__module__,
            'required_timeframes': getattr(strategy_class, 'required_timeframes', []),
            'min_required_periods': getattr(strategy_class, 'min_required_periods', 0),
            'default_params': getattr(strategy_class, 'default_params', {}),
            'docstring': inspect.getdoc(strategy_class)
        }
        
    def reload_strategy(self, name: str) -> bool:
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            logger.warning(f"Cannot reload unknown strategy: {name}")
            return False
        try:
            module = importlib.import_module(strategy_class.__module__)
            importlib.reload(module)
            reloaded_class = getattr(module, strategy_class.__name__)
            self._register_strategy(reloaded_class) # Re-register with potentially new class object
            logger.info(f"Successfully reloaded strategy: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reload strategy {name}: {e}", exc_info=True)
            return False
            
    def validate_strategy_params(
        self, name: str, params: Dict[str, Any]
    ) -> Tuple[bool, List[str]]: # Renamed from InvalidStrategyParamsError
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            return False, [f"Strategy '{name}' not found"]
        errors = []
        try:
            # Temporarily instantiate to trigger Pydantic validation if params is a model
            # or the strategy's own validate_params method
            strategy_class(params=params) # Assuming __init__ or validate_params handles it
            return True, []
        except InvalidStrategyParamsError as e: # Catch specific param error from strategy
            errors.append(str(e))
        except Exception as e: # Catch other instantiation errors (e.g., Pydantic if params is a model)
            errors.append(f"Validation error during instantiation: {e}")
        return False, errors
        
    def clear_instances(self):
        self.strategy_instances.clear()
        logger.debug("Cleared all strategy instances")

# Singleton global - CAUTION: This creates an instance when the module is imported.
# This can be problematic if settings or other dependencies are not yet configured.
# It's often better to instantiate StrategyLoader where needed, or use a factory/DI.
# For now, keeping it as per the original structure.
# strategy_loader = StrategyLoader() # Commented out to avoid premature initialization

# Utility functions - these would need strategy_loader to be instantiated.
# Consider passing an instance or making them methods of a class that holds the loader.
# def get_strategy(loader_instance: StrategyLoader, name: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> BaseStrategy:
#     return loader_instance.create_strategy(name, params, **kwargs)

# def list_available_strategies(loader_instance: StrategyLoader) -> List[str]:
#     return [s['name'] for s in loader_instance.list_strategies()]

# def reload_all_strategies(loader_instance: StrategyLoader):
#     for name in [s['name'] for s in loader_instance.list_strategies()]:
#         loader_instance.reload_strategy(name)

if __name__ == "__main__":
    # Setup basic logging for standalone test
    logger.remove()
    import sys
    logger.add(sys.stderr, level="DEBUG")

    # Instantiate loader for testing
    test_loader = StrategyLoader()
    
    # Manually trigger discovery if not done sufficiently in __init__
    # test_loader._discover_strategies(test_loader.implementations_path) # Already called in __init__

    print("\nAvailable strategies after initialization and discovery:")
    listed_strategies = test_loader.list_strategies()
    if listed_strategies:
        for info in listed_strategies:
            print(f"  - {info['name']} (Class: {info['class_name']}, v{info['version']}): {info['description']}")
    else:
        print("  No strategies found.")

    # Test creating an instance of a known strategy (if any were loaded)
    # Replace 'SMACrossStrategy' with an actual strategy_class.name that you expect to be loaded
    # For example, if BbandsVolumeRsiStrategy was loaded:
    target_strategy_name_for_test = "BbandsVolumeRsiStrategy" # Use the 'name' attribute of the class

    if test_loader.get_strategy_class(target_strategy_name_for_test):
        print(f"\nAttempting to create strategy: {target_strategy_name_for_test}")
        try:
            # Pass necessary kwargs if the strategy __init__ expects them (e.g. pair_symbol)
            strategy_instance = test_loader.create_strategy(
                target_strategy_name_for_test, 
                params={'bbands_period': 21}, # Example param override
                pair_symbol="BTCUSDT_Test" # Example kwarg for strategy constructor
            )
            print(f"Successfully created instance of {target_strategy_name_for_test}: {strategy_instance}")
            print(f"Instance parameters: {strategy_instance.params}")

            # Test reloading
            print(f"\nAttempting to reload: {target_strategy_name_for_test}")
            test_loader.reload_strategy(target_strategy_name_for_test)

        except StrategyLoadError as e:
            print(f"Error (StrategyLoadError) creating strategy {target_strategy_name_for_test}: {e}")
        except StrategyError as e:
            print(f"Error (StrategyError) creating strategy {target_strategy_name_for_test}: {e}")
        except Exception as e:
            print(f"Unexpected error creating strategy {target_strategy_name_for_test}: {e}")
    else:
        print(f"\nStrategy '{target_strategy_name_for_test}' not found, cannot test instance creation.")

    # Example for sma_cross if it's expected to be loaded
    # Note: sma_cross.py had a Pydantic-like error in the logs.
    # If that's fixed, this test might pass.
    sma_strategy_name = "SMACrossStrategy" # This is BaseStrategy.name
    if test_loader.get_strategy_class(sma_strategy_name):
        print(f"\nAttempting to create strategy: {sma_strategy_name}")
        try:
            sma_instance = test_loader.create_strategy(sma_strategy_name, params={'fast_period': 9}, pair_symbol="ETHUSDT_Test")
            print(f"Successfully created instance of {sma_strategy_name}: {sma_instance}")
        except Exception as e:
             print(f"Error creating {sma_strategy_name}: {e}")
    else:
        print(f"\nStrategy '{sma_strategy_name}' not found.")
