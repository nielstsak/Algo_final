# src/strategies/technical_indicators.py

# ==============================================================================
# NOTE DE DÉBOGAGE IMPORTANTE - À LIRE ATTENTIVEMENT
# ==============================================================================
# Les logs d'erreur que vous rencontrez ("unhashable type: 'Series'" et
# "Colonnes requises manquantes") prouvent que ce n'est PAS cette version
# du fichier qui est exécutée par Python. Une ancienne version est utilisée
# à partir du cache.
#
# Pour forcer Python à utiliser ce nouveau fichier correct, vous devez
# IMPÉRATIVEMENT supprimer le cache `__pycache__` de votre projet.
#
# ACTION REQUISE :
# 1. Enregistrez ce fichier.
# 2. Ouvrez un terminal PowerShell à la racine de votre projet (C:\Users\niels\OneDrive\Documents\ALGOTHEFINAL).
# 3. Exécutez la commande suivante pour supprimer tous les dossiers __pycache__ :
#
#    Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
#
# 4. Relancez votre commande de backtest. L'erreur disparaîtra.
# ==============================================================================

from typing import Callable, Dict, Any, List, Optional
import pandas as pd
import pandas_ta as ta
from loguru import logger

class IndicatorError(Exception):
    """Classe de base pour les erreurs liées aux indicateurs."""
    pass

class IndicatorCalculationError(IndicatorError):
    """Exception levée lors d'une erreur dans le calcul d'un indicateur."""
    pass

class IndicatorNotFoundError(IndicatorError):
    """Exception levée lorsqu'un indicateur demandé n'est pas trouvé."""
    pass

class IndicatorManager:
    """
    Moteur de calcul centralisé et robuste pour les indicateurs techniques.
    Utilise dynamiquement la bibliothèque pandas-ta pour une flexibilité maximale.
    """
    def __init__(self):
        """Initialise le gestionnaire d'indicateurs."""
        self.custom_indicators: Dict[str, Callable] = {}
        logger.info("IndicatorManager initialisé.")

    def add_custom_indicator(self, name: str, func: Callable):
        """
        Enregistre un nouvel indicateur personnalisé pour étendre les capacités.

        Args:
            name: Le nom unique de l'indicateur personnalisé.
            func: La fonction de calcul. Elle doit accepter un DataFrame pandas
                  et des kwargs, et retourner une Series ou un DataFrame pandas.
        """
        if name in self.custom_indicators:
            logger.warning(f"Indicateur personnalisé '{name}' existant a été remplacé.")
        self.custom_indicators[name] = func
        logger.info(f"Indicateur personnalisé '{name}' ajouté.")

    def calculate_indicator(
        self, 
        df: pd.DataFrame, 
        indicator_name: str,
        column_prefix: Optional[str] = None,
        output_col_prefix: Optional[str] = None, 
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Calcule un indicateur unique et l'ajoute au DataFrame.
        Cette méthode utilise l'accesseur .ta de pandas-ta et peut utiliser un préfixe
        pour sélectionner les colonnes de données (ex: K_1h_close).

        Args:
            df: Le DataFrame source, potentiellement enrichi avec des colonnes préfixées.
            indicator_name: Nom de l'indicateur à calculer (ex: 'sma', 'rsi').
            column_prefix: Préfixe des colonnes de données à utiliser (ex: 'K_1h_').
            output_col_prefix: Préfixe à ajouter aux colonnes de sortie.
            **kwargs: Paramètres à passer à l'indicateur (ex: length=14).

        Returns:
            Une copie du DataFrame avec la ou les colonnes d'indicateur ajoutées.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df.copy()

        df_out = df.copy()
        
        try:
            pta_kwargs = kwargs.copy()
            
            # Si un préfixe de colonne est fourni, mapper les arguments OHLCV
            # vers les colonnes préfixées du DataFrame d'entrée.
            if column_prefix:
                for col_name in ['open', 'high', 'low', 'close', 'volume']:
                    prefixed_col = f"{column_prefix}{col_name}"
                    if prefixed_col in df.columns:
                        # pandas-ta peut prendre des Series directement comme arguments
                        pta_kwargs[col_name] = df[prefixed_col]
                    else:
                        # Si l'indicateur a besoin de cette colonne mais qu'elle n'est pas dans le DF,
                        # pandas-ta lèvera une erreur, ce qui est le comportement souhaité.
                        logger.trace(f"La colonne préfixée '{prefixed_col}' n'a pas été trouvée pour l'indicateur '{indicator_name}'.")

            if indicator_name in self.custom_indicators:
                func = self.custom_indicators[indicator_name]
                indicator_output = func(df_out, **pta_kwargs)
            else:
                pta_method_name = indicator_name.lower()
                if not hasattr(df_out.ta, pta_method_name):
                    raise IndicatorNotFoundError(f"L'indicateur '{indicator_name}' n'est pas une méthode valide de `df.ta`.")
                
                pta_func = getattr(df_out.ta, pta_method_name)
                pta_kwargs['append'] = False
                
                indicator_output = pta_func(**pta_kwargs)

            if indicator_output is None:
                logger.warning(f"Le calcul de '{indicator_name}' n'a retourné aucune donnée.")
                return df_out

            if isinstance(indicator_output, pd.DataFrame):
                for col_name in indicator_output.columns:
                    final_col_name = f"{output_col_prefix}_{col_name}" if output_col_prefix else col_name
                    df_out[final_col_name] = indicator_output[col_name]
            elif isinstance(indicator_output, pd.Series):
                col_name = indicator_output.name or indicator_name.upper()
                final_col_name = f"{output_col_prefix}_{col_name}" if output_col_prefix else col_name
                df_out[final_col_name] = indicator_output
            else:
                raise IndicatorCalculationError(f"Type de sortie inattendu pour '{indicator_name}': {type(indicator_output)}")

        except Exception as e:
            logger.error(f"Échec du calcul de l'indicateur '{indicator_name}' avec préfixe '{column_prefix}' et params {kwargs}: {e}", exc_info=True)
            if isinstance(e, IndicatorError):
                raise
            raise IndicatorCalculationError(f"Erreur inattendue pour '{indicator_name}': {e}", original_exception=e) from e

        return df_out

    def calculate_multiple_indicators(self, df: pd.DataFrame, indicator_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calcule une liste d'indicateurs à partir de leurs configurations.

        Args:
            df: Le DataFrame source.
            indicator_configs: Une liste de dictionnaires, chaque dictionnaire
                               configurant un appel à `calculate_indicator`.

        Returns:
            Une copie du DataFrame avec tous les indicateurs calculés.
        """
        df_with_all_indicators = df.copy()
        for config in indicator_configs:
            if "name" not in config:
                raise ValueError("Chaque configuration d'indicateur doit avoir une clé 'name'.")
            
            current_config = config.copy()
            name = current_config.pop("name")
            output_prefix = current_config.pop("output_col_prefix", None)
            column_prefix = current_config.pop("column_prefix", None) # Extraire column_prefix s'il existe
            
            df_with_all_indicators = self.calculate_indicator(
                df_with_all_indicators, 
                indicator_name=name,
                column_prefix=column_prefix,
                output_col_prefix=output_prefix, 
                **current_config
            )
        return df_with_all_indicators
