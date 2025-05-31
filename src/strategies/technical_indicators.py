# src/strategies/technical_indicators.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import njit, prange
import time
from typing import Callable, Dict, Any, List, Tuple, Optional, Union
from loguru import logger
import inspect # Pour inspecter les paramètres des fonctions pandas_ta

# Configuration initiale de Loguru (peut être centralisée ailleurs dans le projet)
# logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

# --- Exceptions Personnalisées ---
class IndicatorError(Exception):
    """Classe de base pour les erreurs liées aux indicateurs."""
    pass

class IndicatorCalculationError(IndicatorError):
    """Exception levée lors d'une erreur dans le calcul d'un indicateur."""
    pass

class IndicatorValidationError(IndicatorError):
    """Exception levée lors d'une erreur de validation d'un indicateur."""
    pass

class IndicatorNotFoundError(IndicatorError):
    """Exception levée lorsqu'un indicateur demandé n'est pas trouvé."""
    pass


# --- Indicateurs Personnalisés (Exemples) ---

@njit(nogil=True, cache=True)
def sma_numba_core(series: np.ndarray, length: int) -> np.ndarray:
    """
    Calcule une Moyenne Mobile Simple (SMA) optimisée avec Numba.
    Nécessite un tableau NumPy en entrée.
    Gère les NaNs au début de la série.
    """
    n = len(series)
    result = np.full(n, np.nan) # Initialiser avec NaN
    if length <= 0 or length > n:
        return result

    # Calculer la somme pour la première fenêtre valide
    current_sum = 0.0
    valid_points_in_first_window = 0
    for i in range(length):
        if not np.isnan(series[i]):
            current_sum += series[i]
            valid_points_in_first_window += 1
    
    if valid_points_in_first_window == length: # S'assurer que la première fenêtre est complète
        result[length - 1] = current_sum / length

    # Calculer pour les fenêtres suivantes
    for i in range(length, n):
        # Soustraire l'élément qui sort de la fenêtre et ajouter le nouvel élément
        # Gérer les NaN dans la fenêtre glissante
        term_out = series[i - length]
        term_in = series[i]
        
        if np.isnan(term_in): # Si la nouvelle valeur est NaN, la SMA devient NaN
            current_sum = np.nan # Ou une autre logique pour gérer les NaN internes
        elif np.isnan(term_out) and not np.isnan(current_sum): # Si l'ancienne était NaN mais la somme actuelle est valide
             # Recalculer la somme de la fenêtre actuelle si l'ancienne valeur était NaN
             # Cela peut être coûteux. Une alternative est de propager NaN.
             # Pour une SMA simple, si une valeur dans la fenêtre est NaN, la SMA est NaN.
             # Cependant, pandas_ta semble ignorer les NaN dans sa somme.
             # Pour une implémentation Numba robuste, il faut être explicite.
             # Ici, on propage NaN si une nouvelle valeur est NaN.
             # Si on veut ignorer les NaN comme pandas_ta.sma(nan_policy='omit'), la logique est plus complexe.
             # Pour cet exemple, on garde une logique plus simple.
             pass # La somme reste, mais la moyenne sera affectée si term_out était important.
                 # Une approche plus simple : si une valeur est NaN, la moyenne est NaN pour cette fenêtre.
                 # Pour l'instant, on simule un comportement où on essaie de continuer.
        
        if not np.isnan(current_sum):
            if not np.isnan(term_out):
                 current_sum -= term_out
            if not np.isnan(term_in):
                 current_sum += term_in
            else: # Si term_in est NaN, le résultat sera NaN
                current_sum = np.nan

        if not np.isnan(current_sum):
            result[i] = current_sum / length
        else:
            result[i] = np.nan
            # Si current_sum devient NaN, essayer de réinitialiser pour la prochaine fenêtre valide
            # Cela rend la logique complexe. Pour une SMA, si une valeur est NaN, le résultat est NaN.
            # La logique ci-dessus est simplifiée et pourrait ne pas correspondre exactement à pandas_ta
            # dans tous les cas de figure avec des NaN internes.
            # Réinitialisation simple pour la prochaine fenêtre si current_sum est devenu NaN:
            if i + length <= n:
                next_window_sum = 0.0
                valid_count = 0
                for k_idx in range(length):
                    val = series[i + 1 - length + k_idx]
                    if not np.isnan(val):
                        next_window_sum += val
                        valid_count +=1
                if valid_count == length:
                    current_sum = next_window_sum
                else:
                    current_sum = np.nan # Reste NaN si la fenêtre suivante n'est pas complète

    return result

def custom_sma_numba(df: pd.DataFrame, length: int, column: str = 'close', output_col_name: Optional[str] = None) -> pd.Series:
    """
    Wrapper pour l'indicateur SMA Numba, prenant un DataFrame Pandas.
    Args:
        df (pd.DataFrame): DataFrame contenant la colonne de prix.
        length (int): Période de la SMA.
        column (str): Nom de la colonne à utiliser pour le calcul (par défaut 'close').
        output_col_name (Optional[str]): Nom de la colonne de sortie. Si None, généré automatiquement.
    Returns:
        pd.Series: Série Pandas avec les valeurs de la SMA.
    """
    if column not in df.columns:
        raise IndicatorCalculationError(f"Colonne '{column}' non trouvée dans le DataFrame.")
    if length <= 0:
        raise IndicatorCalculationError("La période (length) doit être positive.")
    
    # pandas_ta.sma gère mieux les types et les NaN initiaux.
    # Numba est plus performant sur des arrays NumPy purs.
    # Convertir en NumPy array, Numba gère les float64 par défaut.
    series_np = df[column].to_numpy(dtype=np.float64) 
    
    sma_values = sma_numba_core(series_np, length)
    
    if output_col_name is None:
        output_col_name = f"CUSTOM_SMA_NUMBA_{length}"
        
    return pd.Series(sma_values, index=df.index, name=output_col_name)


def custom_rsi(df: pd.DataFrame, length: int, column: str = 'close', output_col_name: Optional[str] = None) -> pd.Series:
    """
    Exemple d'un indicateur RSI personnalisé simple (pour démonstration, pandas_ta.rsi est plus complet).
    Args:
        df (pd.DataFrame): DataFrame contenant la colonne de prix.
        length (int): Période du RSI.
        column (str): Nom de la colonne à utiliser (par défaut 'close').
        output_col_name (Optional[str]): Nom de la colonne de sortie.
    Returns:
        pd.Series: Série Pandas avec les valeurs du RSI.
    """
    if column not in df.columns:
        raise IndicatorCalculationError(f"Colonne '{column}' non trouvée pour Custom RSI.")
    if length <= 0:
        raise IndicatorCalculationError("La période RSI (length) doit être positive.")

    delta = df[column].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    
    # Pour les périodes suivantes, utiliser une EMA-like (Wilder's smoothing)
    # Note: pandas_ta.rsi utilise une méthode de lissage plus robuste (EMA ou RMA)
    # Ceci est une simplification pour l'exemple.
    for i in range(length, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (length - 1) + gain.iloc[i]) / length
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (length - 1) + loss.iloc[i]) / length

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Remplacer les infinis par NaN (peut arriver si avg_loss est 0)
    rsi.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if output_col_name is None:
        output_col_name = f"CUSTOM_RSI_{length}"
        
    return pd.Series(rsi, index=df.index, name=output_col_name)


# --- Gestionnaire d'Indicateurs ---
class IndicatorManager:
    """
    Gère le calcul, la validation et les tests de performance des indicateurs techniques.
    """
    def __init__(self):
        """Initialise le gestionnaire d'indicateurs."""
        self.custom_indicators: Dict[str, Dict[str, Any]] = {}
        self._register_default_custom_indicators()
        logger.info("IndicatorManager initialisé.")

    def _register_default_custom_indicators(self):
        """Enregistre les indicateurs personnalisés définis dans ce module."""
        self.add_custom_indicator(
            name="custom_sma_numba",
            func=custom_sma_numba,
            params_info={"length": "int", "column": "str (default 'close')"},
            description="Custom Simple Moving Average calculated with Numba."
        )
        self.add_custom_indicator(
            name="custom_rsi",
            func=custom_rsi,
            params_info={"length": "int", "column": "str (default 'close')"},
            description="Custom Relative Strength Index (simplified version)."
        )

    def add_custom_indicator(
        self,
        name: str,
        func: Callable,
        params_info: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Ajoute un indicateur personnalisé à la bibliothèque.
        STR-012: Créer une bibliothèque d'indicateurs custom

        Args:
            name (str): Nom unique pour l'indicateur personnalisé.
            func (Callable): Fonction qui calcule l'indicateur.
                             Elle doit prendre un DataFrame Pandas comme premier argument
                             et retourner une Series Pandas ou un DataFrame.
            params_info (Optional[Dict[str, str]]): Informations sur les paramètres de la fonction
                                                     (ex: {"length": "int", "offset": "int (optional)"}).
            description (Optional[str]): Description de l'indicateur.
        """
        if name in self.custom_indicators:
            logger.warning(f"L'indicateur personnalisé '{name}' existe déjà et va être remplacé.")
        
        self.custom_indicators[name] = {
            "function": func,
            "params_info": params_info or {},
            "description": description or ""
        }
        logger.info(f"Indicateur personnalisé '{name}' ajouté.")

    def _get_pandas_ta_function(self, indicator_name: str) -> Optional[Callable]:
        """Tente de récupérer une fonction d'indicateur de pandas_ta."""
        if hasattr(ta, indicator_name) and callable(getattr(ta, indicator_name)):
            return getattr(ta, indicator_name)
        # pandas_ta stocke aussi des stratégies/méthodes sur l'accesseur .ta
        # Par exemple, df.ta.rsi()
        # Pour une utilisation plus générique, on peut essayer d'appeler via l'accesseur
        # si la fonction directe n'est pas trouvée.
        # Cependant, la plupart des indicateurs sont des fonctions directes dans le module ta.
        return None

    def _get_pandas_ta_indicator_params(self, indicator_func: Callable) -> List[str]:
        """Inspecte une fonction pandas_ta pour obtenir ses paramètres."""
        try:
            sig = inspect.signature(indicator_func)
            return [p.name for p in sig.parameters.values() if p.name not in ['self', 'df', 'kwargs']]
        except ValueError: # Peut arriver pour certaines fonctions C built-in
            return []


    def calculate_indicator(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        output_col_prefix: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Calcule un indicateur spécifié (soit de pandas_ta, soit personnalisé).
        STR-011: Intégrer pandas-ta pour les indicateurs
        STR-012: Utiliser la bibliothèque d'indicateurs custom

        Args:
            df (pd.DataFrame): DataFrame d'entrée (doit contenir les colonnes OHLCV nécessaires).
            indicator_name (str): Nom de l'indicateur (ex: "rsi", "sma", "custom_sma_numba").
            output_col_prefix (Optional[str]): Préfixe pour les colonnes de sortie.
                                               Si None, utilise le nom de l'indicateur ou les noms par défaut.
            **kwargs: Paramètres spécifiques à l'indicateur (ex: length=14 pour RSI).

        Returns:
            pd.DataFrame: DataFrame original avec la/les colonne(s) de l'indicateur ajouté(es).

        Raises:
            IndicatorNotFoundError: Si l'indicateur n'est ni dans pandas_ta ni personnalisé.
            IndicatorCalculationError: Si une erreur survient pendant le calcul.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("L'entrée 'df' doit être un DataFrame Pandas.")
        if df.empty:
            logger.warning("Le DataFrame d'entrée est vide. Retour d'un DataFrame vide.")
            return df.copy()

        df_out = df.copy()
        
        # Essayer les indicateurs personnalisés d'abord
        if indicator_name in self.custom_indicators:
            try:
                custom_indicator_info = self.custom_indicators[indicator_name]
                indicator_func = custom_indicator_info["function"]
                
                # Préparer les arguments pour la fonction custom
                func_params = inspect.signature(indicator_func).parameters
                custom_kwargs = {}
                if 'output_col_name' in func_params and output_col_prefix: # Gérer le nom de la colonne de sortie
                    # Si la fonction custom prend output_col_name, on le construit
                    # Sinon, on suppose que la fonction nomme sa sortie elle-même ou retourne un DF avec des noms standards.
                    # Ceci est une heuristique.
                    base_name = output_col_prefix if output_col_prefix else indicator_name.upper()
                    param_str = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
                    custom_kwargs['output_col_name'] = f"{base_name}_{param_str}" if param_str else base_name


                # Passer uniquement les kwargs pertinents à la fonction custom
                valid_kwargs_for_custom = {k: v for k, v in kwargs.items() if k in func_params}
                custom_kwargs.update(valid_kwargs_for_custom)


                result = indicator_func(df_out, **custom_kwargs) # Passer df_out pour que la fonction puisse y accéder

                if isinstance(result, pd.Series):
                    col_name = result.name if result.name else custom_kwargs.get('output_col_name', indicator_name.upper())
                    df_out[col_name] = result
                elif isinstance(result, pd.DataFrame):
                    # Si le préfixe est donné, l'appliquer aux nouvelles colonnes
                    new_cols = {col: f"{output_col_prefix}_{col}" if output_col_prefix else col for col in result.columns}
                    df_out = pd.concat([df_out, result.rename(columns=new_cols)], axis=1)
                else:
                    raise IndicatorCalculationError(
                        f"L'indicateur personnalisé '{indicator_name}' a retourné un type inattendu: {type(result)}"
                    )
                logger.debug(f"Indicateur personnalisé '{indicator_name}' calculé.")

            except Exception as e:
                logger.error(f"Erreur lors du calcul de l'indicateur personnalisé '{indicator_name}': {e}")
                raise IndicatorCalculationError(f"Custom indicator '{indicator_name}' failed: {e}") from e
        
        # Essayer pandas_ta ensuite
        else:
            pandas_ta_func = self._get_pandas_ta_function(indicator_name.lower())
            if pandas_ta_func:
                try:
                    # pandas_ta fonctions attendent souvent 'close', 'high', 'low', 'volume' directement.
                    # Elles peuvent aussi prendre un DataFrame et utiliser l'accesseur .ta
                    # df.ta.rsi(length=14)
                    # Pour une approche unifiée, on peut utiliser l'accesseur .ta si disponible,
                    # ou appeler la fonction directement.
                    # La plupart des indicateurs pandas_ta ajoutent les colonnes au DF ou retournent une Series/DF.
                    
                    # Simplification: on suppose que l'utilisateur fournit les colonnes OHLCV standard.
                    # pandas_ta est assez flexible pour les trouver.
                    # On passe le DataFrame entier et les kwargs.
                    
                    # Filtrer les kwargs pour ne passer que ceux attendus par la fonction pandas_ta
                    # et ceux reconnus par l'accesseur .ta (comme 'close', 'high', 'low', 'volume', 'open')
                    known_ohlcv_params = {'open', 'high', 'low', 'close', 'volume'}
                    func_specific_params = self._get_pandas_ta_indicator_params(pandas_ta_func)
                    
                    valid_kwargs = {}
                    for k, v in kwargs.items():
                        if k in func_specific_params or k in known_ohlcv_params:
                            valid_kwargs[k] = v
                    
                    # Certaines fonctions pandas_ta modifient le DataFrame en place via l'accesseur .ta
                    # d'autres retournent une Series/DataFrame.
                    # Ex: df.ta.rsi(length=14, append=True) vs rsi_series = ta.rsi(df['close'], length=14)
                    # On va privilégier l'appel direct de la fonction si possible, sinon l'accesseur.

                    indicator_output = None
                    if hasattr(df_out.ta, indicator_name.lower()): # Essayer avec l'accesseur .ta
                        # Cloner les kwargs pour ajouter 'append=True' sans affecter l'original
                        accessor_kwargs = valid_kwargs.copy()
                        accessor_kwargs['append'] = True # Pour que les colonnes soient ajoutées à df_out
                        getattr(df_out.ta, indicator_name.lower())(**accessor_kwargs)
                        # Les colonnes ont été ajoutées à df_out par pandas_ta
                        indicator_output_generated = True 
                    elif callable(pandas_ta_func): # Essayer l'appel direct
                        # Déterminer la colonne principale (souvent 'close')
                        main_col_name = kwargs.get('close', 'close') if 'close' in kwargs else 'close'
                        if main_col_name not in df_out.columns:
                             main_col_name = 'close' # Fallback
                        
                        if main_col_name not in df_out.columns and indicator_name.lower() not in ['entropy', 'hurst']: # Certains n'ont pas besoin de close
                            raise IndicatorCalculationError(f"Colonne '{main_col_name}' requise pour {indicator_name} non trouvée.")

                        # Préparer les arguments pour la fonction pandas_ta
                        # Certaines fonctions attendent des Series (ex: ta.rsi(df['close'], ...))
                        # D'autres peuvent prendre le DataFrame et des noms de colonnes (ex: ta.atr(df, high='high', ...))
                        # On essaie de deviner. Si 'close' est dans les kwargs, on suppose qu'il faut passer la Series.
                        # Sinon, on passe le DataFrame.
                        
                        # Construction des arguments pour pandas_ta
                        call_args = {}
                        sig_params = inspect.signature(pandas_ta_func).parameters

                        # Séries OHLCV
                        for ohlcv_param in ['open', 'high', 'low', 'close', 'volume']:
                            if ohlcv_param in sig_params and ohlcv_param in df_out.columns:
                                call_args[ohlcv_param] = df_out[ohlcv_param]
                        
                        # Autres paramètres (length, std, etc.)
                        for p_name, p_val in valid_kwargs.items():
                            if p_name not in call_args and p_name in sig_params: # Ne pas écraser les séries OHLCV
                                call_args[p_name] = p_val
                        
                        # Cas où la fonction attend une seule série (ex: 'close' pour ta.rsi)
                        # et que 'close' n'est pas déjà un paramètre nommé dans sig_params
                        # (ce qui est rare, car ils sont généralement nommés).
                        # On va supposer que si 'close' est le seul paramètre positionnel non-OHLCV,
                        # c'est la série principale.
                        # Cette heuristique est fragile. pandas_ta est conçu pour être flexible.
                        # Le plus simple est souvent de laisser l'utilisateur spécifier la colonne via kwargs.
                        
                        # Appel de la fonction
                        indicator_output = pandas_ta_func(**call_args)
                        indicator_output_generated = False # Marquer que la sortie doit être fusionnée

                    else:
                         raise IndicatorNotFoundError(f"Impossible d'appeler l'indicateur pandas_ta '{indicator_name}'.")


                    if not indicator_output_generated and indicator_output is not None:
                        if isinstance(indicator_output, pd.Series):
                            col_name = indicator_output.name if indicator_output.name else indicator_name.upper()
                            if output_col_prefix:
                                col_name = f"{output_col_prefix}_{col_name}"
                            df_out[col_name] = indicator_output
                        elif isinstance(indicator_output, pd.DataFrame):
                            # Si le préfixe est donné, l'appliquer aux nouvelles colonnes
                            new_cols_pta = {col: f"{output_col_prefix}_{col}" if output_col_prefix else col for col in indicator_output.columns}
                            df_out = pd.concat([df_out, indicator_output.rename(columns=new_cols_pta)], axis=1)
                        else:
                             raise IndicatorCalculationError(f"pandas_ta indicator '{indicator_name}' returned unexpected type: {type(indicator_output)}")
                    
                    # Si output_col_prefix est fourni et que pandas_ta a ajouté des colonnes avec des noms par défaut,
                    # essayer de les renommer. C'est complexe car les noms de sortie de pandas_ta varient.
                    # Pour l'instant, on se fie à `append=True` ou à la fusion manuelle.
                    # Si output_col_prefix est donné et qu'on a utilisé l'accesseur .ta avec append=True,
                    # il faudrait identifier les colonnes ajoutées et les renommer.
                    # Exemple: si df.ta.rsi(length=14, append=True) ajoute "RSI_14", et prefix="MY",
                    # on voudrait "MY_RSI_14". C'est difficile à généraliser.
                    # La solution actuelle est que si l'accesseur est utilisé, le préfixe n'est pas appliqué automatiquement.
                    # Si la fonction est appelée directement et retourne Series/DF, le préfixe est appliqué.

                    logger.debug(f"Indicateur pandas_ta '{indicator_name}' calculé.")

                except Exception as e:
                    logger.error(f"Erreur lors du calcul de l'indicateur pandas_ta '{indicator_name}': {e}")
                    raise IndicatorCalculationError(f"pandas_ta indicator '{indicator_name}' failed: {e}") from e
            else:
                raise IndicatorNotFoundError(
                    f"Indicateur '{indicator_name}' non trouvé dans pandas_ta ou indicateurs personnalisés."
                )
        
        # Assurer que l'index est conservé
        if not df_out.index.equals(df.index):
            logger.warning("L'index du DataFrame a été modifié pendant le calcul de l'indicateur. Tentative de réalignement.")
            df_out = df_out.reindex(df.index)
            
        return df_out

    def calculate_multiple_indicators(
        self,
        df: pd.DataFrame,
        indicator_configs: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calcule plusieurs indicateurs à partir d'une liste de configurations.

        Args:
            df (pd.DataFrame): DataFrame d'entrée.
            indicator_configs (List[Dict[str, Any]]): Liste de dictionnaires, chaque dict
                configurant un indicateur. Format:
                {"name": "sma", "length": 20, "column": "close", "output_col_prefix": "SMA20"}
                {"name": "custom_rsi", "length": 14, "output_col_prefix": "MyRSI"}

        Returns:
            pd.DataFrame: DataFrame avec tous les indicateurs calculés.
        """
        df_with_all_indicators = df.copy()
        for config in indicator_configs:
            if "name" not in config:
                raise ValueError("Chaque configuration d'indicateur doit avoir une clé 'name'.")
            
            name = config.pop("name")
            output_prefix = config.pop("output_col_prefix", None) # Utiliser le préfixe s'il est fourni
            
            try:
                df_with_all_indicators = self.calculate_indicator(
                    df_with_all_indicators,
                    indicator_name=name,
                    output_col_prefix=output_prefix,
                    **config # Le reste des clés sont les paramètres de l'indicateur
                )
            except IndicatorError as e: # Attraper nos exceptions personnalisées
                logger.error(f"Échec du calcul de l'indicateur '{name}' avec config {config}: {e}")
                # Optionnel: continuer avec les autres indicateurs ou relancer l'erreur
                # Pour l'instant, on logue et on continue.
            except Exception as e:
                logger.error(f"Erreur inattendue pour indicateur '{name}' avec config {config}: {e}")


        return df_with_all_indicators

    def validate_indicator_output(
        self,
        df_with_indicator: pd.DataFrame,
        indicator_col_name: str,
        expected_lookback: Optional[int] = None,
        value_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    ) -> List[str]:
        """
        Valide la sortie d'un calcul d'indicateur.
        STR-014: Implémenter la validation des indicateurs

        Args:
            df_with_indicator (pd.DataFrame): DataFrame contenant la colonne de l'indicateur.
            indicator_col_name (str): Nom de la colonne de l'indicateur à valider.
            expected_lookback (Optional[int]): Nombre attendu de NaNs initiaux dus au lookback.
            value_range (Optional[Tuple[Optional[float], Optional[float]]]):
                Plage de valeurs attendue pour l'indicateur (min_val, max_val).
                Utiliser None pour une borne non définie (ex: (0, None) pour >= 0).

        Returns:
            List[str]: Liste des messages d'erreur de validation. Vide si valide.
        """
        errors = []

        if indicator_col_name not in df_with_indicator.columns:
            errors.append(f"Colonne d'indicateur '{indicator_col_name}' non trouvée dans le DataFrame.")
            return errors # Inutile de continuer si la colonne n'existe pas

        indicator_series = df_with_indicator[indicator_col_name]

        # 1. Vérifier les NaNs initiaux (lookback)
        if expected_lookback is not None and expected_lookback > 0:
            if len(indicator_series) >= expected_lookback:
                initial_values = indicator_series.iloc[:expected_lookback-1] # Les length-1 premières valeurs devraient être NaN pour une SMA(length)
                if not initial_values.isnull().all():
                    num_initial_nans = initial_values.isnull().sum()
                    # Permettre une petite flexibilité, car le calcul exact des NaNs initiaux peut varier
                    # (ex: pandas_ta.sma(10) a 9 NaNs initiaux)
                    # On s'attend à ce que les `expected_lookback - 1` premières valeurs soient NaN.
                    if num_initial_nans < (expected_lookback -1) :
                         errors.append(
                            f"'{indicator_col_name}': Nombre de NaNs initiaux inattendu. "
                            f"Attendu: >= {expected_lookback-1}, Trouvé: {num_initial_nans} "
                            f"sur les {expected_lookback-1} premières valeurs."
                        )
                # Vérifier que la première valeur non-NaN est à l'index attendu
                first_valid_index = indicator_series.first_valid_index()
                if first_valid_index is not None:
                    first_valid_loc = df_with_indicator.index.get_loc(first_valid_index)
                    if first_valid_loc < expected_lookback -1 :
                         errors.append(
                            f"'{indicator_col_name}': Première valeur non-NaN trouvée à l'index {first_valid_loc}, "
                            f"attendu à l'index >= {expected_lookback-1}."
                        )
            else:
                 errors.append(f"'{indicator_col_name}': Pas assez de données ({len(indicator_series)}) pour vérifier le lookback de {expected_lookback}.")


        # 2. Vérifier les NaNs après la période de lookback (ne devrait pas y en avoir beaucoup)
        series_after_lookback = indicator_series.iloc[expected_lookback:] if expected_lookback else indicator_series
        if series_after_lookback.isnull().any():
            # Permettre un petit nombre de NaN dus à des données d'entrée manquantes, mais pas trop.
            if series_after_lookback.isnull().sum() > len(series_after_lookback) * 0.1: # Si >10% de NaN
                errors.append(f"'{indicator_col_name}': Trop de NaNs ({series_after_lookback.isnull().sum()}) après la période de lookback.")

        # 3. Vérifier la plage de valeurs
        if value_range:
            min_val, max_val = value_range
            valid_series = indicator_series.dropna()
            if len(valid_series) > 0: # Seulement si des valeurs non-NaN existent
                if min_val is not None and (valid_series < min_val).any():
                    errors.append(f"'{indicator_col_name}': Des valeurs sont inférieures à la limite min ({min_val}). Min trouvé: {valid_series.min()}")
                if max_val is not None and (valid_series > max_val).any():
                    errors.append(f"'{indicator_col_name}': Des valeurs sont supérieures à la limite max ({max_val}). Max trouvé: {valid_series.max()}")
        
        # 4. Vérifier les infinis
        if np.isinf(indicator_series).any():
            errors.append(f"'{indicator_col_name}': Contient des valeurs infinies.")

        if not errors:
            logger.debug(f"Validation de '{indicator_col_name}' réussie.")
        else:
            logger.warning(f"Validation de '{indicator_col_name}' échouée: {errors}")
            
        return errors

    def measure_performance(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        repetitions: int = 10,
        **kwargs: Any
    ) -> Dict[str, float]:
        """
        Mesure le temps d'exécution pour le calcul d'un indicateur.
        STR-015: Créer les tests de performance

        Args:
            df (pd.DataFrame): DataFrame d'entrée.
            indicator_name (str): Nom de l'indicateur à tester.
            repetitions (int): Nombre de répétitions pour la mesure.
            **kwargs: Paramètres de l'indicateur.

        Returns:
            Dict[str, float]: Dictionnaire contenant 'total_time_ms' et 'avg_time_ms'.
        """
        if repetitions <= 0:
            raise ValueError("Le nombre de répétitions doit être positif.")

        timings = []
        for _ in range(repetitions):
            start_time = time.perf_counter()
            try:
                _ = self.calculate_indicator(df.copy(), indicator_name, **kwargs) # Utiliser df.copy() pour éviter modif en place
            except IndicatorError as e:
                logger.error(f"Erreur de calcul pendant le test de performance pour '{indicator_name}': {e}")
                return {"error": str(e)}
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000) # Convertir en millisecondes

        total_time_ms = sum(timings)
        avg_time_ms = total_time_ms / repetitions
        
        logger.info(
            f"Performance pour '{indicator_name}' ({len(df)} lignes, {repetitions} reps): "
            f"Total: {total_time_ms:.2f} ms, Moyenne: {avg_time_ms:.2f} ms"
        )
        return {"total_time_ms": total_time_ms, "avg_time_ms": avg_time_ms, "repetitions": repetitions, "data_length": len(df)}

    def get_available_indicators(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retourne une liste des indicateurs disponibles (pandas_ta et custom).
        """
        available = {"pandas_ta": [], "custom": []}

        # Indicateurs pandas_ta (ceci est une liste non exhaustive, pandas_ta en a beaucoup)
        # On pourrait lister dynamiquement, mais c'est complexe à cause de l'accesseur .ta
        # Pour l'instant, on liste quelques exemples courants.
        # Une meilleure approche serait d'utiliser `pandas_ta.Category` ou d'inspecter le module `ta`.
        
        # Tentative de lister les fonctions directes de pandas_ta
        pta_indicators_info = []
        if hasattr(ta, "__all__"): # si __all__ est défini, il liste les exports publics
            pta_func_names = ta.__all__
        else: # sinon, essayer d'inspecter le module
            pta_func_names = [name for name, func in inspect.getmembers(ta, inspect.isfunction) if not name.startswith("_")]

        for pta_name in pta_func_names:
            try:
                func = getattr(ta, pta_name)
                if callable(func):
                    sig = inspect.signature(func)
                    params_list = [
                        f"{p.name}: {p.annotation}" if p.annotation != inspect.Parameter.empty else p.name
                        for p in sig.parameters.values()
                        if p.name not in ['self', 'kwargs', 'append', 'df', 'open_', 'high', 'low', 'close', 'volume'] # Exclure les params génériques
                    ]
                    # Garder seulement les paramètres pertinents pour la configuration de l'indicateur
                    config_params = [p.name for p in sig.parameters.values() if p.default != inspect.Parameter.empty and p.name not in ['self', 'kwargs', 'append', 'df', 'open_', 'high', 'low', 'close', 'volume']]

                    pta_indicators_info.append({
                        "name": pta_name,
                        "params_info": ", ".join(params_list) if params_list else "None",
                        "configurable_params": config_params, # Paramètres avec des valeurs par défaut
                        "description": inspect.getdoc(func) or "N/A"
                    })
            except Exception:
                pass # Ignorer les erreurs d'introspection pour certains objets
        available["pandas_ta"] = sorted(pta_indicators_info, key=lambda x: x['name'])


        # Indicateurs personnalisés
        for name, info in self.custom_indicators.items():
            available["custom"].append({
                "name": name,
                "params_info": info.get("params_info", {}),
                "description": info.get("description", "")
            })
        available["custom"] = sorted(available["custom"], key=lambda x: x['name'])
        
        return available

# --- Exemple d'Utilisation ---
if __name__ == "__main__":
    # Créer un DataFrame de test
    data = {
        'open': np.random.rand(100) * 100 + 50,
        'high': np.random.rand(100) * 100 + 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100 + 75,
        'volume': np.random.rand(100) * 1000 + 100
    }
    start_date = pd.to_datetime("2023-01-01")
    index = pd.date_range(start_date, periods=100, freq='1D')
    df_test = pd.DataFrame(data, index=index)

    # S'assurer que high > low, open, close et low < high, open, close
    df_test['high'] = df_test[['open', 'high', 'low', 'close']].max(axis=1) + np.random.rand(100) * 10
    df_test['low'] = df_test[['open', 'high', 'low', 'close']].min(axis=1) - np.random.rand(100) * 10
    df_test.loc[df_test['low'] < 0, 'low'] = 0.01 # Éviter les prix négatifs

    logger.info("--- Initialisation de IndicatorManager ---")
    manager = IndicatorManager()

    logger.info("\n--- Indicateurs Disponibles ---")
    available_inds = manager.get_available_indicators()
    logger.info(f"Pandas TA: {len(available_inds['pandas_ta'])} indicateurs (liste partielle). Exemple: {available_inds['pandas_ta'][0]['name'] if available_inds['pandas_ta'] else 'N/A'}")
    logger.info(f"Custom: {[ind['name'] for ind in available_inds['custom']]}")

    logger.info("\n--- Calcul d'un indicateur pandas_ta (RSI) ---")
    try:
        df_with_rsi = manager.calculate_indicator(df_test.copy(), "rsi", length=14, output_col_prefix="PTA")
        logger.info(f"RSI calculé. Colonnes: {df_with_rsi.columns}")
        logger.info(df_with_rsi[['close', 'PTA_RSI_14']].tail())
        validation_errors_rsi = manager.validate_indicator_output(df_with_rsi, "PTA_RSI_14", expected_lookback=14, value_range=(0, 100))
        if validation_errors_rsi:
            logger.warning(f"Erreurs de validation RSI: {validation_errors_rsi}")
        else:
            logger.success("Validation RSI réussie.")

    except IndicatorError as e:
        logger.error(f"Erreur RSI: {e}")

    logger.info("\n--- Calcul d'un indicateur custom (custom_sma_numba) ---")
    try:
        df_with_custom_sma = manager.calculate_indicator(df_test.copy(), "custom_sma_numba", length=10, column='close')
        logger.info(f"Custom SMA Numba calculé. Colonnes: {df_with_custom_sma.columns}")
        logger.info(df_with_custom_sma[['close', 'CUSTOM_SMA_NUMBA_10']].tail())
        validation_errors_sma = manager.validate_indicator_output(df_with_custom_sma, "CUSTOM_SMA_NUMBA_10", expected_lookback=10)
        if validation_errors_sma:
            logger.warning(f"Erreurs de validation Custom SMA: {validation_errors_sma}")
        else:
            logger.success("Validation Custom SMA réussie.")
            
    except IndicatorError as e:
        logger.error(f"Erreur Custom SMA: {e}")

    logger.info("\n--- Calcul de plusieurs indicateurs ---")
    indicator_set = [
        {"name": "sma", "length": 20, "output_col_prefix": "SMA_SHORT"},
        {"name": "ema", "length": 50, "column": "close", "output_col_prefix": "EMA_LONG"},
        {"name": "bbands", "length": 20, "std": 2, "output_col_prefix": "BB"},
        {"name": "custom_rsi", "length": 7, "output_col_prefix": "MyFastRSI"}
    ]
    df_multi = manager.calculate_multiple_indicators(df_test.copy(), indicator_set)
    logger.info(f"Indicateurs multiples calculés. Nouvelles colonnes: {[col for col in df_multi.columns if col not in df_test.columns]}")
    logger.info(df_multi.tail())


    logger.info("\n--- Test de Performance (pandas_ta SMA) ---")
    perf_sma_pta = manager.measure_performance(df_test, "sma", length=10, repetitions=50)
    logger.info(f"Résultats perf pandas_ta SMA: {perf_sma_pta}")

    logger.info("\n--- Test de Performance (custom_sma_numba) ---")
    # Numba a un coût de compilation la première fois
    logger.info("Premier appel (avec compilation Numba)...")
    manager.calculate_indicator(df_test.copy(), "custom_sma_numba", length=10)
    logger.info("Appels suivants (compilé)...")
    perf_sma_custom = manager.measure_performance(df_test, "custom_sma_numba", length=10, repetitions=50)
    logger.info(f"Résultats perf custom_sma_numba: {perf_sma_custom}")
    
    logger.info("\n--- Test de Performance (pandas_ta RSI) ---")
    perf_rsi_pta = manager.measure_performance(df_test, "rsi", length=14, repetitions=50)
    logger.info(f"Résultats perf pandas_ta RSI: {perf_rsi_pta}")

    logger.info("\n--- Test de Performance (custom_rsi) ---")
    perf_rsi_custom = manager.measure_performance(df_test, "custom_rsi", length=14, repetitions=50)
    logger.info(f"Résultats perf custom_rsi: {perf_rsi_custom}")

    logger.info("\n--- Test avec indicateur non existant ---")
    try:
        manager.calculate_indicator(df_test, "non_existant_indicator")
    except IndicatorNotFoundError as e:
        logger.error(f"Erreur attendue: {e}")

