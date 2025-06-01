# src/strategies/technical_indicators.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import njit
import time
from typing import Callable, Dict, Any, List, Tuple, Optional, Union
from loguru import logger
import inspect

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

# --- Indicateurs Personnalisés ---
@njit(nogil=True, cache=True)
def sma_numba_core(series: np.ndarray, length: int) -> np.ndarray:
    n = len(series)
    result = np.full(n, np.nan)
    if length <= 0: return result
    if length == 1: return series.copy()
    # Initialiser la somme pour la première fenêtre valide
    current_sum = 0.0
    valid_values_in_window = 0
    for i in range(length):
        if not np.isnan(series[i]):
            current_sum += series[i]
            valid_values_in_window += 1
    
    # Calculer la SMA pour la première fenêtre si elle est valide
    if valid_values_in_window == length:
        result[length - 1] = current_sum / length
    else:
        result[length - 1] = np.nan

    # Calculer la SMA pour les fenêtres suivantes
    for i in range(length, n):
        # Soustraire la valeur qui sort de la fenêtre
        if not np.isnan(series[i - length]):
            current_sum -= series[i - length]
            valid_values_in_window -=1
        
        # Ajouter la nouvelle valeur qui entre dans la fenêtre
        if not np.isnan(series[i]):
            current_sum += series[i]
            valid_values_in_window +=1
        
        if valid_values_in_window == length:
            result[i] = current_sum / length
        else:
            # Si la fenêtre contient des NaNs, la moyenne est NaN
            # ou si le nombre de valeurs valides n'est pas égal à length
            result[i] = np.nan
            # Réinitialiser pour la prochaine fenêtre potentiellement valide
            # Recalculer la somme pour la fenêtre actuelle si on veut gérer les NaNs internes
            # Pour une SMA stricte, si un NaN est dans la fenêtre, le résultat est NaN.
            # La logique ci-dessus suppose que si on perd une valeur valide et qu'on en gagne une valide,
            # on continue. Si on veut que tout NaN dans la fenêtre rende le résultat NaN :
            current_window_for_nan_check = series[i - length + 1 : i + 1]
            if np.any(np.isnan(current_window_for_nan_check)):
                 result[i] = np.nan
            # else: # Si on veut calculer la moyenne même avec moins de 'length' points non-NaN
            #    if valid_values_in_window > 0 : result[i] = current_sum / valid_values_in_window
            #    else: result[i] = np.nan

    return result

def custom_sma_numba(df: pd.DataFrame, length: int, column: str = 'close', output_col_name: Optional[str] = None) -> pd.Series:
    if column not in df.columns:
        raise IndicatorCalculationError(f"Colonne '{column}' non trouvée pour custom_sma_numba.")
    if length <= 0:
        raise IndicatorCalculationError("La période (length) pour custom_sma_numba doit être positive.")
    
    series_np = df[column].to_numpy(dtype=np.float64)
    sma_values = sma_numba_core(series_np, length)
    
    final_name = output_col_name 
    if final_name is None: 
        final_name = f"ORPHAN_CUSTOM_SMA_NUMBA_{length}"
        if column != 'close': final_name += f"_{column.upper()}"
    return pd.Series(sma_values, index=df.index, name=final_name)

def custom_rsi(df: pd.DataFrame, length: int, column: str = 'close', output_col_name: Optional[str] = None) -> pd.Series:
    if column not in df.columns:
        raise IndicatorCalculationError(f"Colonne '{column}' non trouvée pour Custom RSI.")
    if length <= 0:
        raise IndicatorCalculationError("La période RSI (length) doit être positive.")

    delta = df[column].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    # Utiliser ewm pour la moyenne mobile exponentielle, comme dans beaucoup d'implémentations de RSI
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    
    # Remplacer les infinis par NaN (peut arriver si avg_loss est 0)
    rsi_values.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Les premières `length` valeurs seront NaN à cause de min_periods=length dans ewm
    # et la première valeur de diff() est NaN.
    # Donc, on s'attend à `length` NaNs au début de rsi_values.
    if length > 0 and len(rsi_values) >= length:
         rsi_values.iloc[:length] = np.nan # Assurer que les premières 'length' sont NaN
                                           # car diff() cause un NaN, et ewm(min_periods=length)
                                           # en causera length-1 de plus sur gain/loss.
    
    final_name = output_col_name
    if final_name is None:
        final_name = f"ORPHAN_CUSTOM_RSI_{length}"
        if column != 'close': final_name += f"_{column.upper()}"
    return pd.Series(rsi_values, index=df.index, name=final_name)

# --- Gestionnaire d'Indicateurs ---
class IndicatorManager:
    def __init__(self):
        self.custom_indicators: Dict[str, Dict[str, Any]] = {}
        self._register_default_custom_indicators()
        logger.info("IndicatorManager initialisé.")

    def _register_default_custom_indicators(self):
        self.add_custom_indicator(
            name="custom_sma_numba", func=custom_sma_numba,
            params_info={"length": "int", "column": "str (default 'close')"},
            description="Custom Simple Moving Average calculated with Numba."
        )
        self.add_custom_indicator(
            name="custom_rsi", func=custom_rsi,
            params_info={"length": "int", "column": "str (default 'close')"},
            description="Custom Relative Strength Index (using EMA for smoothing)."
        )

    def add_custom_indicator(self, name: str, func: Callable, params_info: Optional[Dict[str, str]] = None, description: Optional[str] = None):
        if name in self.custom_indicators: logger.warning(f"Indicateur custom '{name}' remplacé.")
        self.custom_indicators[name] = {"function": func, "params_info": params_info or {}, "description": description or ""}
        logger.info(f"Indicateur custom '{name}' ajouté.")

    def _get_pta_func(self, indicator_name: str) -> Optional[Callable]:
        try:
            return getattr(ta, indicator_name.lower(), None)
        except Exception:
            return None

    def _generate_final_col_name(self, base_indicator_name: str, prefix: Optional[str], params: Dict[str, Any]) -> str:
        name_parts = []
        if prefix:
            name_parts.append(prefix)
        
        name_parts.append(base_indicator_name.upper())
        
        param_suffix_parts = []
        # Trier les clés pour assurer un nommage cohérent
        for k, v in sorted(params.items()):
            # Ne pas inclure 'column' si c'est 'close' (implicite)
            if k == 'column' and v == 'close': 
                continue
            # Ne pas inclure les paramètres internes comme 'output_col_name'
            if k == 'output_col_name':
                continue
            param_val_str = str(v).replace('.', '_') # Remplacer '.' par '_' pour la lisibilité
            param_suffix_parts.append(f"{k.upper()}{param_val_str}")
        
        if param_suffix_parts:
            name_parts.append("_".join(param_suffix_parts))
            
        return "_".join(name_parts)

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_name: str,
        output_col_prefix: Optional[str] = None, **kwargs: Any
    ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("L'entrée 'df' doit être un DataFrame Pandas.")
        if df.empty: return df.copy()

        df_out = df.copy()
        indicator_output: Optional[Union[pd.Series, pd.DataFrame]] = None
        
        # Copier kwargs pour éviter de modifier le dictionnaire original
        call_params = kwargs.copy()

        if indicator_name in self.custom_indicators:
            try:
                custom_info = self.custom_indicators[indicator_name]
                func = custom_info["function"]
                
                # Générer le nom de colonne final pour l'indicateur custom
                # Le nom de base est le nom de l'indicateur custom lui-même
                # output_col_prefix est le préfixe global
                # Les paramètres (call_params) sont utilisés pour le suffixe
                final_output_name_for_custom = self._generate_final_col_name(indicator_name, output_col_prefix, call_params)
                call_params['output_col_name'] = final_output_name_for_custom
                
                indicator_output = func(df, **call_params) 
                # La fonction custom est maintenant responsable de nommer sa série de sortie
                # avec le `output_col_name` qui lui est passé.
                logger.debug(f"Indicateur personnalisé '{indicator_name}' calculé. Nom de sortie attendu: {final_output_name_for_custom}")
            except Exception as e:
                logger.error(f"Erreur calcul indicateur custom '{indicator_name}': {e}")
                raise IndicatorCalculationError(f"Custom ind. '{indicator_name}' failed: {e}") from e
        else:
            pta_func_direct = self._get_pta_func(indicator_name)
            pta_method_name_accessor = indicator_name.lower()
            
            if pta_func_direct and callable(pta_func_direct):
                try:
                    close_col_name = call_params.pop('close_col', call_params.pop('close', 'close')) # Nom de la colonne source pour 'close'
                    main_series_input: Optional[pd.Series] = None
                    if close_col_name in df.columns:
                        main_series_input = df[close_col_name]
                    
                    # Gérer les cas où 'length=0' est passé, pandas-ta peut le gérer ou le mettre par défaut
                    if 'length' in call_params and call_params['length'] == 0:
                        logger.warning(f"Paramètre 'length=0' pour pandas_ta '{indicator_name}'. "
                                       f"pandas-ta pourrait utiliser une valeur par défaut (ex: 10 pour SMA).")
                        # Pour SMA, length=0 devient length=10 dans pandas-ta.
                        # Pour d'autres, cela pourrait causer une erreur ou un comportement inattendu.
                        # On laisse pandas-ta gérer, mais on logue.

                    if all(col in df.columns for col in ['high', 'low']) and main_series_input is not None and \
                       all(p in inspect.signature(pta_func_direct).parameters for p in ['high','low','close']):
                        indicator_output = pta_func_direct(high=df['high'], low=df['low'], close=main_series_input, **call_params)
                    elif main_series_input is not None:
                         # Cas pour les indicateurs qui prennent une seule série (ex: rsi(close=df['close']))
                        # On doit passer la série explicitement si la fonction attend 'close'
                        sig_params = inspect.signature(pta_func_direct).parameters
                        if 'close' in sig_params and 'close' not in call_params: # Si 'close' est un arg nommé et non dans kwargs
                            call_params_with_series = {'close': main_series_input, **call_params}
                            indicator_output = pta_func_direct(**call_params_with_series)
                        else: # Sinon, passer la série comme premier argument positionnel
                            indicator_output = pta_func_direct(main_series_input, **call_params)
                    else:
                        sig_params = inspect.signature(pta_func_direct).parameters
                        if 'close' not in sig_params and not any(p in sig_params for p in ['open', 'high', 'low', 'volume']):
                            indicator_output = pta_func_direct(df, **call_params)
                        else:
                             raise IndicatorCalculationError(f"Colonne '{close_col_name}' (ou OHLC) non trouvée pour pandas_ta '{indicator_name}'.")
                    logger.debug(f"Indicateur pandas_ta '{indicator_name}' (appel direct) calculé.")
                except Exception as e_direct_call:
                    logger.warning(f"Appel direct à pandas_ta.{indicator_name} échoué: {e_direct_call}. Tentative avec l'accesseur df.ta.")
                    indicator_output = None
            
            if indicator_output is None and hasattr(df.ta, pta_method_name_accessor):
                try:
                    temp_df_for_accessor = df.copy()
                    accessor_kwargs = call_params.copy()
                    accessor_kwargs['append'] = False 
                    
                    result_from_accessor = getattr(temp_df_for_accessor.ta, pta_method_name_accessor)(**accessor_kwargs)
                    
                    if result_from_accessor is not None:
                        indicator_output = result_from_accessor
                    else:
                        logger.warning(f"Accesseeur df.ta.{pta_method_name_accessor} avec append=False a retourné None.")
                    logger.debug(f"Indicateur pandas_ta (accesseur) '{indicator_name}' calculé.")
                except Exception as e_accessor:
                    logger.error(f"Erreur calcul pandas_ta (accesseur) '{pta_method_name_accessor}': {e_accessor}")
                    raise IndicatorCalculationError(f"pandas_ta ind. '{indicator_name}' (accesseur) failed: {e_accessor}") from e_accessor
            elif indicator_output is None and not (pta_func_direct and callable(pta_func_direct)):
                raise IndicatorNotFoundError(f"Indicateur '{indicator_name}' non trouvé.")

        # Fusionner le résultat
        if indicator_output is not None:
            if isinstance(indicator_output, pd.Series):
                # Si c'est un indicateur custom, il a déjà le nom final grâce à output_col_name passé à la fonction custom.
                # Si c'est un indicateur pandas-ta, on construit le nom.
                if indicator_name in self.custom_indicators:
                    final_col_name = indicator_output.name # Doit être déjà bien nommé par la fonction custom
                    if not final_col_name: # Fallback si la fonction custom n'a pas bien nommé
                        final_col_name = self._generate_final_col_name(indicator_name, output_col_prefix, kwargs)
                else: # pandas_ta
                    col_name_from_pta = indicator_output.name
                    if not col_name_from_pta: # Si la série de pandas_ta n'a pas de nom
                        # Utiliser le nom de l'indicateur et ses paramètres comme base
                        base_name = indicator_name.upper()
                        # Pour SMA(length=0), pandas-ta retourne SMA_10 (par défaut)
                        # Le nom de la série est SMA_10, donc col_name_from_pta serait SMA_10
                        # On veut que le nom final reflète les params d'appel (length=0)
                        # Mais aussi le comportement réel (SMA_10).
                        # On va utiliser _generate_final_col_name qui se base sur les kwargs d'appel.
                        # Cependant, si pandas_ta change un paramètre (ex: length=0 -> length=10),
                        # le nom généré par _generate_final_col_name(..., length=0) ne reflètera pas SMA_10.
                        # Le nom de la série retournée par pandas_ta est généralement plus fiable.
                        # Ex: df.ta.sma(length=0) -> pd.Series(name="SMA_10")
                        # Ex: df.ta.rsi(length=14) -> pd.Series(name="RSI_14")
                        # Donc, on peut utiliser col_name_from_pta s'il existe.
                        # S'il n'existe pas, on le construit.
                        params_to_name = kwargs.copy()
                        if 'length' in params_to_name and params_to_name['length'] == 0 and indicator_name.lower() == 'sma':
                            # Cas spécial pour SMA(0) -> SMA_10
                             col_name_from_pta = f"SMA_{ta.SMA(length=0).length}" # Récupérer la longueur par défaut de pandas-ta
                        else:
                             col_name_from_pta = self._generate_final_col_name(indicator_name, None, params_to_name).replace(f"{indicator_name.upper()}_", "") # Enlever le nom de base pour éviter duplication
                    
                    final_col_name = f"{output_col_prefix}_{col_name_from_pta}" if output_col_prefix else col_name_from_pta

                df_out[final_col_name] = indicator_output

            elif isinstance(indicator_output, pd.DataFrame): # Ex: bbands
                for col_in_output_df in indicator_output.columns:
                    final_col_name = f"{output_col_prefix}_{col_in_output_df}" if output_col_prefix else col_in_output_df
                    df_out[final_col_name] = indicator_output[col_in_output_df]
            else:
                raise IndicatorCalculationError(f"Indicateur '{indicator_name}' type sortie inattendu: {type(indicator_output)}")
        
        if not df_out.index.equals(df.index):
            df_out = df_out.reindex(df.index)
        return df_out

    def calculate_multiple_indicators(self, df: pd.DataFrame, indicator_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        df_with_all_indicators = df.copy()
        for config in indicator_configs:
            if "name" not in config:
                raise ValueError("Chaque configuration d'indicateur doit avoir une clé 'name'.")
            
            # Copier la config pour ne pas modifier l'originale
            current_config = config.copy()
            name = current_config.pop("name")
            output_prefix = current_config.pop("output_col_prefix", None)
            
            try:
                df_with_all_indicators = self.calculate_indicator(
                    df_with_all_indicators, indicator_name=name,
                    output_col_prefix=output_prefix, **current_config # Passer le reste de current_config comme kwargs
                )
            except IndicatorError as e:
                logger.error(f"Échec du calcul de l'indicateur '{name}' avec config {config}: {e}")
            except Exception as e: #NOSONAR
                logger.error(f"Erreur inattendue pour indicateur '{name}' avec config {config}: {e}")
        return df_with_all_indicators

    def validate_indicator_output(
        self, df_with_indicator: pd.DataFrame, indicator_col_name: str,
        expected_lookback: Optional[int] = None,
        value_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        input_col_for_nan_check: str = 'close'
    ) -> List[str]:
        errors = []
        if indicator_col_name not in df_with_indicator.columns:
            errors.append(f"Colonne d'indicateur '{indicator_col_name}' non trouvée.")
            return errors

        indicator_series = df_with_indicator[indicator_col_name]

        num_expected_initial_nans = 0
        if expected_lookback is not None and expected_lookback > 0:
            base_name_for_check = indicator_col_name
            # Simplifier le nom pour la vérification (enlever préfixe et suffixe de params)
            # Exemple: TestRSI_RSI_14 -> RSI
            # Exemple: MySMA_CUSTOM_SMA_NUMBA_LENGTH5 -> CUSTOM_SMA_NUMBA
            parts = indicator_col_name.split('_')
            if len(parts) > 1 and parts[0].isupper() and not parts[1].isupper(): # Heuristique pour un préfixe
                base_name_for_check = '_'.join(parts[1:])
            
            # Enlever les suffixes de paramètres pour identifier le type d'indicateur
            # Ex: RSI_14 -> RSI, CUSTOM_SMA_NUMBA_LENGTH5 -> CUSTOM_SMA_NUMBA
            if "RSI" in base_name_for_check.upper():
                 num_expected_initial_nans = expected_lookback # RSI(N) -> N NaNs
            elif "SMA" in base_name_for_check.upper() or "CUSTOM_SMA_NUMBA" in base_name_for_check.upper() :
                 num_expected_initial_nans = expected_lookback -1 # SMA(N) -> N-1 NaNs
            else: 
                 num_expected_initial_nans = expected_lookback -1 # Défaut

        if num_expected_initial_nans > 0:
            if len(indicator_series) >= num_expected_initial_nans :
                first_valid_input_index = df_with_indicator[input_col_for_nan_check].first_valid_index()
                if first_valid_input_index is not None:
                    # Aligner le début de la série d'indicateur avec le premier input valide
                    series_from_valid_input_start = indicator_series.loc[first_valid_input_index:]
                    if len(series_from_valid_input_start) >= num_expected_initial_nans:
                        initial_indicator_values_to_check = series_from_valid_input_start.iloc[:num_expected_initial_nans]
                        if not initial_indicator_values_to_check.isnull().all():
                            errors.append(
                                f"'{indicator_col_name}': NaNs initiaux incorrects. "
                                f"Attendu: les {num_expected_initial_nans} premières valeurs (après 1er input valide) soient NaN. "
                                f"Trouvé: {initial_indicator_values_to_check.isnull().sum()} NaNs sur {len(initial_indicator_values_to_check)}."
                            )
                    # else: # Pas assez de données d'indicateur après le premier input valide pour vérifier tous les NaNs attendus
                        # logger.debug(f"Pas assez de données pour '{indicator_col_name}' après le premier input valide pour vérifier les {num_expected_initial_nans} NaNs initiaux.")
                # else: # Si la colonne d'input est entièrement NaN, l'indicateur devrait aussi être entièrement NaN
                    # if not indicator_series.isnull().all():
                        # errors.append(f"'{indicator_col_name}': Devrait être NaN car input '{input_col_for_nan_check}' est entièrement NaN.")
            # else: # Pas assez de données d'indicateur au total pour vérifier les NaNs attendus
                # logger.debug(f"Pas assez de données pour '{indicator_col_name}' au total pour vérifier les {num_expected_initial_nans} NaNs initiaux.")


        if num_expected_initial_nans >= 0 and len(indicator_series) > num_expected_initial_nans: # Vérifier après les NaNs initiaux
            relevant_input_series = df_with_indicator[input_col_for_nan_check]
            # On ne vérifie que les points où l'indicateur est NaN mais l'input ne l'est pas (et sa fenêtre de lookback non plus)
            indicator_nan_where_input_valid = indicator_series.iloc[num_expected_initial_nans:].isnull() & relevant_input_series.iloc[num_expected_initial_nans:].notnull()
            
            for idx_in_slice, is_problematic_nan in enumerate(indicator_nan_where_input_valid):
                if is_problematic_nan:
                    actual_series_index = indicator_series.index[num_expected_initial_nans + idx_in_slice]
                    
                    # Vérifier la fenêtre de lookback de l'input pour ce point
                    lookback_window_size = expected_lookback if expected_lookback is not None else 1
                    
                    # Trouver l'index de début de la fenêtre dans le DataFrame original
                    original_df_index_loc = df_with_indicator.index.get_loc(actual_series_index)
                    start_input_window_df_loc = max(0, original_df_index_loc - (lookback_window_size -1))
                    
                    input_window_indices_in_df = df_with_indicator.index[start_input_window_df_loc : original_df_index_loc+1]
                    
                    input_window_for_current_indicator = relevant_input_series.loc[input_window_indices_in_df]

                    if not input_window_for_current_indicator.isnull().any():
                        # Tolérance pour RSI qui peut propager NaN plus longtemps
                        is_rsi_indicator = "RSI" in indicator_col_name.upper()
                        if not (is_rsi_indicator and input_window_for_current_indicator.isnull().sum() > 0) : # Si ce n'est pas un RSI avec des NaN dans sa fenêtre
                             errors.append(f"'{indicator_col_name}': NaN inattendu à l'index {actual_series_index.strftime('%Y-%m-%d %H:%M:%S')} où l'entrée et sa fenêtre de lookback sont valides.")
                             break # Un seul exemple suffit
        
        if value_range:
            min_val, max_val = value_range
            valid_series = indicator_series.dropna()
            if len(valid_series) > 0:
                if min_val is not None and (valid_series < min_val).any():
                    errors.append(f"'{indicator_col_name}': Min {valid_series.min()} < limite min {min_val}.")
                if max_val is not None and (valid_series > max_val).any():
                    errors.append(f"'{indicator_col_name}': Max {valid_series.max()} > limite max {max_val}.")
        
        if np.isinf(indicator_series.dropna()).any():
            errors.append(f"'{indicator_col_name}': Contient des valeurs infinies.")

        if not errors: logger.debug(f"Validation de '{indicator_col_name}' réussie.")
        else: logger.warning(f"Validation de '{indicator_col_name}' échouée: {errors}")
        return errors
        
    def measure_performance(
        self, df: pd.DataFrame, indicator_name: str,
        repetitions: int = 10, **kwargs: Any
    ) -> Dict[str, Any]:
        if repetitions <= 0: raise ValueError("Le nombre de répétitions doit être positif.")
        timings = []
        error_occurred = None
        first_result_df_sample = None 
        # Copier les kwargs pour pouvoir les modifier localement si besoin
        current_kwargs = kwargs.copy()

        for i in range(repetitions):
            start_time = time.perf_counter()
            try:
                # Assurer que df n'est pas modifié entre les répétitions
                df_copy_for_repetition = df.copy()
                current_result_df = self.calculate_indicator(df_copy_for_repetition, indicator_name, **current_kwargs)
                if i == 0 and current_result_df is not None: 
                    sample_size = min(5, len(current_result_df))
                    first_result_df_sample = current_result_df.head(sample_size)
            except IndicatorError as e:
                error_occurred = str(e)
                break 
            except Exception as e: #NOSONAR
                error_occurred = f"Unexpected: {str(e)}"
                break
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)

        if error_occurred:
            return {"error": error_occurred, "indicator_name": indicator_name, "params": kwargs, "dataframe_sample_at_error": first_result_df_sample}

        total_time_ms = sum(timings)
        avg_time_ms = total_time_ms / repetitions
        
        logger.info(f"Perf '{indicator_name}' ({len(df)}l, {repetitions}r): Tot:{total_time_ms:.2f}ms, Avg:{avg_time_ms:.2f}ms")
        return {"total_time_ms": total_time_ms, "avg_time_ms": avg_time_ms, "repetitions": repetitions, "data_length": len(df), "result_sample_df": first_result_df_sample}

    def get_available_indicators(self) -> Dict[str, List[Dict[str, Any]]]:
        available = {"pandas_ta": [], "custom": []}
        # Liste plus exhaustive des indicateurs courants de pandas_ta
        common_pta = ["sma", "ema", "dema", "tema", "wma", "hma", "rsi", "macd", "bbands", "atr", "stoch", "adx", "cci", "mom", "roc", "ppo", "vwap", "obv", "cmf", "mfi", "ichimoku", "psar"]
        for pta_name in common_pta:
             pta_func = self._get_pta_func(pta_name)
             if pta_func and callable(pta_func):
                try:
                    sig = inspect.signature(pta_func)
                    params_list = [
                        p.name for p in sig.parameters.values() 
                        # Exclure les paramètres génériques ou ceux gérés par l'accesseur df.ta
                        if p.name not in ['self', 'kwargs', 'append', 'df', 
                                          'open_', 'high_', 'low_', 'close_', 'volume_'] 
                        # Exclure aussi les paramètres qui sont des alias de colonnes OHLCV
                        and p.name not in ['open', 'high', 'low', 'close', 'volume'] 
                    ]
                    available["pandas_ta"].append({
                        "name": pta_name, 
                        "params_info_keys": params_list, 
                        "description": inspect.getdoc(pta_func) or "N/A"
                    })
                except Exception: #NOSONAR
                    pass 
        available["pandas_ta"] = sorted(available["pandas_ta"], key=lambda x: x['name'])
        for name, info in self.custom_indicators.items():
            available["custom"].append({"name": name, "params_info": info.get("params_info", {}), "description": info.get("description", "")})
        available["custom"] = sorted(available["custom"], key=lambda x: x['name'])
        return available

# --- Exemple d'Utilisation ---
if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), colorize=True, format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="DEBUG")

    data = {
        'open':   np.array([10,11,12,11.5,10.5,9.5,8.5,9,10,11,12,13,12.5,11.5,10.5,9,10,11.5, np.nan, np.nan, 13, 14], dtype=float), 
        'high':   np.array([11,12.5,13,12,11,10,9,10.5,11,12,13.5,14,13,12,11,10.5,11,12, np.nan, np.nan, 14, 15], dtype=float),
        'low':    np.array([9.5,10.5,11,10,9,8,7,8.5,9,10,11,12,11.5,10,9,8,9.5,10, np.nan, np.nan, 12, 13], dtype=float),
        'close':  np.array([10.8,12.2,11.5,10.3,9.2,8.1,8.8,10.2,10.8,11.7,13.2,12.3,11.2,10.1,9.3,9.8,11.2,10.5, np.nan, np.nan, 13.5,14.5], dtype=float),
        'volume': np.array([100,150,120,110,90,80,70,85,95,105,115,125,130,140,155,160,170,180, np.nan, np.nan, 190,200], dtype=float)
    }
    start_date = pd.to_datetime("2023-01-01")
    index = pd.date_range(start_date, periods=len(data['close']), freq='D')
    df_test = pd.DataFrame(data, index=index)
    # Assurer la cohérence OHLC
    df_test['high'] = df_test[['open', 'high', 'close']].max(axis=1) 
    df_test['low'] = df_test[['open', 'low', 'close']].min(axis=1)
    df_test.loc[df_test['low'] < 0.001, 'low'] = 0.001 # Éviter les prix négatifs ou nuls

    manager = IndicatorManager()

    logger.info("\n--- Calcul pandas_ta (RSI) ---")
    df_rsi = manager.calculate_indicator(df_test.copy(), "rsi", length=14, output_col_prefix="PTA")
    rsi_col = "PTA_RSI_14" # Le nom de la série retournée par pandas_ta est RSI_14
    if rsi_col in df_rsi: logger.info(f"OK: {rsi_col}\n{df_rsi[['close', rsi_col]].tail(10)}")
    else: logger.error(f"KO: {rsi_col} non trouvé. Colonnes: {df_rsi.columns.tolist()}")
    errs = manager.validate_indicator_output(df_rsi, rsi_col, 14, (0,100))
    if errs: logger.warning(f"Validation {rsi_col}: {errs}") 
    else: logger.success(f"Validation {rsi_col} OK")

    logger.info("\n--- Calcul custom_sma_numba ---")
    # Le nom de colonne final sera généré par _generate_final_col_name et passé à la fonction custom
    # via output_col_name. La fonction custom utilisera ce nom.
    df_sma_custom = manager.calculate_indicator(df_test.copy(), "custom_sma_numba", length=5, column='close', output_col_prefix="MySMA")
    sma_custom_col = "MySMA_CUSTOM_SMA_NUMBA_LENGTH5" # Nom attendu
    if sma_custom_col in df_sma_custom: logger.info(f"OK: {sma_custom_col}\n{df_sma_custom[['close', sma_custom_col]].tail(10)}")
    else: logger.error(f"KO: {sma_custom_col} non trouvé. Colonnes: {df_sma_custom.columns.tolist()}")
    errs = manager.validate_indicator_output(df_sma_custom, sma_custom_col, 5)
    if errs: logger.warning(f"Validation {sma_custom_col}: {errs}") 
    else: logger.success(f"Validation {sma_custom_col} OK")

    logger.info("\n--- Calcul multiple ---")
    cfgs = [{"name": "sma", "length": 10, "output_col_prefix": "S"}, 
            {"name": "custom_rsi", "length": 7, "column":"close", "output_col_prefix": "CR"}]
    df_m = manager.calculate_multiple_indicators(df_test.copy(), cfgs)
    logger.info(f"Colonnes multiples: {[c for c in df_m.columns if c not in df_test.columns]}")
    assert "S_SMA_10" in df_m.columns # Nom de pandas_ta pour SMA(10)
    assert "CR_CUSTOM_RSI_LENGTH7" in df_m.columns # Nom généré pour custom_rsi
    
    logger.info("\n--- Test de Performance (custom_sma_numba avec length=0 pour erreur) ---")
    perf_custom_error = manager.measure_performance(df_test, "custom_sma_numba", length=0, repetitions=3)
    logger.info(f"Résultats perf custom_sma_numba(0): {perf_custom_error}")
    assert "error" in perf_custom_error
