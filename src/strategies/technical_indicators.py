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
    for i in range(n):
        if i < length - 1:
            result[i] = np.nan
        else:
            window = series[i - length + 1 : i + 1]
            if np.any(np.isnan(window)):
                result[i] = np.nan
            else:
                result[i] = np.mean(window)
    return result

def custom_sma_numba(df: pd.DataFrame, length: int, column: str = 'close', output_col_name: Optional[str] = None) -> pd.Series:
    if column not in df.columns:
        raise IndicatorCalculationError(f"Colonne '{column}' non trouvée pour custom_sma_numba.")
    if length <= 0:
        raise IndicatorCalculationError("La période (length) pour custom_sma_numba doit être positive.")
    
    series_np = df[column].to_numpy(dtype=np.float64)
    sma_values = sma_numba_core(series_np, length)
    
    final_name = output_col_name # Doit être fourni par IndicatorManager
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

    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    
    rsi_values.replace([np.inf, -np.inf], np.nan, inplace=True)
    if length > 0 and len(rsi_values) >= length:
        rsi_values.iloc[:length] = np.nan
    
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
        for k, v in sorted(params.items()):
            if k == 'column' and v == 'close': continue
            param_val_str = str(v).replace('.', '_')
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
        
        if indicator_name in self.custom_indicators:
            try:
                custom_info = self.custom_indicators[indicator_name]
                func = custom_info["function"]
                call_kwargs = kwargs.copy()
                
                # Le nom de base est le nom de l'indicateur custom lui-même
                # Les paramètres (kwargs) sont utilisés pour le suffixe
                final_col_name = self._generate_final_col_name(indicator_name, output_col_prefix, kwargs)
                call_kwargs['output_col_name'] = final_col_name
                
                indicator_output = func(df, **call_kwargs) 
                logger.debug(f"Indicateur personnalisé '{indicator_name}' calculé. Nom de sortie: {final_col_name}")
            except Exception as e:
                logger.error(f"Erreur calcul indicateur custom '{indicator_name}': {e}")
                raise IndicatorCalculationError(f"Custom ind. '{indicator_name}' failed: {e}") from e
        else:
            pta_func_direct = self._get_pta_func(indicator_name)
            pta_method_name_accessor = indicator_name.lower()
            
            # Tenter d'abord l'appel direct de la fonction pandas_ta
            if pta_func_direct and callable(pta_func_direct):
                try:
                    pta_kwargs = kwargs.copy()
                    close_col_name = pta_kwargs.pop('close', 'close') # Nom de la colonne source pour 'close'
                    main_series_input: Optional[pd.Series] = None
                    if close_col_name in df.columns:
                        main_series_input = df[close_col_name]
                    
                    if all(col in df.columns for col in ['high', 'low']) and main_series_input is not None and \
                       all(p in inspect.signature(pta_func_direct).parameters for p in ['high','low','close']):
                        indicator_output = pta_func_direct(high=df['high'], low=df['low'], close=main_series_input, **pta_kwargs)
                    elif main_series_input is not None:
                        indicator_output = pta_func_direct(main_series_input, **pta_kwargs)
                    else:
                        sig_params = inspect.signature(pta_func_direct).parameters
                        if 'close' not in sig_params and not any(p in sig_params for p in ['open', 'high', 'low', 'volume']):
                            indicator_output = pta_func_direct(df, **pta_kwargs)
                        else:
                             raise IndicatorCalculationError(f"Colonne '{close_col_name}' (ou OHLC) non trouvée pour pandas_ta '{indicator_name}'.")
                    logger.debug(f"Indicateur pandas_ta '{indicator_name}' (appel direct) calculé.")
                except Exception as e_direct_call:
                    logger.warning(f"Appel direct à pandas_ta.{indicator_name} échoué: {e_direct_call}. Tentative avec l'accesseur df.ta.")
                    indicator_output = None # Réinitialiser pour tenter l'accesseur
            
            # Si l'appel direct a échoué ou n'a pas été tenté, essayer l'accesseur
            if indicator_output is None and hasattr(df.ta, pta_method_name_accessor):
                try:
                    temp_df_for_accessor = df.copy()
                    accessor_kwargs = kwargs.copy()
                    # Forcer append=False pour obtenir la sortie et la nommer nous-mêmes
                    accessor_kwargs['append'] = False 
                    
                    # Appel à la méthode d'accesseur
                    result_from_accessor = getattr(temp_df_for_accessor.ta, pta_method_name_accessor)(**accessor_kwargs)
                    
                    if result_from_accessor is not None:
                        indicator_output = result_from_accessor
                    else:
                        # Si append=False retourne None, cela peut signifier que l'indicateur
                        # ne retourne rien ou qu'il a été conçu pour modifier en place (moins courant avec append=False).
                        # On pourrait essayer avec append=True sur une autre copie pour voir s'il ajoute des colonnes.
                        logger.warning(f"Accesseeur df.ta.{pta_method_name_accessor} avec append=False a retourné None.")
                        # Pour ce cas, on ne fait rien de plus, la fusion plus bas gérera indicator_output = None
                    logger.debug(f"Indicateur pandas_ta (accesseur) '{indicator_name}' calculé.")
                except Exception as e_accessor:
                    logger.error(f"Erreur calcul pandas_ta (accesseur) '{pta_method_name_accessor}': {e_accessor}")
                    raise IndicatorCalculationError(f"pandas_ta ind. '{indicator_name}' (accesseur) failed: {e_accessor}") from e_accessor
            elif indicator_output is None and not (pta_func_direct and callable(pta_func_direct)):
                # Si ni fonction directe ni accesseur n'est trouvé
                raise IndicatorNotFoundError(f"Indicateur '{indicator_name}' non trouvé.")


        # Fusionner le résultat
        if indicator_output is not None:
            if isinstance(indicator_output, pd.Series):
                # Le nom de la série retournée par pandas-ta (ex: RSI_14) ou par la fonction custom.
                col_name_from_indicator = indicator_output.name
                if not col_name_from_indicator: # Si la série n'a pas de nom
                    # Utiliser le nom de l'indicateur et ses paramètres comme base
                    col_name_from_indicator = indicator_name.upper()
                    params_suffix = "_".join(f"{k.upper()}{v}" for k,v in sorted(kwargs.items()))
                    if params_suffix: col_name_from_indicator += f"_{params_suffix}"
                
                final_col_name = f"{output_col_prefix}_{col_name_from_indicator}" if output_col_prefix else col_name_from_indicator
                df_out[final_col_name] = indicator_output

            elif isinstance(indicator_output, pd.DataFrame): # Ex: bbands
                for col_in_output_df in indicator_output.columns:
                    # col_in_output_df est le nom par défaut de pandas-ta (ex: BBL_5_2.0)
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
            
            name = config.pop("name")
            output_prefix = config.pop("output_col_prefix", None)
            
            try:
                df_with_all_indicators = self.calculate_indicator(
                    df_with_all_indicators, indicator_name=name,
                    output_col_prefix=output_prefix, **config
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

        # Déterminer le nombre de NaNs initiaux attendus
        num_expected_initial_nans = 0
        if expected_lookback is not None and expected_lookback > 0:
            # Pour RSI(N) de pandas-ta ou custom_rsi, N NaNs initiaux (indices 0 à N-1)
            # Pour SMA(N) de pandas-ta ou custom_sma, N-1 NaNs initiaux (indices 0 à N-2)
            base_name_for_check = indicator_col_name.split('_')[0] if '_' in indicator_col_name else indicator_col_name
            if "RSI" in base_name_for_check.upper():
                 num_expected_initial_nans = expected_lookback
            elif "SMA" in base_name_for_check.upper():
                 num_expected_initial_nans = expected_lookback -1
            else: # Cas par défaut
                 num_expected_initial_nans = expected_lookback -1


        if num_expected_initial_nans > 0:
            if len(indicator_series) >= num_expected_initial_nans :
                first_valid_input_index = df_with_indicator[input_col_for_nan_check].first_valid_index()
                if first_valid_input_index is not None:
                    series_from_valid_input_start = indicator_series.loc[first_valid_input_index:]
                    if len(series_from_valid_input_start) >= num_expected_initial_nans:
                        initial_indicator_values_to_check = series_from_valid_input_start.iloc[:num_expected_initial_nans]
                        if not initial_indicator_values_to_check.isnull().all():
                            errors.append(
                                f"'{indicator_col_name}': NaNs initiaux incorrects. "
                                f"Attendu: les {num_expected_initial_nans} premières valeurs (après 1er input valide) soient NaN. "
                                f"Trouvé: {initial_indicator_values_to_check.isnull().sum()} NaNs."
                            )
                elif not indicator_series.iloc[:num_expected_initial_nans].isnull().all():
                     errors.append(f"'{indicator_col_name}': Devrait être NaN car input NaN, mais {num_expected_initial_nans} premières valeurs non toutes NaN.")

        # Validation des NaN inattendus APRÈS la période de lookback théorique
        if num_expected_initial_nans > 0 and len(indicator_series) > num_expected_initial_nans:
            relevant_input_series = df_with_indicator[input_col_for_nan_check]
            for i in range(num_expected_initial_nans, len(indicator_series)):
                current_idx = indicator_series.index[i]
                if current_idx not in relevant_input_series.index: continue

                if pd.notna(relevant_input_series.loc[current_idx]) and pd.isna(indicator_series.loc[current_idx]):
                    # Vérifier si un NaN dans la fenêtre de lookback de l'input pourrait expliquer le NaN de l'indicateur
                    lookback_window_size = expected_lookback if expected_lookback is not None else 1 # Fallback
                    start_input_window_idx = max(0, i - (lookback_window_size -1))
                    input_window_indices = indicator_series.index[start_input_window_idx : i+1]
                    
                    valid_input_window_indices = [idx for idx in input_window_indices if idx in relevant_input_series.index]
                    if not valid_input_window_indices: continue

                    input_window_for_current_indicator = relevant_input_series.loc[valid_input_window_indices]
                    
                    # Si la fenêtre d'input est entièrement valide mais l'indicateur est NaN, c'est un problème.
                    # Exception: RSI peut propager NaN plus longtemps à cause de EWM.
                    if not ("RSI" in indicator_col_name.upper() and input_window_for_current_indicator.isnull().sum() > 0): # Tolérance pour RSI
                        if not input_window_for_current_indicator.isnull().any():
                            errors.append(f"'{indicator_col_name}': NaN inattendu à l'index {current_idx.strftime('%Y-%m-%d')} où l'entrée et sa fenêtre de lookback sont valides.")
                            break 
        
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
        # ... (inchangé)
        if repetitions <= 0: raise ValueError("Le nombre de répétitions doit être positif.")
        timings = []
        error_occurred = None
        first_result_df_sample = None 
        for i in range(repetitions):
            start_time = time.perf_counter()
            try:
                current_result_df = self.calculate_indicator(df.copy(), indicator_name, **kwargs)
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
        # ... (inchangé)
        available = {"pandas_ta": [], "custom": []}
        common_pta = ["sma", "ema", "rsi", "macd", "bbands", "atr", "stoch", "adx", "cci", "mom", "roc"]
        for pta_name in common_pta:
             pta_func = self._get_pta_func(pta_name)
             if pta_func and callable(pta_func):
                try:
                    sig = inspect.signature(pta_func)
                    params_list = [
                        p.name for p in sig.parameters.values() 
                        if p.name not in ['self', 'kwargs', 'append', 'df', 'open', 'high', 'low', 'close', 'volume', 
                                          'open_', 'high_', 'low_', 'close_']
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
    # ... (l'exemple reste le même, mais les noms de colonnes attendus dans les logs changeront)
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
    df_test['high'] = df_test[['open', 'high', 'close']].max(axis=1) 
    df_test['low'] = df_test[['open', 'low', 'close']].min(axis=1)
    df_test.loc[df_test['low'] < 0.001, 'low'] = 0.001

    manager = IndicatorManager()

    logger.info("\n--- Calcul pandas_ta (RSI) ---")
    df_rsi = manager.calculate_indicator(df_test.copy(), "rsi", length=14, output_col_prefix="PTA")
    rsi_col = "PTA_RSI_14"
    if rsi_col in df_rsi: logger.info(f"OK: {rsi_col}\n{df_rsi[['close', rsi_col]].tail(10)}")
    else: logger.error(f"KO: {rsi_col} non trouvé. Colonnes: {df_rsi.columns.tolist()}")
    errs = manager.validate_indicator_output(df_rsi, rsi_col, 14, (0,100))
    if errs: logger.warning(f"Validation {rsi_col}: {errs}") 
    else: logger.success(f"Validation {rsi_col} OK")

    logger.info("\n--- Calcul custom_sma_numba ---")
    df_sma_custom = manager.calculate_indicator(df_test.copy(), "custom_sma_numba", length=5, column='close', output_col_prefix="MySMA")
    sma_custom_col = "MySMA_CUSTOM_SMA_NUMBA_LENGTH5" 
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
    # Noms attendus: S_SMA_10, CR_CUSTOM_RSI_LENGTH7
    assert "S_SMA_10" in df_m.columns
    assert "CR_CUSTOM_RSI_LENGTH7" in df_m.columns
    
    logger.info("\n--- Test de Performance (custom_sma_numba avec length=0 pour erreur) ---")
    perf_custom_error = manager.measure_performance(df_test, "custom_sma_numba", length=0, repetitions=3)
    logger.info(f"Résultats perf custom_sma_numba(0): {perf_custom_error}")
    assert "error" in perf_custom_error
