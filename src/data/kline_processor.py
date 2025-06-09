# src/data/kline_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timezone
from loguru import logger
import re

from src.core.constants import Kline
from src.core.exceptions import DataError, KlineValidationError

class KlineProcessor:
    """
    Processeur pour la manipulation et le resampling des klines.
    Gère la génération de klines multi-fréquences (standard et glissantes).
    Version optimisée pour la performance.
    """

    def __init__(self):
        """Initialise le processeur de klines."""
        self.interval_mapping = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': 'D', '3d': '3D', '1w': 'W', '1M': 'MS'
        }
        logger.debug("KlineProcessor initialized")

    def _freq_to_window_size(self, target_freq_str: str, base_freq_str: str = Kline.INTERVAL_1MINUTE) -> int:
        """Calcule la taille de la fenêtre en nombre de barres de base."""
        try:
            target_td = pd.to_timedelta(self.interval_mapping.get(target_freq_str, target_freq_str))
            base_td = pd.to_timedelta(self.interval_mapping.get(base_freq_str, base_freq_str))
            if base_td.total_seconds() == 0:
                raise ValueError("Base frequency cannot be zero.")
            return int(target_td / base_td)
        except (ValueError, TypeError) as e:
            raise DataError(f"Could not determine window size for target '{target_freq_str}' from base '{base_freq_str}'. Error: {e}")

    def _ensure_column_types(self, df: pd.DataFrame):
        """S'assure que les colonnes OHLCV ont les types numériques corrects."""
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'number_of_trades' in df.columns:
            if not str(df['number_of_trades'].dtype) == 'Int64':
                 df['number_of_trades'] = pd.to_numeric(df['number_of_trades'], errors='coerce').astype('Int64')
        return df

    def process_to_multi_rolling_klines(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        Traite un DataFrame de klines 1m pour générer une vue enrichie avec des klines
        glissantes (rolling) sur plusieurs fréquences.

        Args:
            df_1m: DataFrame de klines 1m avec un DatetimeIndex.

        Returns:
            Un DataFrame enrichi avec des colonnes préfixées 'K_<freq>_'.
        """
        if not isinstance(df_1m, pd.DataFrame) or not isinstance(df_1m.index, pd.DatetimeIndex):
            raise DataError("Input must be a pandas DataFrame with a DatetimeIndex.")
        if df_1m.empty:
            logger.warning("Input DataFrame df_1m is empty. Returning an empty DataFrame.")
            return pd.DataFrame()

        df_1m_sorted = df_1m.copy()
        if not df_1m_sorted.index.is_monotonic_increasing:
            df_1m_sorted.sort_index(inplace=True)
        
        df_1m_sorted = self._ensure_column_types(df_1m_sorted)
        
        # Le DataFrame final sera construit à partir de cette liste de séries
        all_series_to_concat = []

        # 1. Préparer les données K_1m_ (valeurs de la bougie précédente)
        for col in Kline.BASE_COLS_FOR_PREFIXING:
            if col in df_1m_sorted.columns:
                prefixed_name = Kline.get_prefixed_col_name(Kline.INTERVAL_1MINUTE, col)
                all_series_to_concat.append(df_1m_sorted[col].shift(1).rename(prefixed_name))

        # 2. Calculer les klines glissantes pour chaque intervalle cible
        for interval_str in Kline.TARGET_ROLLING_INTERVALS:
            window_size = self._freq_to_window_size(interval_str, Kline.INTERVAL_1MINUTE)
            if window_size <= 1:
                continue

            # OPEN: Le premier 'open' de la fenêtre glissante
            open_col = Kline.get_prefixed_col_name(interval_str, Kline.OHLCV_OPEN)
            all_series_to_concat.append(df_1m_sorted['open'].shift(window_size).rename(open_col))

            # HIGH: Le 'high' maximum de la fenêtre
            high_col = Kline.get_prefixed_col_name(interval_str, Kline.OHLCV_HIGH)
            all_series_to_concat.append(df_1m_sorted['high'].rolling(window=window_size, min_periods=window_size).max().rename(high_col))

            # LOW: Le 'low' minimum de la fenêtre
            low_col = Kline.get_prefixed_col_name(interval_str, Kline.OHLCV_LOW)
            all_series_to_concat.append(df_1m_sorted['low'].rolling(window=window_size, min_periods=window_size).min().rename(low_col))
            
            # CLOSE: Le dernier 'close' de la fenêtre
            close_col = Kline.get_prefixed_col_name(interval_str, Kline.OHLCV_CLOSE)
            all_series_to_concat.append(df_1m_sorted['close'].shift(1).rename(close_col))

            # VOLUME et autres colonnes sommables
            for col in ['volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                if col in df_1m_sorted.columns:
                    prefixed_name = Kline.get_prefixed_col_name(interval_str, col)
                    all_series_to_concat.append(df_1m_sorted[col].rolling(window=window_size, min_periods=window_size).sum().rename(prefixed_name))
            
            # KLINE_CLOSE_TIME: Le dernier de la fenêtre
            if 'kline_close_time' in df_1m_sorted.columns:
                kct_col = Kline.get_prefixed_col_name(interval_str, Kline.OHLCV_KLINE_CLOSE_TIME)
                all_series_to_concat.append(df_1m_sorted['kline_close_time'].shift(1).rename(kct_col))

            # IS_KLINE_CLOSED: Toujours True pour les données historiques complètes
            is_closed_col = Kline.get_prefixed_col_name(interval_str, Kline.OHLCV_IS_KLINE_CLOSED)
            all_series_to_concat.append(pd.Series(True, index=df_1m_sorted.index, name=is_closed_col))


        # 3. Concaténer toutes les séries et décaler le résultat final de 1
        # pour s'assurer que les données à un instant T ne contiennent que des informations antérieures.
        result_df = pd.concat(all_series_to_concat, axis=1).shift(1)

        # 4. Nettoyer les lignes initiales qui contiennent des NaNs dus au rolling
        k1m_close_col_for_dropna = Kline.get_prefixed_col_name(Kline.INTERVAL_1MINUTE, Kline.OHLCV_CLOSE)
        if k1m_close_col_for_dropna in result_df.columns:
            result_df.dropna(subset=[k1m_close_col_for_dropna], inplace=True)
        else:
             # Si K_1m_close n'est pas là, utiliser une autre colonne K_1m comme référence pour le nettoyage
             k1m_open_col_for_dropna = Kline.get_prefixed_col_name(Kline.INTERVAL_1MINUTE, Kline.OHLCV_OPEN)
             if k1m_open_col_for_dropna in result_df.columns:
                 result_df.dropna(subset=[k1m_open_col_for_dropna], inplace=True)


        logger.info(f"Finished processing multi-rolling klines. Output shape: {result_df.shape}")
        return result_df

    def resample_klines(
        self,
        df: pd.DataFrame,
        target_freq: str
    ) -> pd.DataFrame:
        """Resample un DataFrame de klines à une fréquence cible standard (non-glissante)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError("DataFrame must have a DatetimeIndex.")
        if df.empty:
            return pd.DataFrame()

        pandas_freq = self.interval_mapping.get(target_freq, target_freq)
        
        agg_rules = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'quote_asset_volume': 'sum', 'number_of_trades': 'sum',
            'taker_buy_base_asset_volume': 'sum', 'taker_buy_quote_asset_volume': 'sum'
        }
        
        df_cleaned = self._ensure_column_types(df.copy())
        
        resampled_df = df_cleaned.resample(pandas_freq, label='left', closed='left').agg(agg_rules)
        resampled_df.dropna(subset=['open'], inplace=True) # Supprime les périodes sans aucune donnée

        return resampled_df

    def validate_klines(
        self,
        df: pd.DataFrame,
        check_continuity: bool = True,
        expected_freq_minutes: Optional[int] = 1 
    ) -> Tuple[bool, List[str]]:
        """Valide la structure et la cohérence d'un DataFrame de klines."""
        errors = []
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Input must be a DataFrame with a DatetimeIndex.")
            return False, errors

        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonically increasing.")

        # Vérifier la continuité
        if check_continuity and expected_freq_minutes and not df.empty:
            expected_delta = pd.Timedelta(minutes=expected_freq_minutes)
            diffs = df.index.to_series().diff().dropna()
            if not diffs.empty and (diffs != expected_delta).any():
                incorrect_intervals = diffs[diffs != expected_delta]
                errors.append(f"Found {len(incorrect_intervals)} gaps or overlaps in data continuity. Example gap: {incorrect_intervals.iloc[0]}")
        
        # Vérifier la logique OHLC
        has_prefixed_cols = any(col.startswith(f"{Kline.ROLLING_KLINE_PREFIX}_") for col in df.columns)
        freq_to_check = [Kline.INTERVAL_1MINUTE] + Kline.TARGET_ROLLING_INTERVALS if has_prefixed_cols else [""]

        for freq in freq_to_check:
            prefix = f"{Kline.ROLLING_KLINE_PREFIX}_{freq}_" if freq else ""
            h, l, o, c = f"{prefix}high", f"{prefix}low", f"{prefix}open", f"{prefix}close"
            if all(col in df.columns for col in [h, l, o, c]):
                invalid_ohlc = df[ (df[h] < df[l]) | (df[o] > df[h]) | (df[c] > df[h]) | (df[o] < df[l]) | (df[c] < df[l]) ]
                if not invalid_ohlc.empty:
                    errors.append(f"Found {len(invalid_ohlc)} rows with invalid OHLC relationship for frequency '{freq}'.")

        return not errors, errors
