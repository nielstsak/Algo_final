# src/data/kline_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from loguru import logger

from src.core.constants import Kline
from src.core.exceptions import DataError, KlineValidationError


class KlineProcessor:
    """
    Processeur pour la manipulation et le resampling des klines.
    Gère la génération de klines multi-fréquences (standard et glissantes).
    """
    
    def __init__(self):
        """Initialise le processeur de klines."""
        # Mapping des intervalles Binance vers les fréquences pandas
        self.interval_mapping = {
            '1m': '1T',
            '3m': '3T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '8h': '8H',
            '12h': '12H',
            '1d': '1D',
            '3d': '3D',
            '1w': '1W',
            '1M': '1M'
        }
        
        # Règles d'agrégation standard
        self.aggregation_rules = Kline.AGGREGATION_RULES_STANDARD.copy()
        
        logger.debug("KlineProcessor initialized")
        
    def resample_klines(
        self,
        df: pd.DataFrame,
        target_freq: str,
        rolling: bool = False,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Génère des klines de fréquence supérieure à partir de klines 1m.
        
        Args:
            df: DataFrame avec klines source (doit avoir un index temporel)
            target_freq: Fréquence cible ('3m', '5m', '15m', '1h', etc.)
            rolling: Si True, génère des klines glissantes
            window_size: Taille de fenêtre pour les klines glissantes (si None, déduit de target_freq)
            
        Returns:
            DataFrame avec klines resamplées
            
        Raises:
            DataError: Si les données d'entrée sont invalides
        """
        # Validation des entrées
        if df.empty:
            raise DataError("Input DataFrame is empty")
            
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError("DataFrame must have a DatetimeIndex")
            
        # Convertir la fréquence cible si nécessaire
        pandas_freq = self.interval_mapping.get(target_freq, target_freq)
        
        if rolling:
            return self._generate_rolling_klines(df, pandas_freq, window_size)
        else:
            return self._generate_standard_klines(df, pandas_freq)
            
    def _generate_standard_klines(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Génère des klines standard (non glissantes) avec pandas resample.
        
        Args:
            df: DataFrame source
            freq: Fréquence pandas (ex: '5T', '1H')
            
        Returns:
            DataFrame avec klines resamplées
        """
        try:
            # Copier pour éviter les modifications in-place
            df_copy = df.copy()
            
            # S'assurer que les colonnes nécessaires existent
            required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'base_asset_volume']
            missing_cols = [col for col in required_cols if col not in df_copy.columns]
            if missing_cols:
                raise DataError(f"Missing required columns: {missing_cols}")
                
            # Resample avec les règles d'agrégation
            resampled = df_copy.resample(freq, label='left', closed='left').agg({
                'open_price': 'first',
                'high_price': 'max',
                'low_price': 'min',
                'close_price': 'last',
                'base_asset_volume': 'sum',
                'quote_asset_volume': 'sum',
                'number_of_trades': 'sum',
                'taker_buy_base_asset_volume': 'sum',
                'taker_buy_quote_asset_volume': 'sum'
            })
            
            # Supprimer les lignes avec des NaN (périodes sans données)
            resampled = resampled.dropna(subset=['open_price', 'close_price'])
            
            # Ajouter les colonnes supplémentaires si présentes
            if 'pair' in df_copy.columns:
                resampled['pair'] = df_copy['pair'].iloc[0]
            if 'is_kline_closed' in df_copy.columns:
                resampled['is_kline_closed'] = True
                
            # Calculer kline_close_time (fin de la période)
            resampled['kline_close_time'] = resampled.index + pd.Timedelta(freq) - pd.Timedelta(milliseconds=1)
            
            logger.debug(f"Generated {len(resampled)} standard klines at {freq} frequency")
            return resampled
            
        except Exception as e:
            logger.error(f"Error generating standard klines: {e}")
            raise DataError(f"Failed to generate standard klines: {e}", original_exception=e)
            
    def _generate_rolling_klines(
        self,
        df: pd.DataFrame,
        freq: str,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Génère des klines glissantes (rolling windows).
        
        Args:
            df: DataFrame source
            freq: Fréquence pandas
            window_size: Taille de la fenêtre en minutes
            
        Returns:
            DataFrame avec klines glissantes
        """
        try:
            # Calculer la taille de fenêtre si non fournie
            if window_size is None:
                window_size = self._freq_to_minutes(freq)
                
            # Copier et trier par index temporel
            df_copy = df.copy().sort_index()
            
            # Créer une liste pour stocker les klines glissantes
            rolling_klines = []
            
            # Parcourir chaque timestamp possible
            for i in range(len(df_copy) - window_size + 1):
                window_start_idx = i
                window_end_idx = i + window_size
                
                # Extraire la fenêtre
                window = df_copy.iloc[window_start_idx:window_end_idx]
                
                # Vérifier que la fenêtre couvre bien la période attendue
                actual_duration = (window.index[-1] - window.index[0]).total_seconds() / 60
                expected_duration = window_size - 1  # -1 car on inclut les deux bornes
                
                if actual_duration != expected_duration:
                    # Skip si la fenêtre n'est pas continue (données manquantes)
                    continue
                    
                # Calculer les agrégations pour cette fenêtre
                kline = {
                    'open_price': window['open_price'].iloc[0],
                    'high_price': window['high_price'].max(),
                    'low_price': window['low_price'].min(),
                    'close_price': window['close_price'].iloc[-1],
                    'base_asset_volume': window['base_asset_volume'].sum(),
                    'quote_asset_volume': window['quote_asset_volume'].sum(),
                    'number_of_trades': window['number_of_trades'].sum(),
                    'taker_buy_base_asset_volume': window['taker_buy_base_asset_volume'].sum(),
                    'taker_buy_quote_asset_volume': window['taker_buy_quote_asset_volume'].sum(),
                    'kline_open_time': window.index[0],
                    'kline_close_time': window.index[-1] + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)
                }
                
                rolling_klines.append(kline)
                
            # Créer le DataFrame résultat
            if rolling_klines:
                result_df = pd.DataFrame(rolling_klines)
                result_df.set_index('kline_open_time', inplace=True)
                
                # Ajouter les colonnes supplémentaires
                if 'pair' in df_copy.columns:
                    result_df['pair'] = df_copy['pair'].iloc[0]
                result_df['is_kline_closed'] = True
                
                logger.debug(f"Generated {len(result_df)} rolling klines with {window_size}m window")
                return result_df
            else:
                logger.warning("No rolling klines generated (insufficient data or gaps)")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error generating rolling klines: {e}")
            raise DataError(f"Failed to generate rolling klines: {e}", original_exception=e)
            
    def _freq_to_minutes(self, freq: str) -> int:
        """
        Convertit une fréquence pandas en nombre de minutes.
        
        Args:
            freq: Fréquence pandas (ex: '5T', '1H')
            
        Returns:
            Nombre de minutes
        """
        # Mapping des unités vers minutes
        unit_mapping = {
            'T': 1,      # Minutes
            'H': 60,     # Heures
            'D': 1440,   # Jours
            'W': 10080   # Semaines
        }
        
        # Extraire le nombre et l'unité
        import re
        match = re.match(r'(\d+)([THDW])', freq)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            return number * unit_mapping.get(unit, 1)
        else:
            # Par défaut, retourner 1 minute
            return 1
            
    def validate_klines(
        self,
        df: pd.DataFrame,
        check_continuity: bool = True,
        max_gap_minutes: int = 5
    ) -> Tuple[bool, List[str]]:
        """
        Valide un DataFrame de klines.
        
        Args:
            df: DataFrame à valider
            check_continuity: Vérifier la continuité temporelle
            max_gap_minutes: Gap maximum autorisé entre klines (en minutes)
            
        Returns:
            Tuple (is_valid, list_of_errors)
        """
        errors = []
        
        # Vérifier que le DataFrame n'est pas vide
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
            
        # Vérifier l'index temporel
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("DataFrame must have a DatetimeIndex")
            
        # Vérifier les colonnes requises
        required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'base_asset_volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            
        # Vérifier les valeurs nulles
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            
        # Vérifier la cohérence OHLC
        invalid_ohlc = df[
            (df['high_price'] < df['low_price']) |
            (df['high_price'] < df['open_price']) |
            (df['high_price'] < df['close_price']) |
            (df['low_price'] > df['open_price']) |
            (df['low_price'] > df['close_price'])
        ]
        if not invalid_ohlc.empty:
            errors.append(f"Invalid OHLC relationships in {len(invalid_ohlc)} rows")
            
        # Vérifier les volumes négatifs
        if 'base_asset_volume' in df.columns and (df['base_asset_volume'] < 0).any():
            errors.append("Negative volumes found")
            
        # Vérifier la continuité temporelle si demandé
        if check_continuity and len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            max_gap = pd.Timedelta(minutes=max_gap_minutes)
            
            large_gaps = time_diffs[time_diffs > max_gap]
            if not large_gaps.empty:
                errors.append(f"Found {len(large_gaps)} gaps larger than {max_gap_minutes} minutes")
                
        # Vérifier l'ordre temporel
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonically increasing")
            
        # Vérifier les doublons dans l'index
        if df.index.has_duplicates:
            errors.append(f"Found {df.index.duplicated().sum()} duplicate timestamps")
            
        is_valid = len(errors) == 0
        return is_valid, errors
        
    def fill_missing_klines(
        self,
        df: pd.DataFrame,
        freq: str = '1T',
        method: str = 'forward'
    ) -> pd.DataFrame:
        """
        Remplit les klines manquantes dans un DataFrame.
        
        Args:
            df: DataFrame avec potentiellement des données manquantes
            freq: Fréquence attendue des klines
            method: Méthode de remplissage ('forward', 'interpolate', 'zero')
            
        Returns:
            DataFrame avec klines manquantes remplies
        """
        if df.empty:
            return df
            
        # Créer un index temporel complet
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        
        # Réindexer le DataFrame
        df_filled = df.reindex(full_index)
        
        # Remplir les valeurs manquantes selon la méthode
        if method == 'forward':
            # Forward fill : utiliser la dernière valeur connue
            df_filled['open_price'] = df_filled['open_price'].fillna(method='ffill')
            df_filled['high_price'] = df_filled['high_price'].fillna(method='ffill')
            df_filled['low_price'] = df_filled['low_price'].fillna(method='ffill')
            df_filled['close_price'] = df_filled['close_price'].fillna(method='ffill')
            # Les volumes sont mis à 0 pour les périodes manquantes
            df_filled['base_asset_volume'] = df_filled['base_asset_volume'].fillna(0)
            df_filled['quote_asset_volume'] = df_filled['quote_asset_volume'].fillna(0)
            df_filled['number_of_trades'] = df_filled['number_of_trades'].fillna(0)
            
        elif method == 'interpolate':
            # Interpolation linéaire pour les prix
            df_filled['open_price'] = df_filled['open_price'].interpolate(method='linear')
            df_filled['high_price'] = df_filled['high_price'].interpolate(method='linear')
            df_filled['low_price'] = df_filled['low_price'].interpolate(method='linear')
            df_filled['close_price'] = df_filled['close_price'].interpolate(method='linear')
            # Volumes à 0
            df_filled['base_asset_volume'] = df_filled['base_asset_volume'].fillna(0)
            
        elif method == 'zero':
            # Remplir avec des valeurs neutres
            price_cols = ['open_price', 'high_price', 'low_price', 'close_price']
            # Utiliser le dernier prix connu pour tous les prix
            last_price = df['close_price'].iloc[-1] if not df.empty else 0
            for col in price_cols:
                df_filled[col] = df_filled[col].fillna(last_price)
            # Volumes à 0
            volume_cols = ['base_asset_volume', 'quote_asset_volume', 'number_of_trades']
            for col in volume_cols:
                if col in df_filled.columns:
                    df_filled[col] = df_filled[col].fillna(0)
                    
        # Copier les autres colonnes
        for col in df.columns:
            if col not in df_filled.columns:
                df_filled[col] = df[col].iloc[0] if not df.empty else None
                
        logger.debug(f"Filled {len(full_index) - len(df)} missing klines using {method} method")
        return df_filled
        
    def merge_klines(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        prefer: str = 'newer'
    ) -> pd.DataFrame:
        """
        Fusionne deux DataFrames de klines.
        
        Args:
            df1: Premier DataFrame
            df2: Deuxième DataFrame
            prefer: Quelle source préférer en cas de conflit ('newer', 'older', 'df1', 'df2')
            
        Returns:
            DataFrame fusionné
        """
        if df1.empty:
            return df2
        if df2.empty:
            return df1
            
        # Combiner les DataFrames
        combined = pd.concat([df1, df2])
        
        # Gérer les doublons selon la préférence
        if prefer == 'newer':
            # Supposer que df2 est plus récent
            combined = combined[~combined.index.duplicated(keep='last')]
        elif prefer == 'older':
            # Supposer que df1 est plus ancien
            combined = combined[~combined.index.duplicated(keep='first')]
        elif prefer == 'df1':
            # Préférer df1
            combined = pd.concat([df2, df1])
            combined = combined[~combined.index.duplicated(keep='last')]
        elif prefer == 'df2':
            # Préférer df2
            combined = pd.concat([df1, df2])
            combined = combined[~combined.index.duplicated(keep='last')]
            
        # Trier par index temporel
        combined.sort_index(inplace=True)
        
        logger.debug(f"Merged klines: {len(df1)} + {len(df2)} -> {len(combined)} rows")
        return combined
        
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule des statistiques sur un DataFrame de klines.
        
        Args:
            df: DataFrame de klines
            
        Returns:
            Dictionnaire avec les statistiques
        """
        if df.empty:
            return {}
            
        stats = {
            'count': len(df),
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'duration_days': (df.index.max() - df.index.min()).days,
            'avg_price': df['close_price'].mean(),
            'min_price': df['low_price'].min(),
            'max_price': df['high_price'].max(),
            'total_volume': df['base_asset_volume'].sum(),
            'avg_volume': df['base_asset_volume'].mean(),
            'price_volatility': df['close_price'].std(),
            'volume_volatility': df['base_asset_volume'].std()
        }
        
        # Calculer le rendement si possible
        if len(df) > 1:
            returns = df['close_price'].pct_change().dropna()
            stats['avg_return'] = returns.mean()
            stats['return_volatility'] = returns.std()
            stats['total_return'] = (df['close_price'].iloc[-1] / df['close_price'].iloc[0] - 1)
            
        return stats


# Exemple d'utilisation
def example_usage():
    """Exemple d'utilisation du KlineProcessor."""
    
    # Créer des données de test
    dates = pd.date_range(start='2024-01-01 00:00:00', periods=60, freq='1T')
    df = pd.DataFrame({
        'open_price': np.random.uniform(40000, 41000, 60),
        'high_price': np.random.uniform(40500, 41500, 60),
        'low_price': np.random.uniform(39500, 40500, 60),
        'close_price': np.random.uniform(40000, 41000, 60),
        'base_asset_volume': np.random.uniform(0.1, 1.0, 60),
        'quote_asset_volume': np.random.uniform(4000, 41000, 60),
        'number_of_trades': np.random.randint(10, 100, 60),
        'taker_buy_base_asset_volume': np.random.uniform(0.05, 0.5, 60),
        'taker_buy_quote_asset_volume': np.random.uniform(2000, 20000, 60),
        'pair': 'BTCUSDC'
    }, index=dates)
    
    # Ajuster pour avoir des OHLC cohérents
    df['high_price'] = df[['open_price', 'high_price', 'close_price']].max(axis=1)
    df['low_price'] = df[['open_price', 'low_price', 'close_price']].min(axis=1)
    
    processor = KlineProcessor()
    
    # Générer des klines 5m standard
    klines_5m = processor.resample_klines(df, '5m', rolling=False)
    print(f"Generated {len(klines_5m)} standard 5m klines")
    print(klines_5m.head())
    
    # Générer des klines 3m glissantes
    klines_3m_rolling = processor.resample_klines(df, '3m', rolling=True)
    print(f"\nGenerated {len(klines_3m_rolling)} rolling 3m klines")
    print(klines_3m_rolling.head())
    
    # Valider les klines
    is_valid, errors = processor.validate_klines(klines_5m)
    print(f"\nValidation result: {'Valid' if is_valid else 'Invalid'}")
    if errors:
        print(f"Errors: {errors}")
        
    # Calculer des statistiques
    stats = processor.calculate_statistics(df)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_usage()