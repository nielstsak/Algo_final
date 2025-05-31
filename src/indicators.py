# src/indicators.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from loguru import logger
import pandas_ta as ta
from datetime import datetime

from src.core.config import settings
from src.core.constants import Kline
from src.strategies.strategy_loader import strategy_loader


class IndicatorCalculator:
    """
    Calcule les indicateurs techniques requis par toutes les stratégies enregistrées.
    Gère le calcul multi-timeframe et optimise les performances.
    """
    
    def __init__(self):
        """Initialise le calculateur d'indicateurs."""
        self.required_indicators = self._extract_required_indicators()
        logger.info(f"IndicatorCalculator initialized with {len(self.required_indicators)} indicators")
        
    def _extract_required_indicators(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extrait les indicateurs requis par toutes les stratégies enregistrées.
        
        Returns:
            Dict avec les indicateurs groupés par fréquence
        """
        indicators_by_freq = {}
        
        # Parcourir toutes les stratégies enregistrées
        for strategy_name, strategy_class in strategy_loader.loaded_strategies.items():
            try:
                # Créer une instance temporaire pour analyser les besoins
                temp_instance = strategy_class()
                
                # Analyser les paramètres pour identifier les indicateurs
                params = temp_instance.default_params
                
                # Fréquence des indicateurs
                freq = params.get('indicator_frequency', '1h')
                if freq not in indicators_by_freq:
                    indicators_by_freq[freq] = []
                
                # Extraire les indicateurs selon le type de stratégie
                indicators = self._extract_strategy_indicators(strategy_name, params)
                
                # Ajouter les indicateurs uniques
                for indicator in indicators:
                    if not self._indicator_exists(indicators_by_freq[freq], indicator):
                        indicators_by_freq[freq].append(indicator)
                        
            except Exception as e:
                logger.warning(f"Failed to extract indicators from {strategy_name}: {e}")
                
        return indicators_by_freq
        
    def _extract_strategy_indicators(self, strategy_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrait les indicateurs spécifiques d'une stratégie.
        
        Args:
            strategy_name: Nom de la stratégie
            params: Paramètres de la stratégie
            
        Returns:
            Liste des indicateurs requis
        """
        indicators = []
        
        # Patterns communs pour identifier les indicateurs
        if 'sma' in strategy_name.lower() or 'ma_' in params:
            # SMA/EMA indicators
            for key, value in params.items():
                if 'period' in key and isinstance(value, int):
                    if 'fast' in key:
                        indicators.append({
                            'type': 'sma',
                            'params': {'length': value},
                            'name': f'SMA_{value}'
                        })
                    elif 'slow' in key:
                        indicators.append({
                            'type': 'sma',
                            'params': {'length': value},
                            'name': f'SMA_{value}'
                        })
                    elif 'ma_' in key:
                        indicators.append({
                            'type': 'sma',
                            'params': {'length': value},
                            'name': f'MA_{value}'
                        })
                        
        # RSI
        if 'rsi' in strategy_name.lower() or 'rsi_period' in params:
            rsi_period = params.get('rsi_period', 14)
            indicators.append({
                'type': 'rsi',
                'params': {'length': rsi_period},
                'name': f'RSI_{rsi_period}'
            })
            
        # Bollinger Bands
        if 'bbands' in strategy_name.lower() or 'bbands_period' in params:
            bb_period = params.get('bbands_period', 20)
            bb_std = params.get('bbands_std_dev', 2.0)
            indicators.append({
                'type': 'bbands',
                'params': {'length': bb_period, 'std': bb_std},
                'name': f'BB_{bb_period}_{bb_std}'
            })
            
        # PSAR
        if 'psar' in strategy_name.lower() or 'psar_step' in params:
            psar_step = params.get('psar_step', 0.02)
            psar_max = params.get('psar_max_step', 0.2)
            indicators.append({
                'type': 'psar',
                'params': {'af0': psar_step, 'af': psar_step, 'max_af': psar_max},
                'name': f'PSAR_{psar_step}_{psar_max}'
            })
            
        # ATR (presque toutes les stratégies l'utilisent pour SL/TP)
        if 'atr_period' in params or 'atr_period_sl_tp' in params:
            atr_period = params.get('atr_period_sl_tp', params.get('atr_period', 14))
            indicators.append({
                'type': 'atr',
                'params': {'length': atr_period},
                'name': f'ATR_{atr_period}'
            })
            
        # Volume MA
        if 'volume_ma_period' in params:
            vol_period = params.get('volume_ma_period', 20)
            indicators.append({
                'type': 'sma',
                'params': {'length': vol_period, 'column': 'volume'},
                'name': f'Volume_MA_{vol_period}'
            })
            
        # ADX
        if 'adx' in strategy_name.lower() or 'adx_period' in params:
            adx_period = params.get('adx_period', 14)
            indicators.append({
                'type': 'adx',
                'params': {'length': adx_period},
                'name': f'ADX_{adx_period}'
            })
            
        return indicators
        
    def _indicator_exists(self, indicators_list: List[Dict], indicator: Dict) -> bool:
        """Vérifie si un indicateur existe déjà dans la liste."""
        for existing in indicators_list:
            if (existing['type'] == indicator['type'] and 
                existing['params'] == indicator['params']):
                return True
        return False
        
    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        base_freq: str = '1m',
        target_freqs: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calcule tous les indicateurs requis pour toutes les fréquences.
        
        Args:
            df: DataFrame avec les données OHLCV de base
            base_freq: Fréquence de base des données
            target_freqs: Fréquences cibles (si None, utilise toutes les fréquences requises)
            
        Returns:
            Dict avec DataFrames par fréquence contenant tous les indicateurs
        """
        if target_freqs is None:
            target_freqs = list(self.required_indicators.keys())
            
        results = {}
        
        # Calculer pour chaque fréquence
        for freq in target_freqs:
            logger.info(f"Calculating indicators for frequency: {freq}")
            
            # Resampler si nécessaire
            if freq == base_freq:
                df_freq = df.copy()
            else:
                df_freq = self._resample_ohlcv(df, base_freq, freq)
                
            if df_freq.empty:
                logger.warning(f"Empty DataFrame after resampling to {freq}")
                continue
                
            # Calculer les indicateurs pour cette fréquence
            if freq in self.required_indicators:
                for indicator_config in self.required_indicators[freq]:
                    try:
                        df_freq = self._calculate_indicator(df_freq, indicator_config)
                    except Exception as e:
                        logger.error(f"Failed to calculate {indicator_config['name']}: {e}")
                        
            results[freq] = df_freq
            
        return results
        
    def _resample_ohlcv(self, df: pd.DataFrame, from_freq: str, to_freq: str) -> pd.DataFrame:
        """
        Resample OHLCV data from one frequency to another.
        
        Args:
            df: Source DataFrame
            from_freq: Source frequency
            to_freq: Target frequency
            
        Returns:
            Resampled DataFrame
        """
        # Mapping des fréquences
        freq_map = {
            '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T',
            '30m': '30T', '1h': '1H', '2h': '2H', '4h': '4H',
            '6h': '6H', '8h': '8H', '12h': '12H', '1d': '1D',
            '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        pandas_freq = freq_map.get(to_freq, to_freq)
        
        # Règles d'agrégation pour OHLCV
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Ajouter d'autres colonnes si présentes
        for col in df.columns:
            if col not in agg_rules and col not in ['pair', 'symbol']:
                if 'volume' in col.lower():
                    agg_rules[col] = 'sum'
                else:
                    agg_rules[col] = 'last'
                    
        # Resampler
        try:
            df_resampled = df.resample(pandas_freq).agg(agg_rules)
            df_resampled.dropna(subset=['close'], inplace=True)
            
            # Préserver certaines colonnes
            if 'pair' in df.columns:
                df_resampled['pair'] = df['pair'].iloc[0]
                
            return df_resampled
            
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return pd.DataFrame()
            
    def _calculate_indicator(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calcule un indicateur spécifique.
        
        Args:
            df: DataFrame source
            config: Configuration de l'indicateur
            
        Returns:
            DataFrame avec l'indicateur ajouté
        """
        indicator_type = config['type']
        params = config['params']
        name_prefix = config['name']
        
        if indicator_type == 'sma':
            column = params.get('column', 'close')
            length = params['length']
            df[f'{name_prefix}'] = ta.sma(df[column], length=length)
            
        elif indicator_type == 'ema':
            column = params.get('column', 'close')
            length = params['length']
            df[f'{name_prefix}'] = ta.ema(df[column], length=length)
            
        elif indicator_type == 'rsi':
            length = params['length']
            df[f'{name_prefix}'] = ta.rsi(df['close'], length=length)
            
        elif indicator_type == 'bbands':
            length = params['length']
            std = params['std']
            bbands = ta.bbands(df['close'], length=length, std=std)
            if bbands is not None:
                df[f'{name_prefix}_LOWER'] = bbands[f'BBL_{length}_{std}']
                df[f'{name_prefix}_MIDDLE'] = bbands[f'BBM_{length}_{std}']
                df[f'{name_prefix}_UPPER'] = bbands[f'BBU_{length}_{std}']
                df[f'{name_prefix}_BANDWIDTH'] = bbands[f'BBB_{length}_{std}']
                
        elif indicator_type == 'atr':
            length = params['length']
            df[f'{name_prefix}'] = ta.atr(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=length
            )
            
        elif indicator_type == 'psar':
            af0 = params.get('af0', 0.02)
            af = params.get('af', 0.02)
            max_af = params.get('max_af', 0.2)
            psar = ta.psar(
                high=df['high'],
                low=df['low'],
                af0=af0,
                af=af,
                max_af=max_af
            )
            if psar is not None:
                df[f'{name_prefix}_LONG'] = psar[f'PSARl_{af0}_{max_af}']
                df[f'{name_prefix}_SHORT'] = psar[f'PSARs_{af0}_{max_af}']
                
        elif indicator_type == 'adx':
            length = params['length']
            adx = ta.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=length
            )
            if adx is not None:
                df[f'{name_prefix}'] = adx[f'ADX_{length}']
                df[f'{name_prefix}_DMP'] = adx[f'DMP_{length}']
                df[f'{name_prefix}_DMN'] = adx[f'DMN_{length}']
                
        return df
        
    def get_required_indicators_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur les indicateurs requis.
        
        Returns:
            Dict avec les infos détaillées
        """
        info = {
            'total_indicators': sum(len(indicators) for indicators in self.required_indicators.values()),
            'frequencies': list(self.required_indicators.keys()),
            'by_frequency': {}
        }
        
        for freq, indicators in self.required_indicators.items():
            info['by_frequency'][freq] = {
                'count': len(indicators),
                'indicators': [ind['name'] for ind in indicators]
            }
            
        return info


# Instance globale
indicator_calculator = IndicatorCalculator()