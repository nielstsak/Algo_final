# src/strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Type

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import create_model, ValidationError

from src.core.constants import Trading, Kline, DataFrameCols
from src.core.exceptions import (
    StrategyError,
    InvalidStrategyParamsError,
    SignalGenerationError,
    IndicatorCalculationError
)
from src.data.enriched_dataframe import EnrichedDataFrame
from src.strategies.params import BaseFixedParams, BaseOptimizableParams


class BaseStrategy(ABC):
    """
    Classe abstraite de base pour toutes les stratégies de trading.
    Fournit une interface standardisée et une configuration déclarative via Pydantic.
    """
    
    # Attributs de classe pour identifier la stratégie
    name: str = "BaseStrategy"
    version: str = "1.0.0"
    description: str = "Abstract base strategy class"
    
    # Intervalles de temps requis par la stratégie
    required_timeframes: List[str] = [Kline.INTERVAL_1MINUTE]
    
    # Nombre minimum de périodes requises pour le calcul
    min_required_periods: int = 100
    
    # 'Slots' pour les modèles Pydantic. A surcharger dans les sous-classes.
    fixed_params_model: Type[BaseFixedParams] = BaseFixedParams
    optimizable_params_model: Type[BaseOptimizableParams] = BaseOptimizableParams
    
    # Paramètres par défaut legacy (maintenu pour compatibilité)
    default_params: Dict[str, Any] = {}
    
    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialise la stratégie avec les paramètres fournis.
        Fusionne les paramètres des modèles Pydantic, les params par défaut, 
        le dictionnaire `params` et les `kwargs`, puis valide le tout.
        
        Args:
            params: Dictionnaire des paramètres de la stratégie.
            **kwargs: Paramètres additionnels passés directement.
        """
        # 1. Obtenir les valeurs par défaut des modèles Pydantic
        fixed_defaults = self.fixed_params_model().model_dump()
        optimizable_defaults = self.optimizable_params_model().model_dump()

        # 2. Fusionner les paramètres avec l'ordre de priorité suivant :
        #    Pydantic < default_params < dictionnaire `params` < kwargs
        all_defaults = {**fixed_defaults, **optimizable_defaults, **self.default_params}
        self.params = {**all_defaults, **(params or {}), **kwargs}
        
        self.validate_params()
        
        # Initialisation des caches et logging
        self._indicators_cache: Optional[pd.DataFrame] = None
        self._signals_cache: Optional[pd.DataFrame] = None
        
        logger.debug(f"Initialized {self.name} with params: {self.params}")

    def validate_params(self) -> None:
        """
        Valide les paramètres de la stratégie en utilisant un modèle Pydantic 
        combiné dynamiquement à partir des modèles `fixed` et `optimizable`.
        Lève InvalidStrategyParamsError en cas d'échec de la validation.
        """
        # Créer un modèle Pydantic combiné à la volée
        CombinedParamsModel = create_model(
            'CombinedParamsModel',
            __base__=(self.fixed_params_model, self.optimizable_params_model)
        )

        try:
            # Valider l'ensemble des paramètres de l'instance
            CombinedParamsModel(**self.params)
        except ValidationError as e:
            # Encapsuler l'erreur Pydantic dans une exception personnalisée
            raise InvalidStrategyParamsError(
                f"Strategy parameter validation failed for {self.__class__.__name__}: {e.errors()}",
                strategy_name=self.name
            ) from e

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)
        
    def _prepare_indicator_data(self, data: Union[Dict[str, pd.DataFrame], "EnrichedDataFrame"]) -> pd.DataFrame:
        indicator_freq = self.get_param('indicator_frequency')
        if not indicator_freq:
            raise InvalidStrategyParamsError(
                strategy_name=self.name,
                parameter_name='indicator_frequency',
                details="Le paramètre 'indicator_frequency' doit être défini."
            )

        if isinstance(data, EnrichedDataFrame):
            try:
                logger.debug(f"[{self.name}] Extraction de la vue '{indicator_freq}' depuis EnrichedDataFrame.")
                return data.get_view(indicator_freq)
            except ValueError as e:
                raise StrategyError(
                    f"Échec de l'obtention de la vue pour la fréquence '{indicator_freq}'. {e}",
                    strategy_name=self.name, original_exception=e
                )
        elif isinstance(data, dict):
            logger.debug(f"[{self.name}] Utilisation du DataFrame '{indicator_freq}' depuis le dictionnaire.")
            df = data.get(indicator_freq)
            if df is None or df.empty:
                raise StrategyError(
                    f"DataFrame requis pour '{indicator_freq}' manquant ou vide.",
                    strategy_name=self.name
                )
            return df
        else:
            raise StrategyError(f"Type de données non supporté: {type(data)}.", strategy_name=self.name)
            
    @abstractmethod
    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], "EnrichedDataFrame"]) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def generate_signals(self, indicators: pd.DataFrame) -> pd.DataFrame:
        pass
        
    def execute(self, klines: Union[Dict[str, pd.DataFrame], "EnrichedDataFrame"], current_position: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            indicators_df = self.calculate_indicators(klines)
            if indicators_df is None or not isinstance(indicators_df, pd.DataFrame):
                 raise StrategyError(f"calculate_indicators pour {self.name} n'a pas retourné un DataFrame valide.", strategy_name=self.name)
            self._indicators_cache = indicators_df
            signals_df = self.generate_signals(indicators_df)
            self._signals_cache = signals_df
            formatted_signals = self._format_signals(signals_df=signals_df, klines_data=klines, current_position=current_position)
            return formatted_signals
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la stratégie {self.name}: {e}", exc_info=True)
            raise StrategyError(f"L'exécution de la stratégie a échoué: {e}", strategy_name=self.name, original_exception=e)
            
    def _format_signals(self, signals_df: pd.DataFrame, klines_data: Union[Dict[str, pd.DataFrame], "EnrichedDataFrame"], current_position: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        formatted_signals = []
        execution_tf = Kline.INTERVAL_1MINUTE
        
        main_df: pd.DataFrame
        try:
            if isinstance(klines_data, EnrichedDataFrame):
                main_df = klines_data.get_view(execution_tf)
            else:
                main_df = klines_data.get(self.get_param('indicator_frequency'))
                if main_df is None:
                    raise StrategyError("Impossible de déterminer le DataFrame principal pour le formatage.")
        except (ValueError, StrategyError) as e:
            logger.warning(f"Impossible d'obtenir '{execution_tf}'. Repli sur la fréquence des indicateurs. Raison: {e}")
            main_df = self._prepare_indicator_data(klines_data)

        if DataFrameCols.SIGNAL.value not in signals_df.columns:
            if 'entry_long' in signals_df.columns and 'entry_short' in signals_df.columns:
                signals_df[DataFrameCols.SIGNAL.value] = 0
                signals_df.loc[signals_df['entry_long'], DataFrameCols.SIGNAL.value] = 1
                signals_df.loc[signals_df['entry_short'], DataFrameCols.SIGNAL.value] = -1
            else:
                logger.warning(f"'signal' manquant dans signals_df pour {self.name}.")
                return []
            
        active_signals = signals_df[signals_df[DataFrameCols.SIGNAL.value] != 0].tail(10)
        
        for idx, row in active_signals.iterrows():
            try:
                current_bar = main_df.loc[idx]
            except KeyError:
                try:
                    nearest_idx_pos = main_df.index.get_indexer([idx], method='nearest')[0]
                    current_bar = main_df.iloc[nearest_idx_pos]
                except (IndexError, KeyError):
                    logger.warning(f"Index proche non trouvé pour {idx} dans {self.name}. Signal ignoré.")
                    continue
            
            if DataFrameCols.CLOSE.value not in current_bar:
                logger.error(f"'{DataFrameCols.CLOSE.value}' manquant. Impossible de déterminer le prix.")
                continue

            signal = {
                'timestamp': idx, 'strategy_name': self.name, 'strategy_version': self.version,
                'signal_type': self._get_signal_type(row[DataFrameCols.SIGNAL.value]),
                'side': Trading.SIDE_BUY if row[DataFrameCols.SIGNAL.value] > 0 else Trading.SIDE_SELL,
                'entry_price': row.get(DataFrameCols.ENTRY_PRICE.value, current_bar[DataFrameCols.CLOSE.value]),
                'stop_loss': row.get(DataFrameCols.STOP_LOSS.value),
                'take_profit': row.get(DataFrameCols.TAKE_PROFIT.value),
                'confidence': row.get(DataFrameCols.CONFIDENCE.value, 0.5),
                'metadata': {
                    'indicators': self._get_signal_metadata(idx, row),
                    'indicator_timeframe': self.get_param('indicator_frequency'),
                    'execution_timeframe': main_df.index.freqstr if hasattr(main_df.index, 'freqstr') else execution_tf,
                    'current_position': current_position
                }
            }
            signal = self._validate_signal_prices(signal, current_bar)
            formatted_signals.append(signal)
            
        return formatted_signals
        
    def _get_signal_type(self, signal_value: float) -> str:
        if signal_value > 0: return Trading.SIGNAL_TYPE_LONG
        elif signal_value < 0: return Trading.SIGNAL_TYPE_SHORT
        else: return Trading.SIGNAL_TYPE_NEUTRAL
            
    def _get_signal_metadata(self, timestamp: pd.Timestamp, signal_row: pd.Series) -> Dict[str, Any]:
        metadata = {}
        if self._indicators_cache is not None and not self._indicators_cache.empty and timestamp in self._indicators_cache.index:
            tf_data = self._indicators_cache.loc[timestamp].to_dict()
            metadata['indicators'] = {k: v for k, v in tf_data.items() if pd.notna(v)}
        return metadata
        
    def _validate_signal_prices(self, signal: Dict[str, Any], current_bar: pd.Series) -> Dict[str, Any]:
        entry_price = signal['entry_price']
        if not isinstance(entry_price, (int, float)) or np.isnan(entry_price):
            logger.warning(f"Prix d'entrée invalide ({entry_price}). Utilisation du prix de clôture.")
            entry_price = current_bar[DataFrameCols.CLOSE.value]
            signal['entry_price'] = entry_price
            if not isinstance(entry_price, (int, float)) or np.isnan(entry_price):
                logger.error(f"Prix d'entrée de repli également invalide.")
                return signal

        def is_valid_price(p): return p is not None and isinstance(p, (int, float)) and not np.isnan(p)

        sl, tp = signal.get('stop_loss'), signal.get('take_profit')
        if signal['signal_type'] == Trading.SIGNAL_TYPE_LONG:
            if is_valid_price(sl) and sl >= entry_price: signal['stop_loss'] = entry_price * 0.98
            if is_valid_price(tp) and tp <= entry_price: signal['take_profit'] = entry_price * 1.02
        elif signal['signal_type'] == Trading.SIGNAL_TYPE_SHORT:
            if is_valid_price(sl) and sl <= entry_price: signal['stop_loss'] = entry_price * 1.02
            if is_valid_price(tp) and tp >= entry_price: signal['take_profit'] = entry_price * 0.98
        return signal
        
    def get_required_lookback(self) -> int:
        return self.min_required_periods
        
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name, 'version': self.version, 'description': self.description,
            'required_timeframes': self.required_timeframes, 'min_required_periods': self.min_required_periods,
            'parameters': self.params, 'default_parameters': self.default_params
        }
        
    def reset(self) -> None:
        self._indicators_cache, self._signals_cache = None, None
        logger.debug(f"Reset strategy {self.name} caches")
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
